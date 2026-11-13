#!/usr/bin/env python3
"""Train two fixed three-epoch LoRAs and evaluate paired entity transfer directions."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from entityqa_common import rows, save, write_rows, sha, now


def idle(gpu):
    device=Path('/dev/nvidia'+gpu)
    if device.exists() and not os.access(device,os.R_OK|os.W_OK):
        raise RuntimeError(f'GPU {gpu} device permission denied for uid {os.getuid()}; waiting for authorized access')
    try:
        values=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'],text=True,stderr=subprocess.STDOUT,timeout=15).strip().split(',')
    except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as error:
        raise RuntimeError(f'GPU {gpu} unavailable: {error}') from error
    if int(values[0])>1000 or int(values[1])>5:
        raise RuntimeError(f'GPU {gpu} is in use: {values}; no existing task will be stopped')


def command(args,modality,smoke=False):
    root,work=args.root.resolve(),args.work.resolve()
    output=work/('smoke' if smoke else 'train')/modality
    cmd=[str(root/'.venv/bin/swift'),'sft','--model',str(args.model.resolve()),
        '--model_type','qwen3_vl','--dataset',str(work/f'data/train_{modality.lower()}.jsonl'),
        '--tuner_type','lora','--lora_rank','64','--lora_alpha','128','--lora_dropout','0.05',
        '--target_modules','all-linear','--freeze_vit','true','--freeze_aligner','true',
        '--torch_dtype','bfloat16','--attn_impl','sdpa',
        '--per_device_train_batch_size','4','--gradient_accumulation_steps','4',
        '--learning_rate','1e-4','--num_train_epochs','3','--lr_scheduler_type','cosine',
        '--warmup_ratio','0.05','--gradient_checkpointing','true','--max_length','2048',
        '--max_pixels','262144','--save_strategy','epoch','--save_total_limit','1',
        '--save_only_model','true','--split_dataset_ratio','0','--logging_steps','1',
        '--seed','42','--data_seed','42','--add_version','false','--output_dir',str(output),
        '--use_chat_template','true','--loss_scale','default','--strict','true',
        '--enable_thinking','false','--report_to','none','--dataset_num_proc','1',
        '--dataloader_num_workers','0','--dataloader_persistent_workers','false',
        '--optim','adamw_torch','--packing','false','--padding_free','false','--weight_decay','0.01']
    if smoke:cmd+=['--max_steps','1','--save_strategy','steps','--save_steps','1','--warmup_ratio','0']
    return cmd,output


def checkpoint(folder,smoke=False):
    import torch
    from safetensors.torch import load_file
    checkpoints=sorted(folder.glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
    if not checkpoints:raise RuntimeError(f'No checkpoint in {folder}')
    ck=checkpoints[-1]
    state=json.loads((ck/'trainer_state.json').read_text())
    assert state['global_step']>0
    if not smoke:assert abs(state['epoch']-3)<1e-6,state['epoch']
    losses=[r['loss'] for r in state['log_history'] if 'loss' in r]
    assert losses and all(math.isfinite(float(x)) for x in losses)
    assert all(math.isfinite(float(r['grad_norm'])) for r in state['log_history'] if 'grad_norm' in r)
    weights=load_file(str(ck/'adapter_model.safetensors'),device='cpu')
    assert weights and all(torch.isfinite(t).all() for t in weights.values())
    assert all('.language_model.' in k and '.visual.' not in k for k in weights)
    assert any(torch.count_nonzero(t)>0 for k,t in weights.items() if '.lora_B.' in k)
    save(folder/'validation.json',dict(checkpoint=str(ck),epoch=state['epoch'],
        global_step=state['global_step'],first_loss=losses[0],last_loss=losses[-1],
        tensor_count=len(weights),adapter_sha256=sha(ck/'adapter_model.safetensors'),
        finite=True,vision_frozen=True,nonzero_lora_B=True))
    return ck


def merge(base,adapter,dest):
    import torch
    from safetensors.torch import load_file,save_file
    torch.set_num_threads(8)
    if (dest/'merge_manifest.json').exists():
        m=json.loads((dest/'merge_manifest.json').read_text())
        assert m['adapter_sha256']==sha(adapter/'adapter_model.safetensors')
        assert all(sha(dest/name)==h for name,h in m['files'].items())
        return
    dest.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((adapter/'adapter_config.json').read_text())
    assert not any(cfg.get(k) for k in ['use_dora','use_rslora','rank_pattern','alpha_pattern','modules_to_save'])
    assert cfg['bias']=='none'
    weights=load_file(str(adapter/'adapter_model.safetensors'),device='cpu')
    updates={}
    for k,a in weights.items():
        if not k.endswith('.lora_A.weight'):continue
        assert k.startswith('base_model.model.') and '.language_model.' in k
        b=weights[k.replace('.lora_A.weight','.lora_B.weight')]
        key=k[len('base_model.model.'):].replace('.lora_A.weight','.weight')
        updates[key]=(a,b)
    assert len(updates)*2==len(weights)
    touched=[]
    base_hashes={}
    for source in sorted(base.glob('model-*.safetensors')):
        base_hashes[source.name]=sha(source)
        tensors=load_file(str(source),device='cpu')
        for k in sorted(set(tensors)&set(updates)):
            a,b=updates[k]
            delta=(b.float()@a.float())*(cfg['lora_alpha']/cfg['r'])
            assert delta.shape==tensors[k].shape and torch.isfinite(delta).all()
            tensors[k]=(tensors[k].float()+delta).to(tensors[k].dtype).contiguous()
            touched.append(k)
        save_file(tensors,str(dest/source.name),metadata={'format':'pt'})
        del tensors
    assert set(touched)==set(updates)
    for source in base.iterdir():
        if source.is_file() and source.suffix in ['.json','.jinja','.txt','.model']:
            shutil.copy2(source,dest/source.name)
    save(dest/'merge_manifest.json',dict(base=str(base),adapter=str(adapter),
        method='BF16 round(base.float + B.float @ A.float * alpha/r), CPU-only serving conversion',
        numerical_note='BF16 rounding can differ slightly from unmerged runtime LoRA',
        adapter_sha256=sha(adapter/'adapter_model.safetensors'),base_sha256=base_hashes,
        merged_layers=len(touched),touched=touched,
        files={p.name:sha(p) for p in dest.glob('*.safetensors')}))


def run(args):
    root,work,data=args.root.resolve(),args.work.resolve(),args.data.resolve()
    work.mkdir(parents=True,exist_ok=True)
    lock=(work/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.wait_for_data:
        deadline=time.monotonic()+args.gpu_wait_seconds
        qa_lock=(work/'qa_generation.lock').open('a')
        while True:
            ready=(work/'data/manifest.json').exists()
            waiting_stage='preparing_learning_QA'
            waiting_reason='Waiting for complete audited learning QA'
            if ready:
                try:
                    fcntl.flock(qa_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    idle(args.gpu)
                    break
                except BlockingIOError:
                    waiting_reason='QA generator still holds its data lock'
                except RuntimeError as error:
                    waiting_stage='waiting_for_gpu'
                    waiting_reason=str(error)
            save(work/'status.json',dict(stage=waiting_stage,reason=waiting_reason,
                controller_pid=os.getpid(),updated_at=now(),wait_seconds_remaining=int(deadline-time.monotonic())))
            if time.monotonic()>deadline:raise RuntimeError('Timed out waiting for complete QA data and free GPU')
            time.sleep(30)
    manifest=json.loads((work/'data/manifest.json').read_text())
    assert manifest['knowledge_units']==1814 and manifest['test_data_used'] is False
    for name,h in manifest['files'].items():assert sha(work/'data'/name)==h,name
    text,vision=(rows(work/f'data/train_{m}.jsonl') for m in ['text','vision'])
    assert len(text)==len(vision)==1814
    image_roles={r['qid']:r for r in rows(data/'audit/image_roles.jsonl')}
    for t,v in zip(text,vision):
        assert t['id']==v['id'] and t['qid']==v['qid']
        assert t['messages'][-1]==v['messages'][-1]
        assert v['images']==[str((data/image_roles[v['qid']]['learning']['file']).resolve())]
        assert not t.get('images')
    plan=dict(status='running',base=str(args.model.resolve()),data=str(data),
        knowledge_units=1814,train_rows_per_modality=1814,method='attribute QA SFT',
        tuner='LoRA',epochs=3,rank=64,alpha=128,dropout=0.05,learning_rate=1e-4,
        effective_batch=16,micro_batch=4,gradient_accumulation=4,seed=42,
        freeze_vit=True,freeze_aligner=True,target_modules='all-linear (language model)',
        reference='MMKU-Bench arXiv:2603.15117 Appendix D and official *_train_rewrite.json',
        adaptation='User requested 3 epochs; MMKU uses 6. Backbone, target modules, image processing and MCQ/CoT evaluation differ.',
        checkpoint_selection='final epoch fixed in advance; no test-based selection',
        evaluation='Paired 3628 T + 3628 V questions per model, direct/statement reported separately',
        baseline='Same base and identical evaluation inputs/protocol',
        limitations='Repaired source remains candidate data; generated QA is automatically checked, not human-certified',
        dataset_inventory_sha256=sha(data/'package_inventory.json'),data_manifest_sha256=sha(work/'data/manifest.json'),
        gpu=args.gpu,pid=os.getpid(),started_at=now())
    save(work/'experiment.json',plan)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu,HF_HUB_OFFLINE='1',MODELSCOPE_OFFLINE='1',
        MAX_PIXELS='262144',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',NPROC_PER_NODE='1',
        MASTER_ADDR='127.0.0.1',MASTER_PORT='30373',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')

    def execute(cmd,log,stage):
        idle(args.gpu)
        save(work/(stage.replace('/','_')+'_command.json'),cmd)
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('a') as stream:
            proc=subprocess.Popen(cmd,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT)
        save(work/'status.json',dict(stage=stage,child_pid=proc.pid,controller_pid=os.getpid(),log=str(log),updated_at=now()))
        code=proc.wait()
        if code:raise RuntimeError(f'{stage} failed with return code {code}; see {log}')
        # CUDA contexts can take a moment to disappear after process shutdown.
        for _ in range(15):
            try:idle(args.gpu);break
            except RuntimeError:time.sleep(2)
        idle(args.gpu)

    for smoke in [True,False]:
        for modality in ['TEXT','VISION']:
            cmd,folder=command(args,modality,smoke)
            try:ck=checkpoint(folder,smoke)
            except (AssertionError,RuntimeError,FileNotFoundError):
                if folder.exists() and any(folder.iterdir()):
                    raise RuntimeError(f'Incomplete output {folder}; inspect rather than overwrite')
                execute(cmd,work/f'{"smoke" if smoke else "train"}_{modality}.log',
                        f'{"smoke" if smoke else "training"}_{modality}')
                ck=checkpoint(folder,smoke)
            if not smoke:
                save(work/'status.json',dict(stage=f'merge_{modality}',checkpoint=str(ck),updated_at=now(),controller_pid=os.getpid()))
                merge(args.model.resolve(),ck,work/'merged'/modality)
    for arm in ['TEXT','VISION','base']:
        model=args.model.resolve() if arm=='base' else work/'merged'/arm
        output=work/'evaluation'/arm
        cmd=[str(args.infer_python.resolve()),str(root/'scripts/evaluate_entityqa_vllm.py'),
            '--data',str(data),'--model',str(model),'--output',str(output),'--arm',arm,'--gpu',args.gpu]
        execute(cmd,work/f'eval_{arm}.log',f'evaluating_{arm}')
    reports={arm:json.loads((work/f'evaluation/{arm}/summary.json').read_text()) for arm in ['TEXT','VISION','base']}
    assert all(r['status']=='complete' and r['saved']==7256 for r in reports.values())
    configs=[json.loads((work/f'evaluation/{arm}/config.json').read_text()) for arm in ['TEXT','VISION','base']]
    assert len({c['input_sha256'] for c in configs})==1,'Model arms used different evaluation prompts or paired inputs'
    assert len({c['dataset_inventory_sha256'] for c in configs})==1
    directions={}
    for arm,short in [('TEXT','T'),('VISION','V')]:
        for modality in ['T','V']:
            name=f'{short}2{modality}'
            group=reports[arm]['groups'][f'{modality}/all']
            directions[name]=dict(**group,
                baseline_accuracy=reports['base']['groups'][f'{modality}/all']['accuracy'],
                accuracy_gain_percentage_points=100*(group['accuracy']-reports['base']['groups'][f'{modality}/all']['accuracy']))
    save(work/'comparison.json',dict(status='complete',directions=directions,reports=reports,completed_at=now()))
    save(work/'status.json',dict(stage='complete',updated_at=now(),controller_pid=os.getpid()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','work','data','model','infer-python']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gpu',default='1')
    p.add_argument('--wait-for-data',action='store_true')
    p.add_argument('--gpu-wait-seconds',type=int,default=86400)
    args=p.parse_args()
    try:run(args)
    except Exception as error:
        save(args.work/'failure.json',dict(error=repr(error),time=now()))
        save(args.work/'status.json',dict(stage='failed',error=repr(error),updated_at=now(),controller_pid=os.getpid()))
        raise
