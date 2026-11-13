#!/usr/bin/env python3
"""Dense-language and whole-VLM SFT versus matched fresh LoRA controls."""
import argparse
import copy
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import explore_v7_application_binding as prior
import explore_v7_visual_replay as replay
import run_v7_evaluation as ev
from run_v7_full_experiment import ROOT,MODELS,rows,read,save,sha,environment,train_command
from v7_gpu_resources import reserve

OUT=ROOT/'runs/v7_full_parameter_20261018'
BASE=OUT/'merged_warmup'
ARMS=('full_text','full_vision','full_all_vision','lora_text','lora_vision')
LEDGER=prior.previous.source.LEDGER
DATA=prior.previous.source.DATA
LOSS=ROOT/'scripts/v7_equal_example_loss.py'
AUDIT=ROOT/'scripts/v7_full_parameter_audit.py'


def dense_checkpoint(path):
    tracked=sorted([*path.glob('model*.safetensors'),path/'config.json'])
    assert tracked and (path/'config.json').is_file()
    assert any(p.suffix=='.safetensors' for p in tracked)
    for p in ('model.safetensors.index.json','generation_config.json','tokenizer_config.json','tokenizer.json','preprocessor_config.json','processor_config.json','chat_template.jinja'):
        if (path/p).exists():tracked.append(path/p)
    return {'path':str(path),'sha256':{p.name:sha(p) for p in tracked},'kind':'dense'}


def checkpoint(arm,smoke=False):
    if arm=='merged_parent':return BASE
    return OUT/('smoke_'+arm if smoke else arm)/'checkpoint'/('checkpoint-1' if smoke else 'checkpoint-128')


def identity(arm,smoke=False):
    path=checkpoint(arm,smoke)
    return ev.checkpoint(path) if arm.startswith('lora') else dense_checkpoint(path)


def env(gpu):
    z=environment(gpu,DATA)
    z.update(TORCH_EXTENSIONS_DIR=str(OUT/'torch_extensions'),TMPDIR=str(OUT/'tmp'),
        MAX_JOBS='4',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',MASTER_ADDR='127.0.0.1',MASTER_PORT=str(29870+int(gpu)))
    return z


def merge():
    OUT.mkdir(exist_ok=True)
    if (OUT/'merged_parent_manifest.json').exists():
        assert read(OUT/'merged_parent_manifest.json')['model']==dense_checkpoint(BASE)
        return
    prior.validate()
    from transformers import Qwen3VLForConditionalGeneration,AutoProcessor
    from peft import PeftModel
    import torch
    torch.set_num_threads(8)
    parent=prior.checkpoint('warmup')
    model=Qwen3VLForConditionalGeneration.from_pretrained(str(MODELS['qwen3']),dtype=torch.bfloat16,device_map='cpu')
    model=PeftModel.from_pretrained(model,str(parent)).merge_and_unload(safe_merge=True)
    assert not any('lora_' in n for n,_ in model.named_parameters())
    model.save_pretrained(str(BASE),safe_serialization=True,max_shard_size='5GB')
    AutoProcessor.from_pretrained(str(MODELS['qwen3'])).save_pretrained(str(BASE))
    save(OUT/'merged_parent_manifest.json',{'source_base':str(MODELS['qwen3']),'source_adapter':ev.checkpoint(parent),
        'model':dense_checkpoint(BASE),'method':'PEFT safe_merge on CPU, dense BF16. All new arms including fresh zero-delta LoRA start from this exact saved model.'})


def prepare():
    if (OUT/'manifest.json').exists():validate();return
    prior.validate();assert read(prior.OUT/'completion_audit.json')['status']=='pass'
    assert read(OUT/'merged_parent_manifest.json')['model']==dense_checkpoint(BASE)
    for d in ('tmp','torch_extensions','source_snapshots'):(OUT/d).mkdir(exist_ok=True)
    for mod in ('text','vision'):
        shutil.copy2(prior.OUT/(mod+'_apply.jsonl'),OUT/(mod+'.jsonl'))
    for name in ('main_probes.jsonl','diagnostic_probes.jsonl','target_knowledge.jsonl'):
        shutil.copy2(prior.OUT/name,OUT/name)
    ds={'bf16':{'enabled':'auto'},'zero_optimization':{'stage':2,'offload_optimizer':{'device':'cpu','pin_memory':True},
        'allgather_partitions':True,'allgather_bucket_size':20000000,'overlap_comm':False,'reduce_scatter':True,
        'reduce_bucket_size':20000000,'contiguous_gradients':True},
        'optimizer':{'type':'AdamW','params':{'lr':'auto','betas':'auto','eps':'auto','weight_decay':'auto'}},
        'gradient_accumulation_steps':'auto','gradient_clipping':'auto','train_batch_size':'auto',
        'train_micro_batch_size_per_gpu':'auto','steps_per_print':2000,'wall_clock_breakdown':False}
    save(OUT/'zero2_cpu.json',ds)
    save(OUT/'decision_criteria.json',{'fixed_before_training':True,'epochs':1,'checkpoint_step':128,
        'gates':{'source_capacity':.8,'unseen_closed':.6,'paired_gain_each_input_over_merged_parent':.1},
        'failure_policy':'Finish every predeclared arm, preserve failures; do not select checkpoint/rules by scores or extend epochs after observing results.'})
    sources=[prior.OUT/'manifest.json',prior.OUT/'final_artifact_manifest.json',prior.OUT/'summary.json',
        prior.OUT/'text_apply.jsonl',prior.OUT/'vision_apply.jsonl',LOSS,AUDIT,
        ROOT/'scripts/explore_v7_application_binding.py',ROOT/'scripts/explore_v7_visual_replay.py',
        ROOT/'scripts/run_v7_full_experiment.py',ROOT/'scripts/run_v7_evaluation.py',ROOT/'scripts/v7_gpu_resources.py']
    save(OUT/'manifest.json',{'created_at':ev.now(),'script_sha256':sha(Path(__file__)),
        'sources':{str(p):sha(p) for p in sources},'parent':read(OUT/'merged_parent_manifest.json'),
        'files':{p.name:sha(p) for p in OUT.iterdir() if p.suffix in ('.json','.jsonl')},
        'arms':ARMS,'config':{'epochs':1,'steps':128,'batch':4,'accumulation':1,'lr':1e-5,'warmup_ratio':.05,
            'schedule':'cosine','seed':42,'max_length':8192,'loss':'equal_example','weight_decay':.1,
            'adam_betas':[.9,.95],'adam_eps':1e-8,'max_grad_norm':1.},
        'scopes':{'full_text':'All language weights, vision and aligner frozen.',
            'full_vision':'All language weights, vision and aligner frozen.',
            'full_all_vision':'Whole VLM, vision and aligner unfrozen.',
            'lora_text':'Fresh r8/alpha32/dropout.05 LoRA over merged parent, vision and aligner frozen.',
            'lora_vision':'Fresh r8/alpha32/dropout.05 LoRA over merged parent, vision and aligner frozen.'},
        'precision':'Dense training uses BF16 forward/backward and DeepSpeed CPU-offloaded FP32 master weights/AdamW states. LoRA uses torch AdamW with FP32 trainable LoRA. No 8bit weights or optimizer used.',
        'comparison':'Five new arms share identical dense merged warmup initialization, data/order/step/lr/loss. Dense vs LoRA differs update parameterization, precision implementation and dropout. Whole-VLM versus language-only vision also changes visual update scope.',
        'evaluation':'Each arm and merged parent: same1016probes from previous application experiment; no native CoT. Learning-image/oracle diagnostic separate from closed book. Four direction aliases are not independent inputs.',
        'new_answers_planned':6106,'limits':['Same8development rules, single seed, not1000rule experiment or independent confirmation.',
            'Previous1e-4LoRA results are references only; matched controls here use1e-5 and fresh LoRA on merged parent.',
            '512rows contain repetitions and rotated choices. Train and heldout candidate sets are disjoint as previously audited.',
            'VISION training includes textual observations/answer supervision, so even whole-VLM SFT is not pixel-only learning.'],
        'versions':{p:importlib.metadata.version(p) for p in ('torch','transformers','ms-swift','peft','deepspeed','bitsandbytes')}})
    for p in [Path(__file__)]+[p for p in sources if p.suffix=='.py']:shutil.copy2(p,OUT/'source_snapshots'/p.name)
    validate();print('Prepared five512row arms, fixed128steps, six1016probe evaluations.',flush=True)


def validate():
    m=read(OUT/'manifest.json');assert m['script_sha256']==sha(Path(__file__))
    for p,h in m['sources'].items():assert sha(Path(p))==h,p
    for p,h in m['files'].items():assert sha(OUT/p)==h,p
    assert m['parent']['model']==dense_checkpoint(BASE)


def train(arm,gpu,smoke=False):
    folder=OUT/('smoke_'+arm if smoke else arm);folder.mkdir(exist_ok=True)
    if smoke and (folder/'validated.json').exists():return
    cp=checkpoint(arm,smoke)
    if not (cp/'trainer_state.json').exists():
        cmd=train_command('qwen3','sft',OUT/('text.jsonl' if arm.endswith('text') else 'vision.jsonl'),folder/'checkpoint',1)
        for flag,value in [('--model',str(BASE)),('--per_device_train_batch_size','4'),('--gradient_accumulation_steps','1'),
                ('--learning_rate','1e-5'),('--save_strategy','steps'),('--save_total_limit','1')]:cmd[cmd.index(flag)+1]=value
        if arm.startswith('full'):
            cmd[cmd.index('--tuner_type')+1]='full'
            for flag in ('--target_modules','--lora_rank','--lora_alpha','--lora_dropout'):
                i=cmd.index(flag);del cmd[i:i+2]
            cmd+=['--deepspeed',str(OUT/'zero2_cpu.json')]
            if arm=='full_all_vision':
                for flag in ('--freeze_vit','--freeze_aligner'):cmd[cmd.index(flag)+1]='false'
        cmd+=['--save_steps','1' if smoke else '64','--external_plugins',str(LOSS),str(AUDIT),
            '--callbacks','v7_full_parameter_audit','--loss_type','v7_equal_example','--packing','false','--padding_free','false',
            '--model_type','qwen3_vl',
            '--weight_decay','0.1','--adam_beta1','0.9','--adam_beta2','0.95','--adam_epsilon','1e-8',
            '--max_grad_norm','1','--optim','adamw_torch','--attn_impl','sdpa']
        if smoke:cmd+=['--max_steps','1','--save_only_model','true']
        else:
            recover=[p for p in (folder/'checkpoint').glob('checkpoint-*') if (p/'trainer_state.json').exists()]
            if recover:
                resume=max(recover,key=lambda p:read(p/'trainer_state.json')['global_step'])
                cmd+=['--resume_from_checkpoint',str(resume),'--load_args','false']
        e=env(gpu);e['V7_EQUAL_LOSS_AUDIT']=str(folder/'loss_batches.jsonl');e['V7_PARAMETER_AUDIT']=str(folder/'parameter_audit.jsonl')
        if arm.startswith('full'):e['NPROC_PER_NODE']='1'
        save(folder/'train_state.json',{'status':'queued','command':cmd})
        with reserve(LEDGER,gpu,str(folder)+':train',27000 if arm.startswith('full') else 17000) as res:
            with (folder/'train.log').open('a') as log:
                proc=subprocess.Popen(cmd,cwd=ROOT,env=e,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                res.bind(proc.pid);save(folder/'train_state.json',{'status':'running','pid':proc.pid,'command':cmd});code=proc.wait()
            save(folder/'process_exit.json',{'returncode':code});assert code==0,(arm,code)
    state=read(cp/'trainer_state.json');assert state['global_step']==(1 if smoke else 128)
    losses=rows(folder/'loss_batches.jsonl');assert losses and all(math.isfinite(r['loss']) for r in losses)
    save(folder/'checkpoint_manifest.json',{'model':identity(arm,smoke),'epoch':state['epoch'],'step':state['global_step'],'selection':'fixed final'})
    save(folder/'train_state.json',{'status':'complete','checkpoint':str(cp)})
    if smoke:
        probes=rows(OUT/'diagnostic_probes.jsonl');probes=[next(p for p in probes if p['condition']==c) for c in ('unseen_closed','unseen_image')]
        evaluate(arm,gpu,smoke=True,probes=probes)
        save(folder/'validated.json',{'status':'pass','finite_loss':True,'reloaded_text_and_image':True})


class Runner(replay.Runner):
    def load(self):
        os.environ.update(env(self.gpu))
        import torch
        from swift.infer_engine import TransformersEngine
        from swift.utils import seed_everything
        seed_everything(42)
        ident=self.config['model'];lora=ident.get('kind')!='dense'
        self.engine=TransformersEngine(str(BASE) if lora else ident['path'],adapters=[ident['path']] if lora else [],
            model_type='qwen3_vl',torch_dtype=torch.bfloat16,device_map='cuda:0',max_batch_size=4,attn_impl='sdpa')
        self.engine.template.enable_thinking=False

    def audit(self,probes,expected):
        assert len(self.rr)==len(self.seen)==expected
        lookup={p['id']:p for p in probes};assert set(lookup)==self.seen
        for r in self.rr:
            p=lookup[r['id']]
            assert r['input_sha256']==ev.digest({'messages':p['messages'],'images':p.get('images',[])})
            assert r['response']==(r['raw_response']['choices'][0]['message']['content'] or '')
            assert r['predicted']==(ev.letter(r['response']) if r['fmt']=='mcq' else None)
            if r['fmt']=='mcq':assert r['correct']==(r['predicted']==p['gold'])
        model=self.config['model'];path=Path(model['path'])
        assert model==(dense_checkpoint(path) if model.get('kind')=='dense' else ev.checkpoint(path))
        save(self.folder/'eval_state.json',{'status':'complete','saved':expected})


def evaluate(arm,gpu,smoke=False,probes=None):
    if probes is None:probes=rows(OUT/'main_probes.jsonl')+rows(OUT/'diagnostic_probes.jsonl')
    folder=OUT/('smoke_'+arm if smoke else arm)/'evaluation'
    cfg={'model':identity(arm,smoke),'temperature':0,'seed':42,'batch':4,'thinking':False,
        'max_tokens':{'mcq':16,'card':384},'probes_sha256':ev.digest(probes)}
    run=Runner(folder,cfg,gpu)
    if len(run.rr)<len(probes):
        with reserve(LEDGER,gpu,str(folder)+':eval',21000) as res:
            res.bind(os.getpid());run.load()
            for fmt in ('card','mcq'):run.infer([p for p in probes if p['fmt']==fmt])
    run.audit(probes,len(probes))
    save(folder/'report.json',{'status':'complete','metrics':{c:prior.previous.source.metric([r for r in run.rr if r['condition']==c])
        for c in sorted({p['condition'] for p in probes if p['fmt']=='mcq'})}})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True,choices=('merge','prepare','smoke','train','eval','run'))
    ap.add_argument('--arm',choices=ARMS+('merged_parent',));ap.add_argument('--gpu',default='0');a=ap.parse_args()
    if a.stage=='merge':merge()
    elif a.stage=='prepare':prepare()
    else:
        validate()
        with ev.run_lock(OUT/(str(a.arm)+'_'+a.stage+'.lock')):
            if a.stage in ('smoke','train','run'):train(a.arm,a.gpu,a.stage=='smoke')
            if a.stage in ('eval','run'):evaluate(a.arm,a.gpu)
