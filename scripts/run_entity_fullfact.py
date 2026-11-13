#!/usr/bin/env python3
"""Own-server full-fact LoRA training and paired four-direction evaluation."""
import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

from entityqa_common import now, rows, save, sha
from run_entityqa_lora import checkpoint, command, idle, merge


def run(args):
    import torch
    torch.set_num_threads(4)
    root, work, data, base = (getattr(args,k).resolve() for k in ['root','work','data','model'])
    assert socket.gethostname() == 'softengine-KVM', 'Run only on the user\'s own server'
    work.mkdir(parents=True,exist_ok=True)
    lock=(work/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=json.loads((work/'data/manifest.json').read_text())
    assert manifest['version']=='fullfact-v2' and manifest['test_data_used'] is False
    assert manifest['knowledge_units']==1814 and manifest['rows_per_modality']==3628
    for name,h in manifest['files'].items():
        assert sha(work/'data'/name)==h,name
    text,vision=(rows(work/f'data/train_{m}.jsonl') for m in ['text','vision'])
    assert len(text)==len(vision)==3628
    sources={r['id']:r for r in rows(work/'data/learning_inputs.jsonl')}
    for t,v in zip(text,vision):
        assert t['id']==v['id'] and t['qid']==v['qid']
        r=sources[t['knowledge_id']]
        assert t['messages'][-1]['content']==r['learning']
        assert v['messages'][-1]['content']==r['visual_learning']
        assert v['images']==[str((data/p).resolve()) for p in r['images']]
    inventory=json.loads((data/'package_inventory.json').read_text())
    for name,h in inventory['sha256'].items():
        assert sha(data/name)==h,name
    snapshots=work/'source_snapshots'
    scripts=['run_entity_fullfact.py','prepare_entity_fullfact.py','run_entityqa_lora.py',
             'evaluate_entityqa_fullfact_v2.py','answer_readout_entity_v2.py',
             'answer_readout.py','entityqa_common.py','test_entity_fullfact.py']
    if not snapshots.exists():
        snapshots.mkdir()
        for name in scripts:
            shutil.copy2(root/'scripts'/name,snapshots/name)
    for name in scripts:
        assert sha(root/'scripts'/name)==sha(snapshots/name),f'Script changed after freezing: {name}'
    plan=dict(status='running',host=socket.gethostname(),root=str(root),gpu=args.gpu,
        training_framework='ms-swift swift sft',inference_backend='vLLM only for evaluation',
        base=str(base),knowledge_units=1814,rows_per_modality=3628,
        method=manifest['method'],tuner='LoRA',epochs=3,rank=64,alpha=128,
        learning_rate=1e-4,effective_batch=16,micro_batch=4,gradient_accumulation=4,seed=42,
        freeze_vit=True,freeze_aligner=True,checkpoint_selection='Fixed final epoch 3, never test-selected',
        training_reused=False,test_data_used_for_training=False,
        evaluation=dict(questions_per_modality=3628,directions=['T2T','T2V','V2T','V2V'],
            brief_prompted_cot=True,max_tokens=65536,wall_seconds=180,
            parser='entity-readout-v2-full-option-prefix',primary='raw response accuracy and invalid rate',
            constrained_finalization='Separate supplementary score, never described as zero raw invalid'),
        changes_from_v1=['Whole learning paragraph in every supervised answer',
            'One neutral recall and one attribute question per knowledge unit',
            'Anonymized VISION target retained without adding entity name',
            'Versioned parser supports consistent letter plus complete option content'],
        limitations=manifest['limitations'],dataset_inventory_sha256=sha(data/'package_inventory.json'),
        data_manifest_sha256=sha(work/'data/manifest.json'),
        source_sha256={name:sha(snapshots/name) for name in scripts},
        controller_pid=os.getpid(),started_at=now())
    save(work/'experiment.json',plan)
    packages=['ms-swift','torch','transformers','peft','trl','accelerate']
    save(work/'training_environment.json',dict(host=socket.gethostname(),
        python=os.sys.version,packages={p:importlib.metadata.version(p) for p in packages},
        cuda=torch.version.cuda,created_at=now()))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu,HF_HUB_OFFLINE='1',MODELSCOPE_OFFLINE='1',
        MAX_PIXELS='262144',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',NPROC_PER_NODE='1',
        MASTER_ADDR='127.0.0.1',MASTER_PORT='30374',
        TMPDIR=str(work/'tmp'),HF_DATASETS_CACHE=str(work/'dataset_cache'),
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    Path(env['TMPDIR']).mkdir(exist_ok=True)

    def execute(cmd,stage):
        while True:
            try:
                idle(args.gpu)
                break
            except RuntimeError as error:
                save(work/'status.json',dict(stage='waiting_for_free_gpu',next_stage=stage,
                    reason=str(error),controller_pid=os.getpid(),updated_at=now()))
                time.sleep(15)
        log=work/(stage+'.log')
        save(work/(stage+'_command.json'),cmd)
        with log.open('a') as stream:
            process=subprocess.Popen(cmd,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT)
        save(work/'status.json',dict(stage=stage,child_pid=process.pid,
            controller_pid=os.getpid(),host=socket.gethostname(),gpu=args.gpu,log=str(log),updated_at=now()))
        print(json.dumps(dict(stage=stage,child_pid=process.pid,time=now())),flush=True)
        code=process.wait()
        if code:
            raise RuntimeError(f'{stage} failed ({code}); inspect {log}')
        for _ in range(20):
            try:
                idle(args.gpu)
                break
            except RuntimeError:
                time.sleep(2)

    for smoke in [True,False]:
        for modality in ['TEXT','VISION']:
            cmd,folder=command(args,modality,smoke)
            try:
                ck=checkpoint(folder,smoke)
            except (AssertionError,RuntimeError,FileNotFoundError):
                if folder.exists() and any(folder.iterdir()):
                    raise RuntimeError(f'Incomplete output {folder}; inspect instead of overwriting')
                stage=('smoke_' if smoke else 'training_')+modality
                execute(cmd,stage)
                ck=checkpoint(folder,smoke)
            if not smoke:
                save(work/'status.json',dict(stage='merge_'+modality,checkpoint=str(ck),
                    controller_pid=os.getpid(),updated_at=now()))
                merge(base,ck,work/'merged'/modality)

    # Training finishes first. Generation uses the frozen evaluation source.
    for arm in ['TEXT','VISION','base']:
        output=work/'evaluation'/arm
        report=output/'summary.json'
        if report.exists() and json.loads(report.read_text()).get('status')=='complete':
            continue
        cmd=[str(args.infer_python.absolute()),'-u',str(snapshots/'evaluate_entityqa_fullfact_v2.py'),
             '--data',str(data),'--model',str(base if arm=='base' else work/'merged'/arm),
             '--output',str(output),'--arm',arm,'--gpu',args.gpu]
        if arm=='base' and args.baseline_predictions:
            cmd+=['--import-predictions',str(args.baseline_predictions.absolute())]
        execute(cmd,'evaluating_'+arm)
    reports={arm:json.loads((work/f'evaluation/{arm}/summary.json').read_text())
             for arm in ['TEXT','VISION','base']}
    configs=[json.loads((work/f'evaluation/{arm}/config.json').read_text())
             for arm in ['TEXT','VISION','base']]
    assert all(r['status']=='complete' and r['saved']==7256 for r in reports.values())
    assert len({c['input_sha256'] for c in configs})==1
    assert len({c['readout_version'] for c in configs})==1
    directions={}
    for arm,prefix in [('TEXT','T'),('VISION','V')]:
        for modality in ['T','V']:
            result=reports[arm]['groups'][modality+'/all']
            before=reports['base']['groups'][modality+'/all']
            directions[prefix+'2'+modality]=dict(**result,
                baseline_primary_accuracy=before['primary_accuracy'],
                primary_gain_percentage_points=100*(result['primary_accuracy']-before['primary_accuracy']),
                baseline_constrained_accuracy=before['accuracy'])
    save(work/'comparison.json',dict(status='complete',directions=directions,
        reports=reports,completed_at=now()))
    plan.update(status='complete',completed_at=now())
    save(work/'experiment.json',plan)
    save(work/'status.json',dict(stage='complete',controller_pid=os.getpid(),updated_at=now()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['root','work','data','model','infer-python']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--gpu',default='0')
    parser.add_argument('--baseline-predictions',type=Path)
    args=parser.parse_args()
    try:
        run(args)
    except Exception as error:
        save(args.work/'failure.json',dict(error=repr(error),time=now()))
        save(args.work/'status.json',dict(stage='failed',error=repr(error),
            controller_pid=os.getpid(),updated_at=now()))
        raise
