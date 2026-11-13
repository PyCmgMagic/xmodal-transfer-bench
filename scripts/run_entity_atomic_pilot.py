#!/usr/bin/env python3
"""Run two 32-entity TEXT pilot conditions before considering full retraining."""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
from types import SimpleNamespace

from entityqa_common import rows,save,sha,now
from run_entityqa_lora import command,checkpoint,idle,merge


def run(args):
    import torch
    torch.set_num_threads(4)
    root,work,base=(getattr(args,k).resolve() for k in ['root','work','model'])
    assert socket.gethostname()=='softengine-KVM'
    lock=(work/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=json.loads((work/'data/manifest.json').read_text())
    assert manifest['knowledge_units']==32 and not manifest['test_data_used_for_training']
    for name,h in manifest['files'].items():assert sha(work/name)==h,name
    snapshots=work/'source_snapshots'
    scripts=['prepare_entity_atomic_pilot.py','prepare_entity_atomic_eval.py',
        'run_entity_atomic_pilot.py','evaluate_entity_atomic_recall.py','run_entityqa_lora.py',
        'entityqa_common.py','answer_readout.py','answer_readout_entity_v2.py',
        'answer_readout_entity_v3.py','test_entity_atomic_pilot.py']
    if not snapshots.exists():
        snapshots.mkdir()
        for name in scripts:shutil.copy2(root/'scripts'/name,snapshots/name)
        evaluation=(root/'scripts/evaluate_entityqa_fullfact_v2.py').read_text()
        evaluation=evaluation.replace('from answer_readout_entity_v2 import parse_answer, VERSION',
                                      'from answer_readout_entity_v3 import parse_answer, VERSION')
        (snapshots/'evaluate_entityqa_pilot.py').write_text(evaluation)
    hashes={p.name:sha(p) for p in snapshots.glob('*.py')}
    plan=dict(created_at=now(),status='running',host=socket.gethostname(),gpu=args.gpu,
        base=str(base),training_framework='ms-swift swift sft',epochs=3,learning_rate=2e-5,
        rank=64,alpha=128,knowledge_units=32,modality='TEXT only first-stage development pilot',
        conditions=['atomic','atomic_replay'],selection=manifest['selection'],
        development_qids=manifest['development_qids'],source_sha256=hashes,
        train_manifest_sha256=sha(work/'data/manifest.json'),
        evaluation_inventory_sha256=sha(work/'evaluation_data/package_inventory.json'),
        gates=dict(original_mcq_invalid_at_most=0.05,original_text_accuracy_gain_at_least=0.10,
            rotation_max_letter_share_at_most=0.50,trained_attribute_span_hit_at_least=0.50,
            independent_instruction_drop_at_most=0.05),
        limitations=manifest['caveats'],automatic_full_retraining=False)
    save(work/'experiment.json',plan)
    rpc=Path('/tmp/xm-atomic-pilot')
    rpc.mkdir(mode=0o700,exist_ok=True)
    assert rpc.stat().st_uid==os.getuid()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu,HF_HUB_OFFLINE='1',MODELSCOPE_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
        MAX_PIXELS='262144',NPROC_PER_NODE='1',MASTER_ADDR='127.0.0.1',MASTER_PORT='30375',
        TMPDIR=str(rpc),VLLM_RPC_BASE_PATH=str(rpc),VLLM_WORKER_MULTIPROC_METHOD='spawn',
        PYTHONUNBUFFERED='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        HF_DATASETS_CACHE=str(work/'dataset_cache'))

    def execute(cmd,stage):
        while True:
            try:idle(args.gpu);break
            except RuntimeError as error:
                save(work/'status.json',dict(stage='waiting_for_free_gpu',next_stage=stage,
                    reason=str(error),controller_pid=os.getpid(),updated_at=now()))
                time.sleep(15)
        save(work/(stage+'_command.json'),cmd)
        with (work/(stage+'.log')).open('a') as stream:
            process=subprocess.Popen(cmd,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT)
        save(work/'status.json',dict(stage=stage,controller_pid=os.getpid(),child_pid=process.pid,
            host=socket.gethostname(),gpu=args.gpu,updated_at=now()))
        print(json.dumps(dict(stage=stage,pid=process.pid,time=now())),flush=True)
        code=process.wait()
        if code:raise RuntimeError(f'{stage} failed ({code}), inspect its log')
        for _ in range(15):
            try:idle(args.gpu);break
            except RuntimeError:time.sleep(2)

    for arm in ['atomic','atomic_replay']:
        folder=work/arm
        train_args=SimpleNamespace(root=root,work=folder,model=base)
        cmd,dest=command(train_args,'TEXT',False)
        cmd[cmd.index('--learning_rate')+1]='2e-5'
        try:ck=checkpoint(dest)
        except (AssertionError,RuntimeError,FileNotFoundError):
            if dest.exists() and any(dest.iterdir()):
                raise RuntimeError(f'Incomplete existing checkpoint in {dest}; inspect')
            execute(cmd,'training_'+arm)
            ck=checkpoint(dest)
        save(work/'status.json',dict(stage='merging_'+arm,controller_pid=os.getpid(),updated_at=now()))
        merge(base,ck,folder/'merged')
    # Same frozen MCQ prompt and parser for every condition. Reuse base originals
    # only after their input hashes were checked by prepare_entity_atomic_eval.
    for arm in ['base','atomic','atomic_replay']:
        output=work/'evaluation'/arm
        if not (output/'summary.json').exists() or json.loads((output/'summary.json').read_text())['status']!='complete':
            cmd=[str(root/'.venv-vllm/bin/python'),'-u',str(snapshots/'evaluate_entityqa_pilot.py'),
                '--data',str(work/'evaluation_data'),'--model',str(base if arm=='base' else work/arm/'merged'),
                '--output',str(output),'--arm','base' if arm=='base' else 'TEXT','--gpu',args.gpu]
            if arm=='base':cmd+=['--import-predictions',str(work/'data/base_original_dev_predictions.jsonl')]
            execute(cmd,'evaluating_mcq_'+arm)
        diag=work/'diagnostics'/arm/'summary.json'
        if not diag.exists() or json.loads(diag.read_text())['status']!='complete':
            execute([str(root/'.venv/bin/python'),'-u',str(snapshots/'evaluate_entity_atomic_recall.py'),
                '--work',str(work),'--model',str(base),'--arm',arm],'diagnostics_'+arm)
    all_reports={}
    for arm in ['base','atomic','atomic_replay']:
        predictions=rows(work/f'evaluation/{arm}/predictions.jsonl')
        assert len(predictions)==192
        groups={}
        for mod in ['T','V']:
            for kind in ['original','rotation']:
                group=[r for r in predictions if r['modality']==mod and
                       ((r['probe_id']=='rotation')==(kind=='rotation'))]
                counts=Counter(str(r['raw_predicted']) for r in group)
                groups[mod+'/'+kind]=dict(n=len(group),
                    accuracy=sum(r['raw_predicted']==r['gold'] for r in group)/len(group),
                    invalid_rate=sum(r['raw_predicted'] is None for r in group)/len(group),
                    answer_distribution=dict(counts),max_letter_share=max(counts.get(x,0) for x in 'ABCD')/len(group))
        all_reports[arm]=dict(mcq=groups,diagnostics=json.loads((work/f'diagnostics/{arm}/summary.json').read_text()))
    base_report=all_reports['base']
    gates={}
    for arm in ['atomic','atomic_replay']:
        r=all_reports[arm]
        diagnostic=r['diagnostics']['groups']
        g=dict(text_invalid=r['mcq']['T/original']['invalid_rate']<=0.05,
            vision_invalid=r['mcq']['V/original']['invalid_rate']<=0.05,
            text_gain=r['mcq']['T/original']['accuracy']-base_report['mcq']['T/original']['accuracy']>=0.10,
            rotation_balance=r['mcq']['T/rotation']['max_letter_share']<=0.50,
            attribute_recall=diagnostic['trained_attribute_recall']['strict_target_span_hit_rate']>=0.50,
            independent_instruction=diagnostic['independent_instruction']['accuracy']>=
                base_report['diagnostics']['groups']['independent_instruction']['accuracy']-0.05)
        gates[arm]=dict(checks=g,passed=all(g.values()))
    save(work/'comparison.json',dict(status='complete',reports=all_reports,gates=gates,completed_at=now(),
        decision='Pilot only. Full-scale retraining requires passing source recall, instruction and position controls; development QIDs must be excluded from confirmatory scores.'))
    plan.update(status='complete',completed_at=now())
    save(work/'experiment.json',plan)
    save(work/'status.json',dict(stage='complete',controller_pid=os.getpid(),updated_at=now()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','work','model']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gpu',default='0')
    args=p.parse_args()
    try:run(args)
    except Exception as error:
        save(args.work/'status.json',dict(stage='failed',error=repr(error),updated_at=now(),controller_pid=os.getpid()))
        raise
