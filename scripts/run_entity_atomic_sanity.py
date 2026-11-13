#!/usr/bin/env python3
"""Conditional four-entity overfit diagnostic, never a transfer benchmark."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
from types import SimpleNamespace

from entityqa_common import rows,save,sha,write_rows,now
from prepare_entity_atomic_pilot import generic
from run_entityqa_lora import command,checkpoint,idle


def run(args):
    import torch
    torch.set_num_threads(4)
    pilot=args.pilot.resolve()
    root=Path(__file__).resolve().parents[1]
    work=pilot/'sanity_overfit4'
    work.mkdir(exist_ok=True)
    assert socket.gethostname()=='softengine-KVM'
    lock=(work/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def status(stage,**extra):
        save(work/'status.json',dict(stage=stage,pid=os.getpid(),updated_at=now(),**extra))
    # Wait for the first-stage inference to release the GPU. Never interrupt it.
    while not (pilot/'rescored_v4/comparison.json').exists():
        previous=json.loads((pilot/'status.json').read_text())
        if previous['stage']=='failed':raise RuntimeError(previous)
        status('waiting_for_first_stage_evaluation')
        time.sleep(15)
    comparison=json.loads((pilot/'rescored_v4/comparison.json').read_text())
    if any(g['passed'] for g in comparison['gates'].values()):
        status('not_needed_first_stage_passed')
        return
    selected=rows(pilot/'data/learning_inputs.jsonl')[:4]
    ids={r['id'] for r in selected}
    core=[r for r in rows(pilot/'data/entity_training.jsonl')
          if r['knowledge_id'] in ids and r['variant'] in ['fact_0','attribute_0']]
    assert len(core)==8
    repeated=[]
    for repeat in range(64):
        for r in core:
            item=copy.deepcopy(r)
            item['id']+=f'-repeat-{repeat}'
            repeated.append(item)
    training=repeated+[generic(i,'train') for i in range(128)]
    assert len(training)==640
    write_rows(work/'atomic_replay/data/train_text.jsonl',training)
    write_rows(work/'data/retention_dev.jsonl',rows(pilot/'data/retention_dev.jsonl'))
    recall=[r for r in rows(pilot/'data/recall_dev.jsonl') if r['qid'] in {s['qid'] for s in selected}]
    for r in core:
        if r['variant']=='fact_0':
            recall.append(dict(id=r['id']+'-exact_training_prompt',qid=r['qid'],
                kind='exact_training_claim_recall',messages=[r['messages'][0]],
                expected=r['messages'][1]['content']))
    assert len(recall)==12
    write_rows(work/'data/recall_dev.jsonl',recall)
    write_rows(work/'data/learning_inputs.jsonl',selected)
    save(work/'experiment.json',dict(created_at=now(),purpose='Can this exact training chain overfit four learning-only facts?',
        unique_entities=4,unique_factual_QA=8,exact_repetitions_per_QA_per_epoch=64,
        generic_rows_per_epoch=128,total_rows_per_epoch=640,epochs=3,learning_rate=1e-4,
        rank=64,alpha=128,training_framework='ms-swift swift sft',
        selection='First four of the original hash-selected 32 development entities, not selected by their scores.',
        test_questions_used_for_training=False,
        inherited_source_manifest_sha256=sha(pilot/'data/manifest.json'),
        exploratory_followup='Triggered because neither first-stage condition passed its frozen gates.',
        score_interpretation='Overfit/optimization diagnostic only; repeated examples are not independent knowledge and success does not prove transfer.',
        train_sha256=sha(work/'atomic_replay/data/train_text.jsonl')))
    snapshots=work/'source_snapshots'
    snapshots.mkdir(exist_ok=True)
    for name in ['run_entity_atomic_sanity.py','prepare_entity_atomic_pilot.py','entityqa_common.py',
                 'run_entityqa_lora.py','answer_readout.py','answer_readout_entity_v2.py',
                 'answer_readout_entity_v3.py','answer_readout_entity_v4.py']:
        shutil.copy2(root/'scripts'/name,snapshots/name)
    evaluation=(root/'scripts/evaluate_entity_atomic_recall.py').read_text().replace(
        'from answer_readout_entity_v3 import parse_answer','from answer_readout_entity_v4 import parse_answer')
    (snapshots/'evaluate_recall.py').write_text(evaluation)
    base=root/'model_cache/Qwen3-VL-4B-Instruct'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',MODELSCOPE_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
        NPROC_PER_NODE='1',MASTER_ADDR='127.0.0.1',MASTER_PORT='30376',
        PYTHONUNBUFFERED='1',TMPDIR='/tmp/xm-atomic-pilot',
        HF_DATASETS_CACHE=str(work/'dataset_cache'),PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    def execute(cmd,stage):
        while True:
            try:idle('0');break
            except RuntimeError as error:
                status('waiting_for_free_gpu',reason=str(error),next_stage=stage)
                time.sleep(15)
        save(work/(stage+'_command.json'),cmd)
        with (work/(stage+'.log')).open('a') as stream:
            proc=subprocess.Popen(cmd,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT)
        status(stage,child_pid=proc.pid)
        if proc.wait():raise RuntimeError(f'{stage} failed; inspect its log')
        for _ in range(15):
            try:idle('0');break
            except RuntimeError:time.sleep(2)
    config=SimpleNamespace(root=root,work=work/'atomic_replay',model=base)
    cmd,folder=command(config,'TEXT',False)
    try:checkpoint(folder)
    except (RuntimeError,FileNotFoundError,AssertionError):
        if folder.exists() and any(folder.iterdir()):raise RuntimeError('Incomplete sanity checkpoint; inspect')
        execute(cmd,'training')
        checkpoint(folder)
    execute([str(root/'.venv/bin/python'),'-u',str(snapshots/'evaluate_recall.py'),
             '--work',str(work),'--model',str(base),'--arm','atomic_replay'],'checking_original_adapter')
    predictions=rows(work/'diagnostics/atomic_replay/predictions.jsonl')
    groups={}
    for kind in ['trained_attribute_recall','exact_training_claim_recall','neutral_claim_recall','independent_instruction']:
        group=[r for r in predictions if r['kind']==kind]
        groups[kind]=dict(n=len(group),length_limited=sum(r['finish_reason']=='length' for r in group))
        if kind=='independent_instruction':
            groups[kind]['accuracy']=sum(r['correct'] for r in group)/len(group)
        else:
            groups[kind]['strict_target_span_hit_rate']=sum(r['target_span_hit'] for r in group)/len(group)
            groups[kind]['mean_token_f1']=sum(r['token_f1'] for r in group)/len(group)
    save(work/'comparison.json',dict(status='complete',groups=groups,
        interpretation='Only a four-fact overfit diagnostic; inspect replies before choosing any new full-scale regimen.',
        completed_at=now()))
    status('complete',results='comparison.json')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot',type=Path,required=True)
    args=p.parse_args()
    try:run(args)
    except Exception as error:
        save(args.pilot/'sanity_overfit4/status.json',dict(stage='failed',error=repr(error),updated_at=now(),pid=os.getpid()))
        raise
