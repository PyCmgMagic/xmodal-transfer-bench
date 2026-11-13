#!/usr/bin/env python3
"""Fixed-rule prompt ablations, equal-example loss, and explicitly labelled image-dropout bridge."""
import argparse
import copy
import importlib.metadata
import json
import math
import os
import random
import shutil
import subprocess
from pathlib import Path

import confirm_v7_rule_recognition as source
import confirm_v7_paired_application as paired
import explore_v7_visual_replay as replay
import pilot_v7_atomic_supervision as atomic
import run_v7_evaluation as ev
from run_v7_full_experiment import ROOT, rows,read,save,sha,train_command,environment,final_checkpoint
from v7_gpu_resources import reserve

OUT=ROOT/'runs/v7_prompt_loss_bridge_20261016'
PLUGIN=ROOT/'scripts/v7_equal_example_loss.py'
ARMS=('text_equal','vision_equal','vision_equal_dropout')
CONTROLS=('warmup','text_match','text_mixed','vision_mixed')
NEUTRAL='Answer the user request accurately using the rules you learned. Do not invent facts.'


def checkpoint(arm):
    if arm=='warmup':return Path(read(source.PARENT)['path'])
    root=OUT if arm in ARMS else source.OUT
    return Path(read(root/arm/'checkpoint_manifest.json')['path'])


def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'manifest.json').exists():validate();return
    source.validate(source.OUT);paired.validate()
    dropped=[]
    for arm in ARMS:
        base='text_mixed' if arm=='text_equal' else 'vision_mixed'
        rr=copy.deepcopy(rows(source.OUT/(base+'.jsonl')))
        drop=set()
        if arm.endswith('dropout'):
            rng=random.Random('image-dropout-bridge:42')
            for q in source.EXPECTED:
                for channel in ('explain','match'):
                    strata=['ABCD'[i] for i in range(4)] if channel=='match' else [None]
                    for gold in strata:
                        ids=sorted(r['id'] for r in rr if r['qid']==q and r['channel']==channel and (gold is None or r['gold']==gold))
                        rng.shuffle(ids);drop.update(ids[:len(ids)//2])
            assert len(drop)==256
            text={r['id']:r for r in rows(source.OUT/'text_mixed.jsonl')}
            for r in rr:
                if r['id'] in drop:
                    before=ev.digest({'messages':r['messages'],'images':r['images']})
                    r['messages']=copy.deepcopy(text[r['id']]['messages']);r.pop('images')
                    assert r['messages'][-1]['content']==text[r['id']]['messages'][-1]['content']
                    dropped.append({'id':r['id'],'qid':r['qid'],'channel':r['channel'],
                        'before_input_sha256':before,'after_messages_sha256':ev.digest(r['messages'])})
        ev.save_rows(OUT/(arm+'.jsonl'),rr)
    ev.save_rows(OUT/'dropout_rows.jsonl',dropped)
    # Main results have the identical original prompts and option rotations.
    main=[copy.deepcopy(p) for p in rows(source.OUT/'probes.jsonl')]+rows(paired.OUT/'probes.jsonl')
    assert len(main)==568
    ev.save_rows(OUT/'main_probes.jsonl',main)
    # Change system only. No true definitions or observations are inserted.
    diagnostic=[]
    for p in main:
        if p['condition'].startswith(('named_','paired_')):
            z=copy.deepcopy(p);z.update(id='aligned_system:'+p['id'],condition='aligned_'+p['condition'],source_probe_id=p['id'])
            z['messages'][0]={'role':'system','content':atomic.SYSTEM};diagnostic.append(z)
        elif p['condition'] in ('recognition_para_noimage','recognition_para_image'):
            z=copy.deepcopy(p);z.update(id='strict_system:'+p['id'],condition='strict_'+p['condition'],source_probe_id=p['id'])
            z['messages'][0]={'role':'system','content':ev.SYSTEM};diagnostic.append(z)
        elif p['condition']=='recall':
            z=copy.deepcopy(p);z.update(id='neutral_system:'+p['id'],condition='neutral_recall',source_probe_id=p['id'])
            z['messages'][0]={'role':'system','content':NEUTRAL};diagnostic.append(z)
    assert len(diagnostic)==456
    ev.save_rows(OUT/'prompt_probes.jsonl',diagnostic)
    from v7_equal_example_loss import verify
    save(OUT/'loss_unit_validation.json',verify())
    sources=[source.OUT/'manifest.json',source.OUT/'final_artifact_manifest.json',source.OUT/'semantic_reviews.jsonl',
        source.OUT/'text_mixed.jsonl',source.OUT/'vision_mixed.jsonl',source.OUT/'probes.jsonl',paired.OUT/'probes.jsonl',
        source.PARENT,ROOT/'scripts/confirm_v7_rule_recognition.py',ROOT/'scripts/explore_v7_visual_replay.py',
        ROOT/'scripts/run_v7_full_experiment.py',ROOT/'scripts/run_v7_evaluation.py',ROOT/'scripts/v7_gpu_resources.py',PLUGIN]
    for a in CONTROLS:
        sources.append(source.OUT/a/'predictions.jsonl');sources.append(paired.OUT/a/'predictions.jsonl')
    save(OUT/'manifest.json',{'created_at':ev.now(),'script_sha256':sha(Path(__file__)),
        'sources':{str(p):sha(p) for p in sources},'files':{p.name:sha(p) for p in OUT.glob('*.jsonl')},
        'new_training_arms':ARMS,'prompt_control_arms':CONTROLS,'parent':ev.checkpoint(checkpoint('warmup')),
        'train_rows':512,'steps':128,'epochs':1,'batch':4,'accumulation':1,'lr':1e-4,'seed':42,
        'loss':'Mean of within-example mean causal CE. Includes same template suffix labels; singleGPU, packing=false, padding_free=false.',
        'contrast':'text_equal vs original text_mixed; vision_equal vs original vision_mixed: same data,row order,parent,steps; change loss only. vision_equal_dropout vs vision_equal: remove image and its reference prefix on fixed256rows; targets and row order unchanged.',
        'dropout_classification':'Image-derived textual bridge with human-annotated canonical targets, NOT pure visual knowledge acquisition. No teacher generation or score-based filtering.',
        'prompt_contrast':'Change only system: application original strict -> training system; definition recognition training -> strict; recall training -> neutral. User messages/options/images unchanged.',
        'new_responses_planned':3528,'main_per_new_arm':568,'diagnostic_per_control':456,
        'limits':['Same8development rules as confirmation; no new independent replication;1seed.',
                  'No application observations or labels in training. Candidate definitions in match rows disclose rule facts.',
                  'No nativeCoT,no PT,no original images changed; final epoch1checkpoint only.',
                  'Prompt ablations may alter output format; invalid answers counted wrong and reported separately.',
                  'Image dropout changes both image presence and reference prefix, classified as a bridge package, not isolated image-only effect.']})
    snap=OUT/'source_snapshots';snap.mkdir(exist_ok=True)
    for p in [Path(__file__)]+[p for p in sources if p.suffix=='.py']:shutil.copy2(p,snap/p.name)
    validate();print('Prepared3x512training rows;4x456prompt controls;3x568main answers.',flush=True)


def validate():
    m=read(OUT/'manifest.json');assert m['script_sha256']==sha(Path(__file__))
    for path,h in m['sources'].items():assert sha(Path(path))==h,path
    for path,h in m['files'].items():assert sha(OUT/path)==h,path
    assert m['parent']==ev.checkpoint(checkpoint('warmup'))


def train(arm,gpu,smoke=False):
    folder=OUT/('smoke_'+arm if smoke else arm);folder.mkdir(exist_ok=True)
    cpdir=folder/'checkpoint';cp=final_checkpoint(cpdir,1) if not smoke else None
    if smoke and (folder/'validated.json').exists():return
    if not cp:
        recover=[p for p in cpdir.glob('checkpoint-*') if all((p/f).exists() for f in ('optimizer.pt','scheduler.pt','rng_state.pth','adapter_model.safetensors','trainer_state.json'))]
        resume=max(recover,key=lambda p:read(p/'trainer_state.json')['global_step']) if recover and not smoke else None
        cmd=train_command('qwen3','sft',OUT/(arm+'.jsonl'),cpdir,1,resume=resume)
        for flag,value in (('--per_device_train_batch_size','4'),('--gradient_accumulation_steps','1'),('--save_total_limit','3'),('--save_strategy','steps')):
            cmd[cmd.index(flag)+1]=value
        cmd+=['--save_steps','1' if smoke else '32','--external_plugins',str(PLUGIN),'--loss_type','v7_equal_example',
              '--packing','false','--padding_free','false']
        if smoke:cmd+=['--max_steps','1']
        if not resume:cmd+=['--adapters',str(checkpoint('warmup'))]
        env=environment(gpu,source.DATA);env['V7_EQUAL_LOSS_AUDIT']=str(folder/'loss_batches.jsonl')
        save(folder/'train_state.json',{'status':'queued','command':cmd})
        with reserve(source.LEDGER,gpu,str(folder)+':train',26000 if arm.startswith('vision') else 22000) as res:
            with (folder/'train.log').open('a') as log:
                process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                res.bind(process.pid);save(folder/'train_state.json',{'status':'running','pid':process.pid,'command':cmd})
                code=process.wait()
            save(folder/'process_exit.json',{'returncode':code});assert code==0,(arm,code)
        cp=cpdir/('checkpoint-1' if smoke else 'checkpoint-128')
    state=read(cp/'trainer_state.json');args=read(cp/'args.json')
    assert state['global_step']==(1 if smoke else 128)
    if not smoke:assert abs(state['epoch']-1)<1e-6
    assert args['loss_type']=='v7_equal_example' and not args['packing'] and not args['padding_free']
    logs=[r for r in rows(cpdir/'logging.jsonl') if 'loss' in r];assert logs and all(math.isfinite(r['loss']) for r in logs)
    batches=rows(folder/'loss_batches.jsonl');assert batches and all(math.isfinite(r['loss']) for r in batches)
    from safetensors import safe_open
    with safe_open(str(cp/'adapter_model.safetensors'),framework='pt',device='cpu') as f:
        keys=list(f.keys());bs=[k for k in keys if 'lora_B' in k]
        assert len(bs)==252 and all(bool(f.get_tensor(k).count_nonzero()) for k in bs)
        assert not any('.visual.' in k or '.merger.' in k for k in keys)
    save(folder/'checkpoint_manifest.json',{**ev.checkpoint(cp),'epoch':state['epoch'],'global_step':state['global_step'],
        'parent':ev.checkpoint(checkpoint('warmup')),'last_logged_loss':logs[-1]['loss'],'selection':'fixed final checkpoint',
        'frozen_visual_aligner':True,'nonzero_lora_B':252})
    if smoke:
        # Reload and generate on text and original-learning-image inputs.
        pp=rows(OUT/'main_probes.jsonl');probe=[next(p for p in pp if p['condition']==c) for c in ('recognition_para_noimage','recognition_para_image')]
        config=replay.config(cp,{'mcq':16});config['smoke_probes_sha256']=ev.digest(probe)
        run=replay.Runner(folder/'reload',config,gpu)
        if len(run.rr)<2:
            with reserve(source.LEDGER,gpu,str(folder)+':reload',24000) as res:
                res.bind(os.getpid());run.load();run.infer(probe)
        run.audit(probe,2);save(folder/'validated.json',{'status':'pass','train_batch':4,'text_and_image_reload':True})
    save(folder/'train_state.json',{'status':'complete','checkpoint':str(cp)})


def evaluate(arm,gpu,diagnostic=False):
    folder=OUT/('prompt_controls' if diagnostic else 'main')/arm
    pp=rows(OUT/('prompt_probes.jsonl' if diagnostic else 'main_probes.jsonl'))
    config=replay.config(checkpoint(arm),{'mcq':16,'card':384});config['probes_sha256']=ev.digest(pp)
    run=replay.Runner(folder,config,gpu)
    if len(run.rr)<len(pp):
        with reserve(source.LEDGER,gpu,str(folder)+':eval',24000) as res:
            res.bind(os.getpid());run.load()
            for fmt in ('card','mcq'):run.infer([p for p in pp if p['fmt']==fmt])
    run.audit(pp,len(pp))
    save(folder/'report.json',{'status':'complete','metrics':{c:source.metric([r for r in run.rr if r['condition']==c])
        for c in sorted({r['condition'] for r in run.rr if r['fmt']=='mcq'})}})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('prepare','smoke','train','eval','run','prompt'),required=True)
    ap.add_argument('--arm',choices=(*ARMS,*CONTROLS));ap.add_argument('--gpu',default='0');a=ap.parse_args()
    if a.stage=='prepare':prepare()
    else:
        validate()
        with ev.run_lock(OUT/(str(a.arm)+'_'+a.stage+'.lock')):
            if a.stage in ('smoke','train','run'):train(a.arm,a.gpu,a.stage=='smoke')
            if a.stage in ('eval','run'):evaluate(a.arm,a.gpu)
            if a.stage=='prompt':evaluate(a.arm,a.gpu,True)
