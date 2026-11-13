#!/usr/bin/env python3
"""Material-derived, name-sensitive application practice; preserve all original PNGs."""
import argparse
import copy
import json
import math
import os
import random
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import explore_v7_prompt_loss_bridge as previous
import explore_v7_visual_replay as replay
import run_v7_evaluation as ev
from build_v7_experiment_data import name
from run_v7_full_experiment import ROOT, rows, read, save, sha, train_command, environment, final_checkpoint
from v7_gpu_resources import reserve

OUT=ROOT/'runs/v7_application_binding_20261018'
ARMS=('text_apply','vision_apply','vision_apply_replay')
CONTROLS=('warmup','text_equal','vision_equal')
PLUGIN=ROOT/'scripts/v7_equal_example_loss.py'
REFERENCE='Original learning reference:\n<image>\n\n'


def solve(rule,obs):
    flags=[]
    for c in rule['conditions']:
        v=obs[c['field']];want=c['value']
        flags.append({'eq':lambda:v==want,'ge':lambda:v>=want,'lt':lambda:v<want}[c['operator']]())
    return any(flags) if rule['mode']=='any' else all(flags)


def result(rule,obs):
    return '; '.join(f'{letter}={"KEEP" if solve(rule,o) else "OMIT"}' for letter,o in zip('ABCD',obs))


def signature(obs):
    # Exclude even reordered copies of candidate sets across train and generated holdout.
    return ev.digest(sorted(json.dumps(o,sort_keys=True) for o in obs))


def wording(obs):
    return 'The observed candidate set is: '+'; '.join(letter+' has '+', '.join(k+': '+str(v) for k,v in o.items()) for letter,o in zip('ABCD',obs))+'.'


def construct(groups,knowledge):
    rules={q:json.loads(k['fact']['quote']) for q,k in knowledge.items()}
    rng=random.Random('application-binding-fixed-before-eval:20261018:42')
    blocks=[]
    for gi,group in enumerate(groups):
        seen=set()
        for index in range(10):
            for attempt in range(100000):
                obs=([{'shape':rng.choice(('triangle','pentagon','square','circle')),'color':rng.choice(('blue','green','red','black'))} for _ in range(4)]
                     if gi==0 else [{'value':v} for v in rng.sample(range(25),4)])
                sig=signature(obs)
                outputs={q:result(rules[q],obs) for q in group}
                if sig not in seen and len(set(outputs.values()))==4:break
            else:raise ValueError('No distinct-rule observation found')
            seen.add(sig);options=list(outputs.values());rng.shuffle(options)
            # One submitted result is correct under one of the four named rules,
            # wrong under the other three. Four names remain gold ABCD per block.
            report=options[index%4]
            blocks.append({'id':f'g{gi}:b{index}','group':gi,'split':'train' if index<6 else 'holdout',
                'index':index,'rule_ids':group,'observations':obs,'observation_signature':sig,
                'results':outputs,'base_options':options,'submitted_report':report})
    return blocks


def application(block,q,knowledge,task,shift):
    opts=block['base_options'][shift:]+block['base_options'][:shift]
    report=block['submitted_report'];gold='ABCD'[opts.index(block['results'][q])]
    heading=('Apply '+name(knowledge[q])+'. Report KEEP/OMIT for each candidate.' if task=='direct' else
        'A technician submitted this report under '+name(knowledge[q])+': Result: '+report+'. Check the observations and the rule. Report KEEP/OMIT for each candidate.')
    contents=[('Result: '+v+'.' if task=='direct' else ('Accept report. ' if v==report else 'Replace report. ')+'Result: '+v+'.') for v in opts]
    body=heading+' KEEP means selected and OMIT means not selected.\n\nTest input:\n'+wording(block['observations'])
    body+='\n\nOptions:\n'+'\n'.join(l+'. '+v for l,v in zip('ABCD',contents))+ev.SUFFIX
    return {'id':f'{block["id"]}:{task}:{shift}:{q}','qid':q,'block':block['id']+':'+task,
        'group':block['group'],'task':task,'shift':shift,'fmt':'mcq','gold':gold,
        'observation_signature':block['observation_signature'],
        'messages':[{'role':'system','content':ev.SYSTEM},{'role':'user','content':body}]}


def add_reference(p):
    z=copy.deepcopy(p);z['messages'][1]['content']=REFERENCE+z['messages'][1]['content']
    z['images']=[str(previous.source.DATA/'images'/z['qid']/'learning_visual.png')]
    return z


def checkpoint(arm):
    if arm in CONTROLS:return previous.checkpoint(arm)
    return OUT/arm/'checkpoint/checkpoint-128'


def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'manifest.json').exists():validate();return
    previous.validate();previous.source.validate(previous.source.OUT);previous.paired.validate()
    knowledge={k['id']:k for k in rows(previous.source.OUT/'target_knowledge.jsonl')}
    groups=read(previous.source.OUT/'selection.json')['groups']
    assert len(groups)==2 and set(knowledge)==set(previous.source.EXPECTED)
    blocks=construct(groups,knowledge);ev.save_rows(OUT/'blocks.jsonl',blocks)
    ev.save_rows(OUT/'target_knowledge.jsonl',list(knowledge.values()))
    explanation=[]
    forms=['Explain the complete rule named {}.','Recall all conditions and comparisons for {}.',
           'State the exact stored rule for {}.','Describe the complete decision rule for {}.']
    for q,k in knowledge.items():
        for i in range(16):
            explanation.append({'id':f'explain:{q}:{i}','qid':q,'channel':'explain',
                'messages':[{'role':'system','content':previous.atomic.SYSTEM},
                    {'role':'user','content':forms[i%4].format(name(k))},
                    {'role':'assistant','content':previous.source.card(k)}]})
    app=[];extra=[]
    for b in blocks:
        for task in ('direct','diagnosis'):
            for shift in range(4):
                for q in b['rule_ids']:
                    p=application(b,q,knowledge,task,shift)
                    if b['split']=='train':
                        z=copy.deepcopy(p);z['channel']='application'
                        z['messages'].append({'role':'assistant','content':p['gold']});app.append(z)
                        # Two training blocks per topology, two positions; not held out.
                        if b['index']<2 and shift in (0,2):
                            for image in (False,True):
                                e=add_reference(p) if image else copy.deepcopy(p)
                                e.update(id=('seen_image:' if image else 'seen_closed:')+p['id'],
                                    condition='seen_image' if image else 'seen_closed')
                                extra.append(e)
                    else:
                        e=copy.deepcopy(p);e.update(id='unseen_closed:'+p['id'],condition='unseen_closed');extra.append(e)
                        if shift==0:
                            e=add_reference(p);e.update(id='unseen_image:'+p['id'],condition='unseen_image');extra.append(e)
    assert len(explanation)==128 and len(app)==384 and len(extra)==448
    training=explanation+app;random.Random(42).shuffle(training)
    for arm in ARMS:
        rr=[]
        for r in training:
            z=copy.deepcopy(r)
            if arm!='text_apply':
                # Same practice content/answers/order; replay removes only image and reference prefix.
                replayed=arm.endswith('replay') and r['channel']=='application' and int(r['id'].split(':')[1][1:])%2==0
                if not replayed:z=add_reference(z)
                z['replayed_without_learning_image']=replayed
            rr.append(z)
        assert len(rr)==512
        ev.save_rows(OUT/(arm+'.jsonl'),rr)
    # Training has now been fully written. Test observations/labels are read only
    # to audit disjointness and copy the unchanged existing evaluation protocol.
    original=rows(previous.source.OUT/'original_case_audit.jsonl')
    train_signatures={b['observation_signature'] for b in blocks if b['split']=='train'}
    heldout_signatures={b['observation_signature'] for b in blocks if b['split']=='holdout'}
    original_signatures={signature(c['observations']) for c in original}
    assert not train_signatures&heldout_signatures
    assert not (train_signatures|heldout_signatures)&original_signatures,'Generated/original scene collision: stop, do not silently replace.'
    main=rows(previous.OUT/'main_probes.jsonl');assert len(main)==568
    ev.save_rows(OUT/'main_probes.jsonl',main);ev.save_rows(OUT/'diagnostic_probes.jsonl',extra)
    save(OUT/'disjointness.json',{'status':'pass','training_unique_candidate_sets':12,'heldout_unique_candidate_sets':8,
        'overlap_train_holdout':0,'overlap_with_original16_candidate_sets':0,
        'comparison':'Candidate multiset SHA including attribute values; also excludes permutations. Atomic attributes and truth patterns can overlap.'})
    sources=[previous.OUT/'manifest.json',previous.OUT/'final_artifact_manifest.json',previous.OUT/'main_probes.jsonl',
        previous.source.OUT/'selection.json',previous.source.OUT/'target_knowledge.jsonl',previous.source.OUT/'original_case_audit.jsonl',
        ROOT/'scripts/explore_v7_prompt_loss_bridge.py',ROOT/'scripts/explore_v7_visual_replay.py',
        ROOT/'scripts/run_v7_evaluation.py',ROOT/'scripts/run_v7_full_experiment.py',ROOT/'scripts/v7_gpu_resources.py',PLUGIN]
    save(OUT/'manifest.json',{'created_at':ev.now(),'script_sha256':sha(Path(__file__)),
        'sources':{str(p):sha(p) for p in sources},'files':{p.name:sha(p) for p in OUT.iterdir() if p.suffix in ('.json','.jsonl')},
        'arms':ARMS,'controls':CONTROLS,'parent':ev.checkpoint(checkpoint('warmup')),
        'training':'128canonical explanations +384application MCQs. Same8development rules; no independent confirmatory holdout rules.',
        'generation':'6new candidate sets per topology x2tasks x4rule names x4positions =384practice answers. Four rule names yield four distinct correct results on each scene.',
        'vision_scope':'VISION uses original learning PNG plus new textual candidate observations. Canonical explanation targets contain textual facts; MCQ labels are computed from annotated rules. Not pixel-only self-supervision.',
        'replay_scope':'192application rows remove learning image/reference prefix, retain same text practice and answer letter. Explanations128allretainimage. Image-derived closed-book answer supervision, NOT pure VISION.',
        'training_config':{'epochs':1,'batch':4,'accumulation':1,'steps':128,'lr':1e-4,'seed':42,'rank':8,'alpha':32,'dropout':.05,'freeze_vit':True,'freeze_aligner':True,'max_length':8192,'loss':'equal_example','schedule':'cosine,5%warmup','shuffle':True},
        'evaluation':'Original568main prompts unchanged; 448extra diagnostic prompts per arm/control. No external materials in named/paired/unseen_closed/seen_closed. seen_* are capacity controls, not held out. *_image supplies correct original learning image, not closed-book transfer.',
        'limits':['Prior full1000models have seen these rules; no model globally unexposed claim. Selected parent has disjoint task warmup rules.',
                  'Same8development rules and1seed; application practice and explanation/match controls differ in objectives and row allocation; not an isolated single-factor ablation.',
                  'New heldout cases are only textual. Original application PNGs remain the only image tests. All4000+answers include dependent option rotations.',
                  'Fixed finalcheckpoint128only; no checkpoint selection by test accuracy; no nativeCoT,noPT,newimagesnone.']})
    snap=OUT/'source_snapshots';snap.mkdir(exist_ok=True)
    for p in [Path(__file__)]+[p for p in sources if p.suffix=='.py']:shutil.copy2(p,snap/p.name)
    validate();print('Prepared3x512rows;568main+448diagnostics/newarm,448/control.',flush=True)


def validate():
    m=read(OUT/'manifest.json');assert m['script_sha256']==sha(Path(__file__))
    for p,h in m['sources'].items():assert sha(Path(p))==h,p
    for p,h in m['files'].items():assert sha(OUT/p)==h,p
    assert ev.checkpoint(checkpoint('warmup'))==m['parent']


def train(arm,gpu,smoke=False):
    folder=OUT/('smoke_'+arm if smoke else arm);folder.mkdir(exist_ok=True)
    if smoke and (folder/'validated.json').exists():return
    cpdir=folder/'checkpoint';cp=None if smoke else final_checkpoint(cpdir,1)
    if cp is None:
        recover=[p for p in cpdir.glob('checkpoint-*') if all((p/f).exists() for f in ('optimizer.pt','scheduler.pt','rng_state.pth','adapter_model.safetensors','trainer_state.json'))]
        resume=max(recover,key=lambda p:read(p/'trainer_state.json')['global_step']) if recover and not smoke else None
        cmd=train_command('qwen3','sft',OUT/(arm+'.jsonl'),cpdir,1,resume=resume)
        for flag,value in [('--per_device_train_batch_size','4'),('--gradient_accumulation_steps','1'),('--save_strategy','steps')]:cmd[cmd.index(flag)+1]=value
        cmd+=['--save_steps','1' if smoke else '64','--external_plugins',str(PLUGIN),'--loss_type','v7_equal_example','--packing','false','--padding_free','false']
        if not resume:cmd+=['--adapters',str(checkpoint('warmup'))]
        if smoke:cmd+=['--max_steps','1']
        env=environment(gpu,previous.source.DATA);env['V7_EQUAL_LOSS_AUDIT']=str(folder/'loss_batches.jsonl')
        save(folder/'train_state.json',{'status':'queued','command':cmd})
        with reserve(previous.source.LEDGER,gpu,str(folder)+':train',27000 if arm.startswith('vision') else 22000) as res:
            with (folder/'train.log').open('a') as log:
                process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                res.bind(process.pid);save(folder/'train_state.json',{'status':'running','pid':process.pid,'command':cmd});code=process.wait()
            save(folder/'process_exit.json',{'returncode':code});assert code==0,(arm,code)
        cp=cpdir/('checkpoint-1' if smoke else 'checkpoint-128')
    state=read(cp/'trainer_state.json');assert state['global_step']==(1 if smoke else 128)
    losses=[r['loss'] for r in rows(folder/'loss_batches.jsonl')];assert losses and all(math.isfinite(x) for x in losses)
    save(folder/'checkpoint_manifest.json',{**ev.checkpoint(cp),'parent':ev.checkpoint(checkpoint('warmup')),'epoch':state['epoch'],'step':state['global_step'],'selection':'fixed final1epoch'})
    if smoke:
        pp=rows(OUT/'diagnostic_probes.jsonl');probe=[next(p for p in pp if p['condition']==c) for c in ('unseen_closed','unseen_image')]
        cfg=replay.config(cp,{'mcq':16});cfg['probes_sha256']=ev.digest(probe);run=replay.Runner(folder/'reload',cfg,gpu)
        if len(run.rr)<2:
            with reserve(previous.source.LEDGER,gpu,str(folder)+':reload',24000) as res:res.bind(os.getpid());run.load();run.infer(probe)
        run.audit(probe,2);save(folder/'validated.json',{'status':'pass','finite_loss':True,'reload_text_and_image':True})
    save(folder/'train_state.json',{'status':'complete','checkpoint':str(cp)})


def evaluate(arm,gpu):
    probes=rows(OUT/'diagnostic_probes.jsonl')
    if arm in ARMS:probes=rows(OUT/'main_probes.jsonl')+probes
    cfg=replay.config(checkpoint(arm),{'mcq':16,'card':384});cfg['probes_sha256']=ev.digest(probes)
    folder=OUT/arm/'evaluation';run=replay.Runner(folder,cfg,gpu)
    if len(run.rr)<len(probes):
        with reserve(previous.source.LEDGER,gpu,str(folder)+':eval',24000) as res:
            res.bind(os.getpid());run.load()
            for fmt in ('card','mcq'):run.infer([p for p in probes if p['fmt']==fmt])
    run.audit(probes,len(probes))
    save(folder/'report.json',{'status':'complete','metrics':{c:previous.source.metric([r for r in run.rr if r['condition']==c]) for c in sorted({r['condition'] for r in run.rr if r['fmt']=='mcq'})}})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True,choices=('prepare','smoke','train','eval','run'))
    ap.add_argument('--arm',choices=ARMS+CONTROLS);ap.add_argument('--gpu',default='0');a=ap.parse_args()
    if a.stage=='prepare':prepare()
    else:
        validate()
        with ev.run_lock(OUT/(str(a.arm)+'_'+a.stage+'.lock')):
            if a.stage in ('train','smoke','run'):train(a.arm,a.gpu,a.stage=='smoke')
            if a.stage in ('eval','run'):evaluate(a.arm,a.gpu)
