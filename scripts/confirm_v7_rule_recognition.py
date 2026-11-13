#!/usr/bin/env python3
"""Predeclared new-rule comparison of explanation, recognition and mixed SFT."""
import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import explore_v7_rule_recognition as prior
import explore_v7_visual_replay as replay
import run_v7_evaluation as ev
from build_v7_experiment_data import name
from pilot_v7_atomic_supervision import PARENT, msg
from pilot_v7_task_then_transfer import DATA, LEDGER, card, rotate
from pilot_v7_counterfactual_names import solve, independent, assignment
from run_v7_full_experiment import ROOT, rows, read, save, sha, environment, train_command, final_checkpoint
from summarize_v7_counterfactual_names import metric
from v7_gpu_resources import reserve

OUT=ROOT/'runs/v7_recognition_confirm_20261016'
ARMS=tuple(m+'_'+t for m in ('text','vision') for t in ('explain','match','mixed'))
EXPECTED=['scale1k_'+s for s in ('0955_programmatic','0270_programmatic','0085_programmatic','0953_programmatic',
    '0966_programmatic','0963_programmatic','0274_programmatic','0698_ai_generated')]


def prior_inventory(out):
    used=set();files={}
    for p in (ROOT/'runs').glob('v7_*/**/*.jsonl'):
        if out in p.parents or any(x in p.parts for x in ('checkpoint','evaluations','source_snapshots','dataset','analysis')):
            continue
        if p.stat().st_size>4_000_000:continue
        try:rr=rows(p)
        except (ValueError,UnicodeError):continue
        if not rr or len(rr)>=1000:continue
        eligible=p.name.endswith(('knowledge.jsonl','probes.jsonl')) or any(
            any(m.get('role')=='assistant' for m in row.get('messages',[])) for row in rr[:4])
        if not eligible:continue
        ids=set()
        for row in rr:
            q=row.get('qid',row.get('id',row.get('content',{}).get('id','')))
            match=re.match(r'(scale1k_\d{4}_(?:programmatic|ai_generated))',str(q))
            if match:ids.add(match.group(1))
        if ids:used|=ids;files[str(p)]={'sha256':sha(p),'rule_ids':sorted(ids)}
    return used,files


def validate(out):
    m=read(out/'manifest.json');assert m['script_sha256']==sha(Path(__file__))
    for path,h in m['sources'].items():assert sha(Path(path))==h,path
    for path,h in m['files'].items():assert sha(out/path)==h,path
    for r in rows(DATA/'image_hashes.jsonl'):
        assert sha(DATA/r['path'])==r['source_sha256']==r['repaired_sha256']
    assert ev.checkpoint(Path(read(PARENT)['path']))==m['parent']


def observations(text):
    text=text.split('Test input:\n')[1].split('\n\nOptions:')[0]
    parts=re.findall(r'([ABCD]) has (.*?)(?=; [ABCD] has|\.$)',text)
    assert [l for l,_ in parts]==list('ABCD')
    return [{f:int(v) if v.isdigit() else v for f,v in re.findall(r'(\w+): ([^,]+)(?:, |$)',body)} for _,body in parts]


def prepare(out):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():validate(out);return
    used,inventory=prior_inventory(out)
    knowledge=[r['content'] for r in rows(DATA/'audit/rule_knowledge.jsonl')]
    groups=[]
    for topology in ('disjunctive_filter','interval_filter'):
        eligible=[k for k in knowledge if k['id'] not in used and json.loads(k['fact']['quote']).get('selection_topology')==topology]
        ordered=sorted(eligible,key=lambda k:(not k['id'].endswith('programmatic'),
            hashlib.sha256(('recognition-confirm:42:'+k['id']).encode()).hexdigest()))
        groups.append(ordered[:4])
    target=[k for group in groups for k in group];assert [k['id'] for k in target]==EXPECTED
    assert not set(EXPECTED)&used
    ev.save_rows(out/'target_knowledge.jsonl',target)
    save(out/'selection.json',{'groups':[[k['id'] for k in g] for g in groups],
        'excluded_ids':sorted(used),'prior_file_inventory':inventory,
        'algorithm':'Exclude IDs in indexed small pilot training/knowledge/probes, then prefer programmatic originals and sort by SHA256(recognition-confirm:42:ID), take first4per topology. No model scores or target test contents used for selection.',
        'scope':'New to indexed repair development and parent warmup. Historical1000rule models saw all v7; those checkpoints are NOT ancestors. This is new-rule replication, not a wholly untouched dataset.'})
    review=[]
    for k in target:
        rule=json.loads(k['fact']['quote']);lo,hi=rule['conditions']
        evidence=(f"Shape bay {lo['value']}; color bay {hi['value']}; MATCH ANY / OR visible." if lo['field']=='shape' else
            f"Lower {lo['value']} filled/included; upper {hi['value']} open/excluded. " +
            ('Arabic2and4are explicitly pictured.' if k['id'].endswith('ai_generated') else 'Dot counts manually counted in both bays.'))
        review.append({'qid':k['id'],'image':str(DATA/'images'/k['id']/'learning_visual.png'),
                       'status':'pass','method':'Manual viewing before training','evidence':evidence})
    ev.save_rows(out/'learning_image_reviews.jsonl',review)
    # Complete training before opening target evaluation questions/labels.
    orders=[tuple((j+s)%4 for j in base) for base in ((0,1,2,3),(0,2,1,3)) for s in range(4)]
    forms=('Explain the complete rule named {}.','State the exact stored rule for {}.',
           'Recall all conditions and comparisons for {}.','Describe the complete decision rule for {}.')
    for arm in ARMS:
        modality,style=arm.split('_');training=[]
        for group in groups:
            for k in group:
                for i in range(64):
                    channel='match' if style=='match' or (style=='mixed' and i%2) else 'explain'
                    n=i//2 if style=='mixed' else i
                    if channel=='match':
                        p=prior.match_probe(k,group,orders[n%8],(n//8)%4,modality=='vision')
                        z={key:copy.deepcopy(p[key]) for key in ('messages','images','option_rule_ids') if key in p}
                        z.update(gold=p['gold']);z['messages'].append({'role':'assistant','content':p['gold']})
                    else:
                        q=forms[n%4].format(name(k))
                        if modality=='vision':q='<image>Using only this original learning reference, '+q[0].lower()+q[1:]
                        z={'messages':msg(q,card(k))}
                        if modality=='vision':z['images']=[str(DATA/'images'/k['id']/'learning_visual.png')]
                    z.update(id=f'{k["id"]}:{i}',qid=k['id'],channel=channel);training.append(z)
        random.Random(42).shuffle(training)
        assert len(training)==512
        for k in target:
            subset=[r for r in training if r['qid']==k['id']]
            assert len(subset)==64
            match=[r for r in subset if r['channel']=='match']
            if match:assert Counter(r['gold'] for r in match)==Counter({l:len(match)//4 for l in 'ABCD'})
        ev.save_rows(out/(arm+'.jsonl'),training)
    # Frozen original evaluation:16questions,16original test PNGs,4option rotations.
    labels={r['id']:r for r in rows(DATA/'audit/eval_labels.jsonl')}
    raw={d:rows(DATA/d/'eval.jsonl') for d in ('T2T','T2V','V2T','V2V')}
    assert raw['T2T']==raw['V2T'] and raw['T2V']==raw['V2V']
    lookup={k['id']:k for k in target};probes=[];cases=[]
    for r in raw['T2T']:
        if r['qid'] not in lookup:continue
        k=lookup[r['qid']];rule=json.loads(k['fact']['quote']);obs=observations(r['messages'][-1]['content'])
        results=[solve(rule,o) for o in obs];assert results==[independent(rule,o) for o in obs]
        expected=assignment(results);opts=dict(re.findall(r'(?m)^([A-D])\. (.*)$',r['messages'][-1]['content']))
        assert len(opts)==len(set(opts.values()))==4
        matching=[l for l,value in opts.items() if value.split('Result: ')[1].rstrip('.')==expected]
        assert matching==[labels[r['id']]['answer']]
        vr=next(v for v in raw['T2V'] if v['id']==r['id'])
        cases.append({'id':r['id'],'qid':k['id'],'observations':obs,'expected':expected,
            'gold':matching[0],'original_test_image':str(DATA/vr['images'][0]),'image_review':'Manual viewed all16original tests; observations and counts agree.','question_type':labels[r['id']]['question_type']})
        for modality,direction,row in [('T','T2T',r),('V','T2V',vr)]:
            base=ev.prepare_row(row,DATA,direction)
            for shift in range(4):
                q,gold,_=rotate(base['messages'][-1]['content'],labels[r['id']]['answer'],shift)
                p={'id':f'named:{modality}:{r["id"]}:{shift}','qid':k['id'],'source_id':r['id'],'shift':shift,
                    'condition':'named_'+modality,'modality':modality,'task':labels[r['id']]['question_type'],
                    'evaluation_group':labels[r['id']]['evaluation_group'],'group':int(rule['selection_topology']=='interval_filter'),
                    'fmt':'mcq','gold':gold,'messages':[{'role':'system','content':ev.SYSTEM},{'role':'user','content':q}]}
                if modality=='V':p['images']=base['images']
                probes.append(p)
                if shift==0:
                    oracle=copy.deepcopy(p);oracle.update(id='oracle:'+p['id'],condition='oracle_'+modality)
                    oracle['messages'][-1]['content']='Correct learning reference:\n'+card(k)+'\n\n'+q
                    probes.append(oracle)
    assert len(cases)==16 and len(probes)==160
    ev.save_rows(out/'original_case_audit.jsonl',cases)
    for condition in ('recognition_exact_noimage','recognition_exact_image','recognition_para_noimage','recognition_para_image'):
        for group in groups:
            for k in group:
                for shift in range(4):
                    p=prior.match_probe(k,group,tuple((j+shift)%4 for j in range(4)),0,
                        condition.endswith('_image'),'_para_' in condition)
                    p.update(id=f'{condition}:{k["id"]}:{shift}',condition=condition,shift=shift)
                    probes.append(p)
    for condition in ('recall','recall_image','plain_recall'):
        for k in target:
            q='Explain the complete rule named '+name(k)+'.'
            if condition=='recall_image':q='<image>Using this original learning reference, '+q[0].lower()+q[1:]
            p={'id':condition+':'+k['id'],'qid':k['id'],'condition':condition,'fmt':'card',
                'messages':msg(q) if condition!='plain_recall' else [{'role':'user','content':q}]}
            if condition=='recall_image':p['images']=[str(DATA/'images'/k['id']/'learning_visual.png')]
            probes.append(p)
    assert len(probes)==312
    ev.save_rows(out/'probes.jsonl',probes)
    sources=[PARENT,DATA/'audit/rule_knowledge.jsonl',DATA/'audit/eval_labels.jsonl',DATA/'image_hashes.jsonl']
    sources += [DATA/d/'eval.jsonl' for d in raw]
    sources += [ROOT/'scripts'/s for s in ('explore_v7_rule_recognition.py','explore_v7_visual_replay.py',
        'pilot_v7_atomic_supervision.py','pilot_v7_task_then_transfer.py','pilot_v7_counterfactual_names.py',
        'run_v7_full_experiment.py','run_v7_evaluation.py','v7_gpu_resources.py')]
    save(out/'manifest.json',{'script_sha256':sha(Path(__file__)),'created_at':ev.now(),'arms':ARMS,
        'parent':ev.checkpoint(Path(read(PARENT)['path'])),'target_ids':EXPECTED,'training_rows':512,'steps':128,
        'epoch':1,'batch':4,'accumulation':1,'lr':1e-4,'seed':42,'max_length':8192,
        'lora':{'rank':8,'alpha':32,'dropout':0.05,'freeze_vision_aligner':True},
        'design':'2modalities x3supervision types. Explain64/rule; match64/rule; mixed32explanations+32recognitions/rule. Fresh optimizer, same warmup parent, fixed last checkpoint. No target test observations or application labels train any arm.',
        'primary':'Unchanged original16v7questions per input modality (shift0), plus4rotations=64answers per input. Both separate. No external learning material in named inputs.',
        'diagnostics':'Self uses8uncorrected aligned name-only recalls; oracle gives correct rule only on shift0; recognition choices contain rule knowledge and are NOT material-free application.',
        'generations_per_arm':440,'total_arms_including_warmup':7,'new_generations_planned':3080,
        'limits':['Only8newrules,1seed,2selector families; four option positions are dependent repeats.',
            'No match of supervised/input token counts or aggregate candidate-fact exposure; mixed halves exposures for each objective.',
            'Originalv7options are NOT rebuilt to enforce equal rule-independent semantic priors. Warmup baseline and rotations necessary; 25%is random reference, not a guaranteed untrained score.',
            'VISION recognition/mixed input options contain textual candidate definitions. Do not claim absence of text fact supervision.',
            'Dataset and families already developed previously; new-rule replication within same1000rule package, not new independent benchmark.',
            'No PT training,no nativeCoT,no originalPNG or source-data mutation.'],
        'sources':{str(p):sha(p) for p in sources},'files':{p.name:sha(p) for p in out.glob('*.jsonl')}|{'selection.json':sha(out/'selection.json')}})
    snap=out/'source_snapshots';snap.mkdir(exist_ok=True)
    for p in [Path(__file__)]+[p for p in sources if p.suffix=='.py']:shutil.copy2(p,snap/p.name)
    validate(out);print('Prepared6x512training rows,312fixed probes,8newrules.',flush=True)


def train(out,arm,gpu):
    folder=out/arm;folder.mkdir(exist_ok=True);cpdir=folder/'checkpoint'
    cp=final_checkpoint(cpdir,1)
    if cp is None:
        recover=[p for p in cpdir.glob('checkpoint-*') if all((p/f).exists() for f in
            ('optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','adapter_model.safetensors'))]
        resume=max(recover,key=lambda p:read(p/'trainer_state.json')['global_step']) if recover else None
        if cpdir.exists() and not resume:cpdir.rename(folder/('interrupted_'+str(int(__import__('time').time()))))
        cmd=train_command('qwen3','sft',out/(arm+'.jsonl'),cpdir,1,resume=resume)
        for flag,value in (('--per_device_train_batch_size','4'),('--gradient_accumulation_steps','1'),
                           ('--save_total_limit','3'),('--save_strategy','steps')):cmd[cmd.index(flag)+1]=value
        cmd+=['--save_steps','32']
        if not resume:cmd+=['--adapters',read(PARENT)['path']]
        save(folder/'train_state.json',{'status':'queued','command':cmd})
        with reserve(LEDGER,gpu,str(folder)+':train',29000 if arm.startswith('vision') else 22000) as res:
            with (folder/'train.log').open('a') as log:
                process=subprocess.Popen(cmd,cwd=ROOT,env=environment(gpu,DATA),stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                res.bind(process.pid);save(folder/'train_state.json',{'status':'running','pid':process.pid,'command':cmd})
                code=process.wait()
            assert code==0,(arm,code)
        cp=final_checkpoint(cpdir,1)
    assert cp
    st=read(cp/'trainer_state.json');args=read(cp/'args.json')
    assert st['global_step']==128 and abs(st['epoch']-1)<1e-6
    assert args['dataset']==[str(out/(arm+'.jsonl'))] and args['adapters']==[read(PARENT)['path']]
    losses=[r['loss'] for r in rows(cpdir/'logging.jsonl') if 'loss' in r];assert losses and all(math.isfinite(x) for x in losses)
    from safetensors import safe_open
    with safe_open(str(cp/'adapter_model.safetensors'),framework='pt',device='cpu') as f:
        keys=list(f.keys());bs=[k for k in keys if 'lora_B' in k]
        assert len(bs)==252 and all(bool(f.get_tensor(k).count_nonzero()) for k in bs)
        assert not any('.visual.' in k or '.merger.' in k for k in keys)
    save(folder/'checkpoint_manifest.json',{**ev.checkpoint(cp),'parent':ev.checkpoint(Path(read(PARENT)['path'])),
        'epoch':1,'global_step':128,'last_logged_loss':losses[-1],'nonzero_lora_B':252,'selection':'fixed last epoch1; no score selection'})
    save(folder/'train_state.json',{'status':'complete','checkpoint':str(cp)})


def evaluate(out,arm,gpu):
    folder=out/arm
    cp=Path(read(PARENT)['path']) if arm=='warmup' else Path(read(folder/'checkpoint_manifest.json')['path'])
    config=replay.config(cp,{'mcq':16,'card':384});config['probes_sha256']=sha(out/'probes.jsonl')
    run=replay.Runner(folder,config,gpu);pp=rows(out/'probes.jsonl')
    def own():
        recall={r['qid']:r for r in run.rr if r['condition']=='recall'};assert len(recall)==8
        probes=[]
        for p in pp:
            if not p['condition'].startswith('named_'):continue
            z=copy.deepcopy(p);text=recall[p['qid']]['response']
            z.update(id='self:'+p['id'],condition='self_'+p['modality'],recall_response_sha256=ev.digest(text))
            z['messages'][-1]['content']='Your own prior rule recall (uncorrected):\n'+text+'\n\n'+p['messages'][-1]['content']
            probes.append(z)
        return probes
    if len(run.rr)<440:
        with reserve(LEDGER,gpu,str(folder)+':eval',24000) as res:
            res.bind(os.getpid());run.load()
            for fmt in ('card','mcq'):run.infer([p for p in pp if p['fmt']==fmt])
            extra=own();ev.save_rows(folder/'self_probes.jsonl',extra);run.infer(extra)
    run.audit(pp+own(),440)
    metrics={}
    for c in sorted({r['condition'] for r in run.rr if r['fmt']=='mcq'}):
        rr=[r for r in run.rr if r['condition']==c];metrics[c]=metric(rr)
        if c.startswith(('named_','self_','oracle_')):
            metrics[c]['original_order']=metric([r for r in rr if r['shift']==0])
            metrics[c]['by_task']={t:metric([r for r in rr if r['task']==t]) for t in ('direct','diagnosis')}
            metrics[c]['by_group']={str(g):metric([r for r in rr if r['group']==g]) for g in (0,1)}
            metrics[c]['by_rule']={q:metric([r for r in rr if r['qid']==q]) for q in EXPECTED}
    save(folder/'report.json',{'status':'complete','metrics':metrics})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=OUT)
    ap.add_argument('--stage',choices=('prepare','train','eval','run'),required=True)
    ap.add_argument('--arm',choices=(*ARMS,'warmup'));ap.add_argument('--gpu',default='0');a=ap.parse_args();out=a.out.resolve()
    if a.stage=='prepare':prepare(out)
    else:
        validate(out)
        with ev.run_lock(out/(a.arm+'_'+a.stage+'.lock')):
            if a.stage in ('train','run'):train(out,a.arm,a.gpu)
            if a.stage in ('eval','run'):evaluate(out,a.arm,a.gpu)
