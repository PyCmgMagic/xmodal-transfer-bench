#!/usr/bin/env python3
"""Controlled output-protocol and image-context probes; no training or source edits."""
import argparse
import copy
import dataclasses
import fcntl
import json
import os
import re
from collections import Counter
from pathlib import Path

import run_v7_evaluation as ev
from build_v7_experiment_data import name
from pilot_v7_task_then_transfer import DATA, LEDGER, card, validate_sources
from run_v7_full_experiment import ROOT, MODELS, environment, read, rows, save, sha
from v7_gpu_resources import reserve

PILOT = ROOT/'runs/v7_task_then_transfer_20261015/pilot_v2'
ARMS = ('base','warmup','text_alias','vision','full_sft_text','full_sft_vision')


def checkpoint(arm):
    if arm=='base':return None
    if arm.startswith('full_'):
        return ROOT/'runs/v7_full_experiment/jobs'/('qwen3_'+arm.removeprefix('full_'))/'checkpoint/checkpoint-315'
    return Path(read(PILOT/arm/'checkpoint_manifest.json')['path'])


def final_option(text):
    match=re.search(r'(?:^|\n)\s*FINAL:\s*([ABCD])\s*\Z',text)
    return match[1] if match and len(re.findall(r'(?:^|\n)\s*FINAL:',text))==1 else None


def prepare(out):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():
        m=read(out/'manifest.json');assert m['script_sha256']==sha(Path(__file__))
        for p,h in m['sources'].items():assert sha(Path(p))==h,p
        assert sha(out/'probes.jsonl')==m['probes_sha256']
        return
    validate_sources(PILOT)
    ks=rows(PILOT/'target_knowledge.jsonl');kmap={k['id']:k for k in ks}
    old=rows(PILOT/'probes.jsonl');direct=[p for p in old if p['condition'].startswith('direct_')]
    probes=[]
    for p in direct:
        for protocol in ('strict','strict_recall_cue','rule_then_option'):
            z=copy.deepcopy(p);z['id']=protocol+':'+p['id'];z['source_id']=p['id'];z['condition']=protocol+'_'+p['condition'][-1];z['fmt']='rule_option' if protocol=='rule_then_option' else 'mcq'
            if protocol=='strict_recall_cue':
                z['messages'][-1]['content']='Use the rule you learned for the named selector or category guide to solve this question.\n\n'+z['messages'][-1]['content']
            if protocol=='rule_then_option':
                text=z['messages'][-1]['content'];assert text.endswith(ev.SUFFIX)
                z['messages'][0]['content']='Solve the multiple-choice question using your learned rule. First state the applicable rule in one RULE: line, including its exact conditions or thresholds and category mapping. Then finish with one FINAL: A, FINAL: B, FINAL: C, or FINAL: D line. Do not invent rule facts.'
                z['messages'][-1]['content']=text[:-len(ev.SUFFIX)]+'\n\nFirst output RULE: with the exact applicable rule, then finish with FINAL: followed by one option letter.'
            probes.append(z)
    byfamily={rel:[k for k in ks if k['fact']['relation']==rel] for rel in {k['fact']['relation'] for k in ks}}
    donors={k['id']:group[(i+1)%len(group)]['id'] for group in byfamily.values() for i,k in enumerate(group)}
    for k in ks:
        template=next(p for p in old if p['qid']==k['id'] and p['condition']=='card_training_prompt')
        test_image=next(p['images'][0] for p in direct if p['qid']==k['id'] and p['condition']=='direct_V' and p['task']=='direct')
        for mode in ('absent','correct_learning','wrong_learning','test_observation'):
            z=copy.deepcopy(template);z['id']='recall_image_'+mode+':'+k['id'];z['condition']='recall_image_'+mode;z['fmt']='card';z['canonical_card']=card(k)
            if mode!='absent':
                donor=donors[k['id']];path=DATA/'images'/(donor if mode=='wrong_learning' else k['id'])/'learning_visual.png'
                z['images']=[test_image if mode=='test_observation' else str(path)];z['messages'][-1]['content']='<image>\n'+z['messages'][-1]['content']
                if mode=='wrong_learning':z.update(image_donor_qid=donor,image_donor_card=card(kmap[donor]))
            probes.append(z)
    assert len(probes)==128 and len({p['id'] for p in probes})==128
    ev.save_rows(out/'probes.jsonl',probes)
    paths={p for z in probes for p in z.get('images',[])}
    sources=[PILOT/'probes.jsonl',PILOT/'target_knowledge.jsonl',PILOT/'manifest.json',DATA/'image_hashes.jsonl']
    sources += [ROOT/'runs/v7_full_experiment/data'/f'{s}.jsonl' for s in ('sft_text','sft_vision')]
    save(out/'manifest.json',{'script_sha256':sha(Path(__file__)),'probes_sha256':sha(out/'probes.jsonl'),'sources':{str(p):sha(p) for p in sources},'selected_image_sha256':{p:sha(Path(p)) for p in sorted(paths)},'models':{a:ev.checkpoint(checkpoint(a)) if a!='base' else None for a in ARMS},'target_rules':[k['id'] for k in ks],'arms':ARMS,'conditions':dict(Counter(p['condition'] for p in probes)),'protocol':'Development mechanism study, original 8 target rules only. Primary original questions and all original images unchanged. Native thinking remains off; rule_then_option explicitly emits factual rule followed by choice in one generation.','image_factor':'Same canonical name query and same system; only image content/tag changes. Wrong donor fixed cyclically within family, never selected by performance.','scoring':'Strict letter or terminal FINAL: letter; raw rule output retained. Image recall literal and full-card-body equality are sufficient-only diagnostics, not semantic grading.','max_tokens':{'mcq':16,'rule_option':384,'card':384},'batch_size':4,'temperature':0,'seed':42,'full_adapters':'Historical fixed epoch5; pilots fixed epoch1. Full models have no task warmup, so comparisons with pilots are not matched training interventions.','created_at':ev.now()})
    (out/'source_snapshots').mkdir(exist_ok=True)
    import shutil
    for f in (Path(__file__),ROOT/'scripts/pilot_v7_task_then_transfer.py',ROOT/'scripts/run_v7_evaluation.py',ROOT/'scripts/v7_gpu_resources.py'):
        shutil.copy2(f,out/'source_snapshots'/f.name)


def verify(out,arm):
    m=read(out/'manifest.json')
    assert m['script_sha256']==sha(Path(__file__))
    assert sha(out/'probes.jsonl')==m['probes_sha256']
    assert m['models'][arm]==(ev.checkpoint(checkpoint(arm)) if arm!='base' else None)
    for p,h in m['selected_image_sha256'].items():assert sha(Path(p))==h,p


def body(text):
    return text.split(': ',1)[-1].strip()


def record(p,reply,reused=None):
    choice=reply['choices'][0];text=choice['message']['content'] or ''
    predicted=final_option(text) if p['fmt']=='rule_option' else ev.letter(text) if p['fmt']=='mcq' else None
    r={k:p[k] for k in ('id','qid','condition','fmt','task','relation','gold')}
    r.update(response=text,finish_reason=choice['finish_reason'],predicted=predicted,
             correct=(predicted==p['gold']) if p['fmt']!='card' else text.strip()==p['gold'],
             format_error=p['fmt']!='card' and predicted is None,
             prompt_sha256=ev.digest({'messages':p['messages'],'images':p.get('images',[])}),
             raw_response=reply,saved_at=ev.now())
    if p['fmt']=='card':
        r['requested_rule_body_exact']=body(text)==body(p['canonical_card'])
        if 'image_donor_qid' in p:
            r.update(image_donor_qid=p['image_donor_qid'],image_donor_body_exact=body(text)==body(p['image_donor_card']))
    if 'evaluation_group' in p:r['evaluation_group']=p['evaluation_group']
    if reused:r['reused_from']=reused
    return r


def run(out,arm,gpu):
    verify(out,arm);folder=out/arm;folder.mkdir(exist_ok=True)
    with ev.run_lock(folder/'.lock'):
        probes=rows(out/'probes.jsonl');mapping={p['id']:p for p in probes};raw=folder/'predictions.jsonl'
        records,notes=ev.read_rows(raw,allow_partial_tail=True) if raw.exists() else ([],[])
        if notes:ev.save_rows(raw,records)
        assert len(records)==len({r['id'] for r in records})
        for r in records:
            p=mapping[r['id']];assert r['prompt_sha256']==ev.digest({'messages':p['messages'],'images':p.get('images',[])})
        if arm in ('base','warmup','text_alias','vision'):
            previous={r['id']:r for r in rows(PILOT/arm/'predictions.jsonl')}
            seen={r['id'] for r in records}
            for p in probes:
                if not p['condition'].startswith('strict_') or p['condition'].startswith('strict_recall_cue') or p['id'] in seen:continue
                old=previous[p['source_id']];assert old['prompt_sha256']==ev.digest({'messages':p['messages'],'images':p.get('images',[])})
                records.append(record(p,old['raw_response'],str(PILOT/arm/'predictions.jsonl')))
            ev.save_rows(raw,records)
        seen={r['id'] for r in records};pending=[p for p in probes if p['id'] not in seen]
        if pending:
            with reserve(LEDGER,gpu,str(folder),24000) as reservation:
                reservation.bind(os.getpid());os.environ.update(environment(gpu,DATA))
                import torch
                from swift.infer_engine import TransformersEngine,InferRequest,RequestConfig
                from swift.utils import seed_everything
                seed_everything(42);cp=checkpoint(arm)
                engine=TransformersEngine(str(MODELS['qwen3']),adapters=[str(cp)] if cp else None,torch_dtype=torch.bfloat16,device_map='cuda:0',max_batch_size=4,attn_impl='sdpa');engine.template.enable_thinking=False
                for fmt in ('mcq','rule_option','card'):
                    group=[p for p in pending if p['fmt']==fmt]
                    for start in range(0,len(group),4):
                        batch=group[start:start+4]
                        outputs=engine.infer([InferRequest(messages=copy.deepcopy(p['messages']),images=p.get('images',[])) for p in batch],RequestConfig(max_tokens=16 if fmt=='mcq' else 384,temperature=0,seed=42,return_details=True),use_tqdm=False)
                        assert len(outputs)==len(batch)
                        for p,reply in zip(batch,outputs):
                            if isinstance(reply,Exception):raise reply
                            r=record(p,dataclasses.asdict(reply));records.append(r)
                            with raw.open('a') as f:f.write(json.dumps(r,ensure_ascii=False,default=str)+'\n');f.flush();os.fsync(f.fileno())
                        save(folder/'state.json',{'status':'running','pid':os.getpid(),'saved':len(records),'expected':128,'updated_at':ev.now()});print(arm,fmt,start+len(batch),'/',len(group),flush=True)
        assert len(records)==len({r['id'] for r in records})==128
        def metric(rr):
            return {'n':len(rr),'correct':sum(r['correct'] for r in rr),'format_errors':sum(r['format_error'] for r in rr),'truncated':sum(r['finish_reason']=='length' for r in rr),'empty':sum(not r['response'].strip() for r in rr),'reused':sum('reused_from' in r for r in rr),'requested_rule_body_exact':sum(r.get('requested_rule_body_exact',False) for r in rr),'image_donor_body_exact':sum(r.get('image_donor_body_exact',False) for r in rr)}
        metrics={c:{**metric([r for r in records if r['condition']==c]),'by_relation':{rel:metric([r for r in records if r['condition']==c and r['relation']==rel]) for rel in ('set_selection_rule','three_way_category')},'by_task':{t:metric([r for r in records if r['condition']==c and r['task']==t]) for t in sorted({r['task'] for r in records if r['condition']==c})}} for c in sorted({p['condition'] for p in probes})}
        save(folder/'report.json',{'status':'complete','arm':arm,'metrics':metrics,'native_thinking':False,'probes_sha256':sha(out/'probes.jsonl'),'checkpoint':ev.checkpoint(checkpoint(arm)) if arm!='base' else None,'finished_at':ev.now()})
        save(folder/'state.json',{'status':'complete','saved':128,'new_requests':sum('reused_from' not in r for r in records),'finished_at':ev.now()})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'runs/v7_access_diagnostics_20261015');ap.add_argument('--arm',choices=ARMS,default='text_alias');ap.add_argument('--gpu',default='0');ap.add_argument('--prepare-only',action='store_true');a=ap.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    with (out/'.prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);prepare(out)
    if not a.prepare_only:run(out,a.arm,a.gpu)
