#!/usr/bin/env python3
"""Fixed new rules, original scenes, four different name-dependent answers."""
import argparse
import copy
import json
import os
import random
import shutil
import time
from pathlib import Path

import confirm_v7_rule_recognition as pilot
import explore_v7_visual_replay as replay
import run_v7_evaluation as ev
from build_v7_experiment_data import name
from run_v7_full_experiment import rows,read,save,sha
from v7_gpu_resources import reserve

OUT=pilot.OUT/'paired_application'


def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'manifest.json').exists():validate();return
    kk={k['id']:k for k in rows(pilot.OUT/'target_knowledge.jsonl')}
    rules={q:json.loads(k['fact']['quote']) for q,k in kk.items()}
    cases=rows(pilot.OUT/'original_case_audit.jsonl')
    original={r['id']:r for r in rows(pilot.DATA/'T2T/eval.jsonl')}
    blocks=[];probes=[]
    rng=random.Random('counterfactual-original-images:42')
    for group_index,group in enumerate(read(pilot.OUT/'selection.json')['groups']):
        for case in cases:
            if case['qid'] not in group:continue
            obs=case['observations']
            results={q:pilot.assignment([pilot.solve(rules[q],o) for o in obs]) for q in group}
            if len(set(results.values()))!=4:continue
            source_text=original[case['id']]['messages'][-1]['content'].split('Test input:\n')[1].split('\n\nOptions:')[0]
            options=list(results.values());rng.shuffle(options);report=rng.choice(options)
            for task in ('direct','diagnosis'):
                block=f'g{group_index}:{case["id"]}:{task}'
                blocks.append({'id':block,'source_id':case['id'],'rule_ids':group,'rule_results':results,
                    'observation':obs,'base_options':options,'image':case['original_test_image'],'group':group_index,'task':task,
                    'submitted_report':report if task=='diagnosis' else None})
                for shift in range(4):
                    opts=options[shift:]+options[:shift]
                    contents=[('Result: '+v+'.' if task=='direct' else ('Accept report. ' if v==report else 'Replace report. ')+'Result: '+v+'.') for v in opts]
                    for modality in ('T','V'):
                        body=('\n\nTest input:\n'+source_text if modality=='T' else '\n\n<image>')
                        body+='\n\nOptions:\n'+'\n'.join(f'{l}. {v}' for l,v in zip('ABCD',contents))+ev.SUFFIX
                        for q in group:
                            heading=('Apply '+name(kk[q])+'. Report KEEP/OMIT for each candidate.' if task=='direct' else
                                'A technician submitted this report under '+name(kk[q])+': Result: '+report+'. Check the observations and the rule. Report KEEP/OMIT for each candidate.')
                            p={'id':f'paired:{block}:{modality}:{shift}:{q}','qid':q,'block':block,'shift':shift,'group':group_index,
                                'task':task,'modality':modality,'condition':'paired_'+modality,'fmt':'mcq','gold':'ABCD'[opts.index(results[q])],
                                'shared_body_sha256':ev.digest(body),'messages':[{'role':'system','content':ev.SYSTEM},
                                {'role':'user','content':heading+' KEEP means selected and OMIT means not selected.'+body}]}
                            if modality=='V':p['images']=[case['original_test_image']]
                            probes.append(p)
    assert len(blocks)==8 and len(probes)==256
    ev.save_rows(OUT/'blocks.jsonl',blocks);ev.save_rows(OUT/'probes.jsonl',probes)
    save(OUT/'manifest.json',{'script_sha256':sha(Path(__file__)),'created_at':ev.now(),
        'source_manifest_sha256':sha(pilot.OUT/'manifest.json'),'probes_sha256':sha(OUT/'probes.jsonl'),
        'blocks_sha256':sha(OUT/'blocks.jsonl'),'images':{b['image']:sha(Path(b['image'])) for b in blocks},
        'selection':'All scenes among the frozen8newrules original16tests where all4rules within a topology produce distinct outcomes. No score reading or rule-group reselection. Foureligible originalPNG,8blocks; both direct/diagnosis built per scene.',
        'comparability':'Same paired construction/prompts as prior development (now4original scenes vs previous5). Kept separate from untouched original-v7 question results.',
        'controls':'For every same-observation/options block, four names have goldABCD. Any identical valid answer for all4names matches1of4. Fouroption rotations are dependent repeats.',
        'new_generations_per_arm':256,'arms':(*pilot.ARMS,'warmup')})
    shutil.copy2(Path(__file__),OUT/Path(__file__).name);validate()


def validate():
    m=read(OUT/'manifest.json');assert m['script_sha256']==sha(Path(__file__))
    assert sha(pilot.OUT/'manifest.json')==m['source_manifest_sha256']
    assert sha(OUT/'probes.jsonl')==m['probes_sha256'] and sha(OUT/'blocks.jsonl')==m['blocks_sha256']
    for path,h in m['images'].items():assert sha(Path(path))==h
    pp=rows(OUT/'probes.jsonl')
    for key in {(p['block'],p['modality'],p['shift']) for p in pp}:
        group=[p for p in pp if (p['block'],p['modality'],p['shift'])==key]
        assert len(group)==4 and {p['gold'] for p in group}==set('ABCD')
        assert len({p['shared_body_sha256'] for p in group})==1
        assert len({ev.digest(p.get('images',[])) for p in group})==1


def evaluate(arm,gpu):
    validate()
    while not (pilot.OUT/arm/'eval_state.json').exists() or read(pilot.OUT/arm/'eval_state.json').get('status')!='complete':
        save(OUT/(arm+'_wait.json'),{'status':'waiting_for_original_protocol','arm':arm,'updated_at':ev.now()})
        time.sleep(10)
    cp=Path(read(pilot.PARENT)['path']) if arm=='warmup' else Path(read(pilot.OUT/arm/'checkpoint_manifest.json')['path'])
    config=replay.config(cp,{'mcq':16});config['probes_sha256']=sha(OUT/'probes.jsonl')
    folder=OUT/arm;run=replay.Runner(folder,config,gpu);pp=rows(OUT/'probes.jsonl')
    if len(run.rr)<256:
        with reserve(pilot.LEDGER,gpu,str(folder)+':eval',24000) as res:
            res.bind(os.getpid());run.load();run.infer(pp)
    run.audit(pp,256)
    save(folder/'report.json',{'status':'complete','metrics':{c:pilot.metric([r for r in run.rr if r['condition']==c]) for c in ('paired_T','paired_V')}})
    save(OUT/(arm+'_wait.json'),{'status':'complete'})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--arm',choices=(*pilot.ARMS,'warmup'))
    ap.add_argument('--gpu',default='0');a=ap.parse_args()
    if a.prepare:prepare()
    else:
        with ev.run_lock(OUT/(a.arm+'.lock')):evaluate(a.arm,a.gpu)
