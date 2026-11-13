#!/usr/bin/env python3
"""New-rule confirmation of fixed memory+frozen-solver recipe, no score selection."""
import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import explore_v7_memory_router as mem
import run_v7_evaluation as ev
from run_v7_full_experiment import ROOT, rows, read, save, sha

DEV = mem.OUT
OUT = ROOT / 'runs/v7_memory_router_confirm_20261019'
PARENT = OUT / 'frozen_parent'
ARMS = ('base','text_memory','vision_bridge')


def configure():
    mem.OUT = OUT
    mem.PARENT = PARENT
    mem.TRAIN_ARMS = ('text_memory','vision_bridge')
    mem.ARMS = ARMS


def prepare():
    if (OUT / 'confirmation_manifest.json').exists():
        configure(); validate(); return
    OUT.mkdir(exist_ok=True); PARENT.mkdir(exist_ok=True)
    previous = rows(DEV / 'knowledge.jsonl')
    allk = [r['content'] for r in rows(mem.DATA / 'audit/rule_knowledge.jsonl')]
    excluded, files = set(), {}
    for f in sorted((ROOT / 'runs').glob('v7*/**/*knowledge*.jsonl')):
        if any(s in f.parts for s in ('source_snapshots','frozen_sources','preflight_revision_1','preflight_revision_2')):
            continue
        if OUT in f.parents:
            continue
        try:
            rr = rows(f)
        except (ValueError,OSError):
            continue
        if not 1 <= len(rr) <= 200:
            continue
        ids = {r.get('content',r).get('id') for r in rr}
        ids = {i for i in ids if isinstance(i,str) and i.startswith('scale1k_')}
        if ids:
            excluded.update(ids); files[str(f)] = sha(f)
    original_excluded = set(excluded)
    selected = []
    for k in previous:
        family = json.loads(k['fact']['quote'])['structural_subfamily_id']
        candidates = [x for x in allk if x['id'] not in excluded and json.loads(x['fact']['quote'])['structural_subfamily_id'] == family]
        candidates.sort(key=lambda x: hashlib.sha256(('memory-router-confirm:42:'+x['id']).encode()).hexdigest())
        assert candidates, family
        selected.append(candidates[0]); excluded.add(candidates[0]['id'])
    assert len(selected)==18 and not {k['id'] for k in selected} & original_excluded
    ev.save_rows(PARENT / 'knowledge.jsonl',selected)
    save(PARENT / 'manifest.json',{'models':read(DEV / 'manifest.json')['models'],
        'role':'Frozen model identities reused from development; knowledge selection independently fixed below.'})
    # Independently recalculate every actual TEXT question BEFORE new inference.
    import audit_v7_application_protocol_labels as labels
    old_singular=labels.singular
    nouns=set()
    for k in selected:
        rule=json.loads(k['fact']['quote'])
        nouns.update(key.removesuffix('_count').replace('_',' ') for key in rule.get('weights',{}))
        nouns.update(c['value'] for c in rule.get('conditions',[]) if c['field']=='category')
    def source_singular(text):
        term=text.strip()
        for noun in nouns:
            forms={noun,noun+'s',noun+'es'}
            if noun.endswith('y'):forms.add(noun[:-1]+'ies')
            if term in forms:return noun
        return old_singular(term)
    # Exact source vocabulary resolves English plurals such as storage boxes;
    # the original development solver only stripped a final s.
    labels.singular=source_singular
    lookup={k['id']:k for k in selected}
    golds={r['id']:r for r in rows(mem.DATA / 'audit/eval_labels.jsonl')}
    checks=[]
    for p in rows(mem.SOURCE / 'eval_T2T.jsonl'):
        if p['qid'] not in lookup:continue
        k=lookup[p['qid']];rule=json.loads(k['fact']['quote'])
        body=p['messages'][-1]['content'];head,tail=body.split('\n\nTest input:\n')
        observed=tail.split('\n\nOptions:\n')[0]
        if k['fact']['relation']=='binary_predicate':
            import operator,re
            ops={'eq':operator.eq,'ne':operator.ne,'ge':operator.ge,'le':operator.le,'gt':operator.gt,'lt':operator.lt}
            description=observed
            focus=re.search(r'Inspect Candidate ([A-D])',head)
            if focus:
                candidates=list(re.finditer(r'Candidate ([A-D])(?: is |: | has )',observed))
                cc={m[1]:observed[m.start():candidates[i+1].start() if i+1<len(candidates) else len(observed)].strip().rstrip('.') for i,m in enumerate(candidates)}
                description=cc[focus[1]]
            attrs={}
            if 'contains exactly' in description:
                n,kind=re.search(r'contains exactly (\d+) complete (.+?)\.?$',description).groups()
                attrs={'category':source_singular(kind),'value':int(n)}
            elif ' shape' in description:
                attrs['shape']=re.search(r'(\w+) shape',description)[1]
                attrs['color']=re.search(r'is (\w+)',description)[1]
                attrs['texture']=re.search(r'has a (\w+) texture',description)[1]
            else:
                color,kind,material=re.search(r'is a (\w+) (.+?) made of (\w+)',description).groups()
                attrs={'color':color,'category':kind,'material':material}
            flags=[ops[c['operator']](attrs[c['field']],c['value']) for c in rule['conditions']]
            if focus:
                protocol=k['material']['learning']['inspection_recording_protocol']
                wanted=[flags[i-1] for i in protocol['checked_indices']]
                matches=[code for desc,code in protocol['outcome_to_code'].items() if [part.endswith('=Met') for part in desc.split('; ')]==wanted]
                assert len(matches)==1
                answer=matches[0]
            else:
                qualification=all(flags) if rule['mode']=='all' else any(flags)
                answer='; '.join([f'Check {i+1}={"Pass" if v else "Fail"}' for i,v in enumerate(flags)]+[f'Qualification={"Pass" if qualification else "Fail"}'])
        elif k['fact']['relation']=='set_selection_rule':
            import operator,re
            ops={'eq':operator.eq,'ne':operator.ne,'ge':operator.ge,'le':operator.le,'gt':operator.gt,'lt':operator.lt}
            results={}
            for code,description in re.findall(r'([A-D]) has (.*?)(?=; [A-D] has|$)',observed.rstrip('.')):
                attrs=dict(re.findall(r'(\w+): ([^,;]+)',description))
                flags=[ops[c['operator']](float(attrs[c['field']]) if c['field']=='value' else attrs[c['field']],c['value']) for c in rule['conditions']]
                keep=all(flags) if rule['mode']=='all' else any(flags)
                results[code]='KEEP' if keep else 'OMIT'
            assert len(results)==4
            answer='; '.join(f'{c}={v}' for c,v in results.items())
        elif k['fact']['relation']=='three_way_category':
            import re
            from fractions import Fraction
            results={}
            for code,description in labels.cases(observed).items():
                pair=re.search(r'(\d+) (?:marked cells? out of|.+? among) (\d+)',description)
                assert pair,description
                ratio=Fraction(int(pair[1]),int(pair[2]))
                lo=Fraction(str(rule['thresholds']['lower']));hi=Fraction(str(rule['thresholds']['upper']))
                results[code]=rule['category_labels'][0 if ratio<lo else 2 if ratio>hi else 1]
            answer='; '.join(f'{c}={v}' for c,v in results.items())
        else:
            answer=labels.result(rule,k,head,observed)
        if golds[p['id']]['question_type']=='diagnosis':
            submitted=head.split('Result: ',1)[1].split('. ',1)[0].rstrip('.')
            action='Accept report.' if labels.norm(submitted)==labels.norm(answer) and labels.rule_part_ok(head,rule) else 'Replace report.'
        else:action=None
        opts=dict(__import__('re').findall(r'(?m)^([A-D])\. (.*)$',body))
        correct=[letter for letter,option in opts.items() if labels.norm(option.split('Result: ',1)[1])==labels.norm(answer)
                 and labels.rule_part_ok(option,rule) and (action is None or option.startswith(action))]
        assert correct==[golds[p['id']]['answer']],(p['id'],answer,correct,golds[p['id']]['answer'])
        checks.append({'id':p['id'],'qid':p['qid'],'expected_result':answer,'expected_action':action,'unique_correct_option':correct[0],'pass':True})
    assert len(checks)==36
    save(PARENT / 'independent_text_label_audit.json',{'status':'pass','questions':36,'checks':checks,
        'limits':'Actual text observations independently solved. No assertion of complete image semantic certification.'})
    configure();mem.prepare()
    save(OUT / 'confirmation_manifest.json',{'created_at':ev.now(),'script_sha256':sha(Path(__file__)),
        'development_manifest_sha256':sha(DEV / 'manifest.json'), 'memory_implementation_sha256':sha(Path(mem.__file__)),
        'selection':'Same18 structural subfamilies, distinct IDs; SHA256 seed42; exclude indexed small experiment knowledge files. No score-based choice.',
        'excluded_ids':sorted(original_excluded),'exclusion_files':files,'selected_ids':[k['id'] for k in selected],
        'arms':ARMS,'fixed_before_inference':True,
        'recipe':'Exact original compact facts,32presentations/rule,rank32,alpha64,LR1e-4,batch4,1epoch/final144. No hyperparameter retuning. New clean base adapters.',
        'models':'Qwen3-VL-4B only; TEXT and visual+text bridge. Pure-visual control remains development negative result.',
        'protocols':'Original direct short answers and uncorrected name-only memory -> frozen original base worksheet -> original options. Base uses exact same routing. Oracle separately supplied material.',
        'decision':'Source routed >=60% and gain >=15pp over method-matched base, oracle>=80%, format<=2%; report all18rules and both tasks regardless of pass.',
        'limits':['New rule IDs but same families/templates; one seed; development of the recipe used original18rules.',
                  'Historical full1000 dense models saw all rules, but confirmation trains NEW adapters on clean base, no inherited dense checkpoint.',
                  'vision_bridge has additional no-image textual supervision; no pure visual parameter transfer claim.',
                  'Generic parent memory manifest describes source development protocol; this confirmation manifest defines the distinct sample and active arms.']})
    snap=OUT/'source_snapshots';shutil.copy2(Path(__file__),snap/Path(__file__).name)
    validate()


def validate():
    mem.validate()
    m=read(OUT/'confirmation_manifest.json')
    assert m['script_sha256']==sha(Path(__file__)) and m['memory_implementation_sha256']==sha(Path(mem.__file__))
    assert m['development_manifest_sha256']==sha(DEV/'manifest.json')
    assert {k['id'] for k in rows(OUT/'knowledge.jsonl')}==set(m['selected_ids'])
    assert not set(m['selected_ids']) & set(m['excluded_ids'])


def run(gpu):
    configure();validate()
    if gpu=='0':
        mem.train('text_memory',gpu,smoke=True);mem.train('text_memory',gpu);mem.memory('text_memory',gpu)
        mem.memory('base',gpu);mem.solve(['text_memory','base'],gpu)
    else:
        mem.train('vision_bridge',gpu,smoke=True);mem.train('vision_bridge',gpu);mem.memory('vision_bridge',gpu)
        mem.solve(['vision_bridge','oracle'],gpu)


def finish():
    configure();validate();mem.audit();mem.summarize()
    assert read(OUT/'completion_audit.json')['all_requested_complete']
    import summarize_v7_memory_router as combined
    table=[]; summary={}; recall={}
    targets={r['qid']:r['answer'] for r in rows(OUT/'targets.jsonl') if r['arm']=='text_memory'}
    for method in ('direct','routed_base'):
        arms=ARMS+('oracle',) if method=='routed_base' else ARMS
        base=combined.selected('base',method)
        summary[method]={}
        for arm in arms:
            rr=combined.selected(arm,method);summary[method][arm]={}
            for mod in ('T','V'):
                ss=[r for r in rr if r['input_mod']==mod]
                assert len(ss)==36
                s=mem.metric(ss);s['paired']=combined.paired(ss,[r for r in base if r['input_mod']==mod])
                for key in ('task','relation','evaluation_group','qid'):
                    s['by_'+key]={v:mem.metric([r for r in ss if r[key]==v]) for v in sorted({r[key] for r in ss})}
                summary[method][arm][mod]=s
                table.append({'arm':arm,'method':method,'input_mod':mod,'correct':s['correct'],'n':36,'gain':s['paired']['accuracy_gain']})
    norm=lambda s:' '.join(s.split())
    for arm in ARMS:
        rr=rows(OUT/arm/'memory/predictions.jsonl');recall[arm]={}
        for stage in ('blind_recall','image_recall'):
            ss=[r for r in rr if r['stage']==stage]
            recall[arm][stage]={'n':18,'exact_target':sum(norm(r['response'])==norm(targets[r['qid']]) for r in ss),
                                'truncated':sum(r['finish_reason']=='length' for r in ss),'not_complete_semantic_score':True}
    save(OUT/'confirmation_results.json',summary);save(OUT/'target_reconstruction.json',recall)
    save(OUT/'final_inventory.json',{'complete':True,'generations':read(OUT/'completion_audit.json')['new_generations'],
         'smoke_generations':4,'training_jobs':2,'all_original_pngs_unchanged':3000,'table':table})
    print(json.dumps(table,ensure_ascii=False,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('prepare','run','finish'),required=True);ap.add_argument('--gpu',choices=('0','1'),default='0');a=ap.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='run':run(a.gpu)
    else:finish()
