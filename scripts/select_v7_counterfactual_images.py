#!/usr/bin/env python3
"""Select discriminating original images using semantics only, never model scores."""
import argparse
import hashlib
import itertools
import json
import re
from pathlib import Path

from pilot_v7_counterfactual_names import DATA, OUT, solve
from run_v7_full_experiment import rows, read, save


def select():
    knowledge=[r['content'] for r in rows(DATA/'audit/rule_knowledge.jsonl')]
    lookup={k['id']:k for k in knowledge};evals=rows(DATA/'T2T/eval.jsonl');observations={}
    for e in evals:
        if lookup[e['qid']]['fact']['relation']!='set_selection_rule':continue
        text=e['messages'][-1]['content'].split('Test input:\n')[1].split('\n\nOptions:')[0]
        parts=re.findall(r'([ABCD]) has (.*?)(?=; [ABCD] has|\.$)',text)
        if len(parts)!=4:continue
        assert [letter for letter,_ in parts]==list('ABCD')
        arr=[]
        for _,body in parts:
            a={}
            for key,val in re.findall(r'(\w+): ([^,]+)(?:, |$)',body):a[key]=int(val) if val.isdigit() else val
            arr.append(a)
        observations[e['id']]=arr
    result=[]
    for topology in ('disjunctive_filter','interval_filter'):
        eligible=[k for k in knowledge if k['fact']['relation']=='set_selection_rule'
            and k['id'].endswith('programmatic') and json.loads(k['fact']['quote']).get('selection_topology')==topology]
        combos=sorted(itertools.combinations(eligible,4),key=lambda cc:hashlib.sha256(('|'.join(k['id'] for k in cc)).encode()).hexdigest())
        for group in combos:
            ids={k['id'] for k in group};rules=[json.loads(k['fact']['quote']) for k in group];good=[]
            for e in evals:
                if e['qid'] not in ids or e['id'] not in observations:continue
                obs=observations[e['id']]
                try:answers=[tuple(solve(rule,x) for x in obs) for rule in rules]
                except KeyError:continue
                if len(set(answers))==4:good.append(e['id'])
            if len(good)>=2:
                result.append({'topology':topology,'rule_ids':sorted(ids),'eligible_test_ids':good,
                    'observations':{id:observations[id] for id in good}})
                break
        else:raise RuntimeError('No eligible original-image group for '+topology)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=OUT);a=ap.parse_args()
    selected=select();path=a.out/'selection_candidates.json'
    if path.exists():assert read(path)==selected;print('existing selection reproduced exactly; no model scores read')
    else:save(path,selected)
