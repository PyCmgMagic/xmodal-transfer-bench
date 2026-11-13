#!/usr/bin/env python3
"""Independent, exact-rational answer check for selected existing questions."""
import argparse
import json
import operator
import re
from fractions import Fraction
from pathlib import Path

from pilot_v7_task_then_transfer import DATA
from run_v7_full_experiment import rows, save, sha


def audit(out):
    knowledge={k['id']:k for k in rows(out/'target_knowledge.jsonl')+rows(out/'development_knowledge.jsonl')}
    labels={r['id']:r for r in rows(DATA/'audit/eval_labels.jsonl')}
    ops={'eq':operator.eq,'ne':operator.ne,'lt':operator.lt,'le':operator.le,'gt':operator.gt,'ge':operator.ge}
    cutoff=r'lower=(\d+(?:\.\d+)?), upper=(\d+(?:\.\d+)?)'
    checks=[]
    for row in rows(DATA/'T2T/eval.jsonl'):
        if row['qid'] not in knowledge:continue
        rule=json.loads(knowledge[row['qid']]['fact']['quote']);text=row['messages'][-1]['content']
        head,body=text.split('\n\nTest input:\n');observations,choice=body.split('\n\nOptions:\n')
        options=dict(re.findall(r'(?m)^([A-D])\. (.*)$',choice));label=labels[row['id']];results={}
        if 'conditions' in rule:
            for code,attrs in re.findall(r'([A-D]) has (.*?)(?=; [A-D] has|\.$)',observations):
                attrs=dict((a,b.strip()) for a,b in re.findall(r'(\w+): ([^,;]+)',attrs))
                flags=[ops[c['operator']](float(attrs[c['field']]) if c['field']=='value' else attrs[c['field']],c['value']) for c in rule['conditions']]
                keep=sum(flags)>=rule['required_count'] if 'required_count' in rule else all(flags) if rule['mode']=='all' else any(flags)
                results[code]='KEEP' if keep else 'OMIT'
            assert set(results)==set('ABCD'),row['id']
        else:
            parsed=re.findall(r'(?:Candidate ([A-D]): )?The observation has (\d+) .*? out of (\d+) ',observations)
            assert len(parsed)==(4 if label['question_type']=='direct' else 1),(row['id'],parsed)
            lo=Fraction(str(rule['thresholds']['lower']));hi=Fraction(str(rule['thresholds']['upper']))
            for code,n,d in parsed:
                share=Fraction(int(n),int(d))
                results[code or 'X']=rule['category_labels'][0 if share<lo else 2 if share>hi else 1]
        expected='; '.join(f'{k}={v}' for k,v in results.items());correct=[]
        for letter,option in options.items():
            actual=option.split('Result: ',1)[1].rstrip('.');ok=actual==expected
            if 'Cutoffs:' in option:
                low,high=map(Fraction,re.search(cutoff,option).groups())
                ok=ok and low==Fraction(str(rule['thresholds']['lower'])) and high==Fraction(str(rule['thresholds']['upper']))
            if label['question_type']=='diagnosis':
                submitted=head.split('Result: ',1)[1].split('.',1)[0] if 'conditions' in rule else re.search(r'Result: (X=\w+)',head)[1]
                original_correct=submitted==expected
                if 'Cutoffs:' in head:
                    low,high=map(Fraction,re.search(cutoff,head).groups())
                    original_correct=original_correct and low==Fraction(str(rule['thresholds']['lower'])) and high==Fraction(str(rule['thresholds']['upper']))
                ok=ok and option.startswith('Accept report.' if original_correct else 'Replace report.')
            if ok:correct.append(letter)
        assert correct==[label['answer']],(row['id'],correct,label['answer'],expected)
        checks.append({'id':row['id'],'qid':row['qid'],'expected_result':expected,'unique_correct_option':correct[0],'gold':label['answer'],'pass':True})
    assert len(checks)==48
    save(out/'independent_answer_audit.json',{'status':'pass','questions_checked':len(checks),'method':'Independent exact-rational threshold and field-operator solver; checks full options including disposition and cutoffs; target questions never generate training examples.','script_sha256':sha(Path(__file__)),'checks':checks})
    print('All48 selected development/target text questions have one correct option; gold labels verified.')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=Path('runs/v7_task_then_transfer_20261015/pilot_v2'));a=ap.parse_args();audit(a.out.resolve())
