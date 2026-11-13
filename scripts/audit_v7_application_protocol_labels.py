#!/usr/bin/env python3
"""Recompute all 36 selected text answers; never supplies answers to inference."""
import itertools
import json
import operator
import re
from fractions import Fraction

from repair_v7_application_protocol import OUT, rows, save, sha, Path


def norm(text):
    return re.sub(r'\s+', '', text).rstrip('.').lower()


def singular(text):
    words=text.strip().split()
    if words[-1].endswith('s'): words[-1]=words[-1][:-1]
    return ' '.join(words)


def cases(text):
    found=list(re.finditer(r'Candidate ([A-D]): ',text))
    if found:
        return {m[1]:text[m.end():found[i+1].start() if i+1<len(found) else len(text)].strip().rstrip('.') for i,m in enumerate(found)}
    return {'X':text.strip().rstrip('.')}


def result(rule,k,head,obs):
    rel=k['fact']['relation']; ops={'eq':operator.eq,'ne':operator.ne,'ge':operator.ge,'le':operator.le}
    if rel=='binary_predicate':
        def flags(sentence):
            if 'contains exactly' in sentence:
                n,kind=re.search(r'contains exactly (\d+) complete (.+?)\.?$',sentence).groups()
                attrs={'category':singular(kind),'value':int(n)}
            else:
                color,kind,material=re.search(r'is a (\w+) (.+?) made of (\w+)',sentence).groups()
                attrs={'color':color,'category':kind,'material':material}
            return [ops[c['operator']](attrs[c['field']],c['value']) for c in rule['conditions']]
        if 'Inspect Candidate' in head:
            focus=re.search(r'Inspect Candidate ([A-D])',head)[1]
            cc=dict(re.findall(r'Candidate ([A-D])(?: is |: )(.*?)(?= Candidate [A-D]|$)',obs))
            sentence=cc[focus]
            if 'contains exactly' not in sentence:sentence='is a '+sentence.removeprefix('a ')
            ff=flags(sentence); protocol=k['material']['learning']['inspection_recording_protocol']
            wanted=[ff[i-1] for i in protocol['checked_indices']]
            matches=[code for desc,code in protocol['outcome_to_code'].items()
                     if [part.endswith('=Met') for part in desc.split('; ')]==wanted]
            assert len(matches)==1
            return matches[0]
        ff=flags(obs)
        return '; '.join([f'Check {i+1}={"Pass" if v else "Fail"}' for i,v in enumerate(ff)]
                         +[f'Qualification={"Pass" if all(ff) else "Fail"}'])
    if rel=='set_selection_rule':
        rr={}
        for code,desc in re.findall(r'([A-D]) has (.*?)(?=; [A-D] has|$)',obs.rstrip('.')):
            attrs=dict(re.findall(r'(\w+): ([^,;]+)',desc))
            ff=[ops[c['operator']](int(attrs[c['field']]) if c['field']=='value' else attrs[c['field']],c['value']) for c in rule['conditions']]
            rr[code]='KEEP' if all(ff) else 'OMIT'
        assert len(rr)==4
        return '; '.join(f'{c}={v}' for c,v in rr.items())
    if rel=='ordering_priority':
        found=dict(re.findall(r'Candidate ([A-D]) is a (.*?)(?=\. Candidate|\.$)',obs))
        assert len(found)==4
        order=sorted(found,key=lambda c:(rule['priority'].index(found[c]),c))
        return order[0] if rule['output']=='top_id' else ' > '.join(order)
    if rel=='mapping_codebook':
        rr={}
        for c,desc in cases(obs).items():
            item=re.search(r'The observed item is a (.*)',desc)[1].rstrip('.')
            rr[c]=rule['mapping'][item]
        return '; '.join(f'{c}={v}' for c,v in rr.items())
    if rel=='three_way_category':
        rr={}
        for c,desc in cases(obs).items():
            pair=re.search(r'(\d+) (?:marked cells? out of|vases? among) (\d+)',desc)
            assert pair,desc
            value=Fraction(int(pair[1]),int(pair[2]));lo=Fraction(str(rule['thresholds']['lower']));hi=Fraction(str(rule['thresholds']['upper']))
            rr[c]=rule['category_labels'][0 if value<lo else 2 if value>hi else 1]
        return '; '.join(f'{c}={v}' for c,v in rr.items())
    if rel=='scalar_index_rule':
        weights={singular(key.removesuffix('_count').replace('_',' ')):Fraction(str(value)) for key,value in rule['weights'].items()}
        rr={}
        for c,desc in cases(obs).items():
            terms=re.search(r'contains (.*)',desc)[1].split(' and ')
            total=Fraction(str(rule['bias']))
            for term in terms:
                num,item=re.fullmatch(r'(\d+) (.*)',term.rstrip('.')).groups()
                total+=int(num)*weights[singular(item)]
            rr[c]=f'{float(total):.{rule["precision"]}f}'
        return '; '.join(f'{c}={v}' for c,v in rr.items())
    if rel=='finite_state_transition':
        rr={}
        for c,desc in cases(obs).items():
            state,signals=re.search(r'starts in (\w+) and receives the signals (.*)',desc).groups()
            trace=[]
            for sig in signals.split(' then '):
                state=rule['transition'][state][sig.strip().replace(' ','_')];trace.append(state)
            rr[c]=state
        return ' -> '.join(trace) if 'after each signal' in head else '; '.join(f'{c}={v}' for c,v in rr.items())
    if rel=='transformation_procedure':
        seq=re.search(r'starting sequence is (.*)',obs)[1].rstrip('.').split('-')
        for op in rule['operations']:
            for _ in range(op['repeat']):
                if op['op']=='reverse':seq=seq[::-1]
                elif op['op'] in ('rotate_left','rotate_right'):
                    shift=op['positions']%len(seq)
                    if op['op']=='rotate_right':shift=-shift
                    seq=seq[shift:]+seq[:shift]
                elif op['op']=='swap_disjoint_pairs':
                    for i in range(0,len(seq)-1,2):seq[i],seq[i+1]=seq[i+1],seq[i]
                else:raise ValueError(op)
        return ' | '.join(seq)
    if rel=='bipartite_matching':
        left,right=obs.split('; the right slots are ')
        items=dict(re.findall(r'([A-Z]) \(([^)]+)\)',left));slots=dict(re.findall(r'([A-Z]) \(([^)]+)\)',right))
        aliases={'round':'circle','triangular':'triangle','hexagonal':'hexagon','square':'square'}
        slots={c:aliases[s.split()[0]] for c,s in slots.items()}
        material=k['material']['learning']['text']
        compatible={item.strip():re.findall(r'(\w+) slots',desc) for item,desc in re.findall(r'([^.]+?) accepts ([^.]+)\.',material.split('once. ',1)[1])}
        matches=[]
        for dest in itertools.permutations(slots):
            if all(slots[s] in compatible[items[c]] for c,s in zip(items,dest)):
                matches.append(', '.join(f'{c} -> {s}' for c,s in zip(items,dest)))
        assert len(matches)==1,matches
        return matches[0]
    raise ValueError(rel)


def rule_part_ok(text,rule):
    if 'Cutoffs:' in text:
        low,high=re.search(r'lower=(\d+(?:\.\d+)?), upper=(\d+(?:\.\d+)?)',text).groups()
        return Fraction(low)==Fraction(str(rule['thresholds']['lower'])) and Fraction(high)==Fraction(str(rule['thresholds']['upper']))
    if 'Codebook:' in text:
        mapping=dict(re.findall(r'([^:;]+?) maps to (\w+)',text.split('Codebook: ',1)[1].split('. Result:',1)[0]))
        return {key.strip():value for key,value in mapping.items()}==rule['mapping']
    return True


def audit():
    knowledge={k['id']:k for k in rows(OUT/'knowledge.jsonl')};checks=[]
    for p in rows(OUT/'probes.jsonl'):
        if p['protocol']!='strict32' or p['input_mod']!='T':continue
        k=knowledge[p['qid']];rule=json.loads(k['fact']['quote'])
        text=p['messages'][-1]['content'].split('\n\n',1)[1]
        head,tail=text.split('\n\nTest input:\n');obs=tail.split('\n\nOptions:\n')[0]
        expected=result(rule,k,head,obs)
        if p['task']=='diagnosis':
            submitted=head.split('Result: ',1)[1].split('. ',1)[0].rstrip('.')
            action='Accept report.' if norm(submitted)==norm(expected) and rule_part_ok(head,rule) else 'Replace report.'
        else:action=None
        correct=[]
        for letter,option in p['options'].items():
            actual=option.split('Result: ',1)[1]
            if norm(actual)==norm(expected) and rule_part_ok(option,rule) and (action is None or option.startswith(action)):
                correct.append(letter)
        assert correct==[p['gold']],(p['source_id'],expected,correct,p['gold'])
        checks.append({'id':p['source_id'],'qid':p['qid'],'expected_result':expected,'expected_action':action,
                       'unique_correct_option':correct[0],'original_gold':p['gold'],'pass':True})
    assert len(checks)==36
    save(OUT/'independent_text_label_audit.json',{'status':'pass','questions':36,'method':'Independent field/transition/permutation/exact-rational solver on actual rendered text observations; checks complete options and report action. No image correctness claim.',
        'script_sha256':sha(Path(__file__)),'checks':checks})
    print('All 36 text labels independently recomputed and unique.')


if __name__=='__main__':audit()
