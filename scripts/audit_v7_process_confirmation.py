#!/usr/bin/env python3
"""Read-only independent checks of confirmation data and completed predictions."""
import argparse
import hashlib
import json
import operator
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/v7_process_confirmation_20261020'
DATA = ROOT / 'rules_repaired_single_image_v7'


def read(p):
    return json.loads(p.read_text())


def rows(p):
    return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def digest(z):
    return hashlib.sha256(json.dumps(z, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def signature(obs):
    return digest(sorted(json.dumps({k: int(v) if isinstance(v, float) and v.is_integer() else v for k,v in o.items()},sort_keys=True) for o in obs))


def decisions(rule, observations):
    ops = {'eq': operator.eq,'lt':operator.lt,'le':operator.le,'gt':operator.gt,'ge':operator.ge}
    flags = [[ops[c['operator']](o[c['field']],c['value']) for c in rule['conditions']] for o in observations]
    keep = [any(f) if rule['mode']=='any' else all(f) for f in flags]
    return '; '.join(c+'='+('KEEP' if v else 'OMIT') for c,v in zip('ABCD',keep)), flags


def main(final=False):
    m=read(OUT/'manifest.json')
    for path,h in m['sources'].items():
        assert sha(Path(path))==h,path
    for path,h in m['files'].items():
        assert sha(OUT/path)==h,path
    selection=read(OUT/'selection_audit.json')
    selected=set(m['ids'])
    assert len(selected)==8 and not selected & set(selection['excluded_rule_ids'])
    for topology,order in selection['eligibility_order'].items():
        assert set(order[:4])<=selected
        assert order==sorted(order,key=lambda q:hashlib.sha256(('process-confirmation-42:'+q).encode()).hexdigest())
    knowledge={r['id']:r for r in rows(OUT/'knowledge.jsonl')}
    blocks=rows(OUT/'blocks.jsonl')
    assert len(blocks)==512
    signatures=[]
    for b in blocks:
        rule=json.loads(knowledge[b['qid']]['fact']['quote'])
        truth,flags=decisions(rule,b['observations'])
        assert b['results'][b['qid']]==truth
        assert len(set(b['options']))==4 and b['options'].count(truth)==1
        assert b['observation_signature']==signature(b['observations'])
        expected_mask=sum((v=='KEEP')<<i for i,v in enumerate(re.findall(r'=(KEEP|OMIT)',truth)))
        assert expected_mask==int(b['id'].rsplit(':',1)[-1])
        signatures.append(b['observation_signature'])
    assert len(set(signatures))==512
    assert len({b['observation_signature'] for b in blocks if b['split']=='train'})==256
    for q in selected:
        for split in ('train','holdout'):
            for task in ('direct','diagnosis'):
                sub=[b for b in blocks if (b['qid'],b['split'],b['task'])==(q,split,task)]
                assert len(sub)==16
                assert Counter(b['options'].index(b['results'][q]) for b in sub)==dict.fromkeys(range(4),4)
                assert sum(b['submitted_report']==b['results'][q] for b in sub)==4
    probes=rows(OUT/'probes.jsonl');pm={r['id']:r for r in probes}
    assert len(pm)==len(probes)==664
    blockmap={b['id']:b for b in blocks}
    trainrows={a:rows(OUT/(a+'_training.jsonl')) for a in ('definitions','definition_replay','process')}
    assert trainrows['definitions']==trainrows['definition_replay']
    for arm,rr in trainrows.items():
        assert len(rr)==len({r['id'] for r in rr})==640
        assert {r['qid'] for r in rr}==selected
        for r in rr:
            assert not r.get('images') and r['messages'][-1]['role']=='assistant'
            assert '<image>' not in str(r['messages']) and 'Learning reference:' not in str(r['messages'])
            assert not r['id'].startswith('original:')
            if r.get('block'):
                b=blockmap[r['block']]
                assert b['split']=='train'
                gold='ABCD'[b['options'].index(b['results'][r['qid']])]
                text=r['messages'][-1]['content']
                if r['channel']=='direct_application':
                    assert text==gold
                else:
                    assert text.endswith('Answer: '+gold)
                    result,flags=decisions(json.loads(knowledge[r['qid']]['fact']['quote']),b['observations'])
                    assert '\nResult: '+result+'.\n' in text
                    if b['task']=='diagnosis':
                        action='Accept report.' if b['submitted_report']==result else 'Replace report.'
                        assert '\n'+action+'\n' in text
    # Confirm original payloads are retained byte-for-byte apart from protocol system/suffix.
    labels={r['id']:r for r in rows(DATA/'audit/eval_labels.jsonl')}
    for mod,direction in [('T','T2T'),('V','T2V')]:
        formal={r['id']:r for r in rows(DATA/direction/'eval.jsonl') if r['qid'] in selected}
        assert len(formal)==16 and {r['qid'] for r in formal.values()}==selected
        for source_id,r in formal.items():
            p=pm['original:'+mod+':'+source_id]
            assert p['gold']==labels[source_id]['answer']
            source=r['messages'][-1]['content']
            # prepare_row uses the same strict suffix; compare problem before that suffix.
            assert source.split('\n\nAnswer with exactly one uppercase letter',1)[0]==p['messages'][-1]['content'].split('\n\nAnswer with exactly one uppercase letter',1)[0]
    images=rows(DATA/'image_hashes.jsonl');assert len(images)==3000
    for r in images:
        assert sha(DATA/r['path'])==r['source_sha256']==r['repaired_sha256']
    result=dict(status='pass',stage='final' if final else 'data',selected_rules=8,scenes=512,
        training_rows_per_task=640,probes_per_arm=664,original_images_verified=3000,
        frozen_sources_and_labels=True,disjoint_scenes=True,balanced_options=True,
        independent_decisions_and_mask_schedule=True,original_problem_payloads_unchanged=True,
        source_sha256=sha(Path(__file__)))
    if final:
        scored=rows(OUT/'scored_predictions.jsonl');assert len(scored)==2656
        indexed={(r['arm'],r['id']):r for r in scored}
        assert len(indexed)==2656
        marker_not_final=Counter()
        score_totals=Counter()
        for arm in m['arms']:
            rr=rows(OUT/arm/'evaluation/predictions.jsonl')
            assert len(rr)==len({r['id'] for r in rr})==664
            for r in rr:
                p=pm[r['id']]
                assert r['input_sha256']==digest(dict(messages=p['messages'],images=p.get('images',[])))
                assert r['gold']==p['gold'] and r['qid']==p['qid']
                z=indexed[arm,r['id']]
                assert z['response']==r['response'] and z['finish_reason']==r['finish_reason']
                response=r['response'].strip()
                if p['protocol']=='short':
                    answer=response if response in ('A','B','C','D') else None
                elif p['protocol']=='reasoning':
                    markers=re.findall(r'(?im)^\s*Answer:\s*([A-D])\s*[.!]?\s*$',response)
                    answer=markers[0] if len(markers)==1 else None
                    if answer is not None and not re.fullmatch(r'Answer:\s*'+answer+r'\s*[.!]?',response.splitlines()[-1].strip(),re.I):
                        marker_not_final[arm]+=1
                else:
                    answer=None
                normalize=lambda text:' '.join(text.split()).lower()
                correct=normalize(response)==normalize(p['gold']) if p['fmt']=='fact' else answer==p['gold']
                assert z['scored_answer']==answer and z['scored_correct']==correct
                assert z['scored_format_error']==(p['fmt']!='fact' and answer is None)
                score_totals[arm]+=int(correct)
            identity=read(OUT/arm/'evaluation/eval_manifest.json')['model']
            for fn,h in identity['sha256'].items():
                assert sha(Path(identity['path'])/fn)==h
            if arm!='base':
                assert identity['path'].endswith(arm+'/checkpoint/checkpoint-480')
                state=read(Path(identity['path'])/'trainer_state.json')
                assert state['global_step']==480 and state['epoch']==3
        result.update(predictions=2656,model_identities_rechecked=True,final_checkpoints_only=True,
            raw_to_scored_predictions_rechecked=True,per_arm_total_correct=dict(score_totals),
            accepted_reasoning_answer_marker_not_last_line=dict(marker_not_final))
        import torch
        from safetensors import safe_open
        torch.set_num_threads(4)
        common=OUT/'definitions/checkpoint/checkpoint-480/adapter_model.safetensors'
        updates={}
        with safe_open(str(common),framework='pt',device='cpu') as initial:
            for arm in ('definition_replay','process'):
                path=OUT/arm/'checkpoint/checkpoint-480/adapter_model.safetensors'
                with safe_open(str(path),framework='pt',device='cpu') as final_weights:
                    assert set(initial.keys())==set(final_weights.keys())
                    changed=0;delta=0.;scale=0.
                    for k in initial.keys():
                        a,b=initial.get_tensor(k).float(),final_weights.get_tensor(k).float()
                        changed+=not torch.equal(a,b)
                        delta+=float((b-a).square().sum());scale+=float(a.square().sum())
                    assert changed>0
                    updates[arm]=dict(changed_tensors=changed,total_tensors=len(initial.keys()),relative_l2=(delta/scale)**.5)
        result['parameter_updates']=updates
    (OUT/('independent_final_audit.json' if final else 'independent_data_audit.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');a=p.parse_args();main(a.final)
