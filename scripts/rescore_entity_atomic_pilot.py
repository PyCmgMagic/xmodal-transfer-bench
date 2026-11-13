#!/usr/bin/env python3
"""Re-score every pilot condition consistently; never regenerate or change old scores."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time

from answer_readout import options_from_question
from answer_readout_entity_v4 import parse_answer,VERSION
from entityqa_common import rows,save,sha,write_rows,now,eval_payload


def run(args):
    work=args.work.resolve()
    source_hashes_at_start={p.name:sha(p) for p in [Path(__file__),
        Path(__file__).with_name('answer_readout_entity_v4.py'),
        Path(__file__).with_name('answer_readout_entity_v3.py'),
        Path(__file__).with_name('answer_readout_entity_v2.py'),
        Path(__file__).with_name('answer_readout.py')]}
    if args.wait_controller:
        while not (work/'comparison.json').exists():
            status=json.loads((work/'status.json').read_text()) if (work/'status.json').exists() else {}
            if status.get('stage')=='failed':raise RuntimeError(status)
            try:os.kill(args.wait_controller,0)
            except ProcessLookupError:raise RuntimeError('Controller exited before producing comparison.json')
            save(work/'final_status.json',dict(stage='waiting_for_pilot_evaluation',
                waiting_for_controller=args.wait_controller,pid=os.getpid(),updated_at=now()))
            time.sleep(15)
    old=json.loads((work/'comparison.json').read_text())
    assert old['status']=='complete'
    questions={}
    for mod,f in [('T','eval_text.jsonl'),('V','eval_multimodal.jsonl')]:
        for r in rows(work/'evaluation_data/shared'/f):
            questions[r['id'],mod]=eval_payload(r,work/'evaluation_data')['messages'][-1]['content']
    result={}
    source_hashes={}
    for arm in ['base','atomic','atomic_replay']:
        source=work/f'evaluation/{arm}/predictions.jsonl'
        source_hashes[str(source)]=sha(source)
        scored=[]
        for r in rows(source):
            pred,reason=parse_answer(r['raw_response'],options_from_question(questions[r['id'],r['modality']]))
            scored.append(dict(r,v3_raw_predicted=r['raw_predicted'],v4_raw_predicted=pred,v4_parse_reason=reason))
        write_rows(work/f'rescored_v4/{arm}/mcq_predictions.jsonl',scored)
        groups={}
        for mod in ['T','V']:
            for kind in ['original','rotation']:
                group=[r for r in scored if r['modality']==mod and ((r['probe_id']=='rotation')==(kind=='rotation'))]
                counts=Counter(str(r['v4_raw_predicted']) for r in group)
                groups[mod+'/'+kind]=dict(n=len(group),accuracy=sum(r['v4_raw_predicted']==r['gold'] for r in group)/len(group),
                    invalid_rate=sum(r['v4_raw_predicted'] is None for r in group)/len(group),
                    answer_distribution=dict(counts),max_letter_share=max(counts.get(x,0) for x in 'ABCD')/len(group))
        diag_source=work/f'diagnostics/{arm}/predictions.jsonl'
        source_hashes[str(diag_source)]=sha(diag_source)
        diag=[]
        for r in rows(diag_source):
            item=dict(r)
            if r['kind']=='independent_instruction':
                pred,reason=parse_answer(r['response'],options_from_question(r['messages'][0]['content']))
                item.update(v3_predicted=r['predicted'],v4_predicted=pred,v4_parse_reason=reason,v4_correct=pred==r['expected'])
            diag.append(item)
        write_rows(work/f'rescored_v4/{arm}/diagnostic_predictions.jsonl',diag)
        d=json.loads((work/f'diagnostics/{arm}/summary.json').read_text())
        instruction=[r for r in diag if r['kind']=='independent_instruction']
        d['groups']['independent_instruction'].update(accuracy=sum(r['v4_correct'] for r in instruction)/len(instruction),
            invalid_rate=sum(r['v4_predicted'] is None for r in instruction)/len(instruction))
        result[arm]=dict(mcq=groups,diagnostics=d)
    gates={}
    for arm in ['atomic','atomic_replay']:
        r=result[arm]; diagnostic=r['diagnostics']['groups']; b=result['base']
        checks=dict(text_invalid=r['mcq']['T/original']['invalid_rate']<=0.05,
            vision_invalid=r['mcq']['V/original']['invalid_rate']<=0.05,
            text_gain=r['mcq']['T/original']['accuracy']-b['mcq']['T/original']['accuracy']>=0.10,
            rotation_balance=r['mcq']['T/rotation']['max_letter_share']<=0.50,
            attribute_recall=diagnostic['trained_attribute_recall']['strict_target_span_hit_rate']>=0.50,
            independent_instruction=diagnostic['independent_instruction']['accuracy']>=b['diagnostics']['groups']['independent_instruction']['accuracy']-0.05)
        gates[arm]=dict(checks=checks,passed=all(checks.values()))
    save(work/'rescored_v4/comparison.json',dict(status='complete',readout_version=VERSION,
        reports=result,gates=gates,created_at=now(),original_comparison_sha256=sha(work/'comparison.json'),
        source_sha256=source_hashes,protocol_change='Only canonicalize unambiguous LaTeX text wrappers inside boxed single letters; identical change for all conditions; no generation, prompt, or label-dependent readout changes.',
        original_outputs_and_scores_preserved=True,scoring_source_sha256=source_hashes_at_start))
    save(work/'final_status.json',dict(stage='complete',main_results='rescored_v4/comparison.json',
        gates=gates,pid=os.getpid(),updated_at=now()))
    print(json.dumps(dict(gates=gates),ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--wait-controller',type=int)
    run(p.parse_args())
