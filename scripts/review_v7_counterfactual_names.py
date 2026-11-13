#!/usr/bin/env python3
"""Save human semantic review decisions for all fixed48rule recall responses."""
from pathlib import Path
import pilot_v7_counterfactual_names as pilot
import run_v7_evaluation as ev
from run_v7_full_experiment import rows, save, sha


def main():
    expected={k['id']:k['fact']['quote'] for k in rows(pilot.OUT/'target_knowledge.jsonl')}
    result=[];summary={}
    for arm in pilot.ARMS:
        raw=pilot.OUT/arm/'predictions.jsonl'
        for r in rows(raw):
            if r['condition'] not in ('recall','recall_image'):continue
            correct=arm=='text' or arm=='vision' and r['condition']=='recall_image'
            if arm=='warmup':reason='未给出对应具体条件或阈值；复述通用边界说明、MATCH ANY或表示没有该规则，不能算完整知识正确。'
            elif arm=='vision' and r['condition']=='recall':reason='无图输出[2,6]或[1,4]等不相干通用区间；目标形状/颜色或实际区间及开闭边界均未正确调用。'
            else:reason='具体条件、组合方式或区间阈值及开闭边界与原结构规则一致，整段自洽。'
            if arm=='text' and r['condition']=='recall_image' and r['qid'] in ('scale1k_0964_programmatic','scale1k_0965_programmatic'):
                correct=False
                reason=('Lamdal上界误写为12，应为14，核心事实错误。' if '0964' in r['qid'] else 'Vevlev误写为[12,16)，应为[9,14)，核心事实错误。')
            result.append({'arm':arm,'id':r['id'],'qid':r['qid'],'condition':r['condition'],
                'core_facts_correct':correct,'whole_response_consistent':correct,'reason_zh':reason,
                'response':r['response'],'finish_reason':r['finish_reason'],'expected_structured_rule':expected[r['qid']],
                'source_file':str(raw),'source_sha256':sha(raw),'response_sha256':ev.digest(r['response']),
                'method':'Human inspection of all48saved responses; explicit judgments, not literal string matching.'})
        summary[arm]={}
        for condition in ('recall','recall_image'):
            rr=[r for r in result if r['arm']==arm and r['condition']==condition];assert len(rr)==8
            summary[arm][condition]={'n':8,'core_correct':sum(r['core_facts_correct'] for r in rr),
                'consistent_correct':sum(r['whole_response_consistent'] for r in rr),
                'truncated':sum(r['finish_reason']=='length' for r in rr)}
    assert len(result)==48
    ev.save_rows(pilot.OUT/'semantic_reviews.jsonl',result);save(pilot.OUT/'semantic_review_summary.json',summary)


if __name__=='__main__':main()
