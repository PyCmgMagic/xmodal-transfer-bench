#!/usr/bin/env python3
"""Audit missed explicit terminal answers on saved predictions; never regenerate."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import run_v7_evaluation as ev
from run_lora_cot_evaluation import final_letter
from run_lora_recall_apply import parse_answer as legacy_recall_parser
from run_lora_recall_apply_v2 import parse_answer
from rescore_cot_answers import recovered_answer as prior_tolerant_parser

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/answer_readout_audit_20261026'


def scan(records, labels, old_parser, metadata, details):
    stats=Counter();methods=Counter();extra=Counter();extra_examples=[];sources=Counter()
    for row in records:
        response=row['response'];gold=labels[row['id']]['answer']
        old=old_parser(response);new,method=parse_answer(response)
        prev=prior_tolerant_parser(response)
        stats['total']+=1;stats['old_invalid']+=old is None
        stats['new_invalid']+=new is None
        stats['old_correct']+=old==gold;stats['new_correct']+=new==gold
        stats['recovered']+=old is None and new is not None
        stats['recovered_correct']+=old is None and new==gold
        stats['old_valid_changed']+=old is not None and old!=new
        if old is None and new is not None:methods[method]+=1
        if new is None:
            stats['unresolved_truncated']+=row.get('finish_reason')=='length'
            stats['unresolved_no_terminal_answer']+=method=='no_unambiguous_terminal_answer'
        if prev is None and new is not None:
            extra['beyond_prior_tolerant_recovered']+=1
            extra['beyond_prior_tolerant_correct']+=new==gold
        if old is None and new is not None or old is not None and old!=new:
            details.write(json.dumps({**metadata,'id':row['id'],'qid':row.get('qid'),
                       'old_predicted':old,'new_predicted':new,'gold':gold,'correct':new==gold,
                       'parse_method':method,'finish_reason':row.get('finish_reason'),
                       'response':response},ensure_ascii=False)+'\n')
            if len(extra_examples)<3:extra_examples.append({'id':row['id'],'tail':response[-400:],
                                                          'new_predicted':new,'gold':gold})
    n=stats['total']
    return {**metadata,**dict(stats),'old_accuracy':stats['old_correct']/n,
            'new_accuracy':stats['new_correct']/n,
            'accuracy_delta_pp':100*(stats['new_correct']-stats['old_correct'])/n,
            **dict(extra),'recovery_methods':dict(methods),'examples':extra_examples}


def totals(rows):
    keys=['total','old_invalid','new_invalid','old_correct','new_correct','recovered',
          'recovered_correct','old_valid_changed','unresolved_truncated',
          'beyond_prior_tolerant_recovered','beyond_prior_tolerant_correct']
    sums={k:sum(r.get(k,0) for r in rows) for k in keys}
    sums['recovered_fraction_of_old_invalid']=sums['recovered']/sums['old_invalid'] if sums['old_invalid'] else 0
    sums['old_accuracy']=sums['old_correct']/sums['total'] if sums['total'] else 0
    sums['new_accuracy']=sums['new_correct']/sums['total'] if sums['total'] else 0
    return sums


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    full=[];recall=[];source_hashes={}
    work=ROOT/'runs/qwen3_lora_cot_eval_20261025'
    with (OUT/'recovered_answers.jsonl').open('w') as details:
        for report_path in sorted(work.glob('jobs/*/*/report.json')):
            report=json.loads(report_path.read_text());folder=report_path.parent
            labels={r['id']:r for r in ev.read_rows(work/'data'/report['dataset']/'eval_labels.jsonl')[0]}
            raw=folder/'raw.jsonl';records=ev.read_rows(raw)[0]
            assert len(records)==report['total']
            source_hashes[str(raw)]=ev.file_hash(raw)
            meta={'experiment':'full_cot','dataset':report['dataset'],'job':folder.parent.name,
                  'direction':report['direction'],'objective':report['objective']}
            result=scan(records,labels,final_letter,meta,details)
            assert result['old_invalid']==report['invalid'] and result['old_correct']==report['correct']
            full.append(result)
        work=ROOT/'runs/qwen3_lora_recall_apply_20261025'
        labels={r['id']:r for r in ev.read_rows(work/'data/eval_labels.jsonl')[0]}
        for raw in sorted(work.glob('jobs/*/raw.jsonl')):
            # The old run was stopped; include only complete 180-question arms.
            records=ev.read_rows(raw,allow_partial_tail=True)[0];groups=defaultdict(list)
            source_hashes[str(raw)]=ev.file_hash(raw)
            for row in records:
                if row['stage']=='application':groups[row['direction'],row['arm']].append(row)
            for (direction,arm),group in sorted(groups.items()):
                if len(group)!=180:continue
                meta={'experiment':'invalidated_recall_v1','dataset':'rules','job':raw.parent.name,
                      'direction':direction,'arm':arm}
                recall.append(scan(group,labels,lambda s:legacy_recall_parser(s)[0],meta,details))
    result={'created_at':ev.now(),'parser_source':str(ROOT/'scripts/run_lora_recall_apply_v2.py'),
            'parser_sha256':ev.file_hash(ROOT/'scripts/run_lora_recall_apply_v2.py'),
            'audit_sha256':ev.file_hash(__file__),'sources':source_hashes,
            'scope':'Completed cells only; no inference, no alteration of formal reports. Recall v1 is invalidated for name errors; its readout audit does not validate that experiment.',
            'limitation':'Conservative confirmed recoveries: explicit unambiguous terminal answers only. Unresolved does not prove the entire response contains no answer. Recovered answers may be wrong.',
            'full_cot':{'totals':totals(full),'by_dataset':{d:totals([r for r in full if r['dataset']==d]) for d in ['rules','entities']},'cells':full},
            'recall_v1':{'totals':totals(recall),'by_arm':{arm:totals([r for r in recall if r['arm']==arm]) for arm in ['direct_boxed','self_recall_boxed']},'cells':recall}}
    ev.save_json(OUT/'report.json',result)
    columns=['experiment','dataset','job','direction','arm','total','old_invalid','recovered',
             'recovered_correct','new_invalid','old_accuracy','new_accuracy','accuracy_delta_pp',
             'old_valid_changed','beyond_prior_tolerant_recovered','beyond_prior_tolerant_correct']
    with (OUT/'cells.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(full+recall)
    lines=['# 答案解析审计','',
           '只复算已完成方向。明确终答包括boxed、FINAL和末尾自然语言选项，未从推理任意猜字母。',
           '“恢复选项”与“恢复正确题”不同：前者原来判无效但可读出选项，后者该选项同时等于gold。',
           '尚不能解析的回答不一定完全没有表达选择，因此下述为保守确认数量。','',
           '| 范围 | 总回答数 | 原无效数 | 可恢复选项 | 其中实际答对 | 仍未解析 |',
           '|---|---:|---:|---:|---:|---:|']
    for title,r in [('全量CoT-规则',result['full_cot']['by_dataset']['rules']),
                    ('全量CoT-实体',result['full_cot']['by_dataset']['entities']),
                    ('全量CoT合计',result['full_cot']['totals']),
                    ('已作废回忆v1-直接作答',result['recall_v1']['by_arm']['direct_boxed']),
                    ('已作废回忆v1-回忆后作答',result['recall_v1']['by_arm']['self_recall_boxed'])]:
        lines.append(f"| {title} | {r['total']} | {r['old_invalid']} | {r['recovered']} | {r['recovered_correct']} | {r['new_invalid']} |")
    lines+=['','原回忆v1存在名称错误，本审计只说明其判分漏项，不能用其回忆准确率判断方法效果。',
            '全量原协议要求严格FINAL；这里衡量宽容终答解析可恢复的答案，不覆盖冻结的原协议成绩。',
            '单项及完整原始答复见cells.csv、recovered_answers.jsonl，来源哈希及评分规则见report.json。']
    (OUT/'report_zh.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'full_cot':result['full_cot']['totals'],'by_dataset':result['full_cot']['by_dataset'],
                      'recall_v1':result['recall_v1']['totals'],'recall_by_arm':result['recall_v1']['by_arm']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
