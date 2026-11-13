#!/usr/bin/env python3
"""Posthoc semantic reading of frozen diagnostic outputs; preserve strict scores."""
import json
import math
import re
from pathlib import Path

from run_v7_full_experiment import ROOT, read, rows, save, sha

IMAGE = ROOT / 'runs/v7_process_image_diagnostic_20261021'
COUNT = ROOT / 'runs/v7_process_count_diagnostic_20261021'


def main():
    assert read(IMAGE / 'completion_audit.json')['status'] == 'pass'
    assert read(COUNT / 'completion_audit.json')['status'] == 'pass'
    count_review = []
    for r in rows(COUNT / 'evaluation/predictions.jsonl'):
        if r['condition'] != 'count_extract': continue
        parsed = json.loads(re.sub(r'^```(?:json)?\s*|\s*```$', '', r['response'].strip()))
        assert set(parsed) == set('ABCD')
        values = [parsed[c]['value'] if isinstance(parsed[c], dict) else parsed[c] for c in 'ABCD']
        assert all(type(v) in (int, float) and math.isfinite(v) and v >= 0 and int(v) == v for v in values)
        expected = [json.loads(r['gold'])[c]['value'] for c in 'ABCD']
        count_review.append(dict(id=r['id'], expected=expected, observed=values,
            matching_fields=sum(a == b for a, b in zip(values, expected)), scene_correct=values == expected,
            strict_schema_valid=all(isinstance(parsed[c], dict) and 'value' in parsed[c] for c in 'ABCD')))
    count_summary = dict(scope='Posthoc semantic count score, separate from frozen strict nested-schema score.',
        scenes=len(count_review), scenes_correct=sum(r['scene_correct'] for r in count_review),
        fields=4 * len(count_review), fields_correct=sum(r['matching_fields'] for r in count_review),
        strict_schema_valid=sum(r['strict_schema_valid'] for r in count_review), review=count_review)
    assert (count_summary['scenes_correct'], count_summary['fields_correct']) == (6, 30)
    save(COUNT / 'posthoc_semantic_count_summary.json', count_summary)
    recall_review = []
    text_errors = {('0035', 'direct'): 'wood被改为metal', ('0662', 'diagnosis'): 'wood被改为plastic',
                   ('0712', 'diagnosis'): '上界5被改为4'}
    visual_correct = {('0035', 'direct'), ('0035', 'diagnosis'), ('0712', 'diagnosis')}
    for r in rows(IMAGE / 'evaluation/predictions.jsonl'):
        if r['condition'] != 'context_recall': continue
        key = (r['qid'].split('_')[1], r['task'])
        if r['input_mod'] == 'T':
            verdict = 'incorrect' if key in text_errors else 'correct'
            reason = text_errors.get(key, '参数、字段、逻辑及输出顺序正确；允许近义改写')
        elif key in visual_correct:
            verdict, reason = 'correct', '参数、字段和逻辑正确；允许近义改写'
        elif key == ('0662', 'direct'):
            verdict, reason = 'uncertain', 'marker panel被写成material panel，不自动视为相同类别'
        elif key == ('0673', 'direct'):
            verdict, reason = 'incorrect', 'material字段被替换成closure type'
        else:
            verdict, reason = 'incorrect', '错误/遗漏规则字段、参数或规则结构，见保留的原始回答'
        recall_review.append(dict(id=r['id'], modality=r['input_mod'], verdict=verdict, rationale=reason,
                                  gold=r['gold'], response=r['response']))
    semantic_recall = {mod: {v: sum(r['modality'] == mod and r['verdict'] == v for r in recall_review)
                            for v in ('correct', 'incorrect', 'uncertain')} for mod in ('T', 'V')}
    assert semantic_recall['T']['correct'] == 13 and semantic_recall['V']['correct'] == 3
    save(IMAGE / 'posthoc_manual_recall_review.json', dict(scope='Manual posthoc parameter/logic review, no official-score changes.',
        summary=semantic_recall, review=recall_review))
    report = '''# 图片低分诊断：语义复核

固定上一轮8规则/1seed过程模型及16张原题图片；128次主控制和16次补充控制均已完成。以下全部是事后诊断，不能当成正式闭卷迁移成绩，也未改变64规则实验方案。

| 控制 | 正确数 |
|---|---:|
| 原题闭卷文字/图片推理 | 16/16、5/16 |
| 图片＋正确规则 | 9/16 |
| 图片＋正确文字观测 | 12/16 |
| 图片＋正确规则和观测 | 16/16 |
| 图片＋同一模型未经修正的回忆 | 5/16 |

提供规则或观测均能提高分数，两者同时提供时本样本全部答对。这支持规则调用与观测表示都有影响，但oracle差值不唯一分解错误原因。

## 规则调用的直接例子

Kinsem本应为“2≤value≤4”，图片上下文只要求回忆规则时，模型却回答“item tray或metal”的类别/材质规则。补充提示value是托盘内物体数量后，它能正确数出4、2、5、1，却在应用时额外加入“必须是equipment cart”的条件而答错。Roncar、Sanvat也多次把区间规则替换成toolbox/material规则。图片内容会伴随错误规则结构的输出，不能仅靠加强计数来解释这些错误。

完整定义精确匹配为文字10/16、图片2/16；逐条允许近义改写后，文字13/16语义正确，图片3/16正确、1/16类别表述不确定、12/16错误。该回忆任务与正式解题提示不同，因此这些计数不能直接当作解题过程的内部状态或严格因果分解。

## 原“只读图”诊断有提示及格式限制

区间原图没有书写数字，用每个托盘内物体数量表示value。原提示问“displayed numeric value”，有的回答转去描述phone/mask/flashlight类别，因此原3/16字面匹配不能直接称为读图准确率；类别题还存在brush/cleaning brush、cart/equipment cart等别名。

补充明确要求计数的8张图，模型全部返回平铺JSON（如A:4），而指令要求嵌套value字段。冻结的严格格式评分因此是0/8，8个格式错误。单独按语义核对计数为6/8张全对、30/32个候选数量正确；另两处为4数成5、6数成5。严格分数和原始输出均保留，语义复核另存，未把格式问题隐去。

仅给出“value=物体个数”的通用输入约定、不给规则/正确数量，区间题为2/8，与原区间闭卷2/8相同，而且有正确字母伴随错误规则的回答。因此本轮没有证据表明加一条计数说明即可修复迁移。

当前证据支持继续先确认过程学习能否稳定扩大，并保留规则回忆、观测与执行三个诊断环节。不能宣称已经找到唯一机制，也不能根据这16题断言完整1000条的迁移瓶颈或修改正式原图。
'''
    (IMAGE / 'analysis_zh.md').write_text(report)
    save(IMAGE / 'semantic_review_provenance.json', dict(script_sha256=sha(Path(__file__)),
        image_predictions_sha256=sha(IMAGE / 'evaluation/predictions.jsonl'),
        count_predictions_sha256=sha(COUNT / 'evaluation/predictions.jsonl'),
        manual_recall_summary=semantic_recall, strict_results_unchanged=True))
    print(json.dumps(dict(count=count_summary | {'review': 'see file'}, recall=semantic_recall), ensure_ascii=False))


if __name__ == '__main__':
    main()
