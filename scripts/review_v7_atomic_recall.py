#!/usr/bin/env python3
"""Transparent manual semantic annotations for the fixed observed recall outputs.

These annotations were made after reading each raw response and the expected
structured rule. They are not another model evaluation or an automatic grader.
"""
import json
from collections import Counter
from pathlib import Path

import run_v7_evaluation as ev
from pilot_v7_atomic_supervision import OUT, ARMS
from run_v7_full_experiment import read, rows, save, sha


def review(arm, condition, qid, response, exact):
    short = qid.split('_')[1]
    if exact:
        return True, True, 'correct_exact', '完整紧凑规则逐字匹配。'
    if arm == 'atomic_text':
        if condition == 'recall_canonical' and short == '0173':
            return True, True, 'correct_concise', '明确写出 value>=2 和 value<6；省略输出模板不影响核心规则。'
        if condition == 'recall_neutral':
            assert short == '0753'
            return False, False, 'wrong_category_mapping', '阈值和红色设备车定义正确，但把 Kinkir 与 Rersin 的低/中类别互换。'
        if condition == 'recall_task':
            if short in ('0954', '0689', '0957'):
                return False, False, 'case_answer_or_incomplete', '输出 KEEP/OMIT 或结果指令，未复述所需具体规则。'
            if short == '0173':
                return False, False, 'wrong_rule_facts', '将数值区间错换成正方形/红色的析取规则。'
            assert short in ('0753', '0284')
            return False, False, 'wrong_category_mapping', '把提示中的类别顺序当作低中高映射；具体类别错绑。'
        if condition == 'recall_image':
            if short == '0173':
                return True, True, 'correct_concise', 'value>=2 和 value<6 均正确。'
            notes = {'0689': '只写 crate 而非 transport crate，具体类别限定不完整；排除 damaged 正确。',
                     '0753': '阈值和类别正确，但未说具体目标是红色 equipment cart。',
                     '0973': '占用比定义与类别次序存在，但未给两个阈值数字。',
                     '0968': '把边界错写为0.25/0.65，真实0.3/0.85。',
                     '0284': '错换为occupied_regions，且未写两个具体边界。'}
            return False, False, 'incomplete_or_wrong_facts', notes[short]
    if arm == 'original_text':
        if condition == 'recall_neutral':
            return True, True, 'correct_paraphrase', '具体条件或占比定义、阈值、边界和类别均正确；0973沿用原材料中蓝色圆物占用区域的描述。'
        if condition == 'recall_canonical':
            if short == '0954':
                return False, False, 'wrong_rule_family', '应为五边形或白色，回答成标记格数值区间。'
            return True, True, 'correct_paraphrase', '完整具体规则正确；保留原材料的措辞和示例。'
        if condition == 'recall_task':
            if short in ('0753', '0968'):
                return True, True, 'correct_paraphrase', '计算定义、两个阈值、类别及边界均正确。'
            if short == '0173':
                return True, False, 'correct_core_wrong_example', '开头核心区间[2,6)正确，但后续例子将12判为合格、15判为不合格，与区间矛盾。'
            return False, False, 'wrong_rule_facts', '将其他筛选规则/占比定义及阈值套到请求名称上。'
        if condition == 'recall_image':
            notes = {'0954': '说两条件任一，但缺白色条件；只在示例提五边形。',
                     '0689': '只要求intact，缺对象类别；intact也比排除damaged更窄。',
                     '0957': '通用任一条件说明；例子有square/red，但未完整给出两个明确参考条件。',
                     '0173': '上界写成7，真实为6。',
                     '0753': '数值阈值和类别正确，但只说参考物的type/color，没有具体红色设备车。',
                     '0973': '将比例区间改成分子分母各自大小条件。',
                     '0968': '阈值错写0.25/0.75且例子边界判定矛盾。',
                     '0284': '阈值及类别映射不完整或错误。'}
            return False, False, 'incomplete_or_wrong_facts', notes[short]
    if arm == 'atomic_vision':
        if condition == 'recall_image':
            assert short == '0968'
            return False, False, 'wrong_lower_threshold', '下阈值写成0.4，真实为0.3；上阈值0.85及类别对应正确。'
        return False, False, 'wrong_rule_facts', '无学习图时生成与目标无关的年龄、身高、车辆或计数规则；未恢复该条具体规则。'
    if arm == 'original_vision':
        if condition == 'recall_image':
            return False, False, 'generic_reference_without_facts', '主要复现原配文的读取图片说明，未明确给出图中的具体参考属性或阈值，不能作为脱离图片的完整规则。'
        return False, False, 'wrong_or_unresolved_rule', '缺具体规则事实，且添加无关计算、类别/筛选步骤；并非完整正确的请求规则。'
    if arm == 'warmup':
        if condition == 'recall_image':
            return False, False, 'incomplete_rule', '只给通用约定、标题或部分类别关系，缺具体属性、数值或阈值。'
        if condition == 'recall_task' and short in ('0954', '0689', '0957', '0173'):
            return False, False, 'echoed_task_instruction', '复述Apply/Report指令，未给规则条件。'
        if condition == 'recall_canonical' and short in ('0968', '0973'):
            return False, False, 'no_stored_rule_response', '回答没有存储的规则，未提供核心事实。'
        return False, False, 'wrong_or_missing_rule', '提供虚构数值/类别规则或只给通用约定，未恢复该条规则。'
    raise AssertionError((arm, condition, short, response))


def main():
    knowledge = {k['id']: k for k in rows(OUT / 'target_knowledge.jsonl')}
    reviews = []
    for arm in ARMS:
        for path, selected in ((OUT / arm / 'predictions.jsonl', ('recall_canonical', 'recall_task', 'recall_image')),
                               (OUT / 'neutral_recall' / arm / 'predictions.jsonl', ('recall_neutral',))):
            rr = rows(path)
            for r in rr:
                if r['condition'] not in selected:
                    continue
                core, full, category, note = review(arm, r['condition'], r['qid'], r['response'], r['correct'])
                reviews.append({'source': str(path), 'source_sha256': sha(path), 'id': r['id'], 'arm': arm,
                                'qid': r['qid'], 'condition': r['condition'], 'response_sha256': ev.digest(r['response']),
                                'expected_rule': json.loads(knowledge[r['qid']]['fact']['quote']),
                                'response': r['response'], 'core_rule_correct': core, 'fully_consistent_response': full,
                                'category': category, 'notes_zh': note, 'review_method': 'agent_manual_semantic_review'})
    assert len(reviews) == 160
    ev.save_rows(OUT / 'semantic_reviews.jsonl', reviews)
    summary = {}
    for arm in ARMS:
        summary[arm] = {}
        for condition in ('recall_canonical', 'recall_task', 'recall_image', 'recall_neutral'):
            group = [r for r in reviews if r['arm'] == arm and r['condition'] == condition]
            assert len(group) == 8
            summary[arm][condition] = {'n': 8, 'core_rule_correct': sum(r['core_rule_correct'] for r in group),
                                      'fully_consistent': sum(r['fully_consistent_response'] for r in group),
                                      'categories': dict(Counter(r['category'] for r in group))}
    save(OUT / 'semantic_review_summary.json', {'reviewed': 160, 'arms': summary,
                                              'note': 'Manual judgments on fixed responses with source hashes. Full-rule completeness requires concrete facts, not referring to a supplied image. Exact short-fact metrics are NOT treated as semantic accuracy.',
                                              'script_sha256': sha(Path(__file__))})
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
