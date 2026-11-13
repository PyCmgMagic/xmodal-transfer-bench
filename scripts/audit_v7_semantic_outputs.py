#!/usr/bin/env python3
"""Audit saved semantic-pilot outputs without changing formal scores or inference."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path


def load(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def statuses(text):
    # Only read the initial answer, never status mentions inside unsolicited reasons.
    text = re.split(r'\*?\*?Reasoning|—', text, maxsplit=1, flags=re.I)[0]
    found = re.findall(r'\b(Check\s+[123]|Qualification)\s*[:=]\s*(Pass|Fail|N/A|Not Applicable)\b', text, re.I)
    result = {}
    for key, value in found:
        key = re.sub(r'\s+', ' ', key).casefold()
        value = value.casefold()
        if key in result:
            return None
        result[key] = value
    return result


def field_equal(response, gold):
    expected, actual = statuses(gold), statuses(response)
    if not expected or not actual:
        return False
    return all(actual.get(k) == v for k, v in expected.items()) and all(
        k in expected or v in ('n/a', 'not applicable') for k, v in actual.items())


def audit(folder):
    groups = {}
    for name, path in [('base', folder / 'base_control/predictions.jsonl'),
                       ('trained', folder / 'predictions.jsonl')]:
        records = load(path)
        result = {}
        for condition in sorted({r['condition'] for r in records}):
            group = [r for r in records if r['condition'] == condition]
            entry = {'n': len(group), 'strict_correct': sum(r['correct'] for r in group),
                     'response_distribution': dict(Counter(r['response'] for r in group)),
                     'gold_distribution': dict(Counter(r['gold'] for r in group))}
            if condition == 'semantic_practice_diagnosis':
                entry['field_match_correct'] = sum(field_equal(r['response'], r['gold']) for r in group)
                entry['field_match_additions'] = [r['id'] for r in group
                                                 if not r['correct'] and field_equal(r['response'], r['gold'])]
                for label, mixed in [('mixed', True), ('uniform', False)]:
                    selected = [r for r in group if
                                (len({v for k, v in statuses(r['gold']).items() if k.startswith('check ')}) > 1) == mixed]
                    entry[label] = {'n': len(selected), 'strict_correct': sum(r['correct'] for r in selected),
                                    'field_match_correct': sum(field_equal(r['response'], r['gold']) for r in selected)}
            result[condition] = entry
        groups[name] = result
    audit_result = {'groups': groups, 'scope': 'Saved outputs only; formal metrics remain unchanged.',
                    'field_scoring': 'Accept colon/equals, whitespace, and additional N/A checks; require every applicable status and Qualification to match.',
                    'condition_review': 'All 20 trained condition nonmatches were inspected: wrong object, color, material, shape, or numeric requirement; broad box is insufficient for a specific storage box/toolbox requirement.'}
    (folder / 'output_audit.json').write_text(json.dumps(audit_result, ensure_ascii=False, indent=2) + '\n')
    trained = groups['trained']
    code = trained['semantic_record_code_recall']
    diagnosis = trained['semantic_practice_diagnosis']
    base_diagnosis = groups['base']['semantic_practice_diagnosis']
    lines = ['# 原始输出复核', '',
             '这轮1epoch混合监督没有通过原题迁移验证：未微调8/24，全选择题对照7/24，混合监督6/24。24题样本很小，不能据此宣称训练显著降低了准确率；可以确认本轮没有展示改善。', '',
             '## 固定答案与事实错误', '',
             f"- 代码映射回忆24题中，输出分布为 {code['response_distribution']}；固定回答H7也能答对 {code['gold_distribution'].get('H7', 0)}/24，本轮实际 {code['strict_correct']}/24。不能把这些正确答案当作已学会映射的证据。",
             '- 条件回忆仅4/24完整匹配。其余20条逐项检查均存在具体事实错误：例如Nornat要求tray却输出box；Kamnet要求circle却输出box，至少4个却输出1个。没有发现仅因同义改写而误判的正确答案。',
             f"- 检查记录严格匹配为 {base_diagnosis['strict_correct']}/24 → {diagnosis['strict_correct']}/24。统一冒号/等号并允许额外N/A检查项后，未微调为 {base_diagnosis['field_match_correct']}/24，训练后为 {diagnosis['field_match_correct']}/24。严格匹配的提升部分来自格式符合要求。",
             f"- 检查结果既有Pass又有Fail的记录：训练后 {diagnosis['mixed']['strict_correct']}/{diagnosis['mixed']['n']}；全Pass或全Fail的记录：{diagnosis['uniform']['strict_correct']}/{diagnosis['uniform']['n']}。目前正确记录集中在全同状态，不能宣称学会逐条件核验。", '',
             '## 解释边界', '',
             '144个选择题输出均为有效单字母，首个答案token与最终解析完全一致。因此，本轮低分不能用解析器吞掉正确答案解释。训练会生成所需格式，但未稳定建立“规则名称→条件/代码→实际应用”的对应关系。这是当前结果支持的观察；其优化原因是否来自监督分配、更新强度或任务干扰，仍未被独立对照确定。', '',
             '本轮只覆盖Qwen3的12条二值规则和24道原始T2T题；没有修改v7原题、标签或图片，也没有扩展到完整1000条与四方向。下一步应先用训练集内的条件/映射回忆证明规则能被学会，再看独立验证题；不能把盲测接近25%本身当作数据修复成功。', '',
             '正式准确率与所有原始回答保留在report.json、predictions.jsonl。本文的字段匹配仅为补充诊断，不回写正式分数。']
    (folder / 'output_audit_zh.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'code_recall': code, 'diagnosis': diagnosis,
                      'base_diagnosis_field_match': base_diagnosis['field_match_correct']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    audit(parser.parse_args().out.resolve())
