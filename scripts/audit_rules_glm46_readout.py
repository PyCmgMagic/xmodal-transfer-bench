#!/usr/bin/env python3
"""Audit saved rule responses on CPU, preserving original evaluation artifacts."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from zoneinfo import ZoneInfo

from answer_readout import options_from_question
from evaluate_entity_glm46_full import glm_readout

VERSION = 'rules-glm-offline-readout-audit-v1'
BOXES = re.compile(
    r'\\boxed\s*\{([^{}]*)\}|'
    r'<\|begin_of_box\|>((?:(?!<\|begin_of_box\|>|<\|end_of_box\|>)[\s\S])*?)<\|end_of_box\|>')
TRAILING = re.compile(r'[\s.!?$\\()\[\]*_`]*')


def content_key(value):
    """Ignore presentation only; preserve signs, numbers, and relation operators."""
    value = unicodedata.normalize('NFKC', value).strip()
    value = re.sub(r'^Result\s*:\s*', '', value, flags=re.I)
    return re.sub(r'\s+', '', value.rstrip('.').strip()).casefold()


def visible_text(response):
    opens, closes = response.count('<think>'), response.count('</think>')
    if opens == closes:
        # A final opening tag is unfinished even when earlier stray closes balance it.
        text = response.rsplit('</think>', 1)[-1]
        return (None, False) if '<think>' in text else (text.strip(), False)
    if opens == 1 and closes == 2 and re.match(r'^\s*<think>\s*</think>', response):
        # Known GLM output: <think></think> rationale </think> explicit final answer.
        # Inspect only the final answer region, including native GLM box tokens.
        return response.rsplit('</think>', 1)[-1].strip(), True
    return None, False


def terminal_box(text):
    matches = list(BOXES.finditer(text))
    if len(matches) != 1 or not TRAILING.fullmatch(text[matches[0].end():]):
        return None
    match = matches[0]
    # Do not accept an example, conditional, or negated box as a final selection.
    start = max(text.rfind('\n', 0, match.start()), text.rfind('.', 0, match.start())) + 1
    prefix = text[start:match.start()]
    if re.search(r'\b(?:if|unless|not|example|e\.g)\b', prefix, re.I):
        return None
    return (match[1] if match[1] is not None else match[2]).strip()


def reparse(response, options):
    """Gold-independent recovery of explicit final answers, without inference."""
    predicted, reason = glm_readout(response, options)
    if predicted is not None:
        return predicted, 'unchanged/' + reason
    if reason in {'conflicting_answers', 'conditional_boxed_answer', 'letter_content_conflict'}:
        return None, reason
    if not isinstance(response, str):
        return None, reason
    text, duplicate_close = visible_text(response)
    if text is None:
        return None, reason
    if duplicate_close:
        predicted, new_reason = glm_readout(text, options)
        if predicted is not None:
            return predicted, 'duplicate_think_close/' + new_reason
        if new_reason in {'conflicting_answers', 'conditional_boxed_answer'}:
            return None, new_reason
    body = terminal_box(text)
    if body is None or not options:
        return None, reason
    prefixed = re.fullmatch(r'([A-D])[.):]\s+(.+)', body, re.S)
    result = prefixed[2] if prefixed else body
    hits = [letter for letter, value in options.items() if content_key(result) == content_key(value)]
    if prefixed and hits != [prefixed[1]]:
        return None, 'boxed_letter_content_conflict'
    if len(hits) == 1:
        kind = 'boxed_letter_and_full_option' if prefixed else 'boxed_full_option_content'
        if duplicate_close:
            kind = 'duplicate_think_close/' + kind
        return hits[0], kind
    return None, 'boxed_content_not_unique_option'


def category(record, predicted, reason):
    if record['raw_predicted'] is not None:
        return 'already_valid'
    if predicted is not None:
        return 'parser_duplicate_think_close' if reason.startswith('duplicate_think_close/') else 'parser_boxed_option_content'
    if record['stop_reason'] == 'length':
        return 'length_without_final_choice'
    text, _ = visible_text(record['raw_response'])
    if text is None:
        return 'unresolved_thinking_tags'
    body = terminal_box(text)
    if body is None:
        return 'rule_explanation_without_final_option' if re.search(r'worked example', text, re.I) else 'no_final_option'
    if re.match(r'none\b', body, re.I):
        return 'none_of_the_options'
    if reason == 'boxed_letter_content_conflict':
        return 'boxed_letter_content_conflict'
    if re.fullmatch(r'[A-D](?:\s*,\s*[A-D])+', body):
        return 'candidate_ids_not_option_letter'
    return 'boxed_result_matches_no_option'


def snapshot(path):
    payload = path.read_bytes()
    if not payload.endswith(b'\n'):
        payload = payload[:payload.rfind(b'\n') + 1]
    return payload, [json.loads(line) for line in payload.decode('utf-8').splitlines() if line.strip()]


def scores(records, key):
    n = len(records)
    correct = sum(r[key] == r['gold'] for r in records)
    invalid = sum(r[key] is None for r in records)
    return dict(n=n, correct=correct, accuracy=correct / n, invalid=invalid, invalid_rate=invalid / n)


def run(work, output):
    output.mkdir(parents=True, exist_ok=False)
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
    questions = {}
    for mod, name in [('T', 'eval_text.jsonl'), ('V', 'eval_multimodal.jsonl')]:
        for line in (work / 'evaluation_data/shared' / name).read_text().splitlines():
            row = json.loads(line)
            questions[mod, row['id']] = row['messages'][-1]['content']
    manifest, audit, original_count = {}, [], 0
    for arm in ['TEXT', 'base']:
        source = work / 'evaluation' / arm / 'predictions.jsonl'
        payload, records = snapshot(source)
        (output / f'{arm}_predictions_snapshot.jsonl').write_bytes(payload)
        manifest[arm] = dict(source=str(source), sha256=hashlib.sha256(payload).hexdigest(),
            records=len(records), complete=len(records) == 4000, last_saved_at=records[-1]['saved_at'])
        original_count += len(records)
        for row in records:
            question = questions[row['modality'], row['id']]
            options = options_from_question(question)
            assert set(options) == set('ABCD')
            baseline, baseline_reason = glm_readout(row['raw_response'], options)
            assert (baseline, baseline_reason) == (row['raw_predicted'], row['raw_parse_reason'])
            predicted, reason = reparse(row['raw_response'], options)
            assert row['raw_predicted'] is None or row['raw_predicted'] == predicted
            audit.append(dict(row, audit_predicted=predicted, audit_parse_reason=reason,
                audit_category=category(row, predicted, reason), question=question,
                recovered=row['raw_predicted'] is None and predicted is not None))
    assert len(audit) == original_count
    groups = {}
    for arm in ['TEXT', 'base']:
        for mod in 'TV':
            subset = [r for r in audit if r['arm'] == arm and r['modality'] == mod]
            if not subset:
                continue
            invalid = [r for r in subset if r['raw_predicted'] is None]
            recovered = [r for r in subset if r['recovered']]
            groups[arm + '/' + mod] = dict(original=scores(subset, 'raw_predicted'),
                offline_reparsed=scores(subset, 'audit_predicted'), recovered=len(recovered),
                recovered_correct=sum(r['audit_predicted'] == r['gold'] for r in recovered),
                original_invalid_categories=dict(Counter(r['audit_category'] for r in invalid)),
                original_invalid_reason_stop=dict(Counter(r['raw_parse_reason'] + '/' + r['stop_reason'] for r in invalid)),
                repaired_parse_reasons=dict(Counter(r['audit_parse_reason'] for r in recovered)))
    summary = dict(version=VERSION, created_at=stamp, source_snapshots=manifest, groups=groups,
        protocol='Offline only; same rule for base and LoRA; no gold in parsing; no new generation; original scores preserved.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    with (output / 'all_reparsed.jsonl').open('w') as stream:
        for row in audit:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    invalid = [r for r in audit if r['raw_predicted'] is None]
    with (output / 'original_invalid_details.jsonl').open('w') as stream:
        for row in invalid:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    lines = ['# 规则类 GLM 无效回答离线审计', '', f'统计时间：{stamp}', '',
        '训练模型是 TEXT LoRA（3轮）。本报告仅使用已保存的回答，在 CPU 上重解析；没有重新推理。',
        '原始评测文件及原始分数保留。下表为单独标注的离线重解析分数；基座和微调模型使用同一规则。', '',
        '| 模型 | 测试 | 题数 | 原准确率 | 重解析准确率 | 原无效率 | 重解析无效率 | 恢复/其中正确 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for name, group in groups.items():
        old, new = group['original'], group['offline_reparsed']
        lines.append(f"| {name} | {'文字' if name.endswith('/T') else '图片'} | {old['n']} | {old['accuracy']:.2%} | {new['accuracy']:.2%} | {old['invalid_rate']:.2%} | {new['invalid_rate']:.2%} | {group['recovered']}/{group['recovered_correct']} |")
    lines += ['', '## 原无效回答的去向', '', '| 模型/测试 | 分类 | 数量 |', '|---|---|---:|']
    labels = dict(parser_duplicate_think_close='多余的思考结束标签导致误判（已恢复）',
        parser_boxed_option_content='框内完整选项内容未被读取（已恢复）',
        length_without_final_choice='触及 token 上限，未形成可确认的最终答案',
        none_of_the_options='模型回答没有正确选项',
        rule_explanation_without_final_option='只解释规则，没有最终选项',
        boxed_letter_content_conflict='框内选项字母与内容不一致',
        candidate_ids_not_option_letter='输出候选对象列表，未选择答案选项',
        boxed_result_matches_no_option='框内计算/映射结果与全部选项不匹配')
    for name, group in groups.items():
        for kind, count in group['original_invalid_categories'].items():
            lines.append(f'| {name} | {labels.get(kind, kind)} | {count} |')
    lines += ['', '## 保守解析规则', '',
        '1. 原来有效的答案保持原判；不从 gold、正确率或规则材料推断答案。',
        '2. 仅对 `<think></think> ... </think> 最终答案` 这一已确认模式读取最后结束标签后的答案；真正未完成的思考仍无效。',
        '3. 支持 LaTeX 和 GLM 原生框中的完整选项内容。内容必须与一个完整选项唯一一致；有字母前缀时，字母也必须一致。',
        '4. 保留负号、数值及关系符号，防止把不同结果归一化为同一个结果。',
        '5. 不把候选对象 A/B/C/D 当成选项，不猜截断答案，不强制补选。', '',
        '## 核对实例', '',
        '基座原始回答包含 `<think></think> ... </think> \\boxed{D}`，旧解析器按标签数量不同直接拒绝；离线审计只读取明确的最终答案。',
        '微调模型 Didtev 回答把 toolbox/barrier 的权重记成 4/1，且漏掉 +2；训练材料实际为 1/4 再 +2。其“没有正确选项”属于规则回忆/应用错误，不能靠解析改正。',
        '微调模型 Silrov 回答得到 25；训练材料实际为 1×storage_box + 3×supply_tray + 1，测试 3/4 的结果为16（选项 C）。', '',
        '逐题问题、原始回复、原解析、恢复结果及分类见 original_invalid_details.jsonl。',
        'TEXT/base_predictions_snapshot.jsonl 是本次统计的固定输入；若基座尚未完成，应在完成后生成新的审计目录。', '']
    (output / 'REPORT_ZH.md').write_text('\n'.join(lines))
    print(json.dumps(dict(output=str(output), groups=groups, source_snapshots=manifest), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    work = args.work.resolve()
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    run(work, args.output or work / 'analysis' / ('readout_audit_' + stamp))
