#!/usr/bin/env python3
"""Independently verify the frozen 20-rule diagnostic and summarize limits."""
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/v7_learned_facts20_20261021'


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    value = ' '.join(str(value).lower().split())
    return ','.join(sorted(part.strip() for part in value.split(',')))


def metric(records):
    n = len(records)
    correct = sum(r['correct'] for r in records)
    return dict(n=n, correct=correct,
                accuracy_percent=100 * correct / n if n else None,
                format_errors=sum(r['format_error'] for r in records),
                truncated=sum(r['finish_reason'] == 'length' for r in records))


def audit():
    manifest = read(OUT / 'manifest.json')
    for path, expected in manifest['sources'].items():
        assert sha(Path(path)) == expected, path
    for path, expected in manifest['files'].items():
        assert sha(OUT / path) == expected, path
    for path, expected in manifest['images'].items():
        assert sha(Path(path)) == expected, path

    probes = rows(OUT / 'probes.jsonl')
    by_id = {r['id']: r for r in probes}
    assert len(probes) == len(by_id) == 240
    knowledge = {r['id']: r for r in rows(OUT / 'selected_knowledge.jsonl')}
    facts = rows(OUT / 'facts_with_evidence.jsonl')
    training_path = ROOT / 'runs/v7_fullparam_sft_ep10_20261019/data/sft_text.jsonl'
    training = {r['id']: r for r in rows(training_path)}
    assert len(knowledge) == 20 and len(facts) == 60
    assert Counter(r['gold'] for r in facts) == dict.fromkeys('ABCD', 15)
    for f in facts:
        assert len({canonical(v) for v in f['options']}) == 4, f['id']
        assert f['options']['ABCD'.index(f['gold'])] == f['answer_value'], f['id']
        material = training[f['qid']]['messages'][-1]['content']
        assert material == knowledge[f['qid']]['material']['learning']['text']
        assert f['evidence'] in material
        closed = by_id[f['id']]
        oracle = by_id['oracle:' + f['id']]
        assert all(m['role'] != 'assistant' for m in closed['messages'])
        assert oracle['messages'][-1]['content'] == (
            'Correct learning material:\n' + material + '\n\n'
            + closed['messages'][-1]['content'])

    predictions = {}
    metrics, histograms, families, exceptions = {}, {}, {}, []
    for arm in ('base', 'text'):
        records = rows(OUT / arm / 'predictions.jsonl')
        predictions[arm] = {r['id']: r for r in records}
        assert len(records) == len(predictions[arm]) == 240
        assert set(predictions[arm]) == set(by_id)
        for r in records:
            p = by_id[r['id']]
            payload = dict(messages=p['messages'], images=p.get('images', []))
            digest = hashlib.sha256(json.dumps(
                payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            assert digest == r['input_sha256'], r['id']
            raw = r['raw_response']['choices'][0]
            assert r['response'] == (raw['message']['content'] or '')
            parsed = r['response'].strip()
            parsed = parsed if parsed in tuple('ABCD') else None
            assert parsed == r['predicted']
            assert r['gold'] == p['gold']
            assert r['correct'] == (parsed == p['gold'])
            assert r['format_error'] == (parsed is None)
            assert r['finish_reason'] == raw['finish_reason']
            if r['format_error'] or (r['condition'] == 'fact_oracle' and not r['correct']):
                exceptions.append(dict(arm=arm, record=r, probe=p))
        metrics[arm] = {c: metric([r for r in records if r['condition'] == c])
                        for c in manifest['conditions']}
        report = read(OUT / arm / 'report.json')
        assert metrics[arm] == report['metrics']
        assert read(OUT / arm / 'state.json')['status'] == 'complete'
        per_rule = {}
        for qid in knowledge:
            per_rule[qid] = {c: metric([r for r in records
                                       if r['qid'] == qid and r['condition'] == c])
                             for c in manifest['conditions']}
        assert per_rule == report['per_rule']
        histograms[arm] = dict(sorted(Counter(
            p['fact_closed']['correct'] for p in per_rule.values()).items()))
        families[arm] = {rel: metric([r for r in records
                                    if r['condition'] == 'fact_closed' and r['relation'] == rel])
                         for rel in manifest['relations']}

    for pid in by_id:
        a = predictions['base'][pid]['raw_response']
        b = predictions['text'][pid]['raw_response']
        assert a['prompt_token_ids'] == b['prompt_token_ids'], pid
        assert a['images_size'] == b['images_size'], pid
        assert a['usage']['prompt_tokens'] == b['usage']['prompt_tokens'], pid

    # This content-based sensitivity check is post hoc and secondary.
    shared_ids = {f['id'] for f in facts if any(term in f['question'] for term in (
        'inspection record code', 'must all conditions hold', 'includes exactly either threshold'))}
    assert len(shared_ids) == 6
    sensitivity = {arm: metric([r for pid, r in predictions[arm].items()
                               if pid.startswith('fact:') and pid not in shared_ids])
                   for arm in predictions}
    mastered = []
    for qid in knowledge:
        rr = [r for r in predictions['text'].values() if r['qid'] == qid]
        if metric([r for r in rr if r['condition'] == 'fact_closed'])['correct'] == 3:
            mastered.append(dict(qid=qid, metrics={c: metric([r for r in rr if r['condition'] == c])
                                                  for c in manifest['conditions']}))
    result = dict(
        integrity=dict(records_per_arm=240, matched_input_token_sequences=240,
                       image_files_hash_verified=len(manifest['images']),
                       frozen_source_hashes_verified=len(manifest['sources']),
                       frozen_input_hashes_verified=len(manifest['files']),
                       all_derived_metrics_reproduced=True),
        metrics=metrics, per_rule_fact_correct_histogram=histograms,
        family_factual_metrics=families, all_three_facts_correct_trained=mastered,
        sensitivity_excluding_six_shared_conventions=sensitivity,
        excluded_shared_fact_ids=sorted(shared_ids), exceptions=exceptions)
    (OUT / 'audit_results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')

    lines = ['# 20条规则诊断复核', '',
             '复核对象是原始1000条规则TEXT全量SFT第10轮模型与未训练基座；本次未训练20条小模型。', '',
             '## 输入与计分', '',
             '- 两组各240条输出完整且ID唯一；相同问题的实际输入token序列逐条一致。',
             '- 从原始输出重新严格解析A/B/C/D，准确率、格式错误、截断和逐规则统计均复现。',
             '- 冻结的脚本、输入文件、源数据及本次40张原图的SHA256均与manifest一致。此处未重新遍历全部3000张图。',
             '- 训练后文字应用有1条格式错误且输出达到32 token上限，已按错误计入40题分母；60道闭卷事实题均无格式错误。', '',
             '## 逐规则观察', '',
             '- 训练后：7条规则事实题0/3，11条1/3，1条2/3，1条3/3。',
             '- 唯一3/3的是Kamnet；它的闭卷文字应用0/2、原图应用1/2，给材料文字应用2/2。',
             '- 这只是个案。Kamnet的类别选项含有明显不同的类型、记录码又是共用约定，不能据3/3断言完整掌握。', '',
             '## 需要保留的限制', '',
             '1. 一些规则本身只有2或3个合法标签，为凑四选一加入了外部标签；部分数字选项也容易排除。因此不能机械地把25%当作无知识基准，应与相同题目的未训练基座比较。',
             '2. 60题包含6道记录码、条件连接方式或区间边界题。这些知识在多条规则间共用，或可从约定推测；其余54题是评测后按内容作的补充统计，不替代60题主结果。',
             '3. Kamnet的“required value”问题实际对应至少4个（value >= 4）中的阈值4，措辞应明确为最低数量；本轮保留原始输入和结果，不在评测完成后改题。',
             '4. 三道事实题只检查规则的部分信息；辨认不等于完整回忆或熟练应用。闭卷低分说明本协议下未展示出可靠知识提取，不能证明权重中完全没有相关知识。',
             '5. 给材料事实题接近满分，但文字应用只有65%，提示还需要检查应用能力及应用题的可答性；不能把这35%的错误全部归因于模型推理。', '',
             '## 补充统计', '',
             '| 排除6道共用约定题后（54题） | 正确率 |',
             '|---|---:|']
    for arm, label in [('base', '未训练基座'), ('text', '训练后')]:
        m = sensitivity[arm]
        lines.append(f'| {label} | {m["accuracy_percent"]:.2f}%（{m["correct"]}/{m["n"]}） |')
    lines += ['', '基座唯一给材料事实题错误：Vodmer中panel只兼容circle，但模型选了square和circle；材料和标准答案一致，原始回复为A。',
              '训练后唯一格式错误：询问Cotcos变换程序时，模型输出了“至少满足三条条件中的两条”的筛选规则。训练材料中Cotcos只出现于该变换程序；这提示规则提取或类型混淆，不能仅靠延长输出解释。', '',
              '结论：这批样本没有显示训练后闭卷事实辨认优于基座，不能把当前问题归结为“已经学会、只是不懂应用”。同时，材料给定后的应用仍有困难，两部分都需要继续检查。']
    (OUT / 'audit_zh.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: result[k] for k in ('integrity', 'per_rule_fact_correct_histogram',
                                              'sensitivity_excluding_six_shared_conventions')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    audit()
