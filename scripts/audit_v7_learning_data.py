#!/usr/bin/env python3
"""Read-only audit of frozen training files, encoding, labels and shortcuts."""
import copy
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import pilot_v7_learning_first as trial
import run_v7_evaluation as ev
from build_v7_experiment_data import name
from run_v7_full_experiment import ROOT, MODELS, environment, rows, read, save, sha

OUT = ROOT / 'runs/v7_learning_data_audit_20261019'


def norm(s):
    return re.sub(r'\s+', ' ', s).strip()


def run():
    OUT.mkdir(exist_ok=True); trial.validate()
    os.environ.update(environment('0', trial.DATA))
    from swift.model import get_model_processor
    from swift.template import get_template
    _, processor = get_model_processor(str(MODELS['qwen3']), model_type='qwen3_vl', load_model=False)
    template = get_template(processor, template_type='qwen3_vl', max_length=8192, enable_thinking=False)
    template.set_mode('train'); tok = processor.tokenizer
    targets = rows(trial.OUT / 'target_knowledge.jsonl'); replays = rows(trial.OUT / 'replay_knowledge.jsonl')
    lookup = {k['id']: k for k in targets + replays}
    rules = {q: json.loads(k['fact']['quote']) for q, k in lookup.items()}
    blocks = {b['id']: b for b in rows(trial.OUT / 'blocks.jsonl')}
    details, stats, issues = [], {}, []
    sources = {}
    for arm in trial.ARMS:
        file = trial.OUT / (arm + '.jsonl'); sources[str(file)] = sha(file)
        training = rows(file); inputs = defaultdict(list); channels = defaultdict(list)
        label_errors = []; material_errors = []
        letters_by_rule = defaultdict(Counter); by_rule_task = defaultdict(Counter)
        unique_observations = defaultdict(set); exact_material = 0
        for row in training:
            q = row['qid']; k = lookup[q]; rule = rules[q]
            assert row['messages'][-1]['role'] == 'assistant'
            target = row['messages'][-1]['content']; user = row['messages'][-2]['content']
            inputs[ev.digest(row['messages'][:-1])].append(row)
            encoded = template.encode({'messages': copy.deepcopy(row['messages'])})
            ids, labels = encoded['input_ids'], encoded['labels']
            target_ids = [v for v in labels if v != -100]
            decoded = tok.decode(target_ids)
            full = tok.decode(ids)
            assert len(ids) == len(labels) < 8192
            assert target in decoded and all(m['content'] in full for m in row['messages'])
            assert all(v == -100 or v == i for i, v in zip(ids, labels))
            start = next(i for i, v in enumerate(labels) if v != -100)
            assert all(v == -100 for v in labels[:start]) and start > 0
            details.append({'arm': arm, 'id': row['id'], 'channel': row['channel'], 'input_tokens': len(ids),
                            'answer_tokens_with_suffix': len(target_ids), 'decoded_supervision': decoded,
                            'supervised_ids_sha256': ev.digest(target_ids), 'first_answer_token': tok.decode(target_ids[:1]),
                            'last_tokens': [tok.decode([v]) for v in target_ids[-3:]],
                            'all_input_and_answer_text_preserved': True})
            channels[row['channel']].append(details[-1])
            if row['channel'] in ('target_fact', 'independent_fact_replay'):
                assert target == trial.mem.compact(k)
                assert name(k) in user
                if norm(target) in norm(k['material']['learning']['text']): exact_material += 1
                else: material_errors.append(row['id'])
            else:
                b = blocks[row['block']]; obs = b['observations']; unique_observations[q].add(b['observation_signature'])
                answer = trial.independent_assignment(rule, obs)
                assert answer == b['results'][q]
                if row['fmt'] == 'mcq':
                    opts = dict(re.findall(r'(?m)^([A-D])\. (.*)$', user))
                    expected_action = 'Accept report.' if b['submitted_report'] == answer else 'Replace report.'
                    matching = [c for c, text in opts.items() if text.split('Result: ', 1)[1].rstrip('.') == answer
                                and (row['task'] == 'direct' or text.startswith(expected_action))]
                    if matching != [target]: label_errors.append(row['id'])
                    assert len(opts) == len(set(opts.values())) == 4
                    if row['channel'] == 'target_mcq':
                        letters_by_rule[q][target] += 1; by_rule_task[(q, row['task'])][target] += 1
                    assert target_ids == tok.encode(target, add_special_tokens=False) + tok.encode('<|im_end|>\n', add_special_tokens=False)
                else:
                    assert row['fmt'] == 'semantic'
                    assert target.split('Result: ', 1)[1].rstrip('.') == answer
                    assert target.startswith('Rule: ' + trial.mem.compact(k))
                    trace = target.split('\nChecks: ', 1)[1].split('\n', 1)[0]
                    bits = re.findall(r'condition \d+=(Not met|Met)', trace)
                    # Recompute with independent closed-form condition comparisons.
                    want = []
                    for o in obs:
                        for c in rule['conditions']:
                            v, rhs, op = o[c['field']], c['value'], c['operator']
                            flag = v == rhs if op == 'eq' else v >= rhs if op == 'ge' else v > rhs if op == 'gt' else v <= rhs if op == 'le' else v < rhs
                            want.append('Met' if flag else 'Not met')
                    assert bits == want
                    if row['task'] == 'diagnosis':
                        assert ('\nAccept report.' in target) == (b['submitted_report'] == answer)
            if row['channel'].startswith('target_') and row['channel'] != 'target_fact':
                assert trial.mem.compact(k) not in user, 'Target rule material disclosed in application input'
        conflicts = [v for v in inputs.values() if len({r['messages'][-1]['content'] for r in v}) > 1]
        assert not conflicts and not label_errors and not material_errors
        channel_stats = {}
        for channel, rr in channels.items():
            ns = [r['answer_tokens_with_suffix'] for r in rr]
            channel_stats[channel] = {'rows': len(rr), 'input_tokens_min': min(r['input_tokens'] for r in rr),
                                      'input_tokens_max': max(r['input_tokens'] for r in rr),
                                      'supervised_tokens_min': min(ns), 'supervised_tokens_max': max(ns),
                                      'supervised_tokens_total': sum(ns)}
        stats[arm] = {'rows': len(training), 'unique_input_messages': len(inputs),
                      'duplicate_rows_beyond_first': len(training) - len(inputs),
                      'conflicting_identical_inputs': len(conflicts), 'label_errors': label_errors,
                      'fact_targets_are_exact_source_prefixes': True, 'channels': channel_stats,
                      'target_letters_global': dict(sum(letters_by_rule.values(), Counter())),
                      'target_letters_per_rule': {q: dict(cc) for q, cc in letters_by_rule.items()},
                      'rule_only_majority_training_accuracy': sum(max(cc.values()) for cc in letters_by_rule.values()) / sum(sum(cc.values()) for cc in letters_by_rule.values()) if letters_by_rule else None,
                      'rule_and_task_majority_training_accuracy': sum(max(cc.values()) for cc in by_rule_task.values()) / sum(sum(cc.values()) for cc in by_rule_task.values()) if by_rule_task else None,
                      'unique_target_training_observations_per_rule': {q: len(v) for q, v in unique_observations.items() if q in {k['id'] for k in targets}}}
        print('AUDITED', arm, len(training), flush=True)
    ev.save_rows(OUT / 'encoded_rows.jsonl', details)
    # A shared pair of official test modalities must remain byte-for-byte unchanged.
    probes = rows(trial.OUT / 'probes.jsonl')
    original = [p for p in probes if p['condition'] == 'original']
    for mod, direction in [('T', 'T2T'), ('V', 'T2V')]:
        indexed = {p['id']: p for p in rows(trial.SOURCE / f'eval_{direction}.jsonl')}
        for p in original:
            if p['input_mod'] == mod:
                src = indexed[p['source_id']]
                assert p['messages'] == src['messages'] and p.get('images', []) == src.get('images', [])
    # Revalidate whether the old full-run supervision audit applies to the ep10 run.
    olddir = ROOT / 'runs/v7_access_diagnostics_20261015/supervision_audit'
    old = read(olddir / 'supervision_audit.json')
    historical_matches = {}
    for mod in ('text', 'vision'):
        a = ROOT / 'runs/v7_full_experiment/data' / f'sft_{mod}.jsonl'
        b = trial.SOURCE / f'sft_{mod}.jsonl'
        assert old['sources'][str(a)] == sha(a)
        historical_matches[mod] = sha(a) == sha(b)
    assert all(historical_matches.values())
    vg = [s for s in old['by_relation'] if s['arm'] == 'sft_vision']
    tg = [s for s in old['by_relation'] if s['arm'] == 'sft_text']
    historical = {'ep10_training_files_byte_identical_to_prior_audit': historical_matches,
                  'vision_rules_in_parameter_distinct_target_collisions': sum(s['rules_in_collisions'] for s in vg),
                  'text_rules_in_parameter_distinct_target_collisions': sum(s['rules_in_collisions'] for s in tg),
                  'three_way_numeric_coverage': old['three_way_numeric_coverage'],
                  'limit': 'Identical normalized targets do not prove image inputs lack facts or that all semantic supervision is absent.'}
    sources[str(olddir / 'supervision_audit.json')] = sha(olddir / 'supervision_audit.json')
    save(OUT / 'audit.json', {'status': 'pass', 'script_sha256': sha(Path(__file__)), 'sources': sources,
         'encoded_training_rows': len(details), 'all_supervision_and_inputs_preserved': True, 'training': stats,
         'original_main_eval_input_unchanged': True, 'historical_full1000': historical,
         'meaning': 'Structural/label audit passed. Several objective-design weaknesses remain; passing this audit does not establish the data are sufficient for learning.'})
    lines = ['# 学习数据审计：正确性与训练目标分开判断', '',
             '检查实际训练文件，未重训、未修改原题或原图。所有3072行通过模板编码；问题和监督答案均保留，长度远小于8192，未发现相同输入对应冲突答案。', '',
             '本轮应用标签、语义结果、逐条件Met/Not met及报告Accept/Replace均重新计算通过；解释目标为原学习材料的准确前缀。未发现能直接解释选项塌缩的错标或截断。', '',
             '## 已确认的数据设计问题', '',
             '1. 1024行不是1024个独立学习样本。A组只有40个不同输入；目标知识主要是8条材料的4种问法重复。B/C每规则只有20个不同应用场景，direct/diagnosis与语义/字母形式复用这些场景。',
             '2. 全局字母均衡掩盖规则内位置偏好。B/C每规则40道字母训练中，Lavrit的C占24/40；Dostan的C仅2/40。仅按规则名记多数选项即可在训练上达到40%，无需执行规则。这是数据捷径，但不能单独解释全答A。',
             '3. 语义辅助答案先给完整正确规则，再监督条件检查与结果。标准teacher forcing在训练后半段时已提供正确前缀，所以较低语义loss不能证明模型能在字母题中自主调用规则。机制可直接由监督顺序确定；尚无隔离干预证明它是主因。',
             '4. 三个训练目标用不同提示分别练复述、长答案应用和短字母。这样不会产生逐字相同输入的冲突标签，但跨目标能力衔接依然是额外泛化要求；长答案会算不保证短答案会选。',
             '5. 字母题仍监督3个token（字母、结束符、换行）。样本等权没有剔除公共后缀：当后两者正确，字母四选一不确定时，该行均值CE约ln(4)/3=0.462。末段字母行loss接近此量级，实际原答也显示塌缩；这不是“应用已经拟合很好”。', '',
             '## 历史完整1000条另有明确监督不足', '',
             f'已确认10epoch实验与旧审计使用逐字相同的训练文件。视觉监督中有{historical["vision_rules_in_parameter_distinct_target_collisions"]}条，在替换虚构名称后，不同参数的规则具有相同目标文字；文字监督对应为0条。',
             '117条三分类的文字目标都明写两阈值，视觉目标均未同时明写两阈值；很多目标只说“按学习图读取边界”。这说明训练目标没有直接要求输出并区分具体参数，不证明图像本身没有参数。',
             '这些是旧全量视觉训练的问题；不能直接用于解释本轮TEXT-only的失败，本轮目标已经包含具体事实。', '',
             '## 下一步应如何修数据', '',
             '先形成数据修复候选，不覆盖原数据：减少同句机械重复；在每条规则和每种题型内平衡选项位置；逐一覆盖条件只满足其一、均满足、均不满足及区间边界；用新的独立场景做来源学习检查。',
             '应用训练中增加直接判断局部结果的独立监督，避免所有结果都只在正确规则前缀之后训练；保留正式字母题训练，具体参数可作独立辅助目标。是否改善必须做同预算对照，不能承诺靠这些修改必然解决。',
             '这些操作属于学习端，正式测试保持原题、原图和直接答题协议；不把T2T高于T2V作为挑数据条件。', '',
             '本审计没有逐条人工核验1000条原学习图的语义，也没有证明1000条文字事实与形式规则完全一致；不能把本轮3072行审计通过扩大成整个数据集无问题。']
    (OUT / 'report_zh.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'training': stats, 'historical': historical}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__': run()
