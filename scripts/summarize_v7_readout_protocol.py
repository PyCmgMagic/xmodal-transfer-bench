#!/usr/bin/env python3
"""Audit prompt interventions without conflating protocols or repeated samples."""
import csv
import json
from pathlib import Path

import diagnose_v7_system_alignment as alignment
import explore_v7_readout_protocol as trial
import explore_v7_visual_replay as replay
import run_v7_evaluation as ev
from run_v7_full_experiment import ROOT, rows, read, save, sha


def score(m):
    return f"{m['correct']}/{m['n']}（{m['accuracy']:.2%}）"


def main():
    out = trial.OUT
    trial.validate(out)
    alignment.validate()
    review = read(out / 'semantic_review_summary.json')
    assert len(rows(out / 'semantic_reviews.jsonl')) == 120
    reports, aligned, inventory, quality, comparisons = {}, {}, [], {}, []
    paired = []
    for arm in trial.ARMS:
        folder = out / arm
        records = rows(folder / 'predictions.jsonl')
        assert read(folder / 'eval_state.json') == {'status': 'complete', 'saved': 373}
        assert len(records) == len({r['id'] for r in records}) == 373
        probes = rows(out / 'extraction_probes.jsonl') + rows(out / 'recall_probes.jsonl') + trial.application_probes(out, arm, records)
        lookup = {p['id']: p for p in probes}
        assert len(lookup) == 373
        saved_apply = rows(folder / 'application_probes.jsonl')
        assert saved_apply == probes[21:]
        for r in records:
            p = lookup[r['id']]
            assert r['input_sha256'] == ev.digest({'messages': p['messages'], 'images': p.get('images', [])})
            assert r['response'] == (r['raw_response']['choices'][0]['message']['content'] or '')
        reports[arm] = read(folder / 'report.json')
        second = out / 'system_alignment' / arm
        aligned[arm] = read(second / 'report.json')
        assert reports[arm]['status'] == aligned[arm]['status'] == 'complete'
        raw = rows(second / 'predictions.jsonl')
        assert read(second / 'eval_state.json') == {'status': 'complete', 'saved': 168}
        assert len(raw) == len({r['id'] for r in raw}) == 168
        pp = {p['id']: p for p in rows(out / 'system_alignment/probes.jsonl')}
        for r in raw:
            p = pp[r['id']]
            assert r['input_sha256'] == ev.digest({'messages': p['messages'], 'images': p.get('images', [])})
            assert r['response'] == (r['raw_response']['choices'][0]['message']['content'] or '')
        # Paired direct inputs differ only in system content. Original user and image are identical.
        for mod in ('T', 'V'):
            original = {p['id']: p for p in rows(trial.SOURCE / 'probes.jsonl') if p['condition'] == 'named_' + mod and p['shift'] == 0}
            rr = {r['id']: r for r in raw}
            both_right = strict_only = aligned_only = both_wrong = changed = 0
            for sid, p in original.items():
                strict, changed_system = pp['strict:' + sid], pp['aligned:' + sid]
                assert strict['messages'][-1] == changed_system['messages'][-1] == p['messages'][-1]
                assert strict.get('images', []) == changed_system.get('images', []) == p.get('images', [])
                a, b = rr[strict['id']], rr[changed_system['id']]
                changed += a['predicted'] != b['predicted']
                if a['correct'] and b['correct']:
                    both_right += 1
                elif a['correct']:
                    strict_only += 1
                elif b['correct']:
                    aligned_only += 1
                else:
                    both_wrong += 1
            paired.append({'arm': arm, 'modality': mod, 'n': 40, 'both_correct': both_right,
                'strict_only_correct': strict_only, 'aligned_only_correct': aligned_only,
                'both_wrong': both_wrong, 'answer_changed': changed})
        for kind, directory, rr in [('readout_recall', folder, records), ('system_alignment', second, raw)]:
            cp = read(directory / 'eval_manifest.json')['adapter']
            assert ev.checkpoint(Path(cp['path'])) == cp
            path = directory / 'predictions.jsonl'
            inventory.append({'arm': arm, 'kind': kind, 'path': str(path), 'sha256': sha(path), 'rows': len(rr)})
            quality[arm + '/' + kind] = {'n': len(rr), 'missing': 0, 'empty': sum(not r['response'].strip() for r in rr),
                'mcq_format_errors': sum(r['format_error'] for r in rr),
                'truncated_total': sum(r['finish_reason'] == 'length' for r in rr),
                'truncated_mcq': sum(r['finish_reason'] == 'length' and r['fmt'] == 'mcq' for r in rr),
                'truncated_recall': sum(r['finish_reason'] == 'length' and r['fmt'] == 'card' for r in rr)}
        for protocol, m in reports[arm]['metrics'].items():
            comparisons.append({'arm': arm, 'protocol': protocol, 'n': m['n'], 'correct': m['correct'],
                'accuracy': m['accuracy'], 'derived_only': m.get('derived_summary', False)})
        for protocol, m in aligned[arm]['metrics'].items():
            comparisons.append({'arm': arm, 'protocol': protocol, 'n': m['n'], 'correct': m['correct'],
                'accuracy': m['accuracy'], 'derived_only': False})
    assert sum(r['rows'] for r in inventory) == 2705
    for r in rows(out / 'semantic_reviews.jsonl'):
        assert sha(Path(r['source_file'])) == r['source_sha256']
        assert ev.digest(r['response']) == r['response_sha256']
    # Row decomposition is audited separately from model totals and arithmetic helpers.
    rowtruth = {r['source_id']: r for r in rows(out / 'row_truth_review.jsonl')}
    extracts = {}
    for arm in trial.ARMS:
        rr = reports[arm]['extraction_reviews']
        old = read(replay.OUT / 'observation_diagnostic' / arm / 'report.json')['extraction']
        numeric = [r for r in rr if r['mode'] == 'model' and r['group'] == 1]
        sums_match, rows_correct = 0, 0
        for r in numeric:
            parsed = replay.observation_json(r['response']) or {}
            for c in 'ABCD':
                fields = parsed.get(c, {})
                rc = fields.get('row_counts')
                good = isinstance(rc, list) and rc and all(type(v) is int and v >= 0 for v in rc)
                sums_match += bool(good and sum(rc) == fields.get('value'))
                rows_correct += rc == rowtruth[r['source_id']]['row_counts'][c]
        extracts[arm] = {'old_shape_scene_correct': sum(r['scene_correct'] for r in old if r['group'] == 0),
            'new_shape_scene_correct': sum(r['all_fields_correct'] for r in rr if r['group'] == 0),
            'old_numeric_candidates_correct': sum(c['correct'] for r in old if r['group'] == 1 for c in r['field_checks']),
            'new_numeric_candidates_correct': sum(c['correct'] for r in numeric for c in r['field_checks']),
            'numeric_candidate_n': 8, 'exact_row_lists_correct': rows_correct,
            'model_total_matches_own_row_sum': sums_match, 'row_sum_candidates_correct': sum(c['correct'] for r in rr if r['mode'] == 'row_sum' for c in r['field_checks'])}
    hashes = []
    for r in read(replay.OUT / 'source_hash_validation.json')['checked_files']:
        assert sha(Path(r['path'])) == r['sha256'], r['path']
        hashes.append(r)
    save(out / 'source_hash_validation.json', {'status': 'pass', 'checked_files': hashes,
        'original_pngs_unchanged': 3000, 'checked_at': ev.now()})
    summary = {'status': 'complete', 'new_generations': 2705, 'new_training': 0,
        'readout': reports, 'system_alignment': aligned, 'semantic_review': review,
        'extraction_summary': extracts, 'paired_system_outcomes': paired, 'output_quality': quality,
        'limits': 'Eight development rules, five original scenes, one seed, shift0 only. No independent confirmation. Oracle controls and arithmetic helpers kept separate.'}
    save(out / 'summary.json', summary)
    ev.save_rows(out / 'response_inventory.jsonl', inventory)
    save(out / 'paired_system_outcomes.json', paired)
    with (out / 'comparison.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)

    lines = ['# 原图读出方式与系统指令对齐诊断（完成）', '',
        '本轮没有训练模型，使用已有五个Qwen3-VL-4B适配器，检查三件事：明确内部填充色能否消除颜色约定混淆；逐行计数能否改善原图观察读取；无图回忆和直接答题是否依赖训练时系统指令。所有原图、规则和历史实验均保留。', '',
        '## 实际配置与计数', '',
        '共同warmup、TEXT、第一轮VISION、纯图条件续训vision_control、视觉来源文字回放text_replay各生成541次新回答：5次改进观察转录、16次无系统回忆、192次观察干预答题、160次新回忆后答题，加上8次训练系统回忆、160次系统对齐短答案，共2705次。text_replay含文字桥接监督，不能算纯视觉学习。', '',
        '仍然只有8条开发规则、5张原测试图、10个题块、每测试输入模态40个语义问题。本轮只测选项shift0，没有四位置平均；不同协议复用题目、规则和观察，不能把请求数当独立样本。没有原生CoT。逐行数量与规则复述是独立中间输出，原答全部保存。', '',
        '主读出计划在生成前保存；系统指令对照是在观察到user-only回忆失败后追加，追加manifest先于新对照生成。这是开发诊断，非独立预注册。', '',
        '## 原图观察读取', '',
        '形状题明确只取内部填充色、忽略边框，提供全局颜色词表及深蓝黑对应black的约定，没有提供某张图或某个候选的属性答案。数量题要求从上到下逐行列出row_counts，再给出value。两张数值图的正确逐行数量由人工审阅原PNG核对，仅用于评分，不进入模型请求。', '',
        '| 适配器 | 形状颜色场景：旧→新 | 数值候选：旧→新 | 行数量列表正确 | 自报总数等于自身逐行和 |',
        '|---|---:|---:|---:|---:|']
    for arm in trial.ARMS:
        m = extracts[arm]
        lines.append(f'| {arm} | {m["old_shape_scene_correct"]}/3 → {m["new_shape_scene_correct"]}/3 | {m["old_numeric_candidates_correct"]}/8 → {m["new_numeric_candidates_correct"]}/8 | {m["exact_row_lists_correct"]}/8 | {m["model_total_matches_own_row_sum"]}/8 |')
    lines += ['',
        '旧转录来自上一轮保存的同模型输出，本轮只新生成改进转录；旧、新转录进入下游时统一成同样的JSON字段格式，并各自重新答题，使比较尽量只改变观察值。与上一轮直接使用转录原文的分数不能混为同一协议。', '',
        '逐行分解可能把看错的“每行4个”重复用于多个候选，不保证改善。脚本求和只对模型自己生成的row_counts做算术，属于外部计算帮助，单独列出；不会用真值修正任何一个行数量。形状场景在混合汇总中复用模型预测，不额外调用。', '',
        '## 同题应用：观察记录干预', '',
        '自身规则采用上一轮同模型的无图回忆，未经纠正；正确规则为外部能力控制。下表每格40题，hybrid汇总对24道形状题复用新转录结果，对16道数值题使用逐行求和后的新答题结果。', '',
        '| 适配器 | 自身规则＋旧观察 | 自身规则＋新观察 | 自身规则＋求和辅助 | 正确规则＋旧观察 | 正确规则＋新观察 |',
        '|---|---:|---:|---:|---:|---:|']
    for arm in trial.ARMS:
        m = reports[arm]['metrics']
        lines.append('| ' + arm + ' | ' + ' | '.join(score(m[c]) for c in ('obs_self_old', 'obs_self_improved', 'obs_self_hybrid_sum', 'obs_oracle_old', 'obs_oracle_improved')) + ' |')
    lines += ['', '给正确规则时，分开看颜色约定与逐行计数的变化：', '',
        '| 适配器 | 形状颜色：旧观察 | 形状颜色：新观察 | 数值：旧观察 | 数值：新观察 |', '|---|---:|---:|---:|---:|']
    for arm in trial.ARMS:
        m = reports[arm]['metrics']
        lines.append('| ' + arm + ' | ' + ' | '.join(score(m[c]['by_group'][g])
            for g in ('0', '1') for c in ('obs_oracle_old', 'obs_oracle_improved')) + ' |')
    lines += ['',
        '该控制不把正确规则成绩当迁移成绩；观察记录是文字输入，属于分步骤的图片诊断，不能覆盖原图片直接答题成绩。所有失败规则仍纳入，不按成功回忆事后筛选。', '',
        '## 无图回忆：提问及系统指令', '',
        '三个新条件均不提供学习图、候选规则、测试观察或规则定义。plain只有用户问题“Explain the complete rule named NAME.”；page_memory仍无系统指令，提醒从此前学习参考中回忆完整事实；aligned_plain与plain用户问题逐字相同，仅加回训练时原系统指令。', '',
        '| 适配器 | plain，无系统 | page_memory，无系统 | plain＋原训练系统 |', '|---|---:|---:|---:|']
    for arm in trial.ARMS:
        lines.append('| ' + arm + ' | ' + ' | '.join(f'{review[arm][c]["correct"]}/8' for c in ('plain', 'page_memory', 'aligned_plain')) + ' |')
    lines += ['',
        '120份回忆逐条对照原规则的条件、组合方式、区间端点及回答一致性。长回答如果一开始就编造无关游戏/术语，已足以判核心事实失败；截断单独统计，不延长或重试。全部原答和明确人工判定都保留，审核脚本不是通用语义评分器。', '',
        '## 只改系统的直接选择题对照', '',
        '两条件在本轮重新生成，用户题面、图片、选项、最大16token和temperature=0相同。strict使用原选择题系统；aligned在原训练系统后附上完全相同的只答A/B/C/D指令。仍不提供规则事实，不先回忆。', '',
        '| 适配器 | 文字strict | 文字aligned | 图片strict | 图片aligned |', '|---|---:|---:|---:|---:|']
    for arm in trial.ARMS:
        m = aligned[arm]['metrics']
        lines.append('| ' + arm + ' | ' + ' | '.join(score(m[c]) for c in ('strict_T', 'aligned_T', 'strict_V', 'aligned_V')) + ' |')
    lines += ['',
        '原训练系统只含通用任务约定及答事实指令，不含本轮8条目标规则的阈值、形状、颜色或ANY/ALL定义。它包含其他三分类任务的通用边界约定，本轮OR/区间题不依赖这些三分类约定。提示差异可能影响事实调用，但本控制不能解释其内部机制。即使回忆恢复，也必须单独检查选择题成绩。', '',
        '同题转对/转错及输出变化数见paired_system_outcomes.json；请求高度相关，没有计算朴素请求级显著性。', '',
        '## 新提问的自身回忆后答题', '',
        '无图回忆原文直接加入同模态题面，不纠正编造内容。每方向40题；文字和图片始终单列。', '',
        '| 适配器 | plain→文字 | plain→图片 | page_memory→文字 | page_memory→图片 |', '|---|---:|---:|---:|---:|']
    for arm in trial.ARMS:
        m = reports[arm]['metrics']
        lines.append('| ' + arm + ' | ' + ' | '.join(score(m[c]) for c in ('recall_apply_plain_T', 'recall_apply_plain_V', 'recall_apply_page_memory_T', 'recall_apply_page_memory_V')) + ' |')
    lines += ['',
        '这些提问不是按每条规则挑最好结果；两个固定提问均应用于全部规则及全部题。不能从中挑出“最好提示”再当成独立确认的成绩。', '',
        '## 输出质量与文件校验', '',
        f'2705次新生成：缺失{sum(m["missing"] for m in quality.values())}，空输出{sum(m["empty"] for m in quality.values())}，选择题格式错误{sum(m["mcq_format_errors"] for m in quality.values())}，总截断{sum(m["truncated_total"] for m in quality.values())}，其中回忆截断{sum(m["truncated_recall"] for m in quality.values())}、选择题截断{sum(m["truncated_mcq"] for m in quality.values())}。正式分母保留所有请求，无效选择题计错。', '',
        '所有3000张原PNG、原数据来源、基座权重/模板和5个适配器哈希均核对不变；输入和原始回答对应，观察求和和自回忆引用可重建。源文件、脚本及产出哈希单独保存。本轮未修改数据集、未训练模型、未终止无关GPU进程。', '',
        f'- [结论与建议]({out / "interpretation_zh.md"})、[全部指标]({out / "summary.json"})、[CSV]({out / "comparison.csv"})。',
        f'- [全部回忆与人工判断]({out / "semantic_reviews.jsonl"})、[原图逐行计数真值审核]({out / "row_truth_review.jsonl"})。',
        f'- [主manifest]({out / "manifest.json"})、[系统指令对照manifest]({out / "system_alignment/manifest.json"})。', '',
        '本轮仍是相同8条规则和5图的开发诊断，未覆盖完整1000条、其他知识族或三模型；没有新增PT实验。任何后续正式提示修改都应统一应用于学习模态对照，并在独立规则上确认。', '']
    (out / 'report_zh.md').write_text('\n'.join(lines))
    save(out / 'completion_audit.json', {'status': 'complete', 'new_generations': 2705,
        'new_training': 0, 'semantic_recall_reviews': 120, 'original_pngs_unchanged': 3000,
        'unique_rules': 8, 'unique_original_test_scenes': 5, 'option_shift': 0,
        'dataset_is_final': False, 'checked_at': ev.now()})
    print(out / 'report_zh.md')


if __name__ == '__main__':
    main()
