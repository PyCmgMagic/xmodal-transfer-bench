#!/usr/bin/env python3
"""Collect existing experiment results; never launch training or inference."""
import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'runs'
DIRECTIONS = ('T2T', 'T2V', 'V2T', 'V2V')
SOURCES = {}
METRICS = []
CHECKS = []


def track(path):
    path = Path(path)
    SOURCES[str(path.relative_to(ROOT))] = {
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'bytes': path.stat().st_size,
    }
    return path


def load(path):
    return json.loads(track(path).read_text())


def rows(path):
    with track(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def csv_rows(path):
    with track(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def check(condition, name):
    if not condition:
        raise ValueError(name)
    CHECKS.append(name)


def add(run, dataset, model, protocol, direction, n, correct, invalid=None,
        status='complete', scope='full', source=None, objective='', modality='',
        arm='', seed='', test='', note=''):
    n = int(n) if n is not None else None
    correct = int(correct) if correct is not None else None
    invalid = int(invalid) if invalid is not None else None
    if correct is not None:
        check(n > 0 and 0 <= correct <= n, f'metric bounds: {run}/{model}/{protocol}/{direction}/{arm}/{test}')
    if invalid is not None:
        check(0 <= invalid <= n, f'invalid bounds: {run}/{model}/{protocol}/{direction}/{arm}/{test}')
    result = dict(run=run, dataset=dataset, model=model, objective=objective,
                  learning_modality=modality, scope=scope, protocol=protocol,
                  direction=direction, arm=arm, seed=seed, test=test, status=status,
                  total=n, correct=correct, invalid=invalid,
                  accuracy=correct / n if correct is not None else None,
                  source_file=str(Path(source).relative_to(ROOT)) if source else '', note=note)
    METRICS.append(result)
    return result


def verify_results(path, report, parser=None):
    records = rows(path)
    check(len(records) == report['total'], f'row count: {path.relative_to(ROOT)}')
    check(len({r['id'] for r in records}) == len(records), f'unique IDs: {path.relative_to(ROOT)}')
    correct = invalid = 0
    for r in records:
        answer = parser(r['response']) if parser else r['predicted']
        check(answer == r['predicted'] and r['correct'] == (answer == r['gold']),
              f'answer/gold: {path.relative_to(ROOT)}/{r["id"]}')
        correct += answer == r['gold']
        invalid += answer not in ('A', 'B', 'C', 'D')
    check(correct == report['correct'], f'correct count: {path.relative_to(ROOT)}')
    check(invalid == report.get('invalid', report['total'] - report.get('valid', report['total'])),
          f'invalid count: {path.relative_to(ROOT)}')
    return records


def write_csv(path, data):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def table(headers, body):
    return ['| ' + ' | '.join(headers) + ' |',
            '|' + '|'.join('---' for _ in headers) + '|'] + [
        '| ' + ' | '.join(str(v) for v in row) + ' |' for row in body]


def pct(value):
    if value is None:
        return '待完成'
    return f'{(Decimal(str(value)) * 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)}%'


def build(output):
    from run_lora_cot_evaluation import final_letter

    SOURCES.clear()
    METRICS.clear()
    CHECKS.clear()
    output.mkdir(parents=True, exist_ok=True)
    track(Path(__file__).resolve())
    track(ROOT / 'scripts/run_lora_cot_evaluation.py')
    generated = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
    def link(path, label=None):
        return f'[{label or Path(path).name}]({os.path.relpath(path, output)})'

    datasets = []
    for name, folder, expected_train, expected_eval in [
        ('rules', 'rules_repaired_single_image_v7', 1000, 2000),
        ('entities', 'entity1020/merged_1109', 1109, 2218),
    ]:
        root = ROOT / 'data' / folder
        counts = {}
        for direction in DIRECTIONS:
            train, evaluation = rows(root / direction / 'train.jsonl'), rows(root / direction / 'eval.jsonl')
            check(len(train) == expected_train and len(evaluation) == expected_eval,
                  f'dataset row counts: {name}/{direction}')
            counts[direction] = {'train': len(train), 'eval': len(evaluation)}
        manifest = load(root / ('build_manifest.json' if name == 'rules' else 'summary.json'))
        datasets.append(dict(dataset=name, root=str(root.relative_to(ROOT)), knowledge_units=expected_train,
                             questions_per_direction=expected_eval,
                             images=manifest.get('source_images', manifest.get('images')),
                             counts=counts,
                             certification=manifest.get('final_dataset_certified', manifest.get('semantic_and_temporal_certification'))))

    dense = RUNS / 'rules_full_sft_ep10_20261026'
    training = []
    learned = load(RUNS / 'qwen3_lora_learning_ep5_20261024/status.json')
    for r in learned['results']:
        dataset, objective, modality = r['name'].split('_')
        training.append(dict(run='qwen3_lora_learning_ep5_20261024', job=r['name'],
                             dataset=dataset, objective=objective, learning_modality=modality, method='lora',
                             status=r['status'], epoch=r['epoch'], steps=r['global_step'],
                             first_loss=r['first_logged_loss'], last_loss=r['last_logged_loss'],
                             ended_at=datetime.fromisoformat(r['ended_at']).astimezone(ZoneInfo('Asia/Shanghai')).isoformat(),
                             checkpoint=r['checkpoint'], checkpoint_exists=Path(r['checkpoint']).is_dir(),
                             source_file='runs/qwen3_lora_learning_ep5_20261024/status.json'))
    for modality in ('text', 'vision'):
        r = load(dense / 'jobs' / modality / 'state.json')
        training.append(dict(run=dense.name, job=modality, status=r['status'], epoch=r['epoch'],
                             dataset='rules', objective='sft', learning_modality=modality, method='full',
                             steps=r['global_step'], first_loss=r['first_loss'], last_loss=r['last_loss'],
                             ended_at=datetime.fromisoformat(r['ended_at']).astimezone(ZoneInfo('Asia/Shanghai')).isoformat(),
                             checkpoint=r['checkpoint'], checkpoint_exists=Path(r['checkpoint']).is_dir(),
                             source_file=str((dense / 'jobs' / modality / 'state.json').relative_to(ROOT))))
    check(all(r['checkpoint_exists'] for r in training), '10 current final checkpoints exist')
    check(all(r['status'] == 'complete' for r in training), '10 current training jobs complete')
    load(dense / 'manifest.json')
    fresh = []
    fresh_reports = []
    for d in DIRECTIONS:
        path = dense / 'evaluation_strict_b4' / d / 'report.json'
        r = load(path)
        verify_results(path.with_name('results.jsonl'), r, lambda s: s.strip() if s.strip() in ('A', 'B', 'C', 'D') else None)
        fresh_reports.append(r)
        fresh.append(add(dense.name, 'rules', 'qwen3', 'strict_short_b4', d, r['total'], r['correct'],
                         r['total'] - r['valid'], source=path, objective='sft', modality='text' if d.startswith('T') else 'vision'))
        for test, subset in [*r['by_question_type'].items(),
                             *((key, r[key]) for key in ('result_application', 'rule_report_audit', 'bias_watchlist'))]:
            add(dense.name, 'rules', 'qwen3', 'strict_short_b4', d, subset['total'], subset['correct'],
                subset['total'] - subset['valid'], source=path, objective='sft',
                modality='text' if d.startswith('T') else 'vision', test=test,
                note='Subset of the overall score; overlapping subgroup rows are not independent observations.')
    load(dense / 'evaluation_strict_b4/manifest.json')
    pilot_path = dense / 'evaluation_strict/T2T/report.json'
    pilot = load(pilot_path)
    add(dense.name, 'rules', 'qwen3', 'strict_short_b1', 'T2T', pilot['total'], pilot['correct'], 0,
        scope='batch_control', source=pilot_path, note='Same checkpoint/questions as b4; not independent observations.')

    old_path = RUNS / 'v7_full_experiment/summary/comparison.csv'
    old = csv_rows(old_path)
    canonical = [r for r in old if r['direction'] in ({'T2T', 'T2V'} if r['train_modality'] == 'text' else {'V2T', 'V2V'})]
    check(len(old) == 48 and len(canonical) == 24, 'historical LoRA 48 rows -> 24 canonical directions')
    for r in canonical:
        add('v7_full_experiment', 'rules', r['model'], 'strict_short', r['direction'], r['total'], r['correct'],
            int(r['total']) - int(r['valid']), source=old_path, objective=r['objective'], modality=r['train_modality'])
    old_training = load(RUNS / 'v7_full_experiment/training_status.json')
    check(all(r['status'] == 'complete' and r['epochs'] == 5 for r in old_training['jobs']),
          '12 historical LoRA training jobs completed 5 epochs')
    historical_dense_path = RUNS / 'v7_fullparam_sft_ep10_20261019/comparison.csv'
    historical_dense = csv_rows(historical_dense_path)
    for r in historical_dense:
        add('v7_fullparam_sft_ep10_20261019', 'rules', 'qwen3', 'strict_short', r['direction'], r['total'], r['correct'],
            int(r['missing']) + int(r['empty']) + int(r['format_errors']), source=historical_dense_path,
            objective='sft', modality='text' if r['direction'].startswith('T') else 'vision')
    glm_path = RUNS / 'v7_full_experiment/analysis/glm_output_format_20261014/report.json'
    glm = load(glm_path)
    add('v7_full_experiment', 'rules', 'glm46', 'supplemental_empty_think_parse', 'T2T', glm['counts']['total'],
        glm['counts']['correct'], 0, source=glm_path, objective='sft', modality='text')

    full = RUNS / 'qwen3_lora_cot_eval_20261025'
    full_manifest, full_status = load(full / 'manifest.json'), load(full / 'status.json')
    cot = []
    cot_progress = []
    for job in full_manifest['jobs']:
        for d in job['directions']:
            folder = full / 'jobs' / job['name'] / d
            progress = load(folder / 'progress.json') if (folder / 'progress.json').exists() else {'saved': 0}
            raw_path = folder / 'raw.jsonl'
            raw_count = 0
            if raw_path.exists():
                with track(raw_path).open() as handle:
                    raw_ids = [json.loads(line)['id'] for line in handle if line.strip()]
                raw_count = len(raw_ids)
                check(len(set(raw_ids)) == raw_count, f'CoT raw unique IDs: {job["name"]}/{d}')
            check(raw_count == progress['saved'], f'CoT raw count matches progress: {job["name"]}/{d}')
            path = folder / 'report.json'
            expected = 2000 if job['dataset'] == 'rules' else 2218
            complete = path.exists()
            cot_progress.append(dict(job=job['name'], dataset=job['dataset'], direction=d,
                                     saved=progress['saved'], expected=expected, complete=complete))
            if complete:
                r = load(path)
                verify_results(folder / 'results.jsonl', r, final_letter)
                check(math.isclose(r['accuracy'], r['correct'] / r['total']), f'CoT report accuracy: {job["name"]}/{d}')
                cot.append({'job': job['name'], **r})
            metric_direction = ('text_test' if d == 'T2T' else 'vision_test') if job['objective'] == 'base' else d
            add(full.name, job['dataset'], 'qwen3', 'cot_strict_final', metric_direction, expected,
                r['correct'] if complete else None, r['invalid'] if complete else None,
                status='complete' if complete else 'incomplete', source=path if complete else full / 'manifest.json',
                objective=job['objective'], modality=job['learning_modality'], arm=job['name'],
                note='' if complete else f'{progress["saved"]}/{expected} saved; no full-cell accuracy. Source label: {d}.')
    check(sum(r['saved'] for r in cot_progress) == full_status['predictions_saved'], 'CoT saved count matches pause status')
    stale_summary = load(full / 'summary.json')

    repair_root = RUNS / 'answer_output_repair_20261026'
    repair = load(repair_root / 'summary.json')
    repair_status = load(repair_root / 'status.json')
    for r in repair['cells']:
        path = repair_root / 'results' / r['experiment'] / r['job'] / r['direction'] / r['arm'] / 'results.jsonl'
        data = rows(path)
        check(len(data) == r['total'] and len({x['id'] for x in data}) == len(data), f'repair rows: {r["job"]}/{r["direction"]}/{r["arm"]}')
        for pred, score in [('original_predicted', 'original_correct'), ('parsed_predicted', 'parsed_correct'), ('final_predicted', 'combined_correct')]:
            correct = sum(x[pred] == x['gold'] for x in data)
            check(all(x[score] == (x[pred] == x['gold']) for x in data) and correct == r[score],
                  f'repair answer/gold: {r["job"]}/{r["direction"]}/{r["arm"]}/{pred}')
        for protocol, key in [('tolerant_parse', 'parsed'), ('constrained_readout', 'combined')]:
            complete = key == 'parsed' or r['readout_pending'] == 0
            add(repair_root.name, r['dataset'], 'qwen3', protocol, r['direction'], r['total'],
                r[key + '_correct'] if complete else None, r[key + '_invalid'] if complete else None,
                status='complete' if complete else 'incomplete', scope='full' if r['experiment'] == 'full_cot' else '90_rules',
                source=repair_root / 'summary.json', arm=r['job'] + '/' + r['arm'],
                note='Reuses original generation.' if key == 'parsed' else 'Additional inference; not the original closed-book protocol.')

    recall_root = RUNS / 'qwen3_lora_recall_apply_v2_20261026'
    recall = load(recall_root / 'summary.json')
    recall_status = load(recall_root / 'status.json')
    for r in recall['reports']:
        for arm in ('direct_boxed', 'self_recall_boxed'):
            a = r[arm]
            add(recall_root.name, 'rules', 'qwen3', arm, r['direction'], r['total'], a['correct'], a['invalid'],
                scope='90_rules', source=recall_root / 'summary.json', objective=r['objective'], modality=r['learning_modality'], arm=r['job'])

    processes = {}
    for run, scope in [('v7_process_confirmation_20261020', '8_rules'), ('v7_process_scale64_20261021', '64_rules')]:
        path = RUNS / run / 'summary.json'
        processes[run] = load(path)
        for arm, tests in processes[run].items():
            for test, r in tests.items():
                kind, protocol, modality = test.split(':')
                direction = ('text_test' if modality == 'T' else 'vision_test') if arm == 'base' else ('T2T' if modality == 'T' else 'T2V')
                add(run, 'rules', 'qwen3', protocol, direction, r['n'], r['correct'], r['format_errors'],
                    scope=scope, source=path, modality='text' if arm != 'base' else 'none',
                    arm=arm.split(':')[-1], seed=arm.split(':')[0] if ':' in arm else '', test=kind)
    scale_seed = load(RUNS / 'v7_process_scale64_20261021/seed_summary.json')
    cot100_path = RUNS / 'v7_cot100_20261021/vllm/summary.csv'
    cot100 = csv_rows(cot100_path)
    for r in cot100:
        direction = ('text_test' if r['modality'] == 'T' else 'vision_test') if r['model'] == 'base' else ('T2T' if r['modality'] == 'T' else 'T2V')
        add('v7_cot100_20261021/vllm', 'rules', 'qwen3', r['protocol'], direction,
            r['n'], r['correct'], r['invalid'], source=cot100_path, scope='50_rules', arm=r['model'])
    entity_base_path = RUNS / 'entity1020_merged1109_qwen3_base_20261020/reports/statistics.json'
    entity_base = load(entity_base_path)
    entity_baselines = {}
    for modality, r in entity_base['results'].items():
        overall = r['groups']['all']['overall']
        entity_baselines[modality] = overall
        add('entity1020_merged1109_qwen3_base_20261020', 'entities', 'qwen3', 'short_4_option_rotations',
            'text_test' if modality == 'text' else 'vision_test', overall['total'], overall['correct'], overall['invalid'],
            source=entity_base_path, modality='none', note='2218 questions x 4 rotations; not 8872 independent questions; not a CoT control.')
    diagnostic_path = RUNS / 'v7_process_image_diagnostic_20261021/summary.json'
    diagnostic = load(diagnostic_path)
    for condition, r in diagnostic['metrics'].items():
        add('v7_process_image_diagnostic_20261021', 'rules', 'qwen3', condition, 'text_test' if condition.endswith(':T') else 'vision_test',
            r['n'], r['correct'], r['invalid'], scope='8_rules_diagnostic', source=diagnostic_path,
            note='Post-hoc control/exact-match diagnostic; not a formal transfer score.')

    descriptions = {
        'rules_full_sft_ep10_20261026': ('规则1000，全参数SFT各10epoch', '训练完成；batch4四方向完成，batch1仅T2T完成'),
        'qwen3_lora_learning_ep5_20261024': ('规则1000/实体1109，8个LoRA各5epoch', '8/8训练完成，最终检查点在本地'),
        full.name: ('两套数据，SFT/PT，CoT及同协议基座', f'已暂停；{len(cot)}/20方向完成，33990/42180条已保存'),
        recall_root.name: ('规则90条，每方向180题，配对回忆诊断', f'已暂停；{recall["cells_complete"]}/{recall["cells_expected"]}配对方向完成'),
        repair_root.name: ('已有CoT/回忆输出解析与追加约束读取', f'已暂停；{repair["totals"]["readout_pending"]}条读取待完成'),
        'qwen3_lora_recall_apply_20261025': ('回忆v1', '已作废：80/90规则名称错误；只保留审计'),
        'answer_readout_audit_20261026': ('历史输出解析漏项检查', '审计完成；不等于训练或新闭卷评测'),
        'v7_full_experiment': ('规则1000，3模型×SFT/PT×TEXT/VISION', 'strict完成；48行中24个规范四方向单元'),
        'v7_fullparam_sft_ep10_20261019': ('历史规则1000，全参数SFT各10epoch', '训练及四方向评测完成'),
        'entity1020_merged1109_qwen3_base_20261020': ('实体1109，基座，4轮选项轮换', '17744条生成完成；不是训练后迁移结果'),
        'v7_process_confirmation_20261020': ('8规则，TEXT过程学习，1seed', '完成；2656条生成'),
        'v7_process_scale64_20261021': ('64规则，TEXT过程学习，3seed', '完成；53120条生成，无VISION训练臂'),
        'v7_cot100_20261021': ('50规则，各模态100题', 'vllm子目录完成900次生成；父目录旧引擎结果不合并'),
        'v7_process_image_diagnostic_20261021': ('8规则，图像/规则/观测控制', '128次事后诊断完成'),
        'environment_smoke_20261024': ('环境、训练与推理链路检查', '通过；不纳入研究准确率'),
        'xmodal_dataset_reports': ('规则/实体分开报告', '派生报告；本次更新到最新来源'),
    }
    inventory = [dict(run=p.name, scope=descriptions[p.name][0], status=descriptions[p.name][1],
                      path=str(p.relative_to(ROOT))) for p in sorted(RUNS.iterdir()) if p.is_dir() and p.name in descriptions]
    unknown = [p.name for p in RUNS.iterdir() if p.is_dir() and p != output and p.name not in descriptions and not p.name.startswith('experiment_summary_')]
    check(not unknown, f'all run directories classified; unclassified={unknown}')

    lines = ['# 当前实验数据汇总', '', f'整理时间（北京时间）：{generated}', '',
             '以本地各方向报告、逐题结果和暂停标记为准。此目录整理已有产物，不启动或恢复训练、推理。', '',
             '## 当前结论', '',
             '- 最新规则1000条全参数SFT已完成10 epoch和四方向短答案评测：26.05% / 26.80% / 25.85% / 26.65%。四方向均为2000题，无无效答案，整体仍接近四选一25%的随机水平。',
             '- 8个5 epoch LoRA均已训练完成。CoT保存33990/42180条，完成15/20个方向；同协议基座尚未完成，暂不能给出训练相对基座的CoT增益。',
             '- 64规则过程学习在三个seed的原题文字推理均为100%，图片推理均值53.13%；这是TEXT学习小规模证据，不扩展为完整1000规则或VISION学习结论。',
             '- 实体类已有训练后部分结果；旧清单中“尚无实体训练后评测”已过时。严格FINAL低分存在大量输出格式失败，纯解析和追加推理必须分列。', '',
             '## 数据与计分口径', '']
    lines += table(['数据集', '知识/学习条数', '每方向题数', '原图数量', '真实入口'], [
        [r['dataset'], r['knowledge_units'], r['questions_per_direction'], r['images'], link(ROOT / r['root'] / 'README.md')] for r in datasets])
    lines += ['', '方向固定为TEXT检查点→T2T/T2V，VISION检查点→V2T/V2V；基座只区分文字/图片测试。规则、实体分别统计，无效答案计错。', '',
              '规则类104道复合记录题保留偏差观察项，241道完整规则审计与1759道结果应用题应分项；原图未获全部独立语义认证。实体1109组含21组修复候选，语义及时间新颖性认证未完成。VISION训练保留原配文，不能称为无文字纯图片学习。', '',
              '## 最新全参数SFT：完整规则1000条', '',
              f'正式汇总采用完整的 {link(dense / "evaluation_strict_b4/summary.md", "batch4评测")}，固定最终epoch10/checkpoint-630；最大生成4 tokens、temperature=0。', '']
    lines += table(['方向', '正确/总题数', '准确率', 'direct', 'diagnosis', '结果应用(1759题)', '规则审计(241题)', '偏差观察(104题)'], [
        [r['direction'], f'{r["correct"]}/{r["total"]}', pct(r['accuracy']), pct(r['by_question_type']['direct']['accuracy']),
         pct(r['by_question_type']['diagnosis']['accuracy']), pct(r['result_application']['accuracy']), pct(r['rule_report_audit']['accuracy']), pct(r['bias_watchlist']['accuracy'])] for r in fresh_reports])
    lines += ['', f'北京时间2026-10-26 21:59训练完成；本轮完整评测汇总时间为北京时间2026-10-26 23:18。batch1先行T2T为{pct(pilot["accuracy"])}，不与batch4重复累计。两个学习模型均由原始基座独立初始化；TEXT冻结视觉/对齐层，VISION更新全部参数，BF16+8-bit AdamW。优化器和视觉参数范围与历史全参数实验有差异，不能仅按epoch认作同条件复现。', '',
              '## 历史完整规则结果', '', '各方向2000题，以下严格短答案与新CoT分别解读。', '']
    historical_table = []
    for model in ('qwen3', 'qwen25', 'glm46'):
        for objective in ('sft', 'pt'):
            selected = {r['direction']: r for r in canonical if r['model'] == model and r['objective'] == objective}
            historical_table.append([model, objective.upper()] + [pct(float(selected[d]['accuracy'])) for d in DIRECTIONS])
    lines += table(['模型', '历史LoRA目标', *DIRECTIONS], historical_table)
    lines += ['', '历史LoRA均为5 epoch；原表48行含同一学习模型下的重复方向标签，本汇总选取24个规范方向。GLM严格0%涉及格式失败：SFT TEXT→T2T仅宽容空think标签后为515/2000（25.75%），该补充解析不替换原成绩。', '']
    lines += table(['历史全参数实验', *DIRECTIONS], [
        ['当轮基座（历史短答案协议）'] + [pct(float(r['baseline_accuracy'])) for r in historical_dense],
        ['20261019 全参数SFT 10epoch'] + [pct(float(r['accuracy'])) for r in historical_dense]])
    lines += ['', f'来源：{link(old_path)}、{link(historical_dense_path)}。历史训练状态记录{len(old_training["jobs"])}个任务；历史完整权重缺失记录仍按各运行目录说明保留，新训练权重单独列在training.csv。', '',
              '## 5 epoch LoRA：已完成的CoT与答案修复', '',
              f'全量CoT状态：{full_status["status"]}；完成{len(cot)}/20个方向。根summary.json停留在{stale_summary["cells_complete"]}个，本表直接读取15份方向report.json。', '',
              '严格FINAL、纯解析、追加约束读取分别报告；后者向同一模型再发送原题/原测试图及自身回答，增加了一次推理，不能当作原闭卷成绩或纯解析收益。', '']
    repair_lookup = {(r['job'], r['direction']): r for r in repair['cells'] if r['experiment'] == 'full_cot'}
    for dataset, title in [('rules', '规则类'), ('entities', '实体类')]:
        lines += [f'### {title}', '']
        body = []
        for r in cot:
            if r['dataset'] != dataset:
                continue
            a = repair_lookup[(r['job'], r['direction'])]
            body.append([r['job'], r['direction'], r['total'], pct(r['accuracy']), pct(r['invalid'] / r['total']),
                         pct(a['parsed_accuracy']), pct(a['combined_accuracy'])])
        lines += table(['模型/学习方式', '方向', '题数', '原严格FINAL', '原无效率', '纯解析后', '追加读取后'], body) + ['']
    lines += ['### 尚未完成的同协议方向', '']
    lines += table(['模型', '方向/测试条件', '已保存/应有', '准确率'], [
        [r['job'], ('文字测试（原T2T）' if r['direction'] == 'T2T' else '图片测试（原T2V）') if r['job'].endswith('_base') else r['direction'],
         f'{r["saved"]}/{r["expected"]}', '未报告'] for r in cot_progress if not r['complete']])
    lines += ['', '完整训练后CoT方向合计31526条；另有2464条未完成方向输出，不用部分样本率充当完整准确率。实体历史基座采用四轮选项轮换，文字28.64%、图片27.55%（各8872次生成）；协议不同，不作本轮CoT基座对照。', '',
              '## 回忆诊断与小规模过程学习', '',
              f'修正后的回忆v2：{recall_status["status"]}，保存{recall["predictions_saved"]}/{recall["predictions_expected"]}条生成；90条规则，每方向180题，完成3/10个配对方向。', '']
    lines += table(['模型', '方向', '直接答', '先回忆再答', '差值(pp)', '无效数：直接/回忆'], [
        [r['job'], r['direction'], pct(r['direct_boxed']['accuracy']), pct(r['self_recall_boxed']['accuracy']),
         f'{r["delta_pp"]:+.2f}', f'{r["direct_boxed"]["invalid"]}/{r["self_recall_boxed"]["invalid"]}'] for r in recall['reports']])
    lines += ['', '回忆v1因80/90条规则名错误已作废。v2未按回忆正确性筛样，回忆语义正确性尚未认证；VISION→文字方向存在大量无效输出，当前部分结果不足以认定回忆方法整体效果。', '',
              '64规则过程组：固定TEXT学习、三个seed，同一最终检查点分别答文字和原图；每个seed均通过预设来源应用门槛。', '']
    scale = processes['v7_process_scale64_20261021']
    lines += table(['seed', '新文字留出推理(2048题)', '原题T2T推理(128题)', '原题T2V推理(128题)'], [
        [seed, pct(scale[f'{seed}:process']['new_holdout:reasoning:T']['accuracy']),
         pct(scale[f'{seed}:process']['original:reasoning:T']['accuracy']),
         pct(scale[f'{seed}:process']['original:reasoning:V']['accuracy'])] for seed in (42, 43, 44)])
    v = scale_seed['original:reasoning:V']
    lines += ['', f'图片推理三seed均值{pct(v["mean"])}，样本标准差{v["sample_std"] * 100:.2f}个百分点；相同题集的多个seed不视为独立扩充测试集。等步数对照的监督token不同，没有VISION学习组。', '',
              '8规则前置实验：过程组新文字留出246/256（96.09%），原题T2T 16/16、T2V 5/16。图像事后控制中给正确规则9/16、给正确文字观测12/16、两者同时给16/16；用于定位读图/调用/执行问题，不纳入正式迁移分数。', '',
              '50规则CoT诊断（从64规则定义SFT检查点出发，各模态100题，vLLM引擎）：', '']
    lines += table(['模型', '协议', '测试模态', '准确率', '无效数'], [
        [r['model'], r['protocol'], r['modality'], f'{float(r["accuracy_percent"]):.2f}%', r['invalid']] for r in cot100])
    lines += ['', '该诊断900次生成含回忆，不能替代全量CoT；vLLM合并权重与历史Swift短答案输出也不混合评分。', '',
              '## 运行目录索引与后续缺口', '']
    lines += table(['目录', '范围', '实际状态'], [[link(RUNS / r['run'], r['run']), r['scope'], r['status']] for r in inventory])
    lines += ['', '当前缺口：完整CoT仍缺5个方向及同协议基座；回忆v2尚缺7个配对方向；答案追加读取剩276条。原暂停标记保持有效。最新全参数四方向已完成，但本地尚未见与该轮完全相同配置的新基座评测和独立最终完成审计。', '',
              '## 可复用表格与核验', '',
              '- [全部指标 metrics.csv](metrics.csv)：一行一个数据集/范围/训练方式/方向/协议/测试子集；accuracy为0到1，未完成留空，不作跨协议混合平均。',
              '- [训练 training.csv](training.csv)：8个LoRA和2个最新全参数的方式、最终检查点、epoch、step、完成时间（北京时间）及首末日志loss；loss不是准确率。',
              '- [实验索引 experiments.csv](experiments.csv)、[汇总 summary.json](summary.json)、[来源哈希 sources.json](sources.json)、[核验 validation.json](validation.json)。', '',
              '本次重新核对两套数据四方向行数、最新全参数8000条、已完成CoT31526条、答案修复32606条的逐题计数及gold一致性；严格短答案/FINAL还复查输出解析，33990条CoT原始生成逐行核对保存数与唯一ID。历史实验复用已有汇总，本次未重跑模型，也未重新认证图像语义或重新散列全部权重。', '',
              f'重新整理：`python scripts/summarize_current_experiments.py --output {output.relative_to(ROOT)}`。CSV中的source_file相对项目根目录解析；各行附原始来源，重复解析同一回答的不同协议不可累加成独立生成数。', '']

    write_csv(output / 'metrics.csv', METRICS)
    write_csv(output / 'training.csv', training)
    write_csv(output / 'experiments.csv', inventory)
    validation = dict(status='passed', checked_at=generated, check_count=len(CHECKS),
                      current_dense_rows_verified=sum(r['total'] for r in fresh), full_cot_rows_verified=sum(r['total'] for r in cot),
                      repair_rows_verified=sum(r['total'] for r in repair['cells']),
                      full_cot_complete_cells=len(cot), full_cot_root_summary_cells=stale_summary['cells_complete'],
                      full_cot_saved=sum(r['saved'] for r in cot_progress), full_cot_raw_rows_verified=sum(r['saved'] for r in cot_progress),
                      limitations=['Historical aggregate results reused; no model inference.',
                                   'No full weight/image rehash or semantic certification.',
                                   'Different protocols and overlapping result subsets must not be pooled.'])
    summary = dict(generated_at=generated, datasets=datasets, inventory=inventory, training=training,
                   metrics=METRICS, full_cot_progress=cot_progress, validation=validation,
                   recall_progress={k: recall[k] for k in ('cells_complete', 'cells_expected', 'predictions_saved', 'predictions_expected')},
                   answer_repair_progress=repair['totals'],
                   paused_runs={full.name: full_status['status'], recall_root.name: recall_status['status'], repair_root.name: repair_status['status']})
    for name, data in [('summary.json', summary), ('validation.json', validation), ('sources.json', SOURCES)]:
        (output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    (output / 'README_ZH.md').write_text('\n'.join(lines))
    print(json.dumps({'output': str(output), 'metric_rows': len(METRICS), 'sources': len(SOURCES), 'validation': validation}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=RUNS / ('experiment_summary_' + datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d')))
    args = parser.parse_args()
    build(args.output.resolve())
