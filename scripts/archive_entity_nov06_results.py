#!/usr/bin/env python3
"""Archive four entity experiments without modifying any live source experiment."""
import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
SPECS = [
    ('01_GLM_两类QA各重复10次', 'GLM-4.6V-Flash', 10,
     'entity1814_repeat10_glm46_lora_ep3_neutral_v5_20261105', 'train/TEXT', 'atomic_replay'),
    ('02_GLM_两类QA各学1次', 'GLM-4.6V-Flash', 1,
     'entity1814_repeat1_glm46_lora_ep3_neutral_v5_20261106', 'train/TEXT', 'atomic_replay'),
    ('03_Qwen_两类QA各学1次', 'Qwen3-VL-4B-Instruct', 1,
     'entity1814_repeat1_qwen3vl4b_lora_ep3_neutral_v5_20261106', 'train/TEXT', 'TEXT'),
    ('04_Qwen_各重复10次_历史对照', 'Qwen3-VL-4B-Instruct', 10,
     'entity1814_repeat10_lora_ep3_neutral_v5_20261104', 'atomic_replay/train/TEXT', 'atomic_replay'),
]


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def copy_file(source, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(dest.suffix + '.tmp')
    shutil.copy2(source, temp)
    temp.replace(dest)


def copy_tree(source, dest, extensions=None):
    if source.exists():
        for path in sorted(source.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                if extensions is None or path.suffix in extensions:
                    copy_file(path, dest / path.relative_to(source))


def snapshot_rows(source, dest):
    # Read only the size visible at open and keep complete JSONL rows.
    with source.open('rb') as stream:
        size = os.fstat(stream.fileno()).st_size
        raw = stream.read(size)
    cutoff = raw.rfind(b'\n') + 1
    data, tail = raw[:cutoff], raw[cutoff:]
    records = [json.loads(line) for line in data.splitlines() if line.strip()]
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix('.jsonl.tmp')
    temp.write_bytes(data)
    temp.replace(dest)
    return records, dict(source_bytes_at_open=size, copied_bytes=len(data),
        incomplete_tail_bytes_excluded=len(tail), complete_rows=len(records),
        predictions_sha256=hashlib.sha256(data).hexdigest())


def metric(records, config):
    n = len(records)
    correct = sum(r['raw_predicted'] == r['gold'] for r in records)
    invalid = sum(r['raw_predicted'] is None for r in records)
    return dict(n=n, correct=correct, accuracy=correct / n if n else None,
        invalid=invalid, invalid_rate=invalid / n if n else None,
        answer_counts=dict(Counter(str(r['raw_predicted']) for r in records)),
        parse_reasons=dict(Counter(r['raw_parse_reason'] for r in records)),
        cap_counts=dict(Counter(str(r.get('generation_max_tokens', config['max_tokens'])) for r in records)),
        length_stops=sum(r['stop_reason'] == 'length' for r in records))


def percent(value):
    return '—' if value is None else f'{value:.2%}'


def write_csv(path, records):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def build(root):
    output = root / '11.06实验结果'
    output.mkdir(parents=True, exist_ok=True)
    captured = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
    reports, config_rows, summary_rows, group_rows = [], [], [], []
    for dirname, model, repetitions, run_name, train_relative, tuned_arm in SPECS:
        source, dest = root / 'runs' / run_name, output / '实验记录' / dirname
        dest.mkdir(parents=True, exist_ok=True)
        plan, manifest = read(source / 'experiment.json'), read(source / 'data/manifest.json')
        for name in ['experiment.json', 'status.json', 'README_ZH.md', 'RESULTS_ZH.md', 'comparison.json']:
            if (source / name).exists():
                copy_file(source / name, dest / '原始记录' / name)
        for src, dst, ext in [
            ('source_snapshots', 'source_snapshots', {'.py'}),
            ('protocol_revisions', 'protocol_revisions', {'.json', '.py', '.md'}),
            ('data', 'data', {'.json', '.jsonl'}),
            ('evaluation_data', 'evaluation_data', {'.json', '.jsonl'}),
        ]:
            copy_tree(source / src, dest / dst, ext)
        training = source / ('atomic_replay/data/train_text.jsonl' if repetitions == 10 else 'data/train_text.jsonl')
        if training != source / 'data/train_text.jsonl':
            copy_file(training, dest / 'data/train_text.jsonl')
        assert sha(training) == sha(dest / 'data/train_text.jsonl') == plan['train_sha256']
        train = source / train_relative
        validation = read(train / 'validation.json')
        checkpoint = Path(validation['checkpoint'])
        assert checkpoint.is_dir()
        copy_tree(checkpoint, dest / 'adapter', {'.json', '.md', '.safetensors'})
        for name in ['validation.json', 'logging.jsonl', 'args.json']:
            if (train / name).exists():
                copy_file(train / name, dest / 'training' / name)
        for path in source.glob('*_command.json'):
            copy_file(path, dest / 'training' / path.name)
        assert sha(dest / 'adapter/adapter_model.safetensors') == validation['adapter_sha256']
        report = dict(experiment=dirname, model=model, source_run=str(source), repetitions=repetitions,
            train_sha256=plan['train_sha256'], adapter_sha256=validation['adapter_sha256'],
            training_rows_per_epoch=plan['total_rows_per_epoch'], optimizer_steps=validation['global_step'],
            epochs=validation['epoch'], captured_at=captured, evaluations={})
        config_rows.append(dict(实验=dirname,模型=model,学习模态='TEXT',方法='ms-swift LoRA',
            实体数=1814,独立实体QA数=3628,每类QA每轮曝光次数=repetitions,训练轮数=3,
            每轮实体样本=plan['factual_rows_per_epoch'],每轮通用指令=plan['generic_rows_per_epoch'],
            每轮总样本=plan['total_rows_per_epoch'],优化步=validation['global_step'],
            LoRA_rank=64,LoRA_alpha=128,LoRA_dropout=0.05,学习率=0.0001,
            微批量=4,梯度累积=4,有效批量=16,训练文件SHA256=plan['train_sha256'],
            原始目录=str(source)))
        dev = set(manifest['prior_development_qids'])
        for arm in [tuned_arm, 'base']:
            folder = source / 'evaluation' / arm
            if not (folder / 'predictions.jsonl').exists():
                continue
            target = dest / 'evaluation' / arm
            for name in ['config.json', 'summary.json', 'status.json', 'runtime.json', 'import_manifest.json']:
                if (folder / name).exists():
                    copy_file(folder / name, target / name)
            copy_tree(folder / 'configuration_revisions', target / 'configuration_revisions', {'.json'})
            records, snapshot = snapshot_rows(folder / 'predictions.jsonl', target / 'predictions.jsonl')
            assert len(records) == len({(r['id'], r['modality']) for r in records})
            complete = len(records) == 7320
            config = read(target / 'config.json')
            kind, groups = ('基座' if arm == 'base' else 'TEXT LoRA'), {}
            for mod in 'TV':
                originals = [r for r in records if r['modality'] == mod and r['probe_id'] != 'rotation']
                selected_groups = dict(original=originals,
                    rotation=[r for r in records if r['modality'] == mod and r['probe_id'] == 'rotation'],
                    direct=[r for r in originals if r['probe_id'] == 'direct'],
                    statement=[r for r in originals if r['probe_id'] == 'statement'],
                    prior_development50=[r for r in originals if r['qid'] in dev],
                    other1764=[r for r in originals if r['qid'] not in dev])
                for group, subset in selected_groups.items():
                    item = metric(subset, config)
                    groups[f'{mod}/{group}'] = item
                    group_rows.append(dict(实验=dirname,模型=model,模型状态=kind,
                        模态=mod,分组=group,题数=item['n'],正确数=item['correct'],
                        准确率=percent(item['accuracy']),无效数=item['invalid'],
                        无效率=percent(item['invalid_rate']),触及长度上限数=item['length_stops'],
                        实际上限题数=json.dumps(item['cap_counts'], ensure_ascii=False),
                        完成状态='完整' if complete else '进行中快照'))
                item = groups[f'{mod}/original']
                summary_rows.append(dict(实验=dirname,模型=model,模型状态=kind,
                    方向=('T2' if arm != 'base' else '基座→') + mod,
                    已测原题=item['n'],计划原题=3628,正确数=item['correct'],
                    准确率=percent(item['accuracy']),无效数=item['invalid'],
                    无效率=percent(item['invalid_rate']),触及长度上限数=item['length_stops'],
                    实际上限题数=json.dumps(item['cap_counts'], ensure_ascii=False),
                    完成状态='完整' if complete else '进行中快照',
                    推理后端='vLLM（历史）' if run_name.endswith('20261104') else 'ms-swift TransformersEngine',
                    快照时间=captured))
            evaluation = dict(status='complete' if complete else 'partial_snapshot',
                saved=len(records), expected=7320, captured_at=captured, snapshot=snapshot,
                source=str(folder), last_saved_at=records[-1].get('saved_at') if records else None,
                raw_readout_only=True, finalizer_results_excluded=True, groups=groups, config=config)
            save(target / '整理后统计.json', evaluation)
            report['evaluations'][arm] = evaluation
        save(dest / '来源与索引.json', report)
        (dest / 'README.md').write_text(
            f'# {dirname}\n\n原始目录：{source}\n\n'
            f'模型：{model}；TEXT LoRA，3轮，两类QA每轮各曝光{repetitions}次。'
            f'每轮{plan["total_rows_per_epoch"]:,}条训练样本，共{validation["global_step"]:,}步。\n\n'
            'adapter/ 为最终LoRA权重和配置；data/ 为训练样本及审计记录；'
            'evaluation/ 为配置、逐题原始回答及整理后统计.json；'
            '原始记录/ 为源报告；source_snapshots/ 为对应源码。\n\n'
            '主成绩只使用 raw_predicted，不使用约束补选。'
            '原始状态文件若滞后，以逐题预测及整理后统计.json为准。\n')
        reports.append(report)
        print('已整理', dirname, flush=True)
    write_csv(output / '实验配置对照.csv', config_rows)
    write_csv(output / '实验结果汇总.csv', summary_rows)
    write_csv(output / '分组评测结果.csv', group_rows)
    save(output / '结果索引.json', dict(captured_at=captured, experiments=reports,
        primary='Original questions only; raw_predicted, invalids remain in denominator.',
        rotation_excluded_from_primary=True, live_source_unmodified=True))
    lines = ['# 11.06 实验结果', '', f'整理时间：{captured}（北京时间）。', '',
        '本文件夹汇总近期四组实体实验及对应基座，包含配置、训练数据、最终LoRA权重、'
        '源码快照和逐题原始回答。运行中的基座按复制到的完整JSONL行保存快照，原实验不受影响。', '',
        '## 完整评测结果', '',
        '每模态3,628道配对原题。每模态另有32道选项轮换题，单列于分组评测结果.csv。'
        '准确率分母包含无效回答；主成绩只使用 raw_predicted，不混入补选。', '',
        '| 实验 / 模型 | 文字准确率 | 文字无效率（题数） | 图片准确率 | 图片无效率（题数） |',
        '|---|---:|---:|---:|---:|']
    partials = []
    for report in reports:
        for arm, evaluation in report['evaluations'].items():
            label = report['experiment'].split('_', 1)[1] + (' / 基座' if arm == 'base' else ' / TEXT LoRA')
            t, v = [evaluation['groups'][f'{m}/original'] for m in 'TV']
            if evaluation['status'] != 'complete':
                partials.append((label, evaluation))
                continue
            lines.append(f'| {label} | {percent(t["accuracy"])} | {percent(t["invalid_rate"])}（{t["invalid"]}）'
                         f' | {percent(v["accuracy"])} | {percent(v["invalid_rate"])}（{v["invalid"]}） |')
    lines += ['', '## 正在进行的评测快照', '']
    for label, evaluation in partials:
        lines += [f'**{label}**：{evaluation["saved"]:,} / 7,320题，最后写入：{evaluation["last_saved_at"]}。', '']
        for mod, desc in [('T', '文字'), ('V', '图片')]:
            item = evaluation['groups'][f'{mod}/original']
            lines.append(f'- {desc}原题：{item["n"]:,} / 3,628，准确率{percent(item["accuracy"])}，'
                         f'无效率{percent(item["invalid_rate"])}（{item["invalid"]}题）。')
        lines += ['', '这是部分题目统计，不能作为完整基座结论。', '']
    if not partials:
        lines.append('本次整理涉及的评测均已完成。')
    lines += ['', '## 训练设置', '',
        '四组均为1,814个实体、每实体 fact_0 与 attribute_0 两类QA，共3,628条独立QA。'
        '只训练TEXT，冻结视觉编码器和对齐层；ms-swift LoRA、3 epoch、rank=64、alpha=128、'
        'dropout=0.05、学习率1e-4、微批量4、梯度累积4。', '',
        '| 学习方式 | 每轮实体样本 | 每轮通用指令 | 每轮总样本 | 优化步 |',
        '|---|---:|---:|---:|---:|',
        '| 两类各学1次 | 3,628 | 908 | 4,536 | 852 |',
        '| 两类各重复10次 | 36,280 | 9,072 | 45,352 | 8,505 |', '',
        'Qwen与GLM各学1次的训练文件及顺序完全相同；各重复10次的两组也使用相同文件。'
        '重复10次只是重复曝光，不代表每实体有20条独立知识。', '',
        '## 比较时需要保留的区别', '',
        '- 本表只有实体集TEXT学习的T2T/T2V，没有本轮规则集或VISION学习结果。',
        '- GLM/Qwen各学1次均采用1,024 token、简短推理提示与boxed答案，'
        '使用ms-swift TransformersEngine加载未合并LoRA；原生thinking关闭。',
        '- GLM重复10次模型用6,000 token；GLM基座前3,068题用6,000，余下4,252题用1,024，'
        '所以重复次数比较并非严格相同的生成条件。',
        '- 历史Qwen重复10次及其基座使用vLLM、BF16合并权重、65,536 token上限、'
        '180秒限制和循环检测。与新Qwen后端、权重加载方式、截断机制不同，不能把全部差异归因于重复次数。',
        '- GLM/Qwen架构、规模、分词器、模板和图像处理不同；重复次数改变时通用指令量及总计算量也改变。',
        '- 每模态32道轮换题仅作诊断；此前50个开发实体与其余1,764个分开统计，'
        '不宣称其他实体从未被评测。事实/时间有效性及同名Jack Miller歧义继承源数据。', '',
        '## 文件导航', '',
        '- 实验结果汇总.csv：原题准确率、无效率、正确/无效数、输出上限、完成状态。',
        '- 分组评测结果.csv：原题、轮换、direct/statement、开发50/其余1764分组。',
        '- 实验配置对照.csv：四组训练设置；结果索引.json：来源、SHA256和统计。',
        '- 实验记录/：各组adapter/、data/、evaluation/、training/、source_snapshots/。',
        '- 文件校验清单.json：归档文件大小及SHA256。',
        '- 更新整理.py：重新运行即可刷新本文件夹，包含仍在运行的Qwen基座。'
        '在项目根目录运行：.venv/bin/python "11.06实验结果/更新整理.py"。', '',
        '已复制四组LoRA adapter；不重复复制完整基座模型和测试图片。'
        '模型与图片仍按配置及eval JSONL中的绝对路径引用。'
        '这是结果审阅与追溯归档，并非脱离原项目即可直接运行的完整环境。', '']
    (output / 'README.md').write_text('\n'.join(lines))
    if Path(__file__).resolve() != output / '更新整理.py':
        copy_file(Path(__file__).resolve(), output / '更新整理.py')
    files = {str(p.relative_to(output)): dict(bytes=p.stat().st_size, sha256=sha(p))
             for p in sorted(output.rglob('*')) if p.is_file() and p.name != '文件校验清单.json'}
    save(output / '文件校验清单.json', dict(captured_at=captured, files=files,
        total_files=len(files), total_bytes=sum(item['bytes'] for item in files.values())))
    print(json.dumps(dict(output=str(output), captured_at=captured, experiments=len(reports),
        evaluation_arms=sum(len(r['evaluations']) for r in reports), files=len(files),
        total_bytes=sum(item['bytes'] for item in files.values()),
        partial_evaluations=[dict(label=label, saved=e['saved']) for label, e in partials]), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    build(parser.parse_args().root.resolve())
