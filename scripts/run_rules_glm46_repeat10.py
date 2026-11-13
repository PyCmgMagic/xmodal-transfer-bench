#!/usr/bin/env python3
"""Repeat the frozen rule-only GLM learning plan 10 times and compare matched runs."""
import argparse
from collections import Counter
import copy
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

from answer_readout import options_from_question
from audit_rules_glm46_readout import VERSION as AUDIT_VERSION, category, reparse, scores
from entityqa_common import digest, now, rows, save, sha, write_rows
from evaluate_rules_glm46 import tasks_for
from run_entity_glm46_full import BASE, training_command
from run_entityqa_lora import checkpoint as check_adapter, idle

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'runs/rules1000_glm46_lora_ep3_original_20261106'
WORK = ROOT / 'runs/rules1000_repeat10_glm46_lora_ep3_20261107'
REPEATS, EPOCHS, MICRO_BATCH, ACCUMULATION = 10, 3, 4, 4
SOURCE_FILES = [
    'run_rules_glm46_repeat10.py', 'evaluate_rules_glm46.py', 'audit_rules_glm46_readout.py',
    'run_entity_glm46_full.py', 'run_entityqa_lora.py', 'evaluate_entity_glm46_full.py',
    'entityqa_common.py', 'entityqa_prompt_neutral_v5.py', 'answer_readout.py',
    'answer_readout_entity_v2.py', 'answer_readout_entity_v3.py', 'answer_readout_entity_v4.py',
]


def freeze(source, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        assert sha(source) == sha(dest), str(dest)
    else:
        shutil.copy2(source, dest)


def repeated_rows(source):
    result = []
    for repetition in range(1, REPEATS + 1):
        for source_row in source:
            row = copy.deepcopy(source_row)
            row.update(id=f'{source_row["id"]}__repeat{repetition:02d}',
                source_sample_id=source_row['id'], repetition_index=repetition)
            result.append(row)
    return result


def optimizer_steps(count):
    return math.ceil(math.ceil(count / MICRO_BATCH) / ACCUMULATION) * EPOCHS


def prepare(work):
    work.mkdir(parents=True, exist_ok=True)
    previous = json.loads((PARENT / 'experiment.json').read_text())
    assert previous['status'] == 'complete'
    assert previous['modality'] == 'TEXT' and previous['epochs'] == EPOCHS
    assert sha(PARENT / 'data/train_text.jsonl') == previous['train_sha256']
    assert (BASE / 'download_complete.json').is_file()
    for name, expected in previous['source_sha256'].items():
        assert sha(ROOT / 'scripts' / name) == expected, name
    manifest = json.loads((PARENT / 'data/manifest.json').read_text())
    for rel, expected in manifest['files'].items():
        assert sha(PARENT / rel) == expected, rel
    original = rows(PARENT / 'data/training_rows_with_metadata.jsonl')
    unique = rows(PARENT / 'data/rule_unique_training.jsonl')
    retention = rows(PARENT / 'data/retention_training.jsonl')
    assert len(original) == 1252 and len(unique) == 1000 and len(retention) == 252
    assert [dict(id=r['id'], messages=r['messages']) for r in original] == rows(PARENT / 'data/train_text.jsonl')
    core_ids, retention_ids = {r['id'] for r in unique}, {r['id'] for r in retention}
    assert not core_ids & retention_ids and len(core_ids | retention_ids) == 1252
    expanded = repeated_rows(original)
    training = [dict(id=r['id'], messages=r['messages']) for r in expanded]
    assert len(training) == len({r['id'] for r in training}) == 12520
    assert Counter(r['source_sample_id'] for r in expanded) == dict.fromkeys(core_ids | retention_ids, REPEATS)
    assert sum(r['source_sample_id'] in core_ids for r in expanded) == 10000
    assert sum(r['source_sample_id'] in retention_ids for r in expanded) == 2520
    assert Counter(r['answer'] for r in expanded if r['source_sample_id'] in retention_ids) == dict.fromkeys('ABCD', 630)
    source_lookup = {r['id']: r for r in original}
    assert all(r['messages'] == source_lookup[r['source_sample_id']]['messages'] for r in expanded)
    assert all([m['role'] for m in r['messages']] == ['user', 'assistant'] and not r.get('images') for r in training)
    for rel, data in [('data/train_text.jsonl', training), ('data/training_rows_with_metadata.jsonl', expanded)]:
        target = work / rel
        if target.exists():
            assert rows(target) == data, rel
        else:
            write_rows(target, data)
    for rel in ['data/rule_unique_training.jsonl', 'data/retention_training.jsonl',
                'data/learning_inputs.jsonl', 'data/training_template_audit.json']:
        freeze(PARENT / rel, work / 'references/repeat1' / rel)
    # All repeated messages are checked against the exact unique source; audit each unique template.
    from datasets import load_dataset
    from swift.model import get_model_processor
    from swift.template import get_template
    loaded = load_dataset('json', data_files=str(work / 'data/train_text.jsonl'),
        cache_dir=str(work / 'loader_validation_cache'), split='train')
    assert set(loaded.column_names) == {'id', 'messages'} and len(loaded) == 12520
    assert all(dict(a) == b for a, b in zip(loaded, training))
    _, processor = get_model_processor(str(BASE), model_type='glm4v', load_model=False)
    template = get_template(processor, template_type='glm4_5v', max_length=2048,
        enable_thinking=False, loss_scale='default', truncation_strategy='raise')
    template.set_mode('train')
    lengths, supervised = [], []
    for row in original:
        encoded = template.encode(dict(messages=copy.deepcopy(row['messages'])))
        ids, labels = encoded['input_ids'], encoded['labels']
        assert len(ids) == len(labels) <= 2048
        expected = template.template_meta.non_thinking_prefix + row['messages'][-1]['content'] + template.template_meta.suffix[0]
        assert processor.tokenizer.decode([v for v in labels if v != -100]) == expected
        assert any(v == -100 for v in labels) and any(v != -100 for v in labels)
        lengths.append(len(ids))
        supervised.append(sum(v != -100 for v in labels))
    save(work / 'data/template_audit.json', dict(status='pass', unique_messages_checked=1252,
        repeated_rows_exactly_verified=12520, loader_verified_rows=12520,
        max_formatted_tokens=max(lengths), min_supervised_tokens=min(supervised),
        truncated_rows=0, user_prompt_masked=True, checked_at=now()))
    for source in (PARENT / 'evaluation_data').rglob('*'):
        if source.is_file():
            freeze(source, work / 'evaluation_data' / source.relative_to(PARENT / 'evaluation_data'))
    inventory = json.loads((work / 'evaluation_data/package_inventory.json').read_text())
    for rel, expected in inventory['sha256'].items():
        assert sha(work / 'evaluation_data' / rel) == expected, rel
    image_rows = rows(work / 'evaluation_data/image_hashes.jsonl')
    assert len(image_rows) == 3000
    source_images = ROOT / 'data/rules_repaired_single_image_v7'
    for image in image_rows:
        assert sha(source_images / image['path']) == image['repaired_sha256'], image['path']
    save(work / 'data/image_audit.json', dict(status='pass', verified_images=3000, checked_at=now()))
    tasks = tasks_for(work)
    inputs_hash = digest([{k: v for k, v in task.items() if k != 'gold'} for task in tasks])
    assert inputs_hash == previous['evaluation']['input_sha256']
    # Freeze completed matched base and single-exposure responses; no repeated base inference.
    reference_hashes = {}
    for arm in ['base', 'TEXT']:
        summary = json.loads((PARENT / 'evaluation' / arm / 'summary.json').read_text())
        assert summary['status'] == 'complete' and summary['saved'] == 4000
        for name in ['config.json', 'predictions.jsonl', 'summary.json', 'status.json']:
            dest = work / ('evaluation/base' if arm == 'base' else 'references/repeat1/evaluation/TEXT') / name
            freeze(PARENT / 'evaluation' / arm / name, dest)
            reference_hashes[str(dest.relative_to(work))] = sha(dest)
    for name in SOURCE_FILES:
        freeze(ROOT / 'scripts' / name, work / 'source_snapshots' / name)
    plan = dict(status='prepared', created_at=now(), host=socket.gethostname(), gpu='0',
        model='GLM-4.6V-Flash', model_path=str(BASE), environment=str(ROOT / '.venv-glm'),
        framework='ms-swift ' + importlib.metadata.version('ms-swift'),
        transformers=importlib.metadata.version('transformers'), modality='TEXT', tuner='LoRA',
        starts_from_clean_base=True, source_single_exposure_run=str(PARENT), epochs=EPOCHS,
        knowledge_units=1000, unique_rule_samples=1000, unique_generic_samples=252,
        repetitions_per_rule_per_epoch=REPEATS, repetitions_per_generic_per_epoch=REPEATS,
        rule_exposures_total=REPEATS * EPOCHS, factual_rows_per_epoch=10000,
        generic_rows_per_epoch=2520, total_rows_per_epoch=12520,
        planned_optimizer_steps=optimizer_steps(12520), rank=64, alpha=128, dropout=0.05,
        learning_rate=1e-4, micro_batch=MICRO_BATCH, gradient_accumulation=ACCUMULATION,
        effective_batch=16, max_length=2048, train_sha256=sha(work / 'data/train_text.jsonl'),
        parent_train_sha256=previous['train_sha256'],
        source_sha256={name: sha(work / 'source_snapshots' / name) for name in SOURCE_FILES},
        reference_sha256=reference_hashes, checkpoint_selection='fixed final epoch 3',
        evaluation=dict(previous['evaluation'], fresh_matched_base=False,
            matched_base_reused_from=str(PARENT / 'evaluation/base'),
            parser_legacy='entity v4 plus GLM native box normalization',
            offline_report_parser=AUDIT_VERSION,
            primary_report='Uniform conservative offline readout for base/repeat1/repeat10; legacy scores retained'),
        limitations=previous['limitations'] + [
            'Exactly the original samples repeated 10 times, including the generic pool; no generated rule variants.',
            'More repetitions also increase optimizer steps and total token exposure.',
            'Offline readout fixes are applied uniformly to every comparison arm without new inference.',
        ])
    assert plan['planned_optimizer_steps'] == 2349
    target = work / 'experiment.json'
    if target.exists():
        existing = json.loads(target.read_text())
        for key in ['train_sha256', 'parent_train_sha256', 'source_sha256', 'reference_sha256', 'evaluation']:
            assert existing[key] == plan[key], key
        plan = existing
    save(target, plan)
    (work / 'README_ZH.md').write_text(
        '# GLM规则类：原始样本重复10次，TEXT LoRA，3轮\n\n'
        '从干净GLM-4.6V-Flash基座训练。每轮1000条规则各出现10次；'
        '上一轮252条通用指令也各重复10次，维持原始样本配比和内容。'
        '每轮12520条，3轮2349步；每条规则总计学习30次。\n\n'
        'ms-swift，隔离环境.venv-glm，rank64/alpha128/dropout0.05，lr1e-4，'
        'batch4×累积4，bf16/SDPA，冻结视觉层，从基座重新训练。\n\n'
        '训练后自动评测T2T/T2V各2000题，闭卷、简短推理提示、原生thinking关闭、'
        '输出1024 token。复用上一轮同题同配置的基座结果，保留原始解析分数，'
        '并对基座、各学1次、重复10次统一离线重解析后对比。\n')
    return plan


def command(work):
    cmd = training_command(work)
    for flag, value in {'--dataset': str(work / 'data/train_text.jsonl'),
                        '--output_dir': str(work / 'train/TEXT'), '--master_port': '30389'}.items():
        cmd[cmd.index(flag) + 1] = value
    return cmd


def validate(work, plan):
    ck = check_adapter(work / 'train/TEXT')
    state = json.loads((ck / 'trainer_state.json').read_text())
    assert state['global_step'] == plan['planned_optimizer_steps'] == 2349
    assert ck.name == 'checkpoint-2349'
    args = json.loads((ck / 'args.json').read_text())
    assert args['dataset'] == [str(work / 'data/train_text.jsonl')]
    assert Path(args['model']).resolve() == BASE.resolve()
    assert not args.get('adapters') and not args.get('resume_from_checkpoint')
    return ck


def compare(work, plan):
    tasks = {(r['id'], r['modality']): r for r in tasks_for(work)}
    comparison, reference_configs, fingerprints = {}, [], {}
    output = work / 'rescored'
    output.mkdir(exist_ok=True)
    for label, folder in [('base', work / 'evaluation/base'),
                          ('repeat1', work / 'references/repeat1/evaluation/TEXT'),
                          ('repeat10', work / 'evaluation/TEXT')]:
        config = json.loads((folder / 'config.json').read_text())
        reference_configs.append({k: v for k, v in config.items() if k not in {'arm', 'adapter', 'adapter_sha256'}})
        summary = json.loads((folder / 'summary.json').read_text())
        assert summary['status'] == 'complete' and summary['saved'] == 4000
        records = rows(folder / 'predictions.jsonl')
        assert len(records) == len({(r['id'], r['modality']) for r in records}) == 4000
        revised = []
        for record in records:
            task = tasks[record['id'], record['modality']]
            assert record['gold'] == task['gold'] and record['input_sha256'] == digest(task['payload'])
            pred, reason = reparse(record['raw_response'], options_from_question(task['payload']['messages'][-1]['content']))
            assert record['raw_predicted'] is None or record['raw_predicted'] == pred
            revised.append(dict(record, audit_predicted=pred, audit_parse_reason=reason,
                audit_category=category(record, pred, reason),
                recovered=record['raw_predicted'] is None and pred is not None))
        write_rows(output / (label + '.jsonl'), revised)
        groups = {}
        for mod in 'TV':
            selected = [r for r in revised if r['modality'] == mod]
            assert len(selected) == 2000
            subsets = {mod + '/all': selected}
            for field in ['question_type', 'family', 'evaluation_group', 'task_variant']:
                for value in sorted({r[field] for r in selected if r[field]}):
                    subsets[f'{mod}/{field}/{value}'] = [r for r in selected if r[field] == value]
            for name, subset in subsets.items():
                groups[name] = dict(legacy=scores(subset, 'raw_predicted'),
                    offline_reparsed=scores(subset, 'audit_predicted'),
                    recovered=sum(r['recovered'] for r in subset))
        fingerprints[label] = sha(folder / 'predictions.jsonl')
        comparison[label] = dict(groups=groups, raw_predictions_sha256=fingerprints[label])
    assert all(config == reference_configs[0] for config in reference_configs)
    save(work / 'comparison.json', dict(status='complete', completed_at=now(),
        same_inputs_labels_generation_settings_verified=True, offline_parser=AUDIT_VERSION,
        no_new_generation_in_readout=True, reports=comparison))
    lines = ['# GLM规则类：重复10次与各学1次对照', '',
        '统一离线重解析；原始解析分数同时保留。每个模型文字/图片各2000题，无效计错。', '',
        '| 模型 | T准确率 | T无效率 | V准确率 | V无效率 |', '|---|---:|---:|---:|---:|']
    for label, name in [('base', 'GLM基座'), ('repeat1', '规则各学1次/轮'), ('repeat10', '规则重复10次/轮')]:
        t, v = [comparison[label]['groups'][mod + '/all']['offline_reparsed'] for mod in 'TV']
        lines.append(f'| {name} | {t["accuracy"]:.2%} | {t["invalid_rate"]:.2%} | {v["accuracy"]:.2%} | {v["invalid_rate"]:.2%} |')
    lines += ['', 'TEXT LoRA，3轮，从干净基座训练；重复10次为12520条/轮、2349步。',
        '通用指令与规则样本均原样重复10次；不新增规则变体。基座复用上一轮同配置结果。',
        '闭卷、简短推理提示、原生thinking关闭、1024输出token；解析修复不追加推理或强制补选。',
        '原始分数、统一重解析分数及分组结果见comparison.json，逐题结果见rescored。', '']
    (work / 'RESULTS_ZH.md').write_text('\n'.join(lines))


def controller(work):
    assert socket.gethostname() == 'softengine-KVM'
    work.mkdir(parents=True, exist_ok=True)
    lock = (work / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def status(stage, **extra):
        save(work / 'status.json', dict(stage=stage, controller_pid=os.getpid(),
            host=socket.gethostname(), gpu='0', updated_at=now(), **extra))
    status('preparing')
    plan = prepare(work)
    for name, expected in plan['source_sha256'].items():
        assert sha(ROOT / 'scripts' / name) == sha(work / 'source_snapshots' / name) == expected
    for rel, expected in plan['reference_sha256'].items():
        assert sha(work / rel) == expected
    assert sha(work / 'data/train_text.jsonl') == plan['train_sha256']
    tmp = work / 'tmp'
    tmp.mkdir(exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1', MODELSCOPE_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', MAX_PIXELS='262144',
        TMPDIR=str(tmp), PYTHONUNBUFFERED='1', HF_DATASETS_CACHE=str(work / 'dataset_cache'),
        NPROC_PER_NODE='1', MASTER_ADDR='127.0.0.1', MASTER_PORT='30389',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    def execute(cmd, stage):
        while True:
            try:
                idle('0')
                break
            except RuntimeError as error:
                status('waiting_for_free_gpu', next_stage=stage, reason=str(error))
                time.sleep(30)
        save(work / (stage + '_command.json'), cmd)
        log = work / (stage + '.log')
        with log.open('a') as stream:
            child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        status(stage, child_pid=child.pid, log=str(log))
        if child.wait():
            raise RuntimeError(f'{stage} failed; inspect {log}; outputs preserved')
    folder = work / 'train/TEXT'
    if not (folder / 'checkpoint-2349').exists():
        if folder.exists() and any(folder.iterdir()):
            raise RuntimeError(f'Preserving partial training output {folder}')
        execute(command(work), 'training_GLM_rules_repeat10_TEXT')
    checkpoint = validate(work, plan)
    summary = work / 'evaluation/TEXT/summary.json'
    if not summary.exists() or json.loads(summary.read_text()).get('status') != 'complete':
        execute([str(ROOT / '.venv-glm/bin/python'), '-u', str(work / 'source_snapshots/evaluate_rules_glm46.py'),
            '--root', str(ROOT), '--work', str(work), '--arm', 'TEXT', '--checkpoint', str(checkpoint)],
            'evaluating_repeat10_TEXT')
    status('offline_comparison')
    compare(work, plan)
    plan.update(status='complete', completed_at=now())
    save(work / 'experiment.json', plan)
    status('complete', results=str(work / 'RESULTS_ZH.md'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=WORK)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    work = args.work.resolve()
    try:
        if args.prepare_only:
            prepare(work)
        else:
            controller(work)
    except Exception as error:
        save(work / 'status.json', dict(stage='failed', controller_pid=os.getpid(), error=repr(error), updated_at=now()))
        raise
