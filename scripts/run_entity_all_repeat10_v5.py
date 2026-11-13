#!/usr/bin/env python3
"""Full repaired entity set: fixed TEXT LoRA learning and neutral v5 evaluation."""
import argparse
import copy
from collections import Counter
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
from types import SimpleNamespace

from entityqa_common import rows, save, write_rows, sha, digest, now
from entityqa_prompt_neutral_v5 import eval_payload, COT_SUFFIX, PROMPT_VERSION
from prepare_entity_atomic_pilot import SEED, entity_examples, generic
from run_entityqa_lora import command, checkpoint, idle, merge


def balanced_retention(count, dev):
    """Independent synthetic tasks, label-balanced and disjoint from dev prompts."""
    assert count > 0 and count % 4 == 0
    quota = count // 4
    forbidden = {r['messages'][0]['content'] for r in dev}
    seen, accepted, labels = set(), [], Counter()
    index = 0
    rejected = Counter()
    while len(accepted) < count:
        row = generic(index, 'train')
        index += 1
        question = row['messages'][0]['content']
        if labels[row['answer']] >= quota:
            rejected['label_quota'] += 1
        elif question in forbidden:
            rejected['dev_prompt_overlap'] += 1
        elif question in seen:
            rejected['duplicate_train_prompt'] += 1
        else:
            accepted.append(row)
            labels[row['answer']] += 1
            seen.add(question)
        if index > count * 20:
            raise RuntimeError('Synthetic retention pool exhausted')
    assert labels == dict.fromkeys('ABCD', quota)
    assert seen.isdisjoint(forbidden)
    return accepted, dict(candidates=index, accepted=count, rejected=dict(rejected), label_counts=dict(labels))


def swift_training_rows(training):
    """Keep audit metadata separate from the schema used by the chunked JSON loader."""
    return [dict(id=r['id'], messages=copy.deepcopy(r['messages'])) for r in training]


def prepare_learning(root, work):
    """No formal questions, labels or test images are read by this function."""
    fact_path = root / 'runs/entity1025_fullfact_lora_ep3_20261104/data/learning_inputs.jsonl'
    qa_path = root / 'runs/entity1025_repaired_attrqa_lora_ep3_20261103/data/qa_pairs.jsonl'
    reference = root / 'runs/entity50_repeat10_lora_ep3_20261104'
    records = sorted(rows(fact_path), key=lambda r: hashlib.sha256((SEED + r['id']).encode()).hexdigest())
    qas = {r['id']: r for r in rows(qa_path)}
    assert len(records) == len({r['id'] for r in records}) == len({r['qid'] for r in records}) == 1814
    assert set(qas) == {r['id'] for r in records}
    prior50 = rows(reference / 'data/learning_inputs.jsonl')
    assert records[:50] == prior50
    core = []
    for record in records:
        qa = qas[record['id']]
        assert qa['source_sha256'] == digest(record)
        assert qa['answer'] in record['learning']
        samples = [r for r in entity_examples(record, qa) if r['variant'] in ['fact_0', 'attribute_0']]
        assert len(samples) == 2 and samples[0]['messages'][1]['content'] == record['claim']
        assert all(r['messages'][1]['content'].strip() for r in samples)
        core.extend(samples)
    assert core[:100] == rows(reference / 'data/entity_unique_training.jsonl')
    training = []
    for repeat in range(10):
        for row in core:
            item = copy.deepcopy(row)
            item['id'] += f'-repeat-{repeat}'
            training.append(item)
    assert len(training) == 36280
    retention_dev = rows(reference / 'data/retention_dev.jsonl')
    retention_count = 4 * math.ceil(len(training) / 16)
    retention, retention_audit = balanced_retention(retention_count, retention_dev)
    training.extend(retention)
    assert len(training) == 45352 and len(core) == 3628
    assert len({r['id'] for r in training}) == len(training)
    exposures = Counter((r['knowledge_id'], r['variant']) for r in training if 'knowledge_id' in r)
    assert len(exposures) == 3628 and set(exposures.values()) == {10}
    assert not any(r.get('images') for r in training)
    write_rows(work / 'data/training_rows_with_metadata.jsonl', training)
    write_rows(work / 'atomic_replay/data/train_text.jsonl', swift_training_rows(training))
    write_rows(work / 'data/entity_unique_training.jsonl', core)
    write_rows(work / 'data/learning_inputs.jsonl', records)
    write_rows(work / 'data/retention_training.jsonl', retention)
    write_rows(work / 'data/retention_dev.jsonl', retention_dev)
    # Exercise the actual loader before loading a GPU model. The full JSONL spans
    # multiple default-size chunks, unlike the smaller E23 file.
    from datasets import load_dataset
    loaded = load_dataset('json', data_files=str(work / 'atomic_replay/data/train_text.jsonl'),
                          cache_dir=str(work / 'loader_validation_cache'), split='train')
    assert set(loaded.column_names) == {'id', 'messages'} and len(loaded) == len(training)
    assert all(actual['id'] == expected['id'] and actual['messages'] == expected['messages']
               for actual, expected in zip(loaded, training))
    save(work / 'data/loader_validation.json', dict(rows=len(loaded), columns=loaded.column_names,
        actual_hf_json_loader_passed=True, all_training_messages_and_order_verified=True,
        audit_metadata_separate=True))
    # The same 50 learning-only recall entities as E23, for a direct scale check.
    recall = rows(reference / 'data/recall_dev.jsonl')
    assert len(recall) == 150 and {r['qid'] for r in recall} == {r['qid'] for r in prior50}
    write_rows(work / 'data/recall_dev.jsonl', recall)
    save(work / 'data/retention_generation_audit.json', retention_audit)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(root / 'model_cache/Qwen3-VL-4B-Instruct'), local_files_only=True)
    def length(row):
        return len(tokenizer.apply_chat_template(row['messages'], tokenize=True, enable_thinking=False))
    lengths = [length(r) for r in core + retention]
    assert max(lengths) <= 2048, max(lengths)
    save(work / 'data/token_audit.json', dict(unique_factual_QA=len(core), independent_instruction_QA=len(retention),
        max_formatted_tokens=max(lengths), truncation_required=False,
        factual_target_tokens_per_epoch=10 * sum(len(tokenizer.encode(r['messages'][1]['content'], add_special_tokens=False)) for r in core),
        instruction_target_tokens_per_epoch=sum(len(tokenizer.encode(r['messages'][1]['content'], add_special_tokens=False)) for r in retention)))
    manifest = dict(version='entity1814-repeat10-neutral-v5', created_at=now(), knowledge_units=1814,
        selection='All repaired entities; original hash order preserved, no score-based filtering',
        unique_factual_QA=3628, repetitions_per_QA_per_epoch=10, exposures_per_QA_over_3_epochs=30,
        factual_rows_per_epoch=36280, independent_instruction_rows_per_epoch=9072, total_rows_per_epoch=45352,
        planned_optimizer_steps=8505, epochs=3, test_data_used_for_training=False,
        training_source_sha256={str(p): sha(p) for p in [fact_path, qa_path]},
        prior_development_qids=[r['qid'] for r in prior50], other_entity_count=1764,
        recall_probe_entities=50, recall_source='Exact learning-only probes from E23, not formal test questions',
        files={str(p.relative_to(work)): sha(p) for p in sorted(
            list((work / 'data').glob('*.jsonl')) + [work / 'atomic_replay/data/train_text.jsonl'])},
        limitations=['All 1814 repaired candidates retain inherited factual/temporal validity limitations.',
            'Repetition increases exposure, not independent knowledge count.',
            'Compared with 50-entity E23, entity count and instruction pool size change together.',
            'Remaining 1764 entities are reported separately, not certified as never previously evaluated.',
            'Retention pool expanded with exact prompt deduplication and dev disjointness; approximately 20% of rows.',
            'Only TEXT learning is trained; no VISION learning model is launched.'])
    save(work / 'data/manifest.json', manifest)
    return manifest


def prepare_evaluation(root, work, manifest):
    """Freeze the learning files before reading formal evaluation content."""
    for name, expected in manifest['files'].items():
        assert sha(work / name) == expected, name
    data = root / 'data/entity1025_repaired_20261103'
    reference = root / 'runs/entity50_repeat10_lora_ep3_20261104'
    dest = work / 'evaluation_data'
    ids = {r['qid'] for r in rows(work / 'data/learning_inputs.jsonl')}
    all_labels = {r['id']: r for r in rows(data / 'audit/eval_labels.jsonl')}
    assert len(all_labels) == 3628
    reference_labels = {r['id']: r for r in rows(reference / 'evaluation_data/audit/eval_labels.jsonl')}
    tasks = {}
    audit = []
    for mod, file in [('T', 'eval_text.jsonl'), ('V', 'eval_multimodal.jsonl')]:
        original = rows(data / 'shared' / file)
        assert len(original) == 3628 and {r['qid'] for r in original} == ids
        selected = []
        for row in original:
            item = copy.deepcopy(row)
            if item.get('images'):
                item['images'] = [str((data / p).resolve()) for p in item['images']]
            selected.append(item)
        rotation = [r for r in rows(reference / 'evaluation_data/shared' / file) if '-rotation-' in r['id']]
        assert len(rotation) == 32
        for row in rotation:
            assert row['id'] not in all_labels or all_labels[row['id']] == reference_labels[row['id']]
            all_labels[row['id']] = reference_labels[row['id']]
        selected.extend(rotation)
        assert len(selected) == len({r['id'] for r in selected}) == 3660
        for row in selected:
            payload = eval_payload(row, dest)
            assert set(payload) <= {'messages', 'images'}
            assert not any('\\boxed{' + c + '}' in payload['messages'][-1]['content'] for c in 'ABCD')
            tasks[row['id'], mod] = (row, payload)
            audit.append(dict(id=row['id'], modality=mod, qid=row['qid'], input_sha256=digest(payload)))
        write_rows(dest / 'shared' / file, selected)
    assert len(tasks) == 7320
    assert {i for i, m in tasks if m == 'T'} == {i for i, m in tasks if m == 'V'}
    write_rows(dest / 'audit/eval_labels.jsonl', list(all_labels.values()))
    write_rows(dest / 'audit/inference_input_hashes.jsonl', audit)
    save(dest / 'package_inventory.json', dict(version='entity-all-repeat10-neutral-v5', knowledge_units=1814,
        original_questions_per_modality=3628, rotation_questions_per_modality=32,
        source_inventory_sha256=sha(data / 'package_inventory.json'),
        training_manifest_sha256=sha(work / 'data/manifest.json'),
        sha256={str(p.relative_to(dest)): sha(p) for p in sorted(dest.rglob('*.jsonl'))}))
    previous = reference / 'evaluation_neutral_v5_20261104/base'
    config = json.loads((previous / 'config.json').read_text())
    assert config['prompt_protocol'].startswith(PROMPT_VERSION + ':')
    assert config['max_tokens'] == 65536 and config['wall_seconds'] == 180 and config['temperature'] == 0
    assert Path(config['model']).resolve() == root / 'model_cache/Qwen3-VL-4B-Instruct'
    reused = []
    for record in rows(previous / 'predictions.jsonl'):
        key = record['id'], record['modality']
        assert record['input_sha256'] == digest(tasks[key][1])
        assert record['gold'] == all_labels[record['id']]['answer']
        assert record['arm'] == 'base'
        reused.append(record)
    assert len(reused) == 264
    write_rows(work / 'data/base_neutral_verified_predictions.jsonl', reused)
    save(work / 'data/base_reuse_manifest.json', dict(created_at=now(), rows=264,
        source=str(previous), source_sha256=sha(previous / 'predictions.jsonl'),
        source_config_sha256=sha(previous / 'config.json'), same_input_and_label_checked=True,
        same_generation_protocol_checked=True, no_legacy_prompt_outputs_reused=True))
    expected = rows(work / 'data/recall_dev.jsonl')
    expected += [dict(id=r['id'], kind='independent_instruction', messages=[r['messages'][0]], expected=r['answer'])
                 for r in rows(work / 'data/retention_dev.jsonl')[:48]]
    earlier = reference / 'diagnostics/base'
    previous_recall = rows(earlier / 'predictions.jsonl')
    lookup = {r['id']: r for r in expected}
    assert len(previous_recall) == len(expected) == 198
    for r in previous_recall:
        t = lookup[r['id']]
        assert all(r[k] == t[k] for k in ['kind', 'messages', 'expected'])
        assert r['arm'] == 'base' and r['diagnostic_max_tokens'] == 256
    write_rows(work / 'diagnostics/base/predictions.jsonl', previous_recall)
    shutil.copy2(earlier / 'summary.json', work / 'diagnostics/base/summary.json')
    save(work / 'diagnostics/base/import_manifest.json', dict(created_at=now(), rows=198,
        source=str(earlier), source_sha256=sha(earlier / 'predictions.jsonl'), all_probe_messages_and_targets_verified=True))


def metric(group):
    assert group
    return dict(n=len(group), correct=sum(r['raw_predicted'] == r['gold'] for r in group),
        accuracy=sum(r['raw_predicted'] == r['gold'] for r in group) / len(group),
        invalid=sum(r['raw_predicted'] is None for r in group),
        invalid_rate=sum(r['raw_predicted'] is None for r in group) / len(group),
        answer_counts=dict(Counter(str(r['raw_predicted']) for r in group)),
        stop_reasons=dict(Counter(r['stop_reason'] for r in group)))


def summarize(work, manifest):
    reports = {}
    dev = set(manifest['prior_development_qids'])
    for arm in ['atomic_replay', 'base']:
        records = rows(work / 'evaluation' / arm / 'predictions.jsonl')
        assert len(records) == 7320
        groups = {}
        for mod in ['T', 'V']:
            for kind in ['original', 'rotation']:
                subset = [r for r in records if r['modality'] == mod and ((r['probe_id'] == 'rotation') == (kind == 'rotation'))]
                assert len(subset) == (3628 if kind == 'original' else 32)
                item = dict(all=metric(subset))
                if kind == 'original':
                    item.update(prior_development50=metric([r for r in subset if r['qid'] in dev]),
                                other1764=metric([r for r in subset if r['qid'] not in dev]),
                                direct=metric([r for r in subset if r['probe_id'] == 'direct']),
                                statement=metric([r for r in subset if r['probe_id'] == 'statement']))
                groups[mod + '/' + kind] = item
        recall = rows(work / f'diagnostics/{arm}/predictions.jsonl')
        assert len(recall) == 198
        diagnostics = {}
        for kind in ['trained_attribute_recall', 'exact_training_claim_recall', 'neutral_claim_recall', 'independent_instruction']:
            subset = [r for r in recall if r['kind'] == kind]
            item = dict(n=len(subset), length_limited=sum(r['finish_reason'] == 'length' for r in subset))
            if kind == 'independent_instruction':
                item.update(correct=sum(r['correct'] for r in subset),
                            accuracy=sum(r['correct'] for r in subset) / len(subset))
            else:
                item.update(strict_target_span_hit_rate=sum(r['target_span_hit'] for r in subset) / len(subset),
                            exact_string_match_rate=sum(r['response'] == r['expected'] for r in subset) / len(subset),
                            mean_token_f1=sum(r['token_f1'] for r in subset) / len(subset))
            diagnostics[kind] = item
        reports[arm] = dict(mcq=groups, recall50=diagnostics)
    a = {(r['id'], r['modality']): r for r in rows(work / 'evaluation/atomic_replay/predictions.jsonl')}
    b = {(r['id'], r['modality']): r for r in rows(work / 'evaluation/base/predictions.jsonl')}
    assert set(a) == set(b)
    assert all(a[k]['input_sha256'] == b[k]['input_sha256'] and a[k]['gold'] == b[k]['gold'] for k in a)
    save(work / 'comparison.json', dict(status='complete', reports=reports, completed_at=now(),
        same_model_inputs_and_labels_verified=True, primary='Raw generation, v4 readout; supplementary finalization excluded',
        limits=manifest['limitations']))
    lines = ['# 全实体 TEXT 重复 10 次训练与中性 v5 评测结果', '',
        '训练为 ms-swift LoRA、3 epoch；1,814 实体、每实体 2 条独立学习问答、每条每轮重复 10 次。', '',
        '| 模型 | 文字原题准确率 | 文字原题无效率 | 图片原题准确率 | 图片原题无效率 |',
        '|---|---:|---:|---:|---:|']
    for arm, label in [('base', '基座'), ('atomic_replay', 'TEXT LoRA')]:
        t, v = (reports[arm]['mcq'][m + '/original']['all'] for m in ['T', 'V'])
        lines.append(f'| {label} | {t["accuracy"]:.2%} | {t["invalid_rate"]:.2%} | {v["accuracy"]:.2%} | {v["invalid_rate"]:.2%} |')
    lines += ['', '每模态 3,628 道配对原题，选项轮换单列于 comparison.json。此前 50 个开发实体与其余 1,764 个实体也分别统计。', '',
        '回忆抽检只覆盖此前固定 50 个实体，字面片段命中不等于语义准确率。事实与时间有效性限制继承源候选数据。', '',
        '主成绩来自原始回答，约束补选另存；同题同提示基座对照包含验证后复用的 264 条既有生成。', '',
        '本轮只训练 TEXT，结果为 T2T / T2V，不包含新 VISION 学习模型。', '',
        '完整逐题生成见 evaluation/，原始 adapter 检查见 diagnostics/，配置与来源见 experiment.json、data/manifest.json。', '']
    (work / 'RESULTS_ZH.md').write_text('\n'.join(lines))


def run(args):
    import torch
    torch.set_num_threads(4)
    root, work = args.root.resolve(), args.work.resolve()
    assert socket.gethostname() == 'softengine-KVM'
    work.mkdir(exist_ok=True, parents=True)
    lock = (work / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def status(stage, **extra):
        save(work / 'status.json', dict(stage=stage, controller_pid=os.getpid(), host=socket.gethostname(),
            gpu='0', updated_at=now(), **extra))
        print(json.dumps(dict(stage=stage, time=now(), **extra), ensure_ascii=False), flush=True)
    status('preparing_learning')
    if (work / 'data/manifest.json').exists():
        manifest = json.loads((work / 'data/manifest.json').read_text())
        for name, expected in manifest['files'].items():
            assert sha(work / name) == expected, name
    else:
        manifest = prepare_learning(root, work)
    status('preparing_evaluation')
    if not (work / 'data/base_reuse_manifest.json').exists():
        prepare_evaluation(root, work, manifest)
    snapshots = work / 'source_snapshots'
    if not snapshots.exists():
        snapshots.mkdir()
        for name in ['run_entity_all_repeat10_v5.py', 'prepare_entity_atomic_pilot.py', 'run_entityqa_lora.py',
                     'entityqa_common.py', 'entityqa_prompt_neutral_v5.py', 'evaluate_entityqa_neutral_v5.py',
                     'evaluate_entityqa_fullfact_v2.py', 'answer_readout.py', 'answer_readout_entity_v2.py',
                     'answer_readout_entity_v3.py', 'answer_readout_entity_v4.py', 'test_entity_all_repeat10.py']:
            shutil.copy2(root / 'scripts' / name, snapshots / name)
        recall = (root / 'scripts/evaluate_entity_atomic_recall.py').read_text().replace(
            'from answer_readout_entity_v3 import parse_answer', 'from answer_readout_entity_v4 import parse_answer')
        (snapshots / 'evaluate_recall.py').write_text(recall)
    hashes = {p.name: sha(p) for p in snapshots.glob('*.py')}
    base = root / 'model_cache/Qwen3-VL-4B-Instruct'
    data = work / 'evaluation_data'
    inventory = json.loads((data / 'package_inventory.json').read_text())
    for name, expected in inventory['sha256'].items():
        assert sha(data / name) == expected, name
    plan = dict(status='running', created_at=now(), host=socket.gethostname(), gpu='0', base=str(base),
        training_framework='ms-swift swift sft', modality='TEXT only', knowledge_units=1814,
        unique_factual_QA=3628, repetitions_per_QA_per_epoch=10, exposures_per_QA_total=30,
        epochs=3, learning_rate=1e-4, rank=64, alpha=128, dropout=0.05, effective_batch=16,
        factual_rows_per_epoch=36280, generic_rows_per_epoch=9072, total_rows_per_epoch=45352,
        planned_optimizer_steps=8505, checkpoint_selection='Fixed final epoch 3, never test-selected',
        starts_from_clean_base=True, train_sha256=sha(work / 'atomic_replay/data/train_text.jsonl'),
        manifest_sha256=sha(work / 'data/manifest.json'), evaluation_inventory_sha256=sha(data / 'package_inventory.json'),
        source_sha256=hashes, prompt_version=PROMPT_VERSION, cot_suffix=COT_SUFFIX,
        evaluation=dict(original_T2T_questions=3628, original_T2V_questions=3628,
            rotations_per_modality=32, tasks_per_model=7320, brief_cot=True, max_tokens=65536,
            wall_seconds=180, readout='v4', primary='raw accuracy and invalid rate', base_reuse=264,
            original_adapter_recall_probe_entities=50, diagnostic_max_tokens=256),
        limitations=manifest['limitations'], launch_additional_training=False)
    if (work / 'experiment.json').exists():
        prior = json.loads((work / 'experiment.json').read_text())
        for key in ['train_sha256', 'manifest_sha256', 'evaluation_inventory_sha256', 'source_sha256', 'cot_suffix']:
            assert prior[key] == plan[key], key
        plan = prior
    else:
        save(work / 'experiment.json', plan)
    if args.prepare_only:
        status('prepared', training_rows=45352, optimizer_steps=8505)
        return
    tmp = Path('/tmp/xm-entity-all-repeat10-v5')
    tmp.mkdir(mode=0o700, exist_ok=True)
    assert tmp.stat().st_uid == os.getuid()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1', MODELSCOPE_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', MAX_PIXELS='262144',
        NPROC_PER_NODE='1', MASTER_ADDR='127.0.0.1', MASTER_PORT='30378', TMPDIR=str(tmp),
        VLLM_RPC_BASE_PATH=str(tmp), VLLM_WORKER_MULTIPROC_METHOD='spawn', PYTHONUNBUFFERED='1',
        HF_DATASETS_CACHE=str(work / 'dataset_cache'), PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    def wait_gpu(stage):
        while True:
            try:
                idle('0')
                return
            except RuntimeError as error:
                status('waiting_for_free_gpu', next_stage=stage, reason=str(error))
                time.sleep(15)
    def execute(cmd, stage):
        wait_gpu(stage)
        save(work / (stage + '_command.json'), cmd)
        with (work / (stage + '.log')).open('a') as stream:
            proc = subprocess.Popen(cmd, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
        status(stage, child_pid=proc.pid, log=str(work / (stage + '.log')))
        if proc.wait():
            raise RuntimeError(stage + ' failed; inspect log; existing outputs retained')
        for _ in range(15):
            try:
                idle('0')
                break
            except RuntimeError:
                time.sleep(2)
    # Training and final checkpoint validation before any new scoring.
    cmd, folder = command(SimpleNamespace(root=root, work=work / 'atomic_replay', model=base), 'TEXT', False)
    try:
        ck = checkpoint(folder)
    except (RuntimeError, FileNotFoundError, AssertionError):
        if folder.exists() and any(folder.iterdir()):
            raise RuntimeError('Incomplete training folder; inspect before any resume, do not overwrite')
        execute(cmd, 'training_TEXT')
        ck = checkpoint(folder)
    # Original unmerged adapter, same fixed 50 recall entities as the smaller trial.
    diag = work / 'diagnostics/atomic_replay/summary.json'
    if not diag.exists() or json.loads(diag.read_text())['status'] != 'complete':
        execute([str(root / '.venv/bin/python'), '-u', str(snapshots / 'evaluate_recall.py'),
            '--work', str(work), '--model', str(base), '--arm', 'atomic_replay'], 'diagnostics_atomic_replay')
    # In-memory conversion avoids the repeated cold disk loading stalls in E23.
    serving = Path('/dev/shm') / (work.name + '-TEXT-serving')
    tuned_summary = work / 'evaluation/atomic_replay/summary.json'
    if not tuned_summary.exists() or json.loads(tuned_summary.read_text())['status'] != 'complete':
        wait_gpu('preparing_TEXT_serving')
        status('preparing_TEXT_serving', temporary_model=str(serving))
        assert shutil.disk_usage('/dev/shm').free > 11 * 2**30 or (serving / 'merge_manifest.json').exists()
        if serving.exists():
            assert serving.stat().st_uid == os.getuid() and (serving / 'merge_manifest.json').exists()
        merge(base, ck, serving)
        metadata = work / 'serving_artifacts'
        metadata.mkdir(exist_ok=True)
        shutil.copy2(serving / 'merge_manifest.json', metadata / 'merge_manifest.json')
        for p in serving.iterdir():
            if p.suffix in ['.json', '.jinja', '.txt', '.model']:
                shutil.copy2(p, metadata / p.name)
        save(metadata / 'serving_record.json', dict(created_at=now(), temporary_model=str(serving),
            method='Existing deterministic BF16 base-plus-LoRA conversion in tmpfs; persistent adapter and hashes retained',
            persistent_adapter=str(ck), persistent_base=str(base), retraining=False))
    # The trained model is evaluated first, followed by the complete base comparison.
    for arm in ['atomic_replay', 'base']:
        dest = work / 'evaluation' / arm
        report = dest / 'summary.json'
        if report.exists() and json.loads(report.read_text())['status'] == 'complete':
            continue
        model = serving if arm == 'atomic_replay' else base
        cmd = [str(root / '.venv-vllm/bin/python'), '-u', str(snapshots / 'evaluate_entityqa_neutral_v5.py'),
            '--data', str(data), '--model', str(model), '--output', str(dest),
            '--arm', 'TEXT' if arm == 'atomic_replay' else 'base', '--gpu', '0',
            '--max-tokens', '65536', '--wall-seconds', '180']
        if arm == 'base':
            cmd += ['--import-predictions', str(work / 'data/base_neutral_verified_predictions.jsonl')]
        execute(cmd, 'evaluating_' + arm)
        if arm == 'atomic_replay' and serving.exists():
            expected = json.loads((work / 'serving_artifacts/merge_manifest.json').read_text())['files']
            assert all(sha(serving / name) == h for name, h in expected.items())
            assert all(p.is_file() and not p.is_symlink() and p.stat().st_uid == os.getuid() for p in serving.iterdir())
            shutil.rmtree(serving)
            save(work / 'serving_artifacts/cleanup.json', dict(removed_after_successful_inference=True, updated_at=now()))
    status('summarizing')
    summarize(work, manifest)
    assert all(sha(snapshots / name) == h for name, h in plan['source_sha256'].items())
    plan.update(status='complete', completed_at=now())
    save(work / 'experiment.json', plan)
    status('complete', results='comparison.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        save(args.work / 'status.json', dict(stage='failed', error=repr(error), updated_at=now(), controller_pid=os.getpid()))
        raise
