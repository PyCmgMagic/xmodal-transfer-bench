#!/usr/bin/env python3
"""Queue GLM TEXT LoRA with both existing factual QA types, each seen once per epoch."""
import argparse
from collections import Counter
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

from entityqa_common import digest, now, rows, save, sha, write_rows
from entityqa_prompt_neutral_v5 import eval_payload
from evaluate_entity_glm46_full import generation_details
from run_entity_glm46_full import BASE, training_command
from run_entityqa_lora import checkpoint as check_adapter, idle

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'runs/entity1814_repeat10_glm46_lora_ep3_neutral_v5_20261105'
WORK = ROOT / 'runs/entity1814_repeat1_glm46_lora_ep3_neutral_v5_20261106'
EVAL_MAX_TOKENS = 1024
SOURCE_FILES = [
    'run_entity_glm46_once.py', 'run_entity_glm46_full.py', 'run_entityqa_lora.py',
    'evaluate_entity_glm46_full.py', 'entityqa_common.py', 'entityqa_prompt_neutral_v5.py',
    'answer_readout.py', 'answer_readout_entity_v2.py', 'answer_readout_entity_v3.py',
    'answer_readout_entity_v4.py',
]
MATCHED_EVAL_KEYS = [
    'model', 'dataset_inventory_sha256', 'input_sha256', 'input_count', 'prompt_protocol',
    'stop_sequences', 'temperature', 'seed', 'thinking', 'vision_max_pixels',
    'readout_version', 'batch_size',
]


def single_pass_data(core, retention_pool):
    """Keep both frozen QA types exactly once; preserve the factual/instruction ratio."""
    pairs = Counter((r['knowledge_id'], r['variant']) for r in core)
    assert pairs and set(pairs.values()) == {1}, 'Repeated entity QA in the source'
    assert set(r['variant'] for r in core) == {'fact_0', 'attribute_0'}
    by_entity = Counter(r['knowledge_id'] for r in core)
    assert set(by_entity.values()) == {2}
    assert len({r['qid'] for r in core}) == len(by_entity)
    assert len({r['id'] for r in core}) == len(core)
    assert all([m['role'] for m in r['messages']] == ['user', 'assistant'] for r in core)
    assert all(not r.get('images') for r in core)
    quota = math.ceil(len(core) / 16)
    chosen, counts = [], Counter()
    for row in retention_pool:
        label = row['answer']
        assert label in 'ABCD'
        if counts[label] < quota:
            chosen.append(copy.deepcopy(row))
            counts[label] += 1
    assert counts == dict.fromkeys('ABCD', quota), 'Insufficient balanced retention pool'
    metadata = copy.deepcopy(core) + chosen
    assert len({r['id'] for r in metadata}) == len(metadata)
    assert len({digest(r['messages']) for r in core}) == len(core), 'Repeated factual QA content'
    # Different entities can share a name; preserve the previous factual texts
    # and record ambiguous prompts rather than silently rewriting this arm.
    generic_prompts = [r['messages'][0]['content'] for r in chosen]
    assert len(set(generic_prompts)) == len(generic_prompts)
    assert set(generic_prompts).isdisjoint(r['messages'][0]['content'] for r in core)
    training = [dict(id=r['id'], messages=copy.deepcopy(r['messages'])) for r in metadata]
    return training, metadata, chosen


def prepare(work):
    work = work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    parent_manifest = json.loads((PARENT / 'data/manifest.json').read_text())
    for relative, expected in parent_manifest['files'].items():
        assert sha(PARENT / relative) == expected, relative
    source_plan = json.loads((PARENT / 'experiment.json').read_text())
    assert (BASE / 'download_complete.json').is_file()
    assert sha(BASE / 'download_manifest.json') == source_plan['model_download_manifest_sha256']
    core = rows(PARENT / 'data/entity_unique_training.jsonl')
    retention_pool = rows(PARENT / 'data/retention_training.jsonl')
    learning = rows(PARENT / 'data/learning_inputs.jsonl')
    train, metadata, retention = single_pass_data(core, retention_pool)
    assert len(core) == 3628 and len(learning) == 1814
    assert len(retention) == 908 and len(train) == 4536
    assert {r['id'] for r in learning} == {r['knowledge_id'] for r in core}
    # Verify that the question/answer text is unchanged from the earlier SFT source.
    qa_path = ROOT / 'runs/entity1025_repaired_attrqa_lora_ep3_20261103/data/qa_pairs.jsonl'
    assert sha(qa_path) == parent_manifest['training_source_sha256'][str(qa_path)]
    qas = {r['id']: r for r in rows(qa_path)}
    for row in core:
        if row['variant'] == 'attribute_0':
            qa = qas[row['knowledge_id']]
            assert row['messages'] == [dict(role='user', content=qa['question_text']),
                                       dict(role='assistant', content=qa['answer'])]
    write_rows(work / 'data/train_text.jsonl', train)
    write_rows(work / 'data/training_rows_with_metadata.jsonl', metadata)
    write_rows(work / 'data/entity_unique_training.jsonl', core)
    write_rows(work / 'data/retention_training.jsonl', retention)
    write_rows(work / 'data/learning_inputs.jsonl', learning)
    prompt_counts = Counter(r['messages'][0]['content'] for r in core)
    ambiguous_prompts = {text for text, count in prompt_counts.items() if count > 1}
    save(work / 'data/inherited_prompt_ambiguity.json', dict(
        duplicate_prompt_groups=len(ambiguous_prompts),
        rows=[r for r in core if r['messages'][0]['content'] in ambiguous_prompts],
        action='Preserve both existing QA targets; no new repetition was introduced.'))
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(str(BASE), local_files_only=True)
    lengths = []
    for row in train:
        text = processor.apply_chat_template(row['messages'], tokenize=False,
            add_generation_prompt=False, enable_thinking=False)
        token_ids = processor.tokenizer.encode(text, add_special_tokens=False)
        lengths.append(len(token_ids))
    assert min(lengths) > 1 and max(lengths) <= 2048
    save(work / 'data/token_audit.json', dict(rows=len(train),
        max_formatted_tokens=max(lengths), truncation_required=False))
    from datasets import load_dataset
    loaded = load_dataset('json', data_files=str(work / 'data/train_text.jsonl'),
        cache_dir=str(work / 'loader_validation_cache'), split='train')
    assert set(loaded.column_names) == {'id', 'messages'} and len(loaded) == len(train)
    assert all(dict(a) == b for a, b in zip(loaded, train))
    save(work / 'data/loader_validation.json', dict(rows=len(loaded),
        actual_json_loader_passed=True, all_messages_and_order_verified=True))
    steps = 3 * math.ceil(math.ceil(len(train) / 4) / 4)
    assert steps == 852
    manifest = dict(version='entity1814-twoQA-repeat1-glm-v1', created_at=now(),
        knowledge_units=1814, unique_factual_QA=3628,
        variant_counts=dict(Counter(r['variant'] for r in core)),
        repetitions_per_QA_per_epoch=1, exposures_per_QA_over_3_epochs=3,
        factual_rows_per_epoch=3628, independent_instruction_rows_per_epoch=908,
        total_rows_per_epoch=4536, planned_optimizer_steps=steps, epochs=3,
        test_data_used_for_training=False,
        prior_development_qids=parent_manifest['prior_development_qids'],
        source_run=str(PARENT), core_source_sha256=sha(PARENT / 'data/entity_unique_training.jsonl'),
        source_qa_sha256=sha(qa_path),
        retention_selection='First 227 rows per answer letter from the frozen previous pool; 908 total',
        files={str(p.relative_to(work)): sha(p) for p in sorted((work / 'data').glob('*.jsonl'))})
    save(work / 'data/manifest.json', manifest)
    # Freeze training before opening any formal test questions or labels.
    for relative, expected in manifest['files'].items():
        assert sha(work / relative) == expected
    inventory = json.loads((PARENT / 'evaluation_data/package_inventory.json').read_text())
    for relative, expected in inventory['sha256'].items():
        source, target = PARENT / 'evaluation_data' / relative, work / 'evaluation_data' / relative
        assert sha(source) == expected
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert sha(target) == expected
        else:
            shutil.copy2(source, target)
    target = work / 'evaluation_data/package_inventory.json'
    if target.exists():
        assert sha(target) == sha(PARENT / 'evaluation_data/package_inventory.json')
    else:
        shutil.copy2(PARENT / 'evaluation_data/package_inventory.json', target)
    tasks = []
    for mod, filename in [('T', 'eval_text.jsonl'), ('V', 'eval_multimodal.jsonl')]:
        evaluation = rows(work / 'evaluation_data/shared' / filename)
        assert len(evaluation) == 3660
        for row in evaluation:
            payload = eval_payload(row, work / 'evaluation_data')
            tasks.append(dict(id=row['id'], qid=row['qid'], modality=mod,
                probe_id='rotation' if '-rotation-' in row['id'] else
                    ('direct' if row['id'].endswith('-direct') else 'statement'), payload=payload))
    # The evaluator includes probe_id in its hash; match its source labels here.
    labels = {r['id']: r for r in rows(work / 'evaluation_data/audit/eval_labels.jsonl')}
    for task in tasks:
        task['probe_id'] = labels[task['id']]['probe_id']
    eval_config = json.loads((PARENT / 'evaluation/atomic_replay/config.json').read_text())
    assert digest(tasks) == eval_config['input_sha256']
    snapshot = work / 'source_snapshots'
    snapshot.mkdir(exist_ok=True)
    hashes = {name: sha(ROOT / 'scripts' / name) for name in SOURCE_FILES}
    old = json.loads((work / 'experiment.json').read_text()) if (work / 'experiment.json').exists() else None
    if old:
        assert old['train_sha256'] == sha(work / 'data/train_text.jsonl')
        assert old['source_sha256'] == hashes, 'Source changed after the run was prepared'
    for name in SOURCE_FILES:
        dest = snapshot / name
        if dest.exists():
            assert sha(dest) == hashes[name]
        else:
            shutil.copy2(ROOT / 'scripts' / name, dest)
    plan = old or dict(status='prepared', created_at=now(), model='GLM-4.6V-Flash',
        model_path=str(BASE), starts_from_clean_base=True, modality='TEXT',
        framework='ms-swift 4.5.3; .venv-glm', tuner='LoRA',
        epochs=3, rank=64, alpha=128, dropout=0.05, learning_rate=1e-4,
        micro_batch=4, gradient_accumulation=4, effective_batch=16, max_length=2048,
        knowledge_units=1814, factual_rows_per_epoch=3628, generic_rows_per_epoch=908,
        total_rows_per_epoch=4536, repetitions_per_QA_per_epoch=1,
        planned_optimizer_steps=852, checkpoint_selection='fixed final epoch 3',
        train_sha256=sha(work / 'data/train_text.jsonl'), source_sha256=hashes,
        parent_run=str(PARENT), queue_dependency=str(PARENT / 'status.json'),
        evaluation=dict({k: eval_config[k] for k in MATCHED_EVAL_KEYS}, max_tokens=EVAL_MAX_TOKENS),
        baseline='Reuse verified paired GLM base inputs; report output-cap differences explicitly.',
        comparison='GLM base, GLM two-QA repeat10, GLM two-QA repeat1',
        limitations=[
            'QA content is identical; repetitions fall from 10 to 1 for both existing types.',
            'Generic instruction rows also fall from 9072 to 908 to retain about 20% of the training mix.',
            'Training exposure and optimizer steps differ; this is not an equal-compute comparison.',
            'Source factual/temporal validity and prior development-entity limitations remain.',
            'One inherited fact_0 prompt refers to two different entities named Jack Miller; it is preserved and audited.',
            'Native thinking remains disabled; the unchanged prompt requests brief reasoning.',
            'The new output cap is 1024; historical results retain their original caps and are not a matched-cap control.',
        ])
    save(work / 'experiment.json', plan)
    (work / 'README_ZH.md').write_text(
        '# GLM：保留两类问答、每轮各学一次\n\n'
        '1,814 个实体，每实体保留原来的 fact_0 与 attribute_0 两条 SFT 问答，共 3,628 条。'
        '不改写、不再重复 10 次。另从旧通用指令池按答案平衡取 908 条，维持约 20% 比例；'
        '每轮共 4,536 条，3 epoch，预计 852 个优化步。\n\n'
        '从原始 GLM-4.6V-Flash 重新训练 TEXT LoRA，视觉编码器和对齐层冻结。'
        '当前 GLM 基座评测全部完成且 GPU 空闲后才做独立训练 smoke，再执行正式训练与 T2T/T2V 评测。'
        '每模态 3,628 道原题及 32 道轮换题；保留原提示和答案解析，输出上限改为 1,024 token。'
        '比较时校验相同输入与标签，并明确披露历史结果的不同上限；不重复运行同一基座。\n\n'
        '重复次数、通用指令数量及总训练计算量同时降低；结果不能解释成等计算量消融。\n')
    return plan


def command(work, smoke=False):
    cmd = training_command(work)
    changes = {'--dataset': str(work / 'data/train_text.jsonl'),
               '--output_dir': str(work / ('smoke/TEXT' if smoke else 'train/TEXT')),
               '--master_port': '30386'}
    for option, value in changes.items():
        assert cmd.count(option) == 1
        cmd[cmd.index(option) + 1] = value
    if smoke:
        cmd += ['--max_steps', '2', '--save_strategy', 'steps', '--save_steps', '2',
                '--warmup_ratio', '0']
    return cmd


def validate(work, smoke=False):
    import torch
    torch.set_num_threads(4)
    folder = work / ('smoke/TEXT' if smoke else 'train/TEXT')
    if smoke:
        from safetensors.torch import load_file
        checkpoint = folder / 'checkpoint-2'
        state = json.loads((checkpoint / 'trainer_state.json').read_text())
        assert state['global_step'] == 2
        assert all(math.isfinite(float(r[k])) for r in state['log_history']
                   for k in ['loss', 'grad_norm'] if k in r)
        tensors = load_file(str(checkpoint / 'adapter_model.safetensors'), device='cpu')
        assert tensors and all(torch.isfinite(t).all() for t in tensors.values())
        assert all('.language_model.' in k and '.visual.' not in k for k in tensors)
        assert any(torch.count_nonzero(t).item() for k, t in tensors.items() if '.lora_B.' in k)
        save(folder / 'validation.json', dict(status='pass', global_step=2,
            adapter_sha256=sha(checkpoint / 'adapter_model.safetensors'), finite=True,
            language_only=True, nonzero_lora_B=True))
    else:
        checkpoint = check_adapter(folder)
        state = json.loads((checkpoint / 'trainer_state.json').read_text())
        assert state['global_step'] == 852 and checkpoint.name == 'checkpoint-852'
        args = json.loads((checkpoint / 'args.json').read_text())
        assert args['dataset'] == [str(work / 'data/train_text.jsonl')]
        assert not args.get('adapters') and not args.get('resume_from_checkpoint')
    return checkpoint


def compare(work):
    reports = {}
    new_config = json.loads((work / 'evaluation/atomic_replay/config.json').read_text())
    for label, run, arm in [('GLM 基座', PARENT, 'base'),
                            ('GLM 两类 QA 各重复10次', PARENT, 'atomic_replay'),
                            ('GLM 两类 QA 各学1次', work, 'atomic_replay')]:
        folder = run / 'evaluation' / arm
        config = json.loads((folder / 'config.json').read_text())
        for key in MATCHED_EVAL_KEYS:
            assert config[key] == new_config[key], (label, key)
        if arm == 'base':
            assert config['adapter'] is None
        summary = json.loads((folder / 'summary.json').read_text())
        assert summary['status'] == 'complete'
        records = rows(folder / 'predictions.jsonl')
        assert len(records) == len({(r['id'], r['modality']) for r in records}) == 7320
        reports[label] = dict(groups=summary['groups'], source=str(folder),
            predictions_sha256=sha(folder / 'predictions.jsonl'),
            generation_protocol=generation_details(folder))
    source_records = {(r['id'], r['modality']): r
                      for r in rows(PARENT / 'evaluation/base/predictions.jsonl')}
    for r in rows(work / 'evaluation/atomic_replay/predictions.jsonl'):
        other = source_records[(r['id'], r['modality'])]
        assert r['input_sha256'] == other['input_sha256'] and r['gold'] == other['gold']
    save(work / 'comparison.json', dict(status='complete', reports=reports, completed_at=now(),
        comparison_limitation='Paired inputs and prompts; output caps differ across and within historical arms.'))
    lines = ['# GLM 两类问答每轮各学一次：结果', '',
        '| 模型 | 文字准确率 | 文字无效率 | 图片准确率 | 图片无效率 |',
        '|---|---:|---:|---:|---:|']
    for label, report in reports.items():
        t, v = [report['groups'][m + '/original']['all'] for m in 'TV']
        lines.append(f'| {label} | {t["accuracy"]:.2%} | {t["invalid_rate"]:.2%} | '
                     f'{v["accuracy"]:.2%} | {v["invalid_rate"]:.2%} |')
    lines += ['', '每模态 3,628 道原题，32 道轮换题单列于 comparison.json。', '',
        '三组同题、同提示；输出上限不同，不构成严格相同推理条件的对照。新训练保留两类原有 QA，不重复；'
        '3,628 条实体 QA + 908 条通用指令，3 epoch。重复次数与总计算量均减少。', '']
    for label, report in reports.items():
        lines.append(f'{label}：各生成上限的题数 {report["generation_protocol"]["generation_cap_counts"]}。')
    (work / 'RESULTS_ZH.md').write_text('\n'.join(lines))


def controller(work, parent=PARENT):
    assert socket.gethostname() == 'softengine-KVM'
    work, parent = work.resolve(), parent.resolve()
    assert parent == PARENT
    work.mkdir(parents=True, exist_ok=True)
    lock = (work / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = prepare(work)

    def status(stage, **extra):
        save(work / 'status.json', dict(stage=stage, controller_pid=os.getpid(),
            host=socket.gethostname(), gpu='0', updated_at=now(), **extra))

    while True:
        previous = json.loads((parent / 'status.json').read_text())
        if previous.get('stage') == 'complete':
            assert json.loads((parent / 'comparison.json').read_text())['status'] == 'complete'
            assert json.loads((parent / 'evaluation/base/summary.json').read_text())['status'] == 'complete'
            break
        status('queued_after_GLM_base', dependency=str(parent),
               dependency_stage=previous.get('stage'), training_started=False,
               training_rows_per_epoch=4536, factual_rows_per_epoch=3628, epochs=3)
        time.sleep(30)
    for name, expected in plan['source_sha256'].items():
        assert sha(ROOT / 'scripts' / name) == expected, name
    for relative, expected in json.loads((work / 'data/manifest.json').read_text())['files'].items():
        assert sha(work / relative) == expected, relative
    tmp = Path('/tmp/xm-entity-glm46-once')
    tmp.mkdir(mode=0o700, exist_ok=True)
    assert tmp.stat().st_uid == os.getuid()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1', MODELSCOPE_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
        MAX_PIXELS='262144', TMPDIR=str(tmp), PYTHONUNBUFFERED='1',
        HF_DATASETS_CACHE=str(work / 'dataset_cache'), NPROC_PER_NODE='1',
        MASTER_ADDR='127.0.0.1', MASTER_PORT='30386',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')

    def execute(cmd, stage):
        while True:
            try:
                idle('0')
                break
            except RuntimeError as error:
                status('waiting_for_free_gpu', next_stage=stage, reason=str(error))
                time.sleep(30)
        log = work / (stage + '.log')
        with log.open('a') as stream:
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        status(stage, child_pid=process.pid, log=str(log))
        result = process.wait()
        if result:
            raise RuntimeError(f'{stage} exited with {result}; inspect {log}')

    for smoke in [True, False]:
        folder = work / ('smoke/TEXT' if smoke else 'train/TEXT')
        expected = folder / ('checkpoint-2' if smoke else 'checkpoint-852')
        if not expected.exists():
            if folder.exists() and any(folder.iterdir()):
                raise RuntimeError(f'Preserving interrupted outputs: {folder}')
            execute(command(work, smoke), 'smoke_training' if smoke else 'training_GLM_TEXT')
        checkpoint = validate(work, smoke)
    target = work / 'evaluation/atomic_replay/summary.json'
    if not target.exists() or json.loads(target.read_text()).get('status') != 'complete':
        execute([str(ROOT / '.venv-glm/bin/python'), '-u',
            str(work / 'source_snapshots/evaluate_entity_glm46_full.py'),
            '--root', str(ROOT), '--work', str(work), '--arm', 'glm46_lora',
            '--checkpoint', str(checkpoint), '--max-tokens', str(plan['evaluation']['max_tokens'])],
            'evaluating_glm46_lora')
    compare(work)
    plan.update(status='complete', completed_at=now())
    save(work / 'experiment.json', plan)
    status('complete', results=str(work / 'RESULTS_ZH.md'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=WORK)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    try:
        if args.prepare_only:
            prepare(args.work)
        else:
            controller(args.work)
    except Exception as error:
        save(args.work / 'status.json', dict(stage='failed', controller_pid=os.getpid(),
            error=repr(error), updated_at=now()))
        raise
