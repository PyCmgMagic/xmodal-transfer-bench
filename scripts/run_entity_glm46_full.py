#!/usr/bin/env python3
"""Train and evaluate GLM-4.6V-Flash on the frozen 1,814-entity TEXT plan."""
import argparse
from collections import Counter
import copy
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

from entityqa_common import rows, save, write_rows, sha, digest, now
from evaluate_entity_glm46_full import generation_details

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'runs/entity1814_repeat10_lora_ep3_neutral_v5_20261104'
WORK = ROOT / 'runs/entity1814_repeat10_glm46_lora_ep3_neutral_v5_20261105'
BASE = ROOT / 'model_cache/modelscope/models/ZhipuAI--GLM-4.6V-Flash/snapshots/master'


def prepare(work=WORK, max_tokens=6000):
    work = work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    assert (BASE / 'download_complete.json').is_file()
    parent_manifest = json.loads((PARENT / 'data/manifest.json').read_text())
    for name, expected in parent_manifest['files'].items():
        assert sha(PARENT / name) == expected, name

    for rel in ['atomic_replay/data/train_text.jsonl',
                'data/training_rows_with_metadata.jsonl', 'data/entity_unique_training.jsonl',
                'data/learning_inputs.jsonl', 'data/retention_training.jsonl',
                'data/retention_dev.jsonl', 'data/recall_dev.jsonl',
                'data/manifest.json', 'data/token_audit.json',
                'data/retention_generation_audit.json', 'data/base_reuse_manifest.json']:
        source, target = PARENT / rel, work / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
        assert sha(target) == sha(source), rel
    for rel in ['shared/eval_text.jsonl', 'shared/eval_multimodal.jsonl',
                'audit/eval_labels.jsonl', 'audit/inference_input_hashes.jsonl',
                'package_inventory.json']:
        source, target = PARENT / 'evaluation_data' / rel, work / 'evaluation_data' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
        assert sha(target) == sha(source), rel

    training = rows(work / 'atomic_replay/data/train_text.jsonl')
    metadata = rows(work / 'data/training_rows_with_metadata.jsonl')
    records = rows(work / 'data/learning_inputs.jsonl')
    retention = rows(work / 'data/retention_training.jsonl')
    assert len(training) == 45352 and len(metadata) == len(training)
    assert len(records) == len({r['id'] for r in records}) == 1814
    assert len({r['knowledge_id'] for r in metadata if 'knowledge_id' in r}) == 1814
    assert all(set(r) == {'id', 'messages'} and 'images' not in r for r in training)
    assert all(a['id'] == b['id'] and a['messages'] == b['messages'] for a, b in zip(training, metadata))
    exposures = Counter((r['knowledge_id'], r['variant']) for r in metadata if 'knowledge_id' in r)
    assert len(exposures) == 3628 and set(exposures.values()) == {10}
    assert len(retention) == 9072 and Counter(r['answer'] for r in retention) == dict.fromkeys('ABCD', 2268)
    labels = rows(work / 'evaluation_data/audit/eval_labels.jsonl')
    assert len(labels) == 3660
    for filename in ['eval_text.jsonl', 'eval_multimodal.jsonl']:
        selected = rows(work / 'evaluation_data/shared' / filename)
        assert len(selected) == 3660
        if filename == 'eval_multimodal.jsonl':
            assert all(Path(image).is_file() for row in selected for image in row['images'])

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(str(BASE), local_files_only=True)
    lengths = []
    for row in training:
        token_ids = processor.apply_chat_template(row['messages'], tokenize=True,
            add_generation_prompt=False, enable_thinking=False)
        if hasattr(token_ids, 'input_ids'):
            token_ids = token_ids.input_ids
        if hasattr(token_ids, 'tolist'):
            token_ids = token_ids.tolist()
        if token_ids and isinstance(token_ids[0], list):
            assert len(token_ids) == 1
            token_ids = token_ids[0]
        lengths.append(len(token_ids))
    assert max(lengths) <= 2048
    save(work / 'data/glm_token_audit.json', dict(rows=len(training),
        max_formatted_tokens=max(lengths), truncation_required=False,
        tokenizer='GLM-4.6V-Flash local tokenizer; glm4_5v, thinking disabled'))

    inventory = json.loads((work / 'evaluation_data/package_inventory.json').read_text())
    for rel, expected in inventory['sha256'].items():
        assert sha(work / 'evaluation_data' / rel) == expected, rel
    snapshot_dir = work / 'source_snapshots'
    snapshot_dir.mkdir(exist_ok=True)
    sources = ['run_entity_glm46_full.py', 'evaluate_entity_glm46_full.py',
               'entityqa_common.py', 'entityqa_prompt_neutral_v5.py',
               'answer_readout.py', 'answer_readout_entity_v2.py',
               'answer_readout_entity_v3.py', 'answer_readout_entity_v4.py']
    for name in sources:
        source, dest = ROOT / 'scripts' / name, snapshot_dir / name
        if source.resolve() != dest.resolve():
            shutil.copy2(source, dest)
    source_hashes = {p.name: sha(p) for p in snapshot_dir.glob('*.py')}
    plan = dict(status='prepared', created_at=now(), host=socket.gethostname(), gpu='0',
        model='GLM-4.6V-Flash', model_path=str(BASE), model_download_manifest_sha256=sha(BASE / 'download_manifest.json'),
        model_download_complete=json.loads((BASE / 'download_complete.json').read_text()),
        training_framework='ms-swift 4.5.3; isolated .venv-glm with Transformers 5.16.1',
        modality='TEXT only', knowledge_units=1814, unique_factual_QA=3628,
        repetitions_per_QA_per_epoch=10, exposures_per_QA_total=30,
        epochs=3, learning_rate=1e-4, rank=64, alpha=128, dropout=0.05,
        effective_batch=16, micro_batch=4, gradient_accumulation=4,
        factual_rows_per_epoch=36280, generic_rows_per_epoch=9072,
        total_rows_per_epoch=45352, planned_optimizer_steps=8505,
        starts_from_clean_base=True, checkpoint_selection='fixed final epoch 3',
        source_training_run=str(PARENT), source_training_sha256=sha(PARENT / 'atomic_replay/data/train_text.jsonl'),
        train_sha256=sha(work / 'atomic_replay/data/train_text.jsonl'),
        train_content_identical_to_Qwen_E24=True,
        evaluation=dict(original_T2T_questions=3628, original_T2V_questions=3628,
            rotations_per_modality=32, tasks_per_model=7320, prompt_version='entity-neutral-box-v5',
            prompt_suffix='Compare the options with what you know. Explain briefly and do not overthink. End with the final option letter in \\boxed{}.',
            native_thinking=False, max_tokens=max_tokens,
            stop_sequences=[r'\boxed{A}', r'\boxed{B}', r'\boxed{C}', r'\boxed{D}'],
            batch_size=4,
            parser='entity v4 plus GLM native boxed-letter token normalization',
            original_and_rotation_metrics_separate=True),
        source_sha256=source_hashes,
        limitations=['The 50 previously used entities are development entities; other 1764 are not certified untouched.',
            'Training data, sample count, repetition and optimizer schedule match Qwen E24; model size, tokenizer, chat template and output token budget differ.',
            'Native GLM thinking is disabled to avoid long loops; the prompt is unchanged, and short prompted reasoning remains requested.',
            'The inherited repaired facts have not been independently verified for factual/temporal validity.'])
    if (work / 'experiment.json').exists():
        existing = json.loads((work / 'experiment.json').read_text())
        for key in ['train_sha256', 'model_download_manifest_sha256']:
            assert existing[key] == plan[key], key
        plan = existing
        plan['evaluation']['native_thinking'] = False
        plan['evaluation']['max_tokens'] = max_tokens
        plan['evaluation']['stop_sequences'] = [r'\boxed{A}', r'\boxed{B}',
                                                r'\boxed{C}', r'\boxed{D}']
        plan['source_sha256'] = source_hashes
        plan['token_audit_sha256'] = sha(work / 'data/glm_token_audit.json')
        save(work / 'experiment.json', plan)
    else:
        plan['token_audit_sha256'] = sha(work / 'data/glm_token_audit.json')
        save(work / 'experiment.json', plan)
    plan['evaluation']['cap_scope'] = 'Pending generations only; completed predictions retain their original cap.'
    save(work / 'experiment.json', plan)
    save(work / 'status.json', dict(stage='prepared', controller_pid=os.getpid(), host=socket.gethostname(),
        updated_at=now(), training_rows=45352, optimizer_steps=8505))
    return plan


def training_command(work):
    output = work / 'train/TEXT'
    return [str(ROOT / '.venv-glm/bin/python'), '-m', 'torch.distributed.run',
        '--nproc_per_node', '1', '--master_addr', '127.0.0.1', '--master_port', '30381',
        str(ROOT / '.venv/lib/python3.10/site-packages/swift/cli/sft.py'),
        '--model', str(BASE), '--model_type', 'glm4v', '--template', 'glm4_5v',
        '--dataset', str(work / 'atomic_replay/data/train_text.jsonl'),
        '--tuner_type', 'lora', '--lora_rank', '64', '--lora_alpha', '128', '--lora_dropout', '0.05',
        '--target_modules', 'all-linear', '--freeze_vit', 'true', '--freeze_aligner', 'true',
        '--torch_dtype', 'bfloat16', '--attn_impl', 'sdpa', '--per_device_train_batch_size', '4',
        '--gradient_accumulation_steps', '4', '--learning_rate', '1e-4', '--num_train_epochs', '3',
        '--lr_scheduler_type', 'cosine', '--warmup_ratio', '0.05', '--gradient_checkpointing', 'true',
        '--max_length', '2048', '--max_pixels', '262144', '--save_strategy', 'epoch',
        '--save_total_limit', '1', '--save_only_model', 'true', '--split_dataset_ratio', '0',
        '--logging_steps', '1', '--seed', '42', '--data_seed', '42', '--add_version', 'false',
        '--output_dir', str(output), '--use_chat_template', 'true', '--loss_scale', 'default',
        '--strict', 'true', '--enable_thinking', 'false', '--report_to', 'none',
        '--dataset_num_proc', '1', '--dataloader_num_workers', '0',
        '--dataloader_persistent_workers', 'false', '--optim', 'adamw_torch',
        '--packing', 'false', '--padding_free', 'false', '--weight_decay', '0.01']


def validate_checkpoint(folder):
    import torch
    from safetensors.torch import load_file
    checkpoints = sorted(folder.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1]))
    assert checkpoints and checkpoints[-1].name == 'checkpoint-8505'
    checkpoint = checkpoints[-1]
    state = json.loads((checkpoint / 'trainer_state.json').read_text())
    assert state['global_step'] == 8505 and abs(state['epoch'] - 3) < 1e-6
    losses = [float(item['loss']) for item in state['log_history'] if 'loss' in item]
    assert losses and all(math.isfinite(value) for value in losses)
    weights = load_file(str(checkpoint / 'adapter_model.safetensors'), device='cpu')
    assert weights and all(torch.isfinite(tensor).all() for tensor in weights.values())
    assert all('.language_model.' in key and '.visual.' not in key for key in weights)
    assert any(torch.count_nonzero(value).item() for key, value in weights.items() if '.lora_B.' in key)
    result = dict(checkpoint=str(checkpoint), epoch=state['epoch'], global_step=state['global_step'],
        first_loss=losses[0], final_logged_loss=losses[-1], finite=True,
        vision_frozen=True, nonzero_lora_B=True, adapter_sha256=sha(checkpoint / 'adapter_model.safetensors'))
    save(folder / 'validation.json', result)
    return checkpoint


def controller(work, max_tokens=6000, allow_cap_reduction=False):
    assert socket.gethostname() == 'softengine-KVM'
    work = work.resolve()
    lock = (work / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    prepare(work, max_tokens)
    tmp = Path('/tmp/xm-entity-glm46-full')
    tmp.mkdir(mode=0o700, exist_ok=True)
    assert tmp.stat().st_uid == os.getuid()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1', MODELSCOPE_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
        MAX_PIXELS='262144', NPROC_PER_NODE='1', MASTER_ADDR='127.0.0.1', MASTER_PORT='30381',
        TMPDIR=str(tmp), HF_DATASETS_CACHE=str(work / 'dataset_cache'),
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', PYTHONUNBUFFERED='1')

    def idle():
        from run_entityqa_lora import idle as check_idle
        while True:
            try:
                check_idle('0')
                return
            except RuntimeError as error:
                save(work / 'status.json', dict(stage='waiting_for_free_gpu', reason=str(error), updated_at=now()))
                time.sleep(15)

    def execute(command, stage, log_name, python=None):
        idle()
        with (work / log_name).open('a') as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=env,
                stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        save(work / 'status.json', dict(stage=stage, controller_pid=os.getpid(), child_pid=process.pid,
            host=socket.gethostname(), gpu='0', log=str(work / log_name), updated_at=now()))
        code = process.wait()
        if code:
            raise RuntimeError(f'{stage} failed with {code}; inspect {log_name}')
        for _ in range(20):
            try:
                idle()
                break
            except RuntimeError:
                time.sleep(2)

    folder = work / 'train/TEXT'
    checkpoint_path = folder / 'checkpoint-8505'
    if checkpoint_path.exists():
        checkpoint = validate_checkpoint(folder)
    else:
        if folder.exists() and any(folder.iterdir()):
            raise RuntimeError('Partial GLM training output exists; preserving it for inspection')
        execute(training_command(work), 'training_GLM_TEXT', 'training_GLM_TEXT.log')
        checkpoint = validate_checkpoint(folder)

    for arm in ['glm46_lora', 'base']:
        target = work / 'evaluation' / ('atomic_replay' if arm == 'glm46_lora' else 'base')
        summary = target / 'summary.json'
        if summary.is_file() and json.loads(summary.read_text()).get('status') == 'complete':
            continue
        cmd = [str(ROOT / '.venv-glm/bin/python'), '-u',
            str(work / 'source_snapshots/evaluate_entity_glm46_full.py'),
            '--root', str(ROOT), '--work', str(work), '--arm', arm,
            '--max-tokens', str(max_tokens)]
        if allow_cap_reduction:
            cmd.append('--allow-cap-reduction')
        execute(cmd, 'evaluating_' + arm, 'evaluating_' + arm + '.log')
    comparison, protocols = {}, {}
    for arm, label in [('base', 'GLM 基座'), ('atomic_replay', 'GLM TEXT LoRA')]:
        summary = json.loads((work / 'evaluation' / arm / 'summary.json').read_text())
        assert summary['status'] == 'complete'
        comparison[label] = summary['groups']
        protocols[label] = generation_details(work / 'evaluation' / arm)
    save(work / 'comparison.json', dict(status='complete', model='GLM-4.6V-Flash',
        reports=comparison, generation_protocols=protocols,
        comparison_limitation='Output caps may differ across and within arms; inspect generation_protocols.',
        qwen_reference=str(PARENT / 'RESULTS_ZH.md'),
        same_training_rows_as_Qwen_E24=True, completed_at=now()))
    lines = ['# GLM-4.6V-Flash 全实体 TEXT 评测结果', '',
        '| 模型 | T2T 准确率 | T2T 无效率 | T2V 准确率 | T2V 无效率 |',
        '|---|---:|---:|---:|---:|']
    for arm, label in [('base', 'GLM 基座'), ('atomic_replay', 'GLM TEXT LoRA')]:
        groups = comparison[label]
        t = groups['T/original']['all']
        v = groups['V/original']['all']
        lines.append(f'| {label} | {t["accuracy"]:.2%} | {t["invalid_rate"]:.2%} | {v["accuracy"]:.2%} | {v["invalid_rate"]:.2%} |')
    lines += ['', '每模态 3,628 道原题；选项轮换各 32 题单列。此前 50 个开发实体与其余 1,764 个分别统计。', '',
        '使用中性 v5 提示，要求简短推理。GLM 原生 thinking 关闭，保留 boxed 答案停止标记和原始答案解析。'
        '已完成预测保留原上限；后续预测采用新上限。不同上限的结果不能视为严格相同推理条件的对照。', '',
        '训练 1,814 实体、每实体 2 条独立 QA、每条每轮重复 10 次；3 轮 ms-swift LoRA。训练行与 Qwen E24 完全相同。', '',
        '完整逐题生成见 evaluation/；训练配置见 experiment.json。', '']
    for label, protocol in protocols.items():
        lines.append(f'{label}：各生成上限的题数 {protocol["generation_cap_counts"]}。')
    (work / 'RESULTS_ZH.md').write_text('\n'.join(lines))
    plan = json.loads((work / 'experiment.json').read_text())
    plan.update(status='complete', completed_at=now())
    save(work / 'experiment.json', plan)
    save(work / 'status.json', dict(stage='complete', controller_pid=os.getpid(),
        results='comparison.json', updated_at=now()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=WORK)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--max-tokens', type=int, default=6000)
    parser.add_argument('--allow-cap-reduction', action='store_true')
    args = parser.parse_args()
    try:
        if args.prepare_only:
            prepare(args.work, args.max_tokens)
        else:
            controller(args.work, args.max_tokens, args.allow_cap_reduction)
    except Exception as error:
        target = args.work.resolve()
        save(target / 'status.json', dict(stage='failed', error=repr(error), updated_at=now(),
            controller_pid=os.getpid()))
        raise
