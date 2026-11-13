#!/usr/bin/env python3
"""Train Qwen on the exact GLM single-pass TEXT corpus, then evaluate paired T2T/T2V."""
import argparse
import copy
from collections import Counter
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
from types import SimpleNamespace

from entityqa_common import digest, now, rows, save, sha
from evaluate_entity_qwen_once import tasks_for
from run_entityqa_lora import command as qwen_command, checkpoint as check_adapter, idle

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'model_cache/Qwen3-VL-4B-Instruct'
SOURCE = ROOT / 'runs/entity1814_repeat1_glm46_lora_ep3_neutral_v5_20261106'
WORK = ROOT / 'runs/entity1814_repeat1_qwen3vl4b_lora_ep3_neutral_v5_20261106'
SOURCE_FILES = [
    'run_entity_qwen_once.py', 'evaluate_entity_qwen_once.py', 'run_entityqa_lora.py',
    'evaluate_entity_glm46_full.py', 'entityqa_common.py', 'entityqa_prompt_neutral_v5.py',
    'answer_readout.py', 'answer_readout_entity_v2.py', 'answer_readout_entity_v3.py',
    'answer_readout_entity_v4.py',
]


def frozen_copy(source, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        assert sha(dest) == sha(source), str(dest)
    else:
        shutil.copy2(source, dest)


def prepare(work):
    work = work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    assert (BASE / 'download_complete.json').is_file()
    source_plan = json.loads((SOURCE / 'experiment.json').read_text())
    assert source_plan['status'] == 'complete'
    source_manifest = json.loads((SOURCE / 'data/manifest.json').read_text())
    for relative, expected in source_manifest['files'].items():
        assert sha(SOURCE / relative) == expected
        frozen_copy(SOURCE / relative, work / relative)
    frozen_copy(SOURCE / 'data/inherited_prompt_ambiguity.json', work / 'data/inherited_prompt_ambiguity.json')
    training = rows(work / 'data/train_text.jsonl')
    metadata = rows(work / 'data/training_rows_with_metadata.jsonl')
    core = rows(work / 'data/entity_unique_training.jsonl')
    generic = rows(work / 'data/retention_training.jsonl')
    assert len(training) == len(metadata) == 4536 and len(core) == 3628 and len(generic) == 908
    assert sha(work / 'data/train_text.jsonl') == source_plan['train_sha256']
    assert len({r['id'] for r in training}) == len(training)
    assert all(set(r) == {'id', 'messages'} and not r.get('images') for r in training)
    assert all(a['id'] == b['id'] and a['messages'] == b['messages'] for a, b in zip(training, metadata))
    exposures = Counter((r['knowledge_id'], r['variant']) for r in metadata if 'knowledge_id' in r)
    assert len(exposures) == 3628 and set(exposures.values()) == {1}
    assert Counter(r['variant'] for r in core) == dict(fact_0=1814, attribute_0=1814)
    assert len({r['knowledge_id'] for r in core}) == 1814
    assert Counter(r['answer'] for r in generic) == dict.fromkeys('ABCD', 227)
    from datasets import load_dataset
    loaded = load_dataset('json', data_files=str(work / 'data/train_text.jsonl'),
        cache_dir=str(work / 'loader_validation_cache'), split='train')
    assert set(loaded.column_names) == {'id', 'messages'}
    assert len(loaded) == len(training) and all(dict(a) == b for a, b in zip(loaded, training))
    save(work / 'data/loader_validation.json', dict(status='pass', rows=len(training), exact_rows_verified=True))
    from swift.model import get_model_processor
    from swift.template import get_template
    _, processor = get_model_processor(str(BASE), model_type='qwen3_vl', load_model=False)
    template = get_template(processor, template_type='qwen3_vl', max_length=2048,
        enable_thinking=False, loss_scale='default', truncation_strategy='raise')
    template.set_mode('train')
    lengths, supervised = [], []
    for row in training:
        encoded = template.encode(dict(messages=copy.deepcopy(row['messages'])))
        ids, labels = encoded['input_ids'], encoded['labels']
        assert len(ids) == len(labels) <= 2048
        target = processor.tokenizer.decode([x for x in labels if x != -100])
        assert target == row['messages'][-1]['content'] + template.template_meta.suffix[0], (row['id'], target)
        assert any(x == -100 for x in labels) and any(x != -100 for x in labels)
        assert row['messages'][0]['content'] in processor.tokenizer.decode(ids)
        lengths.append(len(ids))
        supervised.append(sum(x != -100 for x in labels))
    save(work / 'data/training_template_audit.json', dict(status='pass', rows=len(training),
        max_formatted_tokens=max(lengths), min_supervised_tokens=min(supervised),
        all_supervised_text_exactly_answer_plus_template_eos=True, user_prompt_masked=True,
        truncated_rows=0, checked_at=now()))
    manifest = dict(source_manifest, version='entity1814-twoQA-repeat1-qwen-v1', created_at=now(),
        source_run=str(SOURCE), train_sha256=sha(work / 'data/train_text.jsonl'))
    save(work / 'data/manifest.json', manifest)
    # Training is frozen before formal test inputs are opened.
    inventory = json.loads((SOURCE / 'evaluation_data/package_inventory.json').read_text())
    for relative, expected in inventory['sha256'].items():
        assert sha(SOURCE / 'evaluation_data' / relative) == expected
        frozen_copy(SOURCE / 'evaluation_data' / relative, work / 'evaluation_data' / relative)
    frozen_copy(SOURCE / 'evaluation_data/package_inventory.json', work / 'evaluation_data/package_inventory.json')
    tasks = tasks_for(work)
    input_hash = digest([{k: v for k, v in t.items() if k != 'gold'} for t in tasks])
    assert input_hash == source_plan['evaluation']['input_sha256']
    hashes = {name: sha(ROOT / 'scripts' / name) for name in SOURCE_FILES}
    for name in SOURCE_FILES:
        frozen_copy(ROOT / 'scripts' / name, work / 'source_snapshots' / name)
    path = work / 'experiment.json'
    plan = json.loads(path.read_text()) if path.exists() else dict(
        status='prepared', created_at=now(), host=socket.gethostname(), gpu='0',
        model='Qwen3-VL-4B-Instruct', model_path=str(BASE), starts_from_clean_base=True,
        framework='ms-swift ' + importlib.metadata.version('ms-swift'),
        transformers=importlib.metadata.version('transformers'), environment=str(ROOT / '.venv'),
        modality='TEXT', tuner='LoRA', epochs=3, rank=64, alpha=128, dropout=0.05,
        learning_rate=1e-4, micro_batch=4, gradient_accumulation=4, effective_batch=16,
        max_length=2048, knowledge_units=1814, unique_factual_QA=3628,
        repetitions_per_QA_per_epoch=1, factual_rows_per_epoch=3628, generic_rows_per_epoch=908,
        total_rows_per_epoch=4536, planned_optimizer_steps=852,
        checkpoint_selection='Fixed final epoch 3; no selection using test scores.',
        train_sha256=sha(work / 'data/train_text.jsonl'), source_sha256=hashes,
        source_run=str(SOURCE), same_training_rows_and_order_as_GLM_repeat1=True,
        evaluation=dict(input_sha256=input_hash, tasks_per_model=7320, max_tokens=1024,
            prompt_protocol='entity-neutral-box-v5', temperature=0, seed=42, batch_size=4,
            brief_prompted_reasoning=True, native_thinking=False,
            backend='ms-swift TransformersEngine', adapter='unmerged',
            base_protocol='Fresh Qwen base evaluation with identical 1024-token settings.'),
        limitations=[
            'Text learning only; reports T2T/T2V, not V2T/V2V.',
            'Inherited entity factual/temporal validity and development-entity limitations remain.',
            'One same-name Jack Miller fact_0 prompt has conflicting targets, preserved to match the GLM corpus.',
            'Qwen and GLM differ in tokenizer, architecture, processor, template and parameter count.',
            'Historical repeat10 results have different output caps/backends and training compute.',
        ])
    assert plan['train_sha256'] == sha(work / 'data/train_text.jsonl')
    assert plan['source_sha256'] == hashes
    save(path, plan)
    (work / 'README_ZH.md').write_text(
        '# Qwen：两类 QA，每轮各学一次\n\n'
        'Qwen3-VL-4B-Instruct 从干净基座训练 TEXT LoRA，3 epoch。'
        '与 GLM 单次学习使用完全相同的 4,536 行数据及顺序：1,814 个实体、'
        '每实体 fact_0 与 attribute_0 各一条，共 3,628 条；另有 908 条通用指令。\n\n'
        '先做 2 步独立训练检查与文字/图片推理检查，再从基座执行 852 步正式训练。'
        '训练完成后评测同一批 T2T/T2V 原题各 3,628 题，另每模态 32 道选项轮换题。'
        '新 Qwen 模型和 Qwen 基座都以 1,024 token 上限、简短推理提示、boxed 答案进行评测。'
        '使用 ms-swift TransformersEngine 加载原始 adapter，逐题保存原始回答。\n')
    return plan


def command(work, smoke=False):
    cmd, folder = qwen_command(SimpleNamespace(root=ROOT, work=work, model=BASE), 'TEXT', False)
    if smoke:
        folder = work / 'smoke/TEXT'
        cmd[cmd.index('--output_dir') + 1] = str(folder)
        cmd += ['--max_steps', '2', '--save_strategy', 'steps', '--save_steps', '2', '--warmup_ratio', '0']
    return cmd


def validate(work, smoke=False):
    folder = work / ('smoke/TEXT' if smoke else 'train/TEXT')
    adapter = check_adapter(folder, smoke=smoke)
    state = json.loads((adapter / 'trainer_state.json').read_text())
    assert state['global_step'] == (2 if smoke else 852)
    args = json.loads((adapter / 'args.json').read_text())
    assert args['dataset'] == [str(work / 'data/train_text.jsonl')]
    assert Path(args['model']).resolve() == BASE
    assert not args.get('adapters') and not args.get('resume_from_checkpoint')
    return adapter


def compare(work):
    reports = {}
    reference = None
    for label, run, arm in [('Qwen 基座', work, 'base'), ('Qwen 两类 QA 各学1次', work, 'TEXT'),
                            ('GLM 两类 QA 各学1次', SOURCE, 'atomic_replay')]:
        folder = run / 'evaluation' / arm
        config = json.loads((folder / 'config.json').read_text())
        assert config['max_tokens'] == 1024
        assert config['input_sha256'] == json.loads((work / 'experiment.json').read_text())['evaluation']['input_sha256']
        summary = json.loads((folder / 'summary.json').read_text())
        assert summary['status'] == 'complete'
        records = rows(folder / 'predictions.jsonl')
        assert len(records) == len({(r['id'], r['modality']) for r in records}) == 7320
        indexed = {(r['id'], r['modality']): (r['gold'], r['input_sha256']) for r in records}
        if reference is None:
            reference = indexed
        else:
            assert indexed == reference
        reports[label] = dict(groups=summary['groups'], source=str(folder),
            predictions_sha256=sha(folder / 'predictions.jsonl'), max_tokens=config['max_tokens'])
    save(work / 'comparison.json', dict(status='complete', reports=reports,
        same_inputs_and_labels_verified=True, completed_at=now()))
    lines = ['# Qwen 两类 QA 每轮各学一次：结果', '',
        '| 模型 | T2T 准确率 | T2T 无效率 | T2V 准确率 | T2V 无效率 |',
        '|---|---:|---:|---:|---:|']
    for label, report in reports.items():
        t, v = [report['groups'][m + '/original']['all'] for m in 'TV']
        lines.append(f'| {label} | {t["accuracy"]:.2%} | {t["invalid_rate"]:.2%} | '
                     f'{v["accuracy"]:.2%} | {v["invalid_rate"]:.2%} |')
    lines += ['', '每模态 3,628 道原题；32 道轮换题及开发实体分组单列于 comparison.json。',
        'Qwen 与 GLM 单次学习的训练内容、顺序、轮数、优化步和题目一致；生成上限均为 1,024 token。'
        '两种模型的架构、参数规模、分词器和模板不同。', '']
    (work / 'RESULTS_ZH.md').write_text('\n'.join(lines))


def controller(work):
    assert socket.gethostname() == 'softengine-KVM'
    work = work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    lock = (work / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def status(stage, **extra):
        save(work / 'status.json', dict(stage=stage, controller_pid=os.getpid(),
            host=socket.gethostname(), gpu='0', updated_at=now(), **extra))
    status('preparing')
    plan = prepare(work)
    tmp = work / 'tmp'
    tmp.mkdir(exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1', MODELSCOPE_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
        MAX_PIXELS='262144', TMPDIR=str(tmp), PYTHONUNBUFFERED='1',
        HF_DATASETS_CACHE=str(work / 'dataset_cache'), NPROC_PER_NODE='1',
        MASTER_ADDR='127.0.0.1', MASTER_PORT='30387',
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
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        status(stage, child_pid=process.pid, log=str(log))
        if process.wait():
            raise RuntimeError(f'{stage} failed; inspect {log}; outputs preserved')
    evaluator = work / 'source_snapshots/evaluate_entity_qwen_once.py'
    for smoke in [True, False]:
        folder = work / ('smoke/TEXT' if smoke else 'train/TEXT')
        expected = folder / ('checkpoint-2' if smoke else 'checkpoint-852')
        if not expected.exists():
            if folder.exists() and any(folder.iterdir()):
                raise RuntimeError(f'Preserving interrupted training outputs: {folder}')
            execute(command(work, smoke), 'smoke_training' if smoke else 'training_Qwen_TEXT')
        adapter = validate(work, smoke)
        if smoke:
            summary = work / 'evaluation_smoke/TEXT/summary.json'
            if not summary.exists() or json.loads(summary.read_text()).get('status') != 'complete':
                execute([str(ROOT / '.venv/bin/python'), '-u', str(evaluator), '--root', str(ROOT),
                    '--work', str(work), '--arm', 'TEXT', '--checkpoint', str(adapter), '--smoke'],
                    'smoke_paired_inference')
    for arm in ['TEXT', 'base']:
        summary = work / 'evaluation' / arm / 'summary.json'
        if summary.exists() and json.loads(summary.read_text()).get('status') == 'complete':
            continue
        cmd = [str(ROOT / '.venv/bin/python'), '-u', str(evaluator), '--root', str(ROOT),
               '--work', str(work), '--arm', arm]
        if arm == 'TEXT':
            cmd += ['--checkpoint', str(adapter)]
        execute(cmd, 'evaluating_' + arm)
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
