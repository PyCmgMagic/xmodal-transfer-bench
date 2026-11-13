#!/usr/bin/env python3
"""Queue original-definition GLM rule SFT and matched base/TEXT evaluations on local GPU 0."""
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

from entityqa_common import digest, now, rows, save, sha, write_rows
from evaluate_rules_glm46 import PROMPT_VERSION, tasks_for
from run_entity_glm46_full import BASE, training_command
from run_entityqa_lora import checkpoint as check_adapter, idle

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/rules_repaired_single_image_v7'
WORK = ROOT / 'runs/rules1000_glm46_lora_ep3_original_20261106'
DEPENDENCY = ROOT / 'runs/entity1814_repeat1_qwen3vl4b_lora_ep3_neutral_v5_20261106'
ORIGINAL_SFT = ROOT / 'runs/qwen3_lora_learning_ep5_20261024/data/rules_sft_text.jsonl'
GENERIC_POOL = ROOT / 'runs/entity1814_repeat1_glm46_lora_ep3_neutral_v5_20261106/data/retention_training.jsonl'
SOURCE_FILES = [
    'run_rules_glm46_lora.py', 'evaluate_rules_glm46.py', 'run_entity_glm46_full.py',
    'run_entityqa_lora.py', 'evaluate_entity_glm46_full.py', 'entityqa_common.py',
    'entityqa_prompt_neutral_v5.py', 'answer_readout.py', 'answer_readout_entity_v2.py',
    'answer_readout_entity_v3.py', 'answer_readout_entity_v4.py',
]


def frozen_copy(source, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        assert sha(source) == sha(dest), str(dest)
    else:
        shutil.copy2(source, dest)


def instruction_mix(pool, factual_count):
    quota = math.ceil(factual_count / 16)
    counts, selected = Counter(), []
    for row in pool:
        label = row['answer']
        assert label in 'ABCD'
        if counts[label] < quota:
            selected.append(copy.deepcopy(row))
            counts[label] += 1
    assert counts == dict.fromkeys('ABCD', quota)
    return selected


def prepare(work):
    work = work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    assert (BASE / 'download_complete.json').is_file()
    source_learning = rows(DATA / 'shared/train_text.jsonl')
    original = rows(ORIGINAL_SFT)
    assert len(source_learning) == len(original) == 1000
    assert len({r['id'] for r in source_learning}) == len({r['qid'] for r in source_learning}) == 1000
    core = []
    for source, sft in zip(source_learning, original):
        assert [m['role'] for m in source['messages']] == ['assistant']
        assert [m['role'] for m in sft['messages']] == ['user', 'assistant']
        assert sft['messages'][-1]['content'] == source['messages'][0]['content']
        assert not sft.get('images') and '<image>' not in json.dumps(sft['messages'])
        core.append(dict(id=source['id'], qid=source['qid'], variant='original_definition',
            messages=copy.deepcopy(sft['messages'])))
    assert len({digest(r['messages']) for r in core}) == 1000
    retention = instruction_mix(rows(GENERIC_POOL), len(core))
    assert len(retention) == 252
    metadata = core + retention
    training = [dict(id=r['id'], messages=copy.deepcopy(r['messages'])) for r in metadata]
    assert len(training) == len({r['id'] for r in training}) == 1252
    prompts = [r['messages'][0]['content'] for r in training]
    assert len(prompts) == len(set(prompts)), 'Conflicting or repeated training prompt'
    for rel, records in [
        ('data/train_text.jsonl', training), ('data/training_rows_with_metadata.jsonl', metadata),
        ('data/rule_unique_training.jsonl', core), ('data/retention_training.jsonl', retention),
        ('data/learning_inputs.jsonl', source_learning),
    ]:
        path = work / rel
        if path.exists():
            assert rows(path) == records, rel
        else:
            write_rows(path, records)
    from datasets import load_dataset
    loaded = load_dataset('json', data_files=str(work / 'data/train_text.jsonl'),
        cache_dir=str(work / 'loader_validation_cache'), split='train')
    assert set(loaded.column_names) == {'id', 'messages'}
    assert len(loaded) == len(training) and all(dict(a) == b for a, b in zip(loaded, training))
    from swift.model import get_model_processor
    from swift.template import get_template
    _, processor = get_model_processor(str(BASE), model_type='glm4v', load_model=False)
    template = get_template(processor, template_type='glm4_5v', max_length=2048,
        enable_thinking=False, loss_scale='default', truncation_strategy='raise')
    template.set_mode('train')
    lengths, supervised = [], []
    suffix = template.template_meta.suffix[0]
    prefix = template.template_meta.non_thinking_prefix
    for row in training:
        encoded = template.encode(dict(messages=copy.deepcopy(row['messages'])))
        ids, labels = encoded['input_ids'], encoded['labels']
        assert len(ids) == len(labels) <= 2048
        text = processor.tokenizer.decode([x for x in labels if x != -100])
        assert text == prefix + row['messages'][-1]['content'] + suffix, (row['id'], repr(text[-120:]))
        assert row['messages'][0]['content'] in processor.tokenizer.decode(ids)
        assert any(x == -100 for x in labels) and any(x != -100 for x in labels)
        lengths.append(len(ids))
        supervised.append(sum(x != -100 for x in labels))
    save(work / 'data/training_template_audit.json', dict(status='pass', rows=len(training),
        actual_json_loader_passed=True, max_formatted_tokens=max(lengths),
        min_supervised_tokens=min(supervised),
        all_supervised_text_exactly_template_prefix_answer_and_eos=True,
        supervised_empty_thinking_prefix=prefix, user_prompt_masked=True,
        truncated_rows=0, checked_at=now()))
    manifest = dict(version='rules1000-original-definition-glm-lora-v1', created_at=now(),
        knowledge_units=1000, original_rule_SFT_rows=1000, repetitions_per_rule_per_epoch=1,
        generic_rows_per_epoch=252, total_rows_per_epoch=1252, epochs=3, planned_optimizer_steps=237,
        training_uses_eval_questions_or_labels=False, original_target_verbatim=True,
        source_sha256={str(path): sha(path) for path in [
            DATA / 'shared/train_text.jsonl', ORIGINAL_SFT, GENERIC_POOL]},
        files={str(p.relative_to(work)): sha(p) for p in sorted((work / 'data').glob('*.jsonl'))})
    save(work / 'data/manifest.json', manifest)
    # No formal test inputs or labels are opened until the training set is frozen.
    for rel, expected in manifest['files'].items():
        assert sha(work / rel) == expected
    for rel in ['shared/eval_text.jsonl', 'shared/eval_multimodal.jsonl', 'audit/eval_labels.jsonl',
                'image_hashes.jsonl', 'validation_report.json', 'README.md']:
        target = work / 'evaluation_data' / rel
        if rel.startswith('shared/'):
            selected = rows(DATA / rel)
            for row in selected:
                if row.get('images'):
                    row['images'] = [str((DATA / p).resolve()) for p in row['images']]
                    assert all(Path(p).is_file() for p in row['images'])
            if target.exists():
                assert rows(target) == selected, rel
            else:
                write_rows(target, selected)
        else:
            frozen_copy(DATA / rel, target)
    labels = rows(work / 'evaluation_data/audit/eval_labels.jsonl')
    assert len(labels) == len({r['id'] for r in labels}) == 2000
    assert Counter(r['answer'] for r in labels) == dict.fromkeys('ABCD', 500)
    assert Counter(r['question_type'] for r in labels) == dict(direct=1000, diagnosis=1000)
    assert {r['qid'] for r in labels} == {r['qid'] for r in source_learning}
    image_inventory = rows(DATA / 'image_hashes.jsonl')
    assert len(image_inventory) == len({r['path'] for r in image_inventory}) == 3000
    # Keep original images by reference; verify their recorded hashes once.
    audit_path = work / 'data/image_verification.json'
    if audit_path.exists():
        previous = json.loads(audit_path.read_text())
        assert previous['image_inventory_sha256'] == sha(DATA / 'image_hashes.jsonl')
        assert previous['images_verified'] == 3000
    else:
        for item in image_inventory:
            assert sha(DATA / item['path']) == item['repaired_sha256'], item['path']
        save(audit_path, dict(status='pass', images_verified=3000, original_pixels_unchanged=True,
            image_inventory_sha256=sha(DATA / 'image_hashes.jsonl'), checked_at=now()))
    inventory_path = work / 'evaluation_data/package_inventory.json'
    inventory = dict(source=str(DATA), source_image_inventory_sha256=sha(DATA / 'image_hashes.jsonl'),
        sha256={str(p.relative_to(work / 'evaluation_data')): sha(p) for p in
                sorted((work / 'evaluation_data').rglob('*.jsonl'))})
    save(inventory_path, inventory)
    tasks = tasks_for(work)
    input_hash = digest([{k: v for k, v in t.items() if k != 'gold'} for t in tasks])
    sources = {name: sha(ROOT / 'scripts' / name) for name in SOURCE_FILES}
    for name in SOURCE_FILES:
        frozen_copy(ROOT / 'scripts' / name, work / 'source_snapshots' / name)
    path = work / 'experiment.json'
    plan = json.loads(path.read_text()) if path.exists() else dict(status='prepared', created_at=now(),
        model='GLM-4.6V-Flash', model_path=str(BASE), environment=str(ROOT / '.venv-glm'),
        framework='ms-swift ' + importlib.metadata.version('ms-swift'),
        transformers=importlib.metadata.version('transformers'), host='softengine-KVM', gpu='0',
        modality='TEXT', tuner='LoRA', epochs=3, starts_from_clean_base=True,
        rank=64, alpha=128, dropout=0.05, learning_rate=1e-4,
        micro_batch=4, gradient_accumulation=4, effective_batch=16, max_length=2048,
        knowledge_units=1000, factual_rows_per_epoch=1000, generic_rows_per_epoch=252,
        total_rows_per_epoch=1252, repetitions_per_rule_per_epoch=1, planned_optimizer_steps=237,
        train_sha256=sha(work / 'data/train_text.jsonl'), source_sha256=sources,
        queue_dependency=str(DEPENDENCY), checkpoint_selection='Fixed final epoch 3',
        evaluation=dict(input_sha256=input_hash, tasks_per_model=4000,
            paired_questions_per_modality=2000, max_tokens=1024, temperature=0, seed=42,
            batch_size=4, prompt_protocol=PROMPT_VERSION, native_thinking=False,
            brief_prompted_reasoning=True, parser='entity v4 plus GLM native box normalization',
            fresh_matched_base=True, closed_book=True, no_recall_or_retrieval=True),
        limitations=[
            'Original explanation SFT, one example per rule; no added application examples.',
            'Only TEXT learning is trained, so directions are T2T and T2V.',
            '104 direct tasks use component outcome recording, rather than original binary qualification.',
            'Source image hashes and symbolic validation do not independently certify image semantics.',
            '241 tasks are classified as rule_report_audit; report this separately from result_application.',
            'The original target contains inherited worked examples; no formal test examples are added.',
            'Generic synthetic instruction rows are not entity factual training.',
        ])
    assert plan['train_sha256'] == sha(work / 'data/train_text.jsonl')
    assert plan['source_sha256'] == sources
    assert plan['evaluation']['input_sha256'] == input_hash
    save(path, plan)
    (work / 'README_ZH.md').write_text(
        '# GLM 规则类原始解释 SFT：TEXT LoRA、3轮\n\n'
        '1,000条v7规则，每条一条原始解释问答，每轮不重复。'
        '另加252条答案平衡的通用指令，每轮1,252条；3轮、预计237步。'
        '从GLM-4.6V-Flash干净基座训练，rank64/alpha128/dropout0.05，'
        '学习率1e-4、batch4×累积4，冻结视觉编码器和对齐层。\n\n'
        '等待现有Qwen基座评测结束且GPU0空闲，再执行2步独立试训、8题文字/图片推理检查、'
        '正式训练、训练后及干净GLM基座评测。两组各4,000题，T/V各2,000题，'
        'direct/diagnosis各1,000；104道组件记录题、规则报告审查及规则族单列。\n\n'
        '正式评测闭卷，不提供学习材料，不回忆检索；要求简短推理并以boxed给出选项。'
        '原生thinking关闭，输出上限1,024 token，温度0。'
        '主成绩使用原始回答，无效计错，不追加约束补选，不按测试挑checkpoint。\n\n'
        '现有规则集构建验证通过、3000张原图哈希核验，但独立人工图像语义认证尚未完成；'
        '104道题测试组件检查与记录约定，结果分项保留。\n')
    return plan


def command(work, smoke=False):
    cmd = training_command(work)
    for option, value in {
        '--dataset': str(work / 'data/train_text.jsonl'),
        '--output_dir': str(work / ('smoke/TEXT' if smoke else 'train/TEXT')),
        '--master_port': '30388',
    }.items():
        cmd[cmd.index(option) + 1] = value
    if smoke:
        cmd += ['--max_steps', '2', '--save_strategy', 'steps', '--save_steps', '2', '--warmup_ratio', '0']
    return cmd


def validate(work, smoke=False):
    folder = work / ('smoke/TEXT' if smoke else 'train/TEXT')
    checkpoint = check_adapter(folder, smoke=smoke)
    state = json.loads((checkpoint / 'trainer_state.json').read_text())
    assert state['global_step'] == (2 if smoke else 237)
    args = json.loads((checkpoint / 'args.json').read_text())
    assert args['dataset'] == [str(work / 'data/train_text.jsonl')]
    assert Path(args['model']).resolve() == BASE.resolve()
    assert not args.get('adapters') and not args.get('resume_from_checkpoint')
    return checkpoint


def compare(work):
    reports, reference, configs = {}, None, []
    for arm in ['base', 'TEXT']:
        folder = work / 'evaluation' / arm
        config = json.loads((folder / 'config.json').read_text())
        configs.append(config)
        summary = json.loads((folder / 'summary.json').read_text())
        assert summary['status'] == 'complete' and summary['saved'] == 4000
        records = rows(folder / 'predictions.jsonl')
        indexed = {(r['id'], r['modality']): (r['input_sha256'], r['gold']) for r in records}
        assert len(indexed) == len(records) == 4000
        if reference is None:
            reference = indexed
        else:
            assert indexed == reference
        reports[arm] = dict(groups=summary['groups'], predictions_sha256=sha(folder / 'predictions.jsonl'))
    ignored = {'arm', 'adapter', 'adapter_sha256'}
    assert {k: v for k, v in configs[0].items() if k not in ignored} == {
        k: v for k, v in configs[1].items() if k not in ignored}
    save(work / 'comparison.json', dict(status='complete', reports=reports, completed_at=now(),
        same_inputs_labels_and_generation_settings_verified=True))
    lines = ['# GLM 规则类 TEXT LoRA：3轮结果', '',
        '| 模型 | 文字准确率 | 文字无效率 | 图片准确率 | 图片无效率 |',
        '|---|---:|---:|---:|---:|']
    for arm, label in [('base', 'GLM基座'), ('TEXT', 'GLM规则TEXT LoRA')]:
        t, v = [reports[arm]['groups'][m + '/all'] for m in 'TV']
        lines.append(f'| {label} | {t["accuracy"]:.2%} | {t["invalid_rate"]:.2%} | '
                     f'{v["accuracy"]:.2%} | {v["invalid_rate"]:.2%} |')
    lines += ['', '每模态2,000题；无效计错。闭卷，简短推理提示，原生thinking关闭，上限1,024 token。',
        'direct/diagnosis、规则族、104道组件记录题和规则报告审查题分项见comparison.json。', '']
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
    while True:
        previous = json.loads((DEPENDENCY / 'status.json').read_text())
        if previous.get('stage') == 'complete':
            assert json.loads((DEPENDENCY / 'comparison.json').read_text())['status'] == 'complete'
            break
        status('queued_after_Qwen_base', dependency=str(DEPENDENCY),
            dependency_stage=previous.get('stage'), training_started=False,
            training_rows_per_epoch=1252, epochs=3, planned_optimizer_steps=237)
        time.sleep(30)
    for name, expected in plan['source_sha256'].items():
        assert sha(ROOT / 'scripts' / name) == expected, name
        assert sha(work / 'source_snapshots' / name) == expected, name
    for rel, expected in json.loads((work / 'data/manifest.json').read_text())['files'].items():
        assert sha(work / rel) == expected
    tmp = work / 'tmp'
    tmp.mkdir(exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1', MODELSCOPE_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', MAX_PIXELS='262144',
        TMPDIR=str(tmp), PYTHONUNBUFFERED='1', HF_DATASETS_CACHE=str(work / 'dataset_cache'),
        NPROC_PER_NODE='1', MASTER_ADDR='127.0.0.1', MASTER_PORT='30388',
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
            raise RuntimeError(f'{stage} failed; inspect {log}; existing outputs preserved')
    evaluator = work / 'source_snapshots/evaluate_rules_glm46.py'
    for smoke in [True, False]:
        folder = work / ('smoke/TEXT' if smoke else 'train/TEXT')
        expected = folder / ('checkpoint-2' if smoke else 'checkpoint-237')
        if not expected.exists():
            if folder.exists() and any(folder.iterdir()):
                raise RuntimeError(f'Preserving interrupted training output {folder}')
            execute(command(work, smoke), 'smoke_training' if smoke else 'training_GLM_rules_TEXT')
        checkpoint = validate(work, smoke)
        if smoke:
            target = work / 'evaluation_smoke/TEXT/summary.json'
            if not target.exists() or json.loads(target.read_text()).get('status') != 'complete':
                execute([str(ROOT / '.venv-glm/bin/python'), '-u', str(evaluator),
                    '--root', str(ROOT), '--work', str(work), '--arm', 'TEXT',
                    '--checkpoint', str(checkpoint), '--smoke'], 'smoke_paired_inference')
    for arm in ['TEXT', 'base']:
        target = work / 'evaluation' / arm / 'summary.json'
        if target.exists() and json.loads(target.read_text()).get('status') == 'complete':
            continue
        cmd = [str(ROOT / '.venv-glm/bin/python'), '-u', str(evaluator),
            '--root', str(ROOT), '--work', str(work), '--arm', arm]
        if arm == 'TEXT':
            cmd += ['--checkpoint', str(checkpoint)]
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
