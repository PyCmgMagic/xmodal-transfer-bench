#!/usr/bin/env python3
"""Fixed development-only stronger-prompt comparison after the current main run."""
import argparse
import copy
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from entityqa_common import rows, save, write_rows, digest, sha, now
from entityqa_prompt_neutral_v5 import eval_payload as neutral_payload, COT_SUFFIX as NEUTRAL_SUFFIX

VERSION = 'entity-explicit-option-v6-development'
SUFFIX = (
    '\n\nThis is a multiple-choice question. You must select exactly one of the listed options. '
    'Briefly explain how the relevant fact matches your chosen option; do not overthink. '
    'A factual statement alone is not a complete answer. '
    'End with exactly one option letter enclosed in \\boxed{}. '
    'Do not write anything after the boxed answer.')


def payload(row, root):
    result = neutral_payload(row, root)
    content = result['messages'][-1]['content']
    assert content.endswith(NEUTRAL_SUFFIX)
    result['messages'][-1]['content'] = content[:-len(NEUTRAL_SUFFIX)] + SUFFIX
    return result


def prepare(root, parent, work):
    if (work / 'plan.json').exists():
        plan = json.loads((work / 'plan.json').read_text())
        for name, h in plan['frozen_files'].items():
            assert sha(work / name) == h, name
        return plan
    work.mkdir(parents=True, exist_ok=True)
    reference = root / 'runs/entity50_repeat10_lora_ep3_20261104/evaluation_data'
    data = work / 'evaluation_data'
    original = {(r['id'], r['modality']): r for r in rows(parent / 'evaluation/atomic_replay/predictions.jsonl')}
    labels = {r['id']: r for r in rows(reference / 'audit/eval_labels.jsonl')}
    baseline, input_audit, label_ids = [], [], set()
    for mod, name in [('T', 'eval_text.jsonl'), ('V', 'eval_multimodal.jsonl')]:
        selected = rows(reference / 'shared' / name)
        assert len(selected) == 132
        assert len({r['qid'] for r in selected}) == 50
        write_rows(data / 'shared' / name, selected)
        for row in selected:
            old = neutral_payload(row, reference)
            new = payload(row, reference)
            assert old.get('images') == new.get('images')
            assert old['messages'][-1]['content'][:-len(NEUTRAL_SUFFIX)] == new['messages'][-1]['content'][:-len(SUFFIX)]
            prior = copy.deepcopy(original[row['id'], mod])
            assert prior['input_sha256'] == digest(old)
            assert prior['gold'] == labels[row['id']]['answer']
            baseline.append(prior)
            label_ids.add(row['id'])
            input_audit.append(dict(id=row['id'], modality=mod, baseline_input_sha256=digest(old),
                                    strengthened_input_sha256=digest(new), question_options_images_unchanged=True))
    write_rows(data / 'audit/eval_labels.jsonl', [labels[i] for i in sorted(label_ids)])
    write_rows(data / 'audit/input_hashes.jsonl', input_audit)
    write_rows(work / 'baseline_v5/predictions.jsonl', baseline)
    save(data / 'package_inventory.json', dict(version=VERSION, entities=50,
        originals_per_modality=100, rotations_per_modality=32,
        selected_before_new_outputs=True, source=str(reference),
        files={str(p.relative_to(data)): sha(p) for p in data.rglob('*.jsonl')}))
    snapshots = work / 'source_snapshots'
    snapshots.mkdir()
    sources = ['probe_entity_prompt_strength_v6.py', 'entityqa_prompt_neutral_v5.py',
               'entityqa_common.py', 'evaluate_entityqa_fullfact_v2.py', 'run_entityqa_lora.py',
               'answer_readout.py', 'answer_readout_entity_v2.py', 'answer_readout_entity_v3.py',
               'answer_readout_entity_v4.py']
    for name in sources:
        shutil.copy2(root / 'scripts' / name, snapshots / name)
    validation = json.loads((parent / 'atomic_replay/train/TEXT/validation.json').read_text())
    assert validation['epoch'] == 3 and validation['global_step'] == 8505
    plan = dict(created_at=now(), version=VERSION, parent=str(parent),
        selection='Same fixed 50 E23 development entities and pre-existing 8-entity rotations, no score-based filtering',
        model='E24 all-entity TEXT LoRA, unchanged checkpoint-8505',
        adapter=validation['checkpoint'], adapter_sha256=validation['adapter_sha256'],
        base=str(root / 'model_cache/Qwen3-VL-4B-Instruct'), tasks=264,
        original_questions_per_modality=100, rotation_questions_per_modality=32,
        baseline_source=str(parent / 'evaluation/atomic_replay/predictions.jsonl'),
        baseline_source_sha256=sha(parent / 'evaluation/atomic_replay/predictions.jsonl'),
        old_suffix=NEUTRAL_SUFFIX, new_suffix=SUFFIX, changes='Only user suffix; same questions, options and images',
        max_tokens=65536, wall_seconds=180, temperature=0, readout='v4',
        schedule='Wait for parent status complete, then idle GPU; never stop running jobs',
        limitations=['Development prompt comparison, not new confirmatory transfer evidence.',
            'V5 uses verified existing model generations; V6 generates new outputs.',
            'Primary metrics use original raw responses; constrained finalization remains supplementary.',
            'Prompt changes as a bundle; cannot isolate one sentence.'],
        frozen_files={str(p.relative_to(work)): sha(p) for p in work.rglob('*') if p.is_file()})
    save(work / 'plan.json', plan)
    save(work / 'status.json', dict(stage='prepared', updated_at=now(), tasks=264))
    return plan


def metric(records):
    n = len(records)
    assert n
    return dict(n=n, correct=sum(r['raw_predicted'] == r['gold'] for r in records),
        accuracy=sum(r['raw_predicted'] == r['gold'] for r in records) / n,
        invalid=sum(r['raw_predicted'] is None for r in records),
        invalid_rate=sum(r['raw_predicted'] is None for r in records) / n,
        answers=dict(Counter(str(r['raw_predicted']) for r in records)),
        stop_reasons=dict(Counter(r['stop_reason'] for r in records)))


def summarize(work):
    before = rows(work / 'baseline_v5/predictions.jsonl')
    after = rows(work / 'strengthened_v6/predictions.jsonl')
    assert len(before) == len(after) == 264
    a, b = ({(r['id'], r['modality']): r for r in group} for group in [before, after])
    assert set(a) == set(b)
    hashes = {(r['id'], r['modality']): r for r in rows(work / 'evaluation_data/audit/input_hashes.jsonl')}
    for key in a:
        assert a[key]['gold'] == b[key]['gold']
        assert a[key]['input_sha256'] == hashes[key]['baseline_input_sha256']
        assert b[key]['input_sha256'] == hashes[key]['strengthened_input_sha256']
    groups = {}
    for mod in ['T', 'V']:
        for probe in ['original', 'rotation']:
            keys = [k for k in a if k[1] == mod and ((a[k]['probe_id'] == 'rotation') == (probe == 'rotation'))]
            groups[mod + '/' + probe] = dict(v5=metric([a[k] for k in keys]),
                v6=metric([b[k] for k in keys]),
                raw_invalid_to_valid=sum(a[k]['raw_predicted'] is None and b[k]['raw_predicted'] is not None for k in keys),
                raw_valid_to_invalid=sum(a[k]['raw_predicted'] is not None and b[k]['raw_predicted'] is None for k in keys),
                wrong_to_correct=sum(a[k]['raw_predicted'] != a[k]['gold'] and b[k]['raw_predicted'] == b[k]['gold'] for k in keys),
                correct_to_wrong=sum(a[k]['raw_predicted'] == a[k]['gold'] and b[k]['raw_predicted'] != b[k]['gold'] for k in keys))
    save(work / 'comparison.json', dict(status='complete', groups=groups, completed_at=now(),
        primary='Raw generation; original questions only; rotations separate',
        limitations='Fixed development set; no change to E24 full-run scores.'))
    lines = ['# 全实体模型：固定 50 实体的提示强度对照', '',
        '模型权重、题目、选项、图片及生成预算保持不变，仅强化用户提示。主指标是原始回答，轮换另列。', '',
        '| 提示 | T2T 准确率 | T2T 无效率 | T2V 准确率 | T2V 无效率 |',
        '|---|---:|---:|---:|---:|']
    for key, label in [('v5', '中性 v5'), ('v6', '明确选项要求 v6')]:
        t, v = (groups[m + '/original'][key] for m in ['T', 'V'])
        lines.append(f'| {label} | {t["accuracy"]:.2%} | {t["invalid_rate"]:.2%} | {v["accuracy"]:.2%} | {v["invalid_rate"]:.2%} |')
    lines += ['', '每模态 100 道原题；每模态另有 32 道轮换题，见 comparison.json。固定开发实体，不作为未参与调参的确认集。', '',
        '原始回复见 baseline_v5/ 与 strengthened_v6/；未修改 E24 主实验结果。', '']
    (work / 'RESULTS_ZH.md').write_text('\n'.join(lines))


def controller(root, parent, work):
    from run_entityqa_lora import idle, merge
    plan = prepare(root, parent, work)
    lock = (work / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def status(stage, **extra):
        save(work / 'status.json', dict(stage=stage, controller_pid=os.getpid(), updated_at=now(), **extra))
    status('waiting_for_main_experiment', parent_status=str(parent / 'status.json'))
    while json.loads((parent / 'status.json').read_text())['stage'] != 'complete':
        time.sleep(15)
    while True:
        try:
            idle('0')
            break
        except RuntimeError as error:
            status('waiting_for_free_gpu', reason=str(error))
            time.sleep(15)
    for name, expected in plan['frozen_files'].items():
        assert sha(work / name) == expected, name
    adapter = Path(plan['adapter'])
    assert sha(adapter / 'adapter_model.safetensors') == plan['adapter_sha256']
    serving = Path('/dev/shm') / (parent.name + '-prompt-strength-v6')
    summary = work / 'strengthened_v6/summary.json'
    if not summary.exists() or json.loads(summary.read_text())['status'] != 'complete':
        status('preparing_serving', temporary_model=str(serving))
        assert shutil.disk_usage('/dev/shm').free > 11 * 2**30 or (serving / 'merge_manifest.json').exists()
        merge(Path(plan['base']), adapter, serving)
        old = json.loads((parent / 'serving_artifacts/merge_manifest.json').read_text())
        new = json.loads((serving / 'merge_manifest.json').read_text())
        assert old['files'] == new['files'] and old['base_sha256'] == new['base_sha256']
        save(work / 'merge_manifest.json', new)
        save(work / 'same_model_verification.json', dict(merged_weight_hashes_match_parent=True,
            original_adapter_sha256=plan['adapter_sha256']))
    if not summary.exists() or json.loads(summary.read_text())['status'] != 'complete':
        env = dict(os.environ, HF_HUB_OFFLINE='1', MODELSCOPE_OFFLINE='1', CUDA_VISIBLE_DEVICES='0',
            TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', VLLM_WORKER_MULTIPROC_METHOD='spawn')
        cmd = [str(root / '.venv-vllm/bin/python'), '-u', str(work / 'source_snapshots/probe_entity_prompt_strength_v6.py'),
            '--mode', 'evaluate', '--root', str(root), '--parent', str(parent), '--work', str(work), '--model', str(serving)]
        save(work / 'evaluation_command.json', cmd)
        with (work / 'evaluation.log').open('a') as stream:
            child = subprocess.Popen(cmd, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
        status('evaluating_strengthened_v6', child_pid=child.pid)
        assert child.wait() == 0, 'Evaluation failed; retained outputs'
    summarize(work)
    if serving.exists():
        expected = json.loads((work / 'merge_manifest.json').read_text())['files']
        assert all(sha(serving / name) == h for name, h in expected.items())
        assert all(p.is_file() and not p.is_symlink() and p.stat().st_uid == os.getuid() for p in serving.iterdir())
        shutil.rmtree(serving)
    status('complete', results='comparison.json')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['prepare', 'controller', 'evaluate'], required=True)
    for name in ['root', 'parent', 'work']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--model', type=Path)
    args = p.parse_args()
    root, parent, work = args.root.resolve(), args.parent.resolve(), args.work.resolve()
    try:
        if args.mode == 'prepare':
            prepare(root, parent, work)
        elif args.mode == 'controller':
            controller(root, parent, work)
        else:
            from answer_readout_entity_v4 import parse_answer, VERSION as READOUT_VERSION
            from evaluate_entityqa_fullfact_v2 import run
            args.data = work / 'evaluation_data'
            args.output = work / 'strengthened_v6'
            args.arm, args.gpu, args.limit = 'TEXT', '0', 0
            args.max_tokens, args.wall_seconds, args.import_predictions = 65536, 180, None
            run(args, payload_builder=payload, cot_suffix=SUFFIX, answer_parser=parse_answer,
                readout_version=READOUT_VERSION, prompt_protocol=VERSION + ': explicit option requirement, brief CoT',
                evaluator_version=6)
    except Exception as error:
        save(work / ('failure_evaluation.json' if args.mode == 'evaluate' else 'status.json'),
             dict(stage='failed', error=repr(error), updated_at=now(), pid=os.getpid()))
        raise
