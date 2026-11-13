#!/usr/bin/env python3
"""Compare greedy saved replies with sampled decoding on 50 randomly selected rules."""
import argparse
from collections import Counter
import copy
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import socket
import subprocess
import time

from audit_rules_glm46_readout import reparse, VERSION as PARSER_VERSION
from entityqa_common import rows, save, write_rows, sha, digest, now
from evaluate_entity_glm46_full import ANSWER_STOP_SEQUENCES, glm_readout
from run_entityqa_lora import idle
from run_rules_qwen32b_rule_given import freeze, summaries

VERSION = 'rules-glm-vision-repeat1-frozen-recall-temperature-pilot-v1'
SOURCES = ['run_rules_glm_temperature_pilot.py', 'run_rules_qwen32b_rule_given.py',
           'audit_rules_glm46_readout.py', 'entityqa_common.py', 'evaluate_entity_glm46_full.py',
           'run_entityqa_lora.py', 'answer_readout.py', 'answer_readout_entity_v2.py',
           'answer_readout_entity_v3.py', 'answer_readout_entity_v4.py', 'entityqa_prompt_neutral_v5.py']
PACKAGES = ['torch', 'transformers', 'ms-swift', 'peft']


def sample_tasks(tasks):
    qids = sorted({t['qid'] for t in tasks})
    assert len(qids) == 1000 and len(tasks) == 4000
    random.Random(20261110).shuffle(qids)
    chosen = set(qids[:50])
    selected = [copy.deepcopy(t) for t in tasks if t['qid'] in chosen]
    assert len(selected) == 200
    assert Counter(t['modality'] for t in selected) == dict(T=100, V=100)
    assert Counter(t['qid'] for t in selected) == dict.fromkeys(chosen, 4)
    for i in range(0, 200, 4):
        batch = selected[i:i + 4]
        assert len({t['qid'] for t in batch}) == 1
        for a, b in [batch[:2], batch[2:]]:
            assert (a['modality'], b['modality']) == ('T', 'V')
            assert all(a[k] == b[k] for k in ['id', 'gold', 'options', 'rule_sha256'])
    return selected


def repeated_sentence(text):
    parts = [re.sub(r'\s+', ' ', s).strip().casefold()
             for s in re.split(r'(?<=[.!?])\s+|\n+', text)]
    counts = Counter(s for s in parts if len(s) >= 30)
    return max(counts.values(), default=0) >= 3


def metrics(records):
    groups = summaries(records)
    for key, group in groups.items():
        mod, kind = key.split('/')
        selected = [r for r in records if r['modality'] == mod and
                    (kind == 'all' or r['question_type'] == kind)]
        group['sentence_repeat_3plus'] = sum(repeated_sentence(r['raw_response']) for r in selected)
        group['sentence_repeat_3plus_rate'] = group['sentence_repeat_3plus'] / len(selected)
    return groups


def verify(work, plan):
    for rel, expected in plan['immutable_sha256'].items():
        assert sha(work / rel) == expected, rel
    for name, expected in plan['source_sha256'].items():
        assert sha(work / 'source_snapshots' / name) == expected, name
    for name, expected in plan['adapter_sha256'].items():
        assert sha(Path(plan['adapter']) / name) == expected, name
    assert {k: importlib.metadata.version(k) for k in PACKAGES} == plan['packages']


def prepare(root, work):
    work.mkdir(parents=True, exist_ok=True)
    if (work / 'experiment.json').exists():
        plan = json.loads((work / 'experiment.json').read_text())
        assert plan['protocol'] == VERSION
        verify(work, plan)
        return plan
    parent = root / 'runs/rules1000_glm46_vision_repeat1_repeat10_ep3_20261109'
    old = json.loads((parent / 'experiment.json').read_text())
    tasks = sample_tasks(rows(parent / 'data/generated_apply_tasks_repeat1.jsonl'))
    write_rows(work / 'data/task_manifest.jsonl', [{k: v for k, v in t.items() if k != 'payload'} for t in tasks])
    write_rows(work / 'data/inference_inputs.jsonl', [t['payload'] for t in tasks])
    freeze(parent / 'experiment.json', work / 'references/parent_before_completion.json')
    freeze(parent / 'evaluation_apply/repeat1/config.json', work / 'references/greedy_generation_config.json')
    for name in SOURCES:
        freeze(root / 'scripts' / name, work / 'source_snapshots' / name)
    plan = dict(protocol=VERSION, status='prepared', created_at=now(), project_root=str(root),
        host=socket.gethostname(), environment=str(root / '.venv-glm'), gpu='0',
        source_run=str(parent), queue_dependency=str(root / 'runs/rules120_qwen38_27b_nf4_rule_given_20261109'),
        queue_policy='After both queued Qwen diagnostics; no competing GPU waiter at an earlier boundary',
        model_path=old['model_path'], adapter=old['arms']['repeat1']['path'],
        adapter_sha256=old['arms']['repeat1']['sha256'], training=False,
        knowledge_units=50, total=200, tasks_per_modality=100,
        selection='Seed20261110 random 50 of all 1000 rule IDs, no conditioning on correctness or invalidity',
        input_policy='Verbatim same questions, images, already generated recall and boxed prompt as source repeat1 apply',
        max_tokens=1024, batch_size=4, native_thinking=False, brief_prompted_reasoning=True,
        temperature=0.8, top_p=0.95, top_k=50, repetition_penalty=1.0, seed=42,
        batch_seed_policy='Set all RNG seeds to 42 + original filtered batch start index before each batch; resumable',
        baseline='Previously saved greedy results on exactly the selected questions, same checkpoint and input hashes',
        parser=PARSER_VERSION, stop_sequences=ANSWER_STOP_SEQUENCES, finalizer=False,
        packages={k: importlib.metadata.version(k) for k in PACKAGES},
        immutable_sha256={str(p.relative_to(work)): sha(p) for folder in ['data', 'references']
                          for p in (work / folder).iterdir() if p.is_file()},
        source_sha256={name: sha(work / 'source_snapshots' / name) for name in SOURCES},
        limitations=[
            'Sampling can reduce loops but cannot repair incorrect recalled rules.',
            'One generation per item and one seed schedule; small pilot, sampling variability not estimated.',
            'Greedy ignores top_p/top_k; sampled condition enables temperature0.8 plus top_p0.95/top_k50.',
            'Sentence repetition is a heuristic: sentences of at least 30 characters repeated at least 3 times.',
        ])
    save(work / 'experiment.json', plan)
    (work / 'README_ZH.md').write_text(
        '# GLM视觉学习：采样解码对照\n\n'
        '从全部1000规则按固定seed20261110随机选50条，每条direct/diagnosis、文字/图片配对，'
        '共200题，文字/图片各100题；不按旧答案正确性或是否循环筛选。\n\n'
        '使用原VISION repeat1 LoRA与已经生成的原回忆，问题、图片、提示词、1024输出上限均保持一致。'
        '原temperature0贪心回复作为同题基线；新条件temperature0.8、top_p0.95、top_k50、'
        'repetition_penalty1.0，batch4。每批显式设置所有随机种子，保留原始回复，不强制补选。\n\n'
        '排在已经排队的Qwen3.8任务后自动运行，不中断现有任务。报告准确率、无效率、截断率、句子重复率与逐题变化。'
        '该试验只检验解码改善；错误回忆不会被替换或纠正。单次采样不代表稳定提升。\n')
    verify(work, plan)
    return plan


def matched_baseline(manifest, inputs, records, allow_missing=False):
    lookup = {(r['id'], r['modality']): r for r in records}
    result = []
    for task, payload in zip(manifest, inputs):
        if allow_missing and (task['id'], task['modality']) not in lookup:
            continue
        r = copy.deepcopy(lookup[task['id'], task['modality']])
        assert r['gold'] == task['gold'] and r['input_sha256'] == digest(payload)
        p, reason = reparse(r['raw_response'], task['options'])
        assert p == r['predicted']
        result.append(r)
    return result


def evaluate(work, plan):
    verify(work, plan)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    source = Path(plan['source_run']) / 'evaluation_apply/repeat1'
    source_status = json.loads((source / 'status.json').read_text())['status']
    assert source_status in {'complete', 'stopped_by_user'}
    manifest, inputs = rows(work / 'data/task_manifest.jsonl'), rows(work / 'data/inference_inputs.jsonl')
    baseline = matched_baseline(manifest, inputs, rows(source / 'predictions.jsonl'),
                                allow_missing=source_status == 'stopped_by_user')
    assert baseline and len(baseline) % 4 == 0
    save(work / 'baseline_coverage.json', dict(source_status=source_status, matched_tasks=len(baseline),
        sampled_tasks=200, comparison='Only matched input hashes enter primary paired comparison; no new greedy answers generated'))
    baseline_path = work / 'references/greedy_same_questions.jsonl'
    if baseline_path.exists():
        assert rows(baseline_path) == baseline
    else:
        write_rows(baseline_path, baseline)
    idle('0')
    import torch
    from swift import TransformersEngine, InferRequest, RequestConfig
    from swift.utils import seed_everything
    torch.set_num_threads(4)
    engine = TransformersEngine(plan['model_path'], model_type='glm4v', template_type='glm4_5v',
        adapters=[plan['adapter']], torch_dtype=torch.bfloat16, attn_impl='sdpa',
        device_map='cuda:0', max_batch_size=4)
    engine.template.enable_thinking = False
    request = RequestConfig(max_tokens=plan['max_tokens'], temperature=plan['temperature'],
        top_p=plan['top_p'], top_k=plan['top_k'], repetition_penalty=plan['repetition_penalty'],
        seed=42, return_details=True, stop=plan['stop_sequences'])
    actual = engine._prepare_generation_config(request)
    assert actual.do_sample and actual.temperature == 0.8 and actual.top_p == 0.95 and actual.top_k == 50
    assert actual.repetition_penalty == 1.0 and actual.max_new_tokens == 1024
    save(work / 'effective_generation_config.json', actual.to_dict())
    output = work / 'evaluation'
    output.mkdir(exist_ok=True)
    path = output / 'predictions.jsonl'
    records = rows(path) if path.exists() else []
    assert len(records) % 4 == 0 and len(records) <= 200
    for i, r in enumerate(records):
        assert (r['id'], r['modality'], r['input_sha256']) == (
            manifest[i]['id'], manifest[i]['modality'], digest(inputs[i]))
    started = time.monotonic()
    for start in range(len(records), 200, 4):
        seed_everything(plan['seed'] + start)
        batch = manifest[start:start + 4]
        replies = engine.infer([InferRequest(**copy.deepcopy(p)) for p in inputs[start:start + 4]],
                               request, use_tqdm=False)
        assert len(replies) == len(batch)
        new = []
        for i, (task, reply) in enumerate(zip(batch, replies), start):
            if isinstance(reply, Exception):
                raise reply
            choice = reply.choices[0]
            raw = choice.message.content or ''
            pred, reason = reparse(raw, task['options'])
            legacy, legacy_reason = glm_readout(raw, task['options'])
            r = {k: v for k, v in task.items() if k != 'options'}
            r.update(raw_response=raw, predicted=pred, parse_reason=reason, correct=pred == task['gold'],
                raw_predicted=legacy, raw_parse_reason=legacy_reason,
                generated_tokens=reply.usage.completion_tokens, stop_reason=str(choice.finish_reason),
                input_sha256=digest(inputs[i]), temperature=0.8, top_p=0.95, top_k=50,
                batch_seed=plan['seed'] + start, generation_max_tokens=1024, saved_at=now())
            new.append(r)
        # Commit a complete batch atomically so resuming preserves the RNG schedule.
        records.extend(new)
        write_rows(path, records)
        progress = dict(status='generating', saved=len(records), total=200,
            groups=metrics(records), updated_at=now(), elapsed_seconds=round(time.monotonic() - started, 1))
        save(output / 'status.json', progress)
        print(json.dumps(dict(saved=len(records), total=200, groups=progress['groups'])), flush=True)
    save(output / 'status.json', dict(status='complete', saved=200, total=200, groups=metrics(records), updated_at=now()))


def summarize(work, plan):
    previous = rows(work / 'references/greedy_same_questions.jsonl')
    all_current = rows(work / 'evaluation/predictions.jsonl')
    assert len(all_current) == 200
    keys = {(r['id'], r['modality']) for r in previous}
    current = [r for r in all_current if (r['id'], r['modality']) in keys]
    assert len(previous) == len(current) > 0
    changes = []
    for a, b in zip(previous, current):
        assert (a['id'], a['modality'], a['gold'], a['input_sha256']) == (
            b['id'], b['modality'], b['gold'], b['input_sha256'])
        changes.append(dict(id=a['id'], modality=a['modality'], gold=a['gold'],
            greedy_prediction=a['predicted'], sampled_prediction=b['predicted'],
            greedy_correct=a['predicted'] == a['gold'], sampled_correct=b['predicted'] == b['gold'],
            greedy_repeated=repeated_sentence(a['raw_response']), sampled_repeated=repeated_sentence(b['raw_response'])))
    write_rows(work / 'paired_changes.jsonl', changes)
    groups = dict(greedy=metrics(previous), sampled_t08=metrics(current))
    save(work / 'comparison.json', dict(status='complete', groups=groups, paired_tasks=len(current),
        auxiliary_sampled_all_tasks=metrics(all_current), completed_at=now(), limitations=plan['limitations'],
        missing_baseline_note='Source repeat1 was stopped by user; primary comparison excludes inputs without saved greedy answers'))
    lines = ['# GLM视觉学习回忆后答题：贪心与采样对照', '',
        '| 解码 | 测试模态 | 题数 | 准确率 | 无效率 | 截断 | 句子重复≥3次 |', '|---|---|---:|---:|---:|---:|---:|']
    for arm, name in [('greedy', 'temperature0'), ('sampled_t08', 'temperature0.8')]:
        for mod in 'TV':
            g = groups[arm][mod + '/all']
            lines.append(f'| {name} | {mod} | {g["n"]} | {g["accuracy"]:.2%} | {g["invalid_rate"]:.2%} | {g["length"]} | {g["sentence_repeat_3plus"]} |')
    lines += ['', f'采样条件随机50规则、200题；主对照仅使用旧评测已有回复的{len(current)}题，缺少贪心回复的题不混入对照。'
        '同一checkpoint、原始回忆、题目、图片、简短推理提示与1024输出上限。',
        '采样开启top_p0.95、top_k50，未加重复惩罚；固定每批随机种子。一次采样结果仅为小规模诊断，错误回忆没有被修复。', '']
    (work / 'RESULTS_ZH.md').write_text('\n'.join(lines))


def readiness(work, plan):
    dependency = Path(plan['queue_dependency'])
    d = json.loads((dependency / 'status.json').read_text())
    if d.get('stage') != 'complete':
        return 'queued_after_qwen38', dict(dependency_stage=d.get('stage'))
    assert json.loads((dependency / 'comparison.json').read_text())['status'] == 'complete'
    assert json.loads((Path(plan['source_run']) / 'status.json').read_text())['stage'] == 'complete'
    return 'ready', {}


def controller(root, work):
    with (work / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = prepare(root, work)
        if plan['status'] == 'complete':
            return
        def status(stage, **extra):
            save(work / 'status.json', dict(stage=stage, controller_pid=os.getpid(),
                updated_at=now(), host=socket.gethostname(), inference_started=stage == 'evaluating', **extra))
        while True:
            stage, detail = readiness(work, plan)
            if stage == 'ready':
                try:
                    idle('0')
                    break
                except RuntimeError as error:
                    stage, detail = 'waiting_for_free_gpu', dict(reason=str(error))
            status(stage, **detail)
            time.sleep(30)
        verify(work, plan)
        cmd = [str(root / '.venv-glm/bin/python'), '-u',
            str(work / 'source_snapshots/run_rules_glm_temperature_pilot.py'),
            '--root', str(root), '--work', str(work), '--stage', 'evaluate']
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1', MODELSCOPE_OFFLINE='1',
            TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', MAX_PIXELS='262144')
        save(work / 'evaluation_command.json', cmd)
        with (work / 'evaluation.log').open('a') as log:
            child = subprocess.Popen(cmd, cwd=root, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        status('evaluating', child_pid=child.pid)
        if child.wait():
            raise RuntimeError('Temperature pilot failed; outputs retained; inspect evaluation.log')
        summarize(work, plan)
        plan.update(status='complete', completed_at=now())
        save(work / 'experiment.json', plan)
        status('complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--stage', choices=['prepare', 'evaluate'])
    args = parser.parse_args()
    root, work = args.root.resolve(), args.work.resolve()
    try:
        if args.stage == 'prepare':
            prepare(root, work)
        elif args.stage == 'evaluate':
            evaluate(work, json.loads((work / 'experiment.json').read_text()))
        else:
            controller(root, work)
    except Exception as error:
        save(work / ('failure_evaluate.json' if args.stage == 'evaluate' else 'status.json'),
             dict(stage='failed', error=repr(error), updated_at=now()))
        raise
