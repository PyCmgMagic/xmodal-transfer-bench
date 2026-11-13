#!/usr/bin/env python3
"""Queue a paired rule-given diagnostic on clean GLM, without changing training runs."""
import argparse
from collections import Counter
import copy
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import socket
import subprocess
import time

from answer_readout import options_from_question
from audit_rules_glm46_readout import VERSION as READOUT_VERSION, reparse, scores
from entityqa_common import digest, now, rows, save, sha, write_rows
from evaluate_entity_glm46_full import ANSWER_STOP_SEQUENCES, configure_generation, glm_readout
from evaluate_rules_glm46 import tasks_for as closed_book_tasks
from run_entity_glm46_full import BASE
from run_entityqa_lora import idle

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'runs/rules1000_glm46_lora_ep3_original_20261106'
DEPENDENCY = ROOT / 'runs/rules1000_repeat10_glm46_lora_ep3_20261107'
WORK = ROOT / 'runs/rules1000_glm46_rule_given_20261108'
PROMPT_VERSION = 'rules-v7-rule-given-verbatim-brief-v1'
SOURCES = ['run_rules_glm46_rule_given.py', 'evaluate_rules_glm46.py',
    'audit_rules_glm46_readout.py', 'evaluate_entity_glm46_full.py',
    'entityqa_common.py', 'entityqa_prompt_neutral_v5.py', 'run_entity_glm46_full.py',
    'run_entityqa_lora.py', 'answer_readout.py', 'answer_readout_entity_v2.py',
    'answer_readout_entity_v3.py', 'answer_readout_entity_v4.py']


def freeze(source, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        assert sha(source) == sha(dest), str(dest)
    else:
        shutil.copy2(source, dest)


def with_rule(payload, material):
    assert set(payload) <= {'messages', 'images'}
    assert [m['role'] for m in payload['messages']] == ['user']
    assert isinstance(material, str) and material.strip() and '<image>' not in material
    result = copy.deepcopy(payload)
    question = result['messages'][0]['content']
    result['messages'][0]['content'] = (
        'Reference rule for this question (use the supplied rule):\n'
        + material + '\n\nQuestion:\n' + question)
    assert result['messages'][0]['content'].count('<image>') == len(result.get('images', []))
    return result


def build_tasks(work):
    learning = rows(work / 'data/learning_inputs.jsonl')
    assert len(learning) == len({r['qid'] for r in learning}) == 1000
    materials = {}
    for row in learning:
        assert [m['role'] for m in row['messages']] == ['assistant']
        materials[row['qid']] = row['messages'][0]['content']
    tasks = []
    for task in closed_book_tasks(work):
        material = materials[task['qid']]
        question = task['payload']['messages'][0]['content']
        options = options_from_question(question)
        assert set(options) == set('ABCD')
        payload = with_rule(task['payload'], material)
        tasks.append(dict(task, payload=payload, options=options,
            rule_sha256=digest(material), closed_book_input_sha256=digest(task['payload'])))
    qids = sorted(materials)
    random.Random(42).shuffle(qids)
    rank = {qid: i for i, qid in enumerate(qids)}
    tasks.sort(key=lambda t: (rank[t['qid']], ['direct', 'diagnosis'].index(t['question_type']), t['modality']))
    assert len(tasks) == 4000
    for i in range(0, 4000, 4):
        batch = tasks[i:i + 4]
        assert len({r['qid'] for r in batch}) == 1
        for offset in [0, 2]:
            a, b = batch[offset:offset + 2]
            assert (a['modality'], b['modality']) == ('T', 'V')
            assert a['id'] == b['id'] and a['gold'] == b['gold']
            assert a['options'] == b['options'] and a['rule_sha256'] == b['rule_sha256']
    return tasks


def input_fingerprint(tasks):
    return digest([{k: v for k, v in task.items() if k != 'gold'} for task in tasks])


def prepare(work):
    work.mkdir(parents=True, exist_ok=True)
    parent = json.loads((PARENT / 'experiment.json').read_text())
    assert parent['status'] == 'complete'
    assert (BASE / 'download_complete.json').is_file()
    for source in (PARENT / 'evaluation_data').rglob('*'):
        if source.is_file():
            freeze(source, work / 'evaluation_data' / source.relative_to(PARENT / 'evaluation_data'))
    for name in ['learning_inputs.jsonl', 'rule_unique_training.jsonl']:
        freeze(PARENT / 'data' / name, work / 'data' / name)
    materials, sft = rows(work / 'data/learning_inputs.jsonl'), rows(work / 'data/rule_unique_training.jsonl')
    assert len(materials) == len(sft) == 1000
    for material, example in zip(materials, sft):
        assert material['qid'] == example['qid']
        assert material['messages'][0]['content'] == example['messages'][-1]['content']
    inventory = json.loads((work / 'evaluation_data/package_inventory.json').read_text())
    for rel, expected in inventory['sha256'].items():
        assert sha(work / 'evaluation_data' / rel) == expected, rel
    # Verify referenced images on CPU, including all learning/test image hashes.
    source_images = ROOT / 'data/rules_repaired_single_image_v7'
    image_rows = rows(work / 'evaluation_data/image_hashes.jsonl')
    assert len(image_rows) == 3000
    for row in image_rows:
        assert sha(source_images / row['path']) == row['repaired_sha256'], row['path']
    tasks = build_tasks(work)
    for mod in 'TV':
        subset = [r for r in tasks if r['modality'] == mod]
        assert len(subset) == 2000
        assert Counter(r['gold'] for r in subset) == dict.fromkeys('ABCD', 500)
        assert Counter(r['question_type'] for r in subset) == dict(direct=1000, diagnosis=1000)
        for task in subset:
            images = task['payload'].get('images', [])
            assert len(images) == (0 if mod == 'T' else 1)
            assert all(Path(path).name.startswith('test_visual_') for path in images)
    # Gold lives in audit metadata only. Inference files contain exactly InferRequest fields.
    for rel, selected in [('inference_inputs.jsonl', [r['payload'] for r in tasks]),
                          ('task_manifest.jsonl', [{k: v for k, v in r.items() if k != 'payload'} for r in tasks])]:
        target = work / 'data' / rel
        if target.exists():
            assert rows(target) == selected
        else:
            write_rows(target, selected)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(BASE), local_files_only=True)
    lengths = [len(tokenizer.encode(t['payload']['messages'][0]['content'], add_special_tokens=False)) for t in tasks]
    assert max(lengths) < 4096, 'Unexpectedly long rule plus question'
    audit = dict(status='pass', checked_at=now(), knowledge_units=1000, total_tasks=4000,
        original_learning_targets_verbatim=True, paired_ids_rules_options_labels_verified=True,
        inference_payload_keys=['messages', 'images'], labels_not_injected=True,
        train_images_not_injected=True, verified_source_images=3000,
        max_text_prompt_tokens=max(lengths), max_output_tokens=1024,
        prompt_length_note='Text tokens only; test images add visual tokens. Model context 131072, max_pixels 262144.',
        inference_inputs_sha256=sha(work / 'data/inference_inputs.jsonl'),
        task_manifest_sha256=sha(work / 'data/task_manifest.jsonl'))
    save(work / 'data/preflight.json', audit)
    for name in SOURCES:
        freeze(ROOT / 'scripts' / name, work / 'source_snapshots' / name)
    plan = dict(status='prepared', created_at=now(), host=socket.gethostname(), gpu='0',
        model='GLM-4.6V-Flash', model_path=str(BASE), adapters=None, training=False,
        framework='ms-swift TransformersEngine', ms_swift=importlib.metadata.version('ms-swift'),
        environment=str(ROOT / '.venv-glm'), source_learning_run=str(PARENT),
        queue_dependency=str(DEPENDENCY), prompt_protocol=PROMPT_VERSION,
        knowledge_units=1000, paired_questions_per_modality=2000, total=4000,
        rule_input_modality='TEXT, original complete learning material',
        question_modalities=['TEXT', 'VISION'],
        order='seed42 shuffled rule IDs; direct T/V then diagnosis T/V per rule',
        input_sha256=input_fingerprint(tasks), max_tokens=1024, temperature=0, seed=42,
        native_thinking=False, brief_prompted_reasoning=True, vision_max_pixels=262144,
        batch_size=4, parser=READOUT_VERSION, no_finalizer=True,
        material_sha256=sha(work / 'data/learning_inputs.jsonl'),
        source_sha256={name: sha(work / 'source_snapshots' / name) for name in SOURCES},
        limitations=[
            'Rule-given open-book application diagnostic, not a learned cross-modal transfer score.',
            'An incorrect model answer is a review candidate, not proof of a dataset defect.',
            'Image semantics are not independently certified by image hashes.',
            '104 direct questions test component recording; 241 questions per modality are rule_report_audit.',
            'Both modalities receive text rules; only the test observation modality changes.',
            'The original learning material includes its existing worked example; no test answer is added.',
        ])
    path = work / 'experiment.json'
    if path.exists():
        existing = json.loads(path.read_text())
        for key in ['input_sha256', 'material_sha256', 'source_sha256', 'max_tokens', 'parser']:
            assert existing[key] == plan[key], key
        plan = existing
    save(path, plan)
    (work / 'README_ZH.md').write_text(
        '# GLM规则原文给定后答题：数据可解性诊断\n\n'
        'GLM-4.6V-Flash干净基座，不加LoRA。全部1000条规则，文字/图片各2000题。'
        '每题将对应原始完整学习材料直接加入用户输入，规则文本与训练目标逐字一致；'
        '保留原题、全部选项、测试图片和简短推理boxed提示，不提供测试答案。\n\n'
        '原生thinking关闭，输出1024 token，temperature0，seed42，batch4。'
        '按打乱后的规则顺序，每条规则direct/diagnosis和文字/图片配对。'
        '等重复10次训练模型的评测完成且GPU0空闲后启动，不中止其他任务。\n\n'
        '报告准确率、无效率、规则族分项，以及配对的两者都对/仅文字对/仅图片对/都错。'
        '错题连同完整规则、问题、选项、原回复存入待审查清单；模型错题不直接判为数据错误。'
        '本实验是开卷规则应用，不能混入闭卷跨模态迁移成绩。\n')
    return plan


def metrics(records):
    result = scores(records, 'predicted')
    result.update(legacy=scores(records, 'raw_predicted'),
        parse_reasons=dict(Counter(r['parse_reason'] for r in records)),
        stop_reasons=dict(Counter(r['stop_reason'] for r in records)))
    return result


def summaries(records):
    result = {}
    for mod in 'TV':
        selected = [r for r in records if r['modality'] == mod]
        if not selected:
            continue
        result[mod + '/all'] = metrics(selected)
        for field in ['question_type', 'family', 'evaluation_group', 'task_variant']:
            for value in sorted({r[field] for r in selected if r[field]}):
                result[f'{mod}/{field}/{value}'] = metrics([r for r in selected if r[field] == value])
    return result


def pair_counts(records):
    indexed = {(r['id'], r['modality']): r for r in records}
    pairs, counts = [], Counter()
    for r in records:
        if r['modality'] != 'T' or (r['id'], 'V') not in indexed:
            continue
        v = indexed[r['id'], 'V']
        assert r['gold'] == v['gold'] and r['rule_sha256'] == v['rule_sha256']
        state = 'both_correct' if r['correct'] and v['correct'] else (
            'text_only_correct' if r['correct'] else 'vision_only_correct' if v['correct'] else 'both_wrong')
        counts[state] += 1
        pairs.append(dict(id=r['id'], qid=r['qid'], question_type=r['question_type'], family=r['family'],
            gold=r['gold'], text_predicted=r['predicted'], vision_predicted=v['predicted'], outcome=state))
    return dict(counts), pairs


def evaluate(work):
    plan = json.loads((work / 'experiment.json').read_text())
    base = Path(plan['model_path'])
    assert (base / 'download_complete.json').is_file()
    tasks = build_tasks(work)
    assert input_fingerprint(tasks) == plan['input_sha256']
    assert sha(work / 'data/learning_inputs.jsonl') == plan['material_sha256']
    output = work / 'evaluation/base_rule_given'
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'predictions.jsonl'
    records = rows(path) if path.exists() else []
    config = dict(model=str(base), adapter=None, input_sha256=plan['input_sha256'],
        input_count=4000, prompt_protocol=PROMPT_VERSION, max_tokens=1024, temperature=0,
        seed=42, thinking=False, vision_max_pixels=262144, batch_size=4, gpu='0',
        readout_version=READOUT_VERSION, stop_sequences=ANSWER_STOP_SEQUENCES,
        framework='ms-swift TransformersEngine', oracle_material=True, finalizer=False)
    configure_generation(output, config, records)
    lookup = {(t['id'], t['modality']): t for t in tasks}
    seen = set()
    for r in records:
        key = r['id'], r['modality']
        assert key in lookup and key not in seen
        assert r['gold'] == lookup[key]['gold'] and r['input_sha256'] == digest(lookup[key]['payload'])
        seen.add(key)
    pending = [r for r in tasks if (r['id'], r['modality']) not in seen]
    if pending:
        os.environ.update(CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1', MODELSCOPE_OFFLINE='1',
            TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', MAX_PIXELS='262144')
        import torch
        from swift import TransformersEngine, InferRequest, RequestConfig
        torch.set_num_threads(4)
        engine = TransformersEngine(str(base), model_type='glm4v', template_type='glm4_5v',
            torch_dtype=torch.bfloat16, attn_impl='sdpa', device_map='cuda:0', max_batch_size=4)
        engine.template.enable_thinking = False
        request = RequestConfig(max_tokens=1024, temperature=0, seed=42,
            return_details=True, stop=ANSWER_STOP_SEQUENCES)
        started = time.monotonic()
        for start in range(0, len(pending), 4):
            batch = pending[start:start + 4]
            replies = engine.infer([InferRequest(**copy.deepcopy(t['payload'])) for t in batch],
                request, use_tqdm=False)
            assert len(replies) == len(batch)
            new = []
            for task, reply in zip(batch, replies):
                if isinstance(reply, Exception):
                    raise reply
                choice = reply.choices[0]
                response = choice.message.content or ''
                legacy, legacy_reason = glm_readout(response, task['options'])
                predicted, reason = reparse(response, task['options'])
                record = {k: v for k, v in task.items() if k not in {'payload', 'options'}}
                record.update(arm='base_rule_given', raw_response=response,
                    raw_predicted=legacy, raw_parse_reason=legacy_reason,
                    predicted=predicted, parse_reason=reason, correct=predicted == task['gold'],
                    finalizer=None, stop_reason=str(choice.finish_reason),
                    generated_tokens=reply.usage.completion_tokens, generation_max_tokens=1024,
                    input_sha256=digest(task['payload']), saved_at=now())
                new.append(record)
            with path.open('a', encoding='utf-8') as stream:
                for record in new:
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            records.extend(new)
            pairs, _ = pair_counts(records)
            progress = dict(stage='generating', saved=len(records), total=4000,
                elapsed_seconds=round(time.monotonic() - started, 1),
                groups=summaries(records), paired_outcomes=pairs, updated_at=now())
            save(output / 'status.json', progress)
            save(output / 'summary.json', progress)
            print(json.dumps(dict(saved=len(records), total=4000,
                groups={mod: progress['groups'].get(mod + '/all') for mod in 'TV'})), flush=True)
        del engine
    assert len(records) == 4000
    pairs, detail = pair_counts(records)
    assert len(detail) == 2000 and sum(pairs.values()) == 2000
    save(output / 'summary.json', dict(status='complete', saved=4000, total=4000,
        groups=summaries(records), paired_outcomes=pairs, completed_at=now(),
        interpretation='Rule-given base model diagnostic; text reference rules for both test modalities.'))
    save(output / 'status.json', dict(status='complete', saved=4000, total=4000, updated_at=now()))
    write_rows(work / 'paired_results.jsonl', detail)
    reviews = []
    materials = {r['qid']: r['messages'][0]['content'] for r in rows(work / 'data/learning_inputs.jsonl')}
    for r in records:
        if not r['correct']:
            task = lookup[r['id'], r['modality']]
            reviews.append(dict(r, learning_rule=materials[r['qid']], options=task['options'],
                exact_inference_payload=task['payload'], review_status='unreviewed_model_error_not_confirmed_data_defect'))
    write_rows(work / 'needs_review.jsonl', reviews)
    group = summaries(records)
    lines = ['# 规则原文给定后答题：GLM基座', '',
        '| 测试输入 | 题数 | 准确率 | 无效率 |', '|---|---:|---:|---:|']
    for mod, name in [('T', '规则文本+文字题'), ('V', '规则文本+图片题')]:
        m = group[mod + '/all']
        lines.append(f'| {name} | {m["n"]} | {m["accuracy"]:.2%} | {m["invalid_rate"]:.2%} |')
    lines += ['', '每条题目都给出对应原始完整规则；干净GLM基座，无训练，无LoRA。',
        '简短推理提示，原生thinking关闭，输出上限1024 token；无效计错。', '',
        '配对结果（共2000对）：' + json.dumps(pairs, ensure_ascii=False), '',
        '详细规则族、direct/diagnosis、rule_report_audit/result_application等分组见evaluation/base_rule_given/summary.json。',
        '错题包含原规则、题目、选项和回复，见needs_review.jsonl。模型错误不直接等于数据错误。',
        '这是提供规则的开卷应用诊断，不属于闭卷学习迁移成绩。', '']
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
    while True:
        previous = json.loads((DEPENDENCY / 'status.json').read_text())
        if previous['stage'] == 'complete':
            break
        status('queued_after_repeat10_evaluation', dependency=str(DEPENDENCY),
            dependency_stage=previous['stage'], rule_given_tasks=4000, inference_started=False)
        time.sleep(30)
    while True:
        try:
            idle('0')
            break
        except RuntimeError as error:
            status('waiting_for_free_gpu', reason=str(error), inference_started=False)
            time.sleep(30)
    for name, expected in plan['source_sha256'].items():
        assert sha(work / 'source_snapshots' / name) == expected, name
    log = work / 'evaluation.log'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1', MODELSCOPE_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', MAX_PIXELS='262144', PYTHONUNBUFFERED='1')
    command = [str(ROOT / '.venv-glm/bin/python'), '-u',
        str(work / 'source_snapshots/run_rules_glm46_rule_given.py'), '--work', str(work), '--evaluate-only']
    save(work / 'evaluation_command.json', command)
    with log.open('a') as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    status('evaluating_rule_given', child_pid=child.pid, log=str(log))
    if child.wait():
        raise RuntimeError(f'Rule-given evaluation failed; inspect {log}; outputs preserved')
    plan.update(status='complete', completed_at=now())
    save(work / 'experiment.json', plan)
    status('complete', results=str(work / 'RESULTS_ZH.md'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=WORK)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args()
    work = args.work.resolve()
    try:
        if args.prepare_only:
            prepare(work)
        elif args.evaluate_only:
            evaluate(work)
        else:
            controller(work)
    except Exception as error:
        target = work / 'evaluation/base_rule_given/failure.json' if args.evaluate_only else work / 'status.json'
        save(target, dict(stage='failed', error=repr(error), updated_at=now()))
        raise
