#!/usr/bin/env python3
"""Apply previously generated rule memories with the same frozen GLM adapters."""
import argparse
from collections import Counter
import copy
import fcntl
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time

from answer_readout import options_from_question
from audit_rules_glm46_readout import VERSION as READOUT_VERSION, reparse
from entityqa_common import digest, now, rows, save, sha, write_rows
from evaluate_entity_glm46_full import ANSWER_STOP_SEQUENCES, configure_generation, glm_readout
from evaluate_rules_glm46 import tasks_for as closed_tasks
from run_entityqa_lora import idle
from run_rules_glm46_free_recall import answer_text
from run_rules_glm46_rule_given import (
    SOURCES as RULE_GIVEN_SOURCES, freeze, pair_counts, summaries, with_rule,
)

VERSION = 'rules-v7-frozen-recall-then-apply-same-adapter-v1'
ARMS = ['repeat10', 'repeat1']
SOURCES = list(dict.fromkeys(RULE_GIVEN_SOURCES + [
    'run_rules_glm46_free_recall.py', 'run_rules_glm46_recall_apply.py']))


def normalized(text):
    return re.sub(r'\s+', ' ', text).strip()


def memory_index(recalls, arm):
    result = {}
    for row in recalls:
        assert row['arm'] == arm and row['qid'] not in result, 'Duplicate or wrong-arm memory'
        text, _ = answer_text(row['raw_response'])
        assert text == row['answer_for_judge'], 'Saved recall does not match its raw response'
        assert '<image>' not in text, 'A recalled rule cannot inject another image'
        result[row['qid']] = row
    return result


def build_tasks(questions, recalls, arm):
    memories = memory_index(recalls, arm)
    assert set(memories) == {q['qid'] for q in questions}, 'Missing or unrelated recalled rule'
    tasks = []
    for question in questions:
        memory = memories[question['qid']]
        material = memory['answer_for_judge']
        # Preserve even an empty memory, never substitute the reference or discard a failed recall.
        if material.strip():
            payload = with_rule(question['payload'], material)
        else:
            payload = copy.deepcopy(question['payload'])
            payload['messages'][0]['content'] = (
                'Reference rule for this question (use the supplied rule):\n'
                + material + '\n\nQuestion:\n' + payload['messages'][0]['content'])
        assert set(payload) <= {'messages', 'images'}
        assert [m['role'] for m in payload['messages']] == ['user']
        assert payload['messages'][0]['content'].count('<image>') == len(payload.get('images', []))
        tasks.append(dict(question, payload=payload,
            options=options_from_question(question['payload']['messages'][0]['content']),
            rule_sha256=digest(material), recall_raw_sha256=digest(memory['raw_response']),
            recall_id=memory['id'], closed_book_input_sha256=digest(question['payload'])))
    assert len({(t['id'], t['modality']) for t in tasks}) == len(tasks)
    return tasks


def store_rows(path, records):
    if path.exists():
        assert rows(path) == records, str(path)
    else:
        write_rows(path, records)


def verify_files(work, plan):
    for relative, expected in plan['immutable_sha256'].items():
        assert sha(work / relative) == expected, relative
    for name, expected in plan['source_sha256'].items():
        assert sha(work / 'source_snapshots' / name) == expected, name


def prepare(root, work):
    work.mkdir(parents=True, exist_ok=True)
    plan_path = work / 'experiment.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        assert plan['project_root'] == str(root) and plan['protocol'] == VERSION
        verify_files(work, plan)
        return plan
    recall_run = root / 'runs/rules1000_glm46_free_recall_ai_judge_20261108'
    given_run = root / 'runs/rules1000_glm46_rule_given_20261108'
    repeat_run = root / 'runs/rules1000_repeat10_glm46_lora_ep3_20261107'
    recall_plan = json.loads((recall_run / 'experiment.json').read_text())
    given_plan = json.loads((given_run / 'experiment.json').read_text())
    assert recall_plan['status'] == given_plan['status'] == 'complete'
    assert given_plan['model_path'] == recall_plan['model_path']
    assert sha(recall_run / 'data/tasks.jsonl') == recall_plan['task_sha256']
    for name, expected in given_plan['source_sha256'].items():
        assert sha(root / 'scripts' / name) == expected, 'Existing protocol changed: ' + name
    for source, target in [
        (recall_run / 'experiment.json', work / 'references/recall_experiment.json'),
        (recall_run / 'data/tasks.jsonl', work / 'references/recall_tasks.jsonl'),
        (given_run / 'experiment.json', work / 'references/rule_given_experiment.json'),
        (given_run / 'data/task_manifest.jsonl', work / 'references/rule_given_task_manifest.jsonl'),
        (given_run / 'evaluation/base_rule_given/predictions.jsonl', work / 'references/base_original_rule.jsonl'),
    ]:
        freeze(source, target)
    for arm in ['base'] + ARMS:
        freeze(repeat_run / f'rescored/{arm}.jsonl', work / f'references/closed_{arm}.jsonl')
    manifest = rows(work / 'references/rule_given_task_manifest.jsonl')
    by_key = {(q['id'], q['modality']): q for q in closed_tasks(given_run)}
    assert len(manifest) == len(by_key) == 4000
    questions = [by_key[m['id'], m['modality']] for m in manifest]
    for metadata, question in zip(manifest, questions):
        assert all(question[k] == metadata[k] for k in question if k != 'payload')
        assert digest(question['payload']) == metadata['closed_book_input_sha256']
        assert options_from_question(question['payload']['messages'][0]['content']) == metadata['options']
    store_rows(work / 'data/question_tasks.jsonl', questions)
    # Hash actual test images, and cross-check against the repaired dataset inventory.
    image_inventory = {r['path']: r['repaired_sha256']
                       for r in rows(given_run / 'evaluation_data/image_hashes.jsonl')}
    images = sorted({image for q in questions for image in q['payload'].get('images', [])})
    assert len(images) == 2000
    verified_images = []
    for image in images:
        path = Path(image)
        relative = 'images/' + path.parent.name + '/' + path.name
        assert path.name.startswith('test_visual_')
        actual = sha(path)
        assert actual == image_inventory[relative], image
        verified_images.append(dict(path=image, sha256=actual))
    store_rows(work / 'data/image_manifest.jsonl', verified_images)
    references = {t['qid']: t['reference'] for t in rows(work / 'references/recall_tasks.jsonl')}
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(recall_plan['model_path'], local_files_only=True)
    arms, audits = {}, {}
    for arm in ARMS:
        ck = recall_plan['arms'][arm]
        for name, expected in ck['sha256'].items():
            assert sha(Path(ck['path']) / name) == expected
            if name != 'adapter_model.safetensors':
                freeze(Path(ck['path']) / name, work / 'references' / arm / name)
        for name in ['responses.jsonl', 'config.json', 'status.json']:
            freeze(recall_run / 'recall' / arm / name, work / 'recall_sources' / arm / name)
        recalls = rows(work / 'recall_sources' / arm / 'responses.jsonl')
        assert len(recalls) == 1000
        assert json.loads((work / 'recall_sources' / arm / 'status.json').read_text())['status'] == 'complete'
        tasks = build_tasks(questions, recalls, arm)
        for mod in 'TV':
            selected = [t for t in tasks if t['modality'] == mod]
            assert len(selected) == 2000
            assert Counter(t['gold'] for t in selected) == dict.fromkeys('ABCD', 500)
            assert Counter(t['question_type'] for t in selected) == dict(direct=1000, diagnosis=1000)
        for i in range(0, 4000, 4):
            batch = tasks[i:i + 4]
            assert len({t['qid'] for t in batch}) == len({t['rule_sha256'] for t in batch}) == 1
            for a, b in [batch[:2], batch[2:]]:
                assert (a['modality'], b['modality']) == ('T', 'V')
                assert a['id'] == b['id'] and a['gold'] == b['gold'] and a['options'] == b['options']
        lengths = [len(tokenizer.encode(t['payload']['messages'][0]['content'], add_special_tokens=False)) for t in tasks]
        assert max(lengths) < 4096
        store_rows(work / f'data/tasks_{arm}.jsonl', tasks)
        store_rows(work / f'data/inference_inputs_{arm}.jsonl', [t['payload'] for t in tasks])
        arms[arm] = dict(ck, input_sha256=digest([t['payload'] for t in tasks]))
        audits[arm] = dict(tasks=4000, rules=1000, max_text_prompt_tokens=max(lengths),
            empty_recalled_rules=sum(not r['answer_for_judge'].strip() for r in recalls),
            recall_length_limited=sum(r['stop_reason'] == 'length' for r in recalls),
            whitespace_normalized_exact_memories=sum(normalized(r['answer_for_judge']) ==
                normalized(references[r['qid']]) for r in recalls))
    for name in SOURCES:
        freeze(root / 'scripts' / name, work / 'source_snapshots' / name)
    immutable = {str(p.relative_to(work)): sha(p)
                 for directory in ['data', 'references', 'recall_sources']
                 for p in (work / directory).rglob('*') if p.is_file()}
    plan = dict(status='prepared', created_at=now(), project_root=str(root),
        host=socket.gethostname(), gpu='0', model='GLM-4.6V-Flash',
        model_path=recall_plan['model_path'], arms=arms, protocol=VERSION,
        environment=str(root / '.venv-glm'), framework='ms-swift TransformersEngine',
        recall_source=str(recall_run), knowledge_units=1000, tasks_per_arm=4000, total=8000,
        execution_order=ARMS, application_model='Same adapter that generated the recalled rule',
        memory_source='Previously saved answer_for_judge, unchanged; never replaced by reference',
        prompt_wrapper='Identical to existing original-rule-given diagnostic',
        max_tokens=1024, native_thinking=False, brief_prompted_reasoning=True,
        temperature=0, seed=42, batch_size=4, vision_max_pixels=262144,
        parser=READOUT_VERSION, stop_sequences=ANSWER_STOP_SEQUENCES,
        no_finalizer=True, training=False, filtering_by_recall_correctness=False,
        immutable_sha256=immutable,
        source_sha256={name: sha(work / 'source_snapshots' / name) for name in SOURCES},
        limitations=[
            'Two-stage memory-assisted rule application, not closed-book direct transfer accuracy.',
            'Stage-one recall used the original training question, with only rule names.',
            'Text memories are supplied for both text and image questions.',
            'Clean-base original-rule results use different weights; same-adapter closed-book comparison is primary.',
            'Recall-stage AI judgment is not used to select, fix, or exclude any memory.',
        ])
    save(work / 'data/preflight.json', dict(status='pass', checked_at=now(), arms=audits,
        verified_test_images=2000, same_questions_options_labels_images=True,
        recall_pairing_by_qid=True, no_reference_fallback=True, labels_not_injected=True))
    plan['immutable_sha256']['data/preflight.json'] = sha(work / 'data/preflight.json')
    save(plan_path, plan)
    (work / 'README_ZH.md').write_text(
        '# GLM先回忆再答规则题\n\n'
        '复用已经独立生成的规则回忆，按qid与题目匹配，原样放到题目前；'
        '回答阶段加载生成该回忆的同一个TEXT LoRA。先跑重复10次组，再跑每轮1次组。'
        '每组1000规则，文字2000题、图片2000题，direct/diagnosis配对，共8000次回答。\n\n'
        '不按回忆正确性筛选，也不使用原规则修正错误回忆。原规则、标准答案、AI裁判结论只用于审计。'
        '保留原题、选项与测试图。使用既有简短解释boxed提示，原生thinking关闭，'
        'max_tokens=1024、temperature=0、seed=42、batch=4，无强制补选。\n\n'
        '主要对比同一LoRA的闭卷答题；另列干净基座闭卷、干净基座给原规则成绩。'
        '后者权重不同，不能把全部差异归因于回忆。规则应用诊断与闭卷迁移成绩分开报告。\n')
    return plan


def load_tasks(work, plan, arm):
    verify_files(work, plan)
    tasks = rows(work / f'data/tasks_{arm}.jsonl')
    assert len(tasks) == 4000
    assert digest([t['payload'] for t in tasks]) == plan['arms'][arm]['input_sha256']
    rebuilt = build_tasks(rows(work / 'data/question_tasks.jsonl'),
                          rows(work / 'recall_sources' / arm / 'responses.jsonl'), arm)
    assert tasks == rebuilt
    return tasks


def append(path, records):
    with path.open('a', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def evaluate(work, arm):
    plan = json.loads((work / 'experiment.json').read_text())
    tasks = load_tasks(work, plan, arm)
    ck = plan['arms'][arm]
    for name, expected in ck['sha256'].items():
        assert sha(Path(ck['path']) / name) == expected
    for image in rows(work / 'data/image_manifest.jsonl'):
        assert sha(Path(image['path'])) == image['sha256']
    output = work / 'evaluation' / arm
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'predictions.jsonl'
    records = rows(path) if path.exists() else []
    config = dict(model=plan['model_path'], adapter=ck['path'],
        adapter_sha256=ck['sha256']['adapter_model.safetensors'], input_sha256=ck['input_sha256'],
        input_count=4000, protocol=VERSION, max_tokens=1024, temperature=0, seed=42,
        native_thinking=False, brief_prompted_reasoning=True, vision_max_pixels=262144,
        batch_size=4, parser=READOUT_VERSION, stop_sequences=ANSWER_STOP_SEQUENCES,
        finalizer=False, recalled_material=True, oracle_material=False,
        memory_file_sha256=sha(work / 'recall_sources' / arm / 'responses.jsonl'))
    configure_generation(output, config, records)
    lookup = {(t['id'], t['modality']): t for t in tasks}
    seen = set()
    for record in records:
        key = record['id'], record['modality']
        assert key not in seen and key in lookup and record['arm'] == arm
        assert record['gold'] == lookup[key]['gold']
        assert record['input_sha256'] == digest(lookup[key]['payload'])
        assert record['rule_sha256'] == lookup[key]['rule_sha256']
        seen.add(key)
    pending = [t for t in tasks if (t['id'], t['modality']) not in seen]
    if pending:
        idle('0')
        os.environ.update(CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1', MODELSCOPE_OFFLINE='1',
            TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', MAX_PIXELS='262144')
        import torch
        from swift import InferRequest, RequestConfig, TransformersEngine
        torch.set_num_threads(4)
        engine = TransformersEngine(plan['model_path'], model_type='glm4v', template_type='glm4_5v',
            adapters=[ck['path']], torch_dtype=torch.bfloat16, attn_impl='sdpa',
            device_map='cuda:0', max_batch_size=4)
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
                raw = choice.message.content or ''
                legacy, legacy_reason = glm_readout(raw, task['options'])
                predicted, reason = reparse(raw, task['options'])
                item = {k: v for k, v in task.items() if k not in {'payload', 'options'}}
                item.update(arm=arm, raw_response=raw, raw_predicted=legacy,
                    raw_parse_reason=legacy_reason, predicted=predicted, parse_reason=reason,
                    correct=predicted == task['gold'], finalizer=None,
                    stop_reason=str(choice.finish_reason), generated_tokens=reply.usage.completion_tokens,
                    generation_max_tokens=1024, input_sha256=digest(task['payload']), saved_at=now())
                new.append(item)
            append(path, new)
            records.extend(new)
            paired, _ = pair_counts(records)
            progress = dict(stage='generating', arm=arm, saved=len(records), total=4000,
                updated_at=now(), elapsed_seconds=round(time.monotonic() - started, 1),
                groups=summaries(records), paired_outcomes=paired)
            save(output / 'status.json', progress)
            save(output / 'summary.json', progress)
            print(json.dumps(dict(arm=arm, saved=len(records), total=4000,
                groups={mod: progress['groups'].get(mod + '/all') for mod in 'TV'})), flush=True)
        del engine
    assert len(records) == 4000
    paired, details = pair_counts(records)
    assert len(details) == 2000
    write_rows(work / f'paired_results_{arm}.jsonl', details)
    save(output / 'summary.json', dict(status='complete', arm=arm, saved=4000, total=4000,
        groups=summaries(records), paired_outcomes=paired, completed_at=now()))
    save(output / 'status.json', dict(status='complete', saved=4000, total=4000, updated_at=now()))


def matched_change(closed, assisted):
    lookup = {(r['id'], r['modality']): r for r in closed}
    outcomes = Counter()
    for new in assisted:
        old = lookup[new['id'], new['modality']]
        assert old['gold'] == new['gold'] and old['qid'] == new['qid']
        state = 'both_correct' if old['correct'] and new['correct'] else (
            'improved' if new['correct'] else 'worsened' if old['correct'] else 'both_wrong')
        outcomes[state] += 1
    return dict(outcomes)


def summarize(work):
    plan = json.loads((work / 'experiment.json').read_text())
    verify_files(work, plan)
    questions = {(r['id'], r['modality']): r for r in rows(work / 'data/question_tasks.jsonl')}
    conditions, records_by_condition = {}, {}
    source_files = [('base_closed', 'references/closed_base.jsonl', True),
        ('base_original_rule', 'references/base_original_rule.jsonl', False)]
    for arm in ARMS:
        source_files.extend([(arm + '_closed', f'references/closed_{arm}.jsonl', True),
                             (arm + '_recalled_rule', f'evaluation/{arm}/predictions.jsonl', False)])
    for condition, relative, rescored in source_files:
        records = rows(work / relative)
        assert len(records) == 4000
        assert len({(r['id'], r['modality']) for r in records}) == 4000
        normalized_records = []
        for record in records:
            question = questions[record['id'], record['modality']]
            assert record['gold'] == question['gold'] and record['qid'] == question['qid']
            options = options_from_question(question['payload']['messages'][0]['content'])
            predicted, reason = reparse(record['raw_response'], options)
            expected = record['audit_predicted'] if rescored else record['predicted']
            assert predicted == expected
            normalized_records.append(dict(record, predicted=predicted, parse_reason=reason,
                correct=predicted == record['gold']))
        records_by_condition[condition] = normalized_records
        conditions[condition] = dict(groups=summaries(normalized_records),
            source=str(work / relative), sha256=sha(work / relative))
    changes = {}
    for arm in ARMS:
        changes[arm] = {mod: matched_change(
            [r for r in records_by_condition[arm + '_closed'] if r['modality'] == mod],
            [r for r in records_by_condition[arm + '_recalled_rule'] if r['modality'] == mod])
            for mod in 'TV'}
    result = dict(status='complete', completed_at=now(), conditions=conditions,
        paired_change_from_same_adapter_closed_book=changes,
        interpretation='Two-stage recall and application; same frozen adapter for both stages; no recall correction or filtering.')
    save(work / 'comparison.json', result)
    lines = ['# GLM先回忆再答规则题', '',
        '| 模型与条件 | 文字准确率 | 文字无效率 | 图片准确率 | 图片无效率 |',
        '|---|---:|---:|---:|---:|']
    labels = {'base_closed': '基座：闭卷', 'base_original_rule': '基座：给原规则',
        'repeat10_closed': '重复10次：闭卷', 'repeat10_recalled_rule': '重复10次：给其回忆',
        'repeat1_closed': '每轮1次：闭卷', 'repeat1_recalled_rule': '每轮1次：给其回忆'}
    for condition in labels:
        g = conditions[condition]['groups']
        t, v = g['T/all'], g['V/all']
        lines.append(f'| {labels[condition]} | {t["accuracy"]:.2%} | {t["invalid_rate"]:.2%} | {v["accuracy"]:.2%} | {v["invalid_rate"]:.2%} |')
    lines += ['', '各条件文字、图片各2000题，qid/题目/选项/标准答案配对，无效计错。',
        '新实验把已经生成的回忆原样放到题目前，回答阶段用生成该回忆的同一LoRA。'
        '不替换成原规则，不按AI评分筛选，不提供标准答案或训练图片。',
        '沿用1024 token、简短解释boxed提示、temperature0、seed42、batch4；原生thinking关闭。',
        '基座给原规则是额外参考，模型权重不同；主要对比同一LoRA闭卷与给其回忆。',
        '本实验属于记忆辅助的两阶段规则应用诊断，与闭卷跨模态迁移成绩分开。',
        '同题改善/退化计数见comparison.json；原回复及回忆溯源见evaluation/*/predictions.jsonl。', '']
    (work / 'RESULTS_ZH.md').write_text('\n'.join(lines))
    plan.update(status='complete', completed_at=now())
    save(work / 'experiment.json', plan)
    save(work / 'status.json', dict(stage='complete', updated_at=now(), results=str(work / 'RESULTS_ZH.md')))


def controller(root, work):
    assert socket.gethostname() == 'softengine-KVM'
    work.mkdir(parents=True, exist_ok=True)
    with (work / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = prepare(root, work)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1', MODELSCOPE_OFFLINE='1',
            TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', MAX_PIXELS='262144',
            PYTHONUNBUFFERED='1')
        for arm in ARMS:
            status_path = work / 'evaluation' / arm / 'status.json'
            if status_path.exists() and json.loads(status_path.read_text()).get('status') == 'complete':
                continue
            while True:
                try:
                    idle('0')
                    break
                except RuntimeError as error:
                    save(work / 'status.json', dict(stage='waiting_for_free_gpu', next_arm=arm,
                        reason=str(error), controller_pid=os.getpid(), updated_at=now()))
                    time.sleep(30)
            verify_files(work, plan)
            command = [str(root / '.venv-glm/bin/python'), '-u',
                str(work / 'source_snapshots/run_rules_glm46_recall_apply.py'),
                '--root', str(root), '--work', str(work), '--stage', arm]
            save(work / f'{arm}_command.json', command)
            log = work / f'{arm}.log'
            with log.open('a') as stream:
                child = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            save(work / 'status.json', dict(stage='applying_recalled_rule', arm=arm,
                controller_pid=os.getpid(), child_pid=child.pid, log=str(log), updated_at=now()))
            if child.wait():
                raise RuntimeError(f'{arm} failed; inspect {log}; outputs retained')
        summarize(work)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--stage', choices=ARMS + ['summarize'])
    args = parser.parse_args()
    work = args.work.resolve()
    try:
        if args.prepare_only:
            prepare(args.root.resolve(), work)
        elif args.stage == 'summarize':
            summarize(work)
        elif args.stage:
            evaluate(work, args.stage)
        else:
            controller(args.root.resolve(), work)
    except Exception as error:
        path = work / ('evaluation/' + args.stage + '/failure.json' if args.stage in ARMS else 'status.json')
        save(path, dict(stage='failed', error=repr(error), updated_at=now()))
        raise
