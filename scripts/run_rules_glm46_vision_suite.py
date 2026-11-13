#!/usr/bin/env python3
"""Queue paired VISION rule LoRAs and three evaluations after the active TEXT suite."""
import argparse
from collections import Counter
import copy
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import socket
import subprocess
import time

from answer_readout import options_from_question
from audit_rules_glm46_readout import VERSION as PARSER_VERSION, reparse
from entityqa_common import digest, now, rows, save, sha, write_rows
from evaluate_entity_glm46_full import ANSWER_STOP_SEQUENCES, configure_generation, glm_readout
from evaluate_rules_glm46 import tasks_for
from run_entity_glm46_full import training_command as original_training_command
from run_entityqa_lora import checkpoint as check_adapter, idle
from run_rules_glm46_free_recall import (
    JUDGE_SYSTEM, controls, enforce_recording_evidence,
    parse_case_judgment, recall as original_recall,
)
from run_rules_glm46_recall_apply import (
    SOURCES as PREVIOUS_SOURCES, append, build_tasks as recalled_tasks, matched_change, normalized,
)
from run_rules_glm46_rule_given import freeze, pair_counts, summaries

VERSION = 'rules1000-glm-vision-original-sft-repeat1-repeat10-ep3-v1'
JUDGE_VERSION = 'rules-vision-name-only-semantic-judge-v1'
ARMS = ['repeat1', 'repeat10']
SOURCES = list(dict.fromkeys(PREVIOUS_SOURCES + ['run_rules_glm46_vision_suite.py']))
STAGES = ['smoke', 'train_repeat1', 'train_repeat10',
          'closed_repeat1', 'closed_repeat10', 'recall_repeat1', 'recall_repeat10',
          'judge', 'apply_repeat1', 'apply_repeat10', 'summarize']


def store_rows(path, records):
    if path.exists():
        assert rows(path) == records, str(path)
    else:
        write_rows(path, records)


def vision_core(source, text_training, dataset):
    text_lookup = {r['qid']: r for r in text_training}
    assert len(text_lookup) == len(text_training) == len(source)
    result = []
    for row in source:
        assert [m['role'] for m in row['messages']] == ['assistant']
        text = text_lookup[row['qid']]
        question = text['messages'][0]['content']
        assert question.startswith('Explain the rule or codebook named ')
        content = row['messages'][0]['content']
        assert content.count('<image>') == len(row['images']) == 1
        images = [str((dataset / p).resolve()) for p in row['images']]
        assert all(Path(p).is_file() and Path(p).name == 'learning_visual.png' for p in images)
        result.append(dict(id=row['id'], qid=row['qid'], variant='original_visual_definition',
            images=images, messages=[
                dict(role='user', content='<image>Using this learning reference image, ' + question[0].lower() + question[1:]),
                dict(role='assistant', content=content.replace('<image>', ''))]))
    assert len({r['qid'] for r in result}) == len(result)
    return result


def repeated_training(unique, repetitions):
    result = []
    for repetition in range(1, repetitions + 1):
        for original in unique:
            row = copy.deepcopy(original)
            row.update(id=f'{original["id"]}__repeat{repetition:02d}',
                source_sample_id=original['id'], repetition_index=repetition)
            result.append(row)
    return result


def training_cmd(root, work, plan, arm, smoke=False):
    cmd = original_training_command(work)
    cmd[0] = str(root / '.venv-glm/bin/python')
    cli_index = next(i for i, value in enumerate(cmd) if value.endswith('/swift/cli/sft.py'))
    cmd[cli_index] = str(root / '.venv/lib/python3.10/site-packages/swift/cli/sft.py')
    folder = work / ('smoke/VISION' if smoke else f'train/{arm}/VISION')
    for flag, value in {
        '--model': plan['model_path'], '--dataset': str(work / f'data/train_{arm}.jsonl'),
        '--output_dir': str(folder), '--master_port': '30391',
        '--per_device_train_batch_size': str(plan['training']['micro_batch']),
        '--gradient_accumulation_steps': str(plan['training']['gradient_accumulation']),
        '--max_length': str(plan['training']['max_length']),
    }.items():
        cmd[cmd.index(flag) + 1] = value
    if smoke:
        cmd += ['--max_steps', '2', '--save_strategy', 'steps', '--save_steps', '2', '--warmup_ratio', '0']
    return cmd


def verify(work, plan):
    for rel, expected in plan['immutable_sha256'].items():
        assert sha(work / rel) == expected, rel
    for name, expected in plan['source_sha256'].items():
        assert sha(work / 'source_snapshots' / name) == expected, name


def prepare(root, work):
    work.mkdir(parents=True, exist_ok=True)
    if (work / 'experiment.json').exists():
        plan = json.loads((work / 'experiment.json').read_text())
        assert plan['project_root'] == str(root) and plan['protocol'] == VERSION
        verify(work, plan)
        return plan
    os.environ.update(HF_HUB_OFFLINE='1', MODELSCOPE_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
        MAX_PIXELS='262144', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    text_run = root / 'runs/rules1000_glm46_lora_ep3_original_20261106'
    dataset = root / 'data/rules_repaired_single_image_v7'
    text_plan = json.loads((text_run / 'experiment.json').read_text())
    assert text_plan['status'] == 'complete'
    model = text_plan['model_path']
    assert (Path(model) / 'download_complete.json').is_file()
    source = rows(dataset / 'shared/train_multimodal.jsonl')
    text_training = rows(text_run / 'data/rule_unique_training.jsonl')
    core = vision_core(source, text_training, dataset)
    assert len(core) == 1000
    # Reuse the historical modality-specific conversion, not the textual rule targets.
    historical_path = root / 'runs/qwen3_lora_learning_ep5_20261024/data/rules_sft_vision.jsonl'
    historical = rows(historical_path)
    assert len(historical) == 1000
    for row, old in zip(core, historical):
        assert row['messages'] == old['messages'] and row['images'] == old['images']
    generic = rows(text_run / 'data/retention_training.jsonl')
    assert len(generic) == 252 and Counter(r['answer'] for r in generic) == dict.fromkeys('ABCD', 63)
    unique = core + [dict(copy.deepcopy(r), images=[]) for r in generic]
    assert len(unique) == len({r['id'] for r in unique}) == 1252
    store_rows(work / 'data/rule_unique_training.jsonl', core)
    store_rows(work / 'data/retention_training.jsonl', generic)
    store_rows(work / 'data/original_visual_learning.jsonl', source)
    for arm, repetitions in [('repeat1', 1), ('repeat10', 10)]:
        metadata = repeated_training(unique, repetitions)
        training = [{k: r[k] for k in ['id', 'messages', 'images']} for r in metadata]
        assert len(training) == len({r['id'] for r in training}) == 1252 * repetitions
        assert Counter(r['source_sample_id'] for r in metadata) == dict.fromkeys([r['id'] for r in unique], repetitions)
        for row in training:
            assert row['messages'][0]['content'].count('<image>') == len(row['images'])
            assert '<image>' not in row['messages'][-1]['content']
            assert all(Path(p).name == 'learning_visual.png' for p in row['images'])
        store_rows(work / f'data/train_{arm}.jsonl', training)
        store_rows(work / f'data/training_metadata_{arm}.jsonl', metadata)
    # Perform the actual Arrow load and multimodal template encode on CPU before queueing.
    from datasets import load_dataset
    from swift.model import get_model_processor
    from swift.template import get_template
    for arm in ARMS:
        loaded = load_dataset('json', data_files=str(work / f'data/train_{arm}.jsonl'),
            cache_dir=str(work / 'loader_validation_cache'), split='train')
        expected = rows(work / f'data/train_{arm}.jsonl')
        assert set(loaded.column_names) == {'id', 'messages', 'images'}
        assert len(loaded) == len(expected) and all(dict(a) == b for a, b in zip(loaded, expected))
    _, processor = get_model_processor(model, model_type='glm4v', load_model=False)
    template = get_template(processor, template_type='glm4_5v', max_length=8192, max_pixels=262144,
        enable_thinking=False, loss_scale='default+ignore_empty_think', truncation_strategy='raise')
    template.set_mode('train')
    lengths, supervision = [], []
    for i, row in enumerate(unique):
        encoded = template.encode({k: copy.deepcopy(row[k]) for k in ['messages', 'images']})
        ids, labels = encoded['input_ids'], encoded['labels']
        assert len(ids) == len(labels) <= 8192, row['id']
        # TemplateArguments automatically adds ignore_empty_think for this hybrid model.
        expected = row['messages'][-1]['content'] + template.template_meta.suffix[0]
        assert processor.tokenizer.decode([v for v in labels if v != -100]) == expected, row['id']
        assert any(v == -100 for v in labels) and any(v != -100 for v in labels)
        lengths.append(len(ids))
        supervision.append(sum(v != -100 for v in labels))
        if (i + 1) % 100 == 0:
            print('CPU template audit', i + 1, '/1252', flush=True)
    training_max_length = max(2048, math.ceil(max(lengths) / 256) * 256)
    save(work / 'data/template_audit.json', dict(status='pass', checked_at=now(),
        actual_arrow_loader_verified=True, rows_repeat1=1252, rows_repeat10=12520,
        unique_templates_checked=1252, visual_templates_checked=1000,
        max_formatted_tokens=max(lengths), training_max_length=training_max_length,
        training_image_max_pixels=262144, effective_loss_scale='default+ignore_empty_think',
        min_supervised_tokens=min(supervision),
        target_exactly_original_visual_explanation=True, user_and_image_tokens_masked=True,
        truncated_rows=0, formal_test_examples_used_for_training=False))
    # Training files are frozen before evaluation labels are read.
    for f in (text_run / 'evaluation_data').rglob('*'):
        if f.is_file():
            freeze(f, work / 'evaluation_data' / f.relative_to(text_run / 'evaluation_data'))
    image_inventory = rows(work / 'evaluation_data/image_hashes.jsonl')
    assert len(image_inventory) == 3000
    for r in image_inventory:
        assert sha(dataset / r['path']) == r['repaired_sha256'], r['path']
    save(work / 'data/image_audit.json', dict(status='pass', verified_images=3000, checked_at=now()))
    # Keep exactly the same shuffled question order used by the completed TEXT diagnostics.
    given_run = root / 'runs/rules1000_glm46_rule_given_20261108'
    manifest = rows(given_run / 'data/task_manifest.jsonl')
    question_lookup = {(r['id'], r['modality']): r for r in tasks_for(work)}
    questions = [question_lookup[m['id'], m['modality']] for m in manifest]
    assert len(questions) == 4000
    for q, m in zip(questions, manifest):
        assert q['gold'] == m['gold'] and q['qid'] == m['qid']
        assert digest(q['payload']) == m['closed_book_input_sha256']
    store_rows(work / 'data/question_tasks.jsonl', questions)
    freeze(text_run / 'data/rule_unique_training.jsonl', work / 'references/text_training.jsonl')
    text_refs = {r['qid']: r['messages'][-1]['content'] for r in text_training}
    visual_refs = {r['qid']: r['messages'][-1]['content'] for r in core}
    recall_tasks = []
    for row in text_training:
        # Deliberately no training image here: this probes stored visual knowledge.
        prompt = copy.deepcopy(row['messages'][0])
        assert '<image>' not in prompt['content']
        recall_tasks.append(dict(id=row['id'], qid=row['qid'], payload=dict(messages=[prompt]),
            reference=text_refs[row['qid']], visual_training_target=visual_refs[row['qid']],
            recording_required='Inspection recording convention:' in text_refs[row['qid']]))
    random.Random(42).shuffle(recall_tasks)
    assert len(recall_tasks) == 1000 and sum(t['recording_required'] for t in recall_tasks) == 104
    store_rows(work / 'data/tasks.jsonl', recall_tasks)
    store_rows(work / 'data/recall_inference_inputs.jsonl', [t['payload'] for t in recall_tasks])
    store_rows(work / 'data/judge_controls.jsonl', judge_controls())
    repeat_run = root / 'runs/rules1000_repeat10_glm46_lora_ep3_20261107'
    for arm in ['base'] + ARMS:
        freeze(repeat_run / f'rescored/{arm}.jsonl', work / f'references/text_closed_{arm}.jsonl')
    freeze(given_run / 'evaluation/base_rule_given/predictions.jsonl', work / 'references/base_original_rule.jsonl')
    dependency = root / 'runs/rules1000_glm46_recall_apply_20261108'
    for name in SOURCES:
        freeze(root / 'scripts' / name, work / 'source_snapshots' / name)
    immutable = {str(p.relative_to(work)): sha(p)
        for folder in ['data', 'references', 'evaluation_data']
        for p in (work / folder).rglob('*') if p.is_file()}
    plan = dict(status='prepared', created_at=now(), project_root=str(root),
        host='softengine-KVM', gpu='0', protocol=VERSION, queue_dependency=str(dependency),
        model='GLM-4.6V-Flash', model_path=model, environment=str(root / '.venv-glm'),
        framework='ms-swift ' + importlib.metadata.version('ms-swift'),
        learning_modality='VISION: original learning image + original modality-specific explanation SFT',
        knowledge_units=1000, arms={}, starts_from_clean_base=True,
        repetitions={'repeat1': 1, 'repeat10': 10}, stages=STAGES,
        training=dict(epochs=3, rank=64, alpha=128, dropout=0.05, learning_rate=1e-4,
            micro_batch=4, gradient_accumulation=4, effective_batch=16, max_length=training_max_length,
            max_length_reason='Smallest multiple of 256 covering every actual visual training template with max_pixels=262144, minimum 2048',
            image_max_pixels=262144, effective_loss_scale='default+ignore_empty_think',
            freeze_vit=True, freeze_aligner=True,
            rows_per_epoch={'repeat1': 1252, 'repeat10': 12520},
            rule_rows_per_epoch={'repeat1': 1000, 'repeat10': 10000},
            generic_rows_per_epoch={'repeat1': 252, 'repeat10': 2520},
            optimizer_steps={'repeat1': 237, 'repeat10': 2349},
            checkpoint_selection='Fixed final epoch 3, independent clean-base training for both arms'),
        input_sha256=digest([t['payload'] for t in recall_tasks]),
        task_sha256=sha(work / 'data/tasks.jsonl'), recall_max_tokens=1024,
        recall_prompt='Same rule-name-only TEXT recall question; NO learning image. Different from the VISION training prompt.',
        recall_reference='Full explicit text rule, including bindings originally learned from the image',
        judge_protocol=JUDGE_VERSION, judge_max_tokens=256, judge_calibration_cases=10,
        max_tokens=1024, native_thinking=False, brief_prompted_reasoning=True,
        temperature=0, seed=42, batch_size=4, parser=PARSER_VERSION,
        inference_image_max_pixels=None,
        inference_image_resolution='Model defaults, matching existing GLM evaluations; MAX_PIXELS env does not cap GLM images',
        evaluation=dict(directions=['V2T', 'V2V'], paired_questions_per_modality=2000,
            closed_book=True, recall_assisted_answering=True, same_adapter_for_recall_and_application=True,
            no_correction_or_filtering_of_recalled_rules=True),
        source_sha256={name: sha(work / 'source_snapshots' / name) for name in SOURCES},
        immutable_sha256=immutable, original_visual_sft_sha256=sha(historical_path),
        limitations=[
            'Original visual target descriptions may refer to image regions instead of naming objects.',
            'Name-only recall requires expressing the learned visual bindings without seeing the learning image.',
            'A repetition of a deictic description without those bindings is not full rule recall.',
            'Visual encoder and aligner remain frozen; language-layer LoRA learns from multimodal inputs.',
            'Original learning image files are unchanged; the training template scales images to max_pixels=262144 without cropping.',
            'Evaluation uses model-default image preprocessing, as in the previous GLM evaluations.',
            'Image hashes do not independently certify visual semantic correctness.',
            'Clean GLM base judges recall; same model family, not certified human ground truth.',
        ])
    save(work / 'experiment.json', plan)
    (work / 'README_ZH.md').write_text(
        '# GLM规则类视觉学习：1次/10次，3轮LoRA\n\n'
        '当前TEXT回忆辅助答题实验全部完成后才占用GPU。自有softengine-KVM GPU0、隔离.venv-glm。'
        '两组均从干净GLM-4.6V-Flash开始；原始参考图片放在user，原视觉学习说明原样作为SFT目标。'
        '每轮各1000/10000条规则，加252/2520条同源通用指令；10次为原样重复。'
        '语言层LoRA，冻结视觉编码器和对齐层。两组各3轮，有效batch16，lr1e-4。'
        '原学习图文件不变，训练时按既有max_pixels=262144缩放、不裁剪；评测保留此前GLM的默认图像预处理。'
        f'视觉图文训练长度审计后设为{training_max_length}，所有样本完整保留、不截断。'
        '先以micro-batch4×累积4做2步试训；若实测显存不足，则归档失败试训并将两组统一为1×16。\n\n'
        '先做独立2步试训，随后分别正式训练1次、10次两组。每组评测：\n\n'
        '1. 直接闭卷答题：V2T/V2V各2000题，qid与选项配对。\n'
        '2. 仅给规则名开放回忆1000条，不给学习图片。干净基座匿名语义评分，'
        '标准为完整显式规则；仅说“参照图片中的对象”而没回忆具体绑定不算完整。'
        '与原VISION训练提问不同，另报与视觉训练说明的一致性以区分背说明和记绑定。\n'
        '3. 回忆原文＋相应题目：仍由同一LoRA回答V2T/V2V各2000题；不筛选或修正错误回忆。\n\n'
        '选择题用既有简短解释boxed提示；回忆/答题均1024 token，原生thinking关闭。'
        '保留原答、gold独立存放、统一保守解析、无效计错，分规则族与题型汇总。'
        'AI裁判256 token、10条已知控制；判定失败单列，不算记忆错误。'
        '若裁判校准失败，保留失败记录并继续答题阶段，不发布未经校准的记忆准确率。\n')
    return plan


def validate_training(work, plan, arm, smoke=False):
    folder = work / ('smoke/VISION' if smoke else f'train/{arm}/VISION')
    ck = check_adapter(folder, smoke=smoke)
    expected_steps = 2 if smoke else plan['training']['optimizer_steps'][arm]
    state = json.loads((ck / 'trainer_state.json').read_text())
    assert state['global_step'] == expected_steps
    args = json.loads((ck / 'args.json').read_text())
    assert args['dataset'] == [str(work / f'data/train_{arm}.jsonl')]
    assert Path(args['model']).resolve() == Path(plan['model_path']).resolve()
    assert not args.get('adapters') and not args.get('resume_from_checkpoint')
    assert args['freeze_vit'] and args['freeze_aligner']
    assert args['num_train_epochs'] == 3
    if not smoke:
        plan['arms'][arm] = dict(path=str(ck), sha256={name: sha(ck / name) for name in
            ['adapter_model.safetensors', 'adapter_config.json', 'args.json', 'trainer_state.json']})
        save(work / 'experiment.json', plan)
    return ck


def application_tasks(work, plan, arm, phase):
    questions = rows(work / 'data/question_tasks.jsonl')
    assert len(questions) == 4000
    if phase == 'apply':
        memories = rows(work / 'recall' / arm / 'responses.jsonl')
        assert len(memories) == 1000
        tasks = recalled_tasks(questions, memories, arm)
    else:
        tasks = []
        for q in questions:
            tasks.append(dict(q, options=options_from_question(q['payload']['messages'][0]['content']),
                rule_sha256='not_supplied', closed_book_input_sha256=digest(q['payload'])))
    store_rows(work / f'data/generated_{phase}_tasks_{arm}.jsonl', tasks)
    return tasks


def answer_questions(work, arm, phase):
    plan = json.loads((work / 'experiment.json').read_text())
    verify(work, plan)
    tasks = application_tasks(work, plan, arm, phase)
    ck = plan['arms'][arm]
    for name, expected in ck['sha256'].items():
        assert sha(Path(ck['path']) / name) == expected
    output = work / ('evaluation_closed' if phase == 'closed' else 'evaluation_apply') / arm
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'predictions.jsonl'
    records = rows(path) if path.exists() else []
    config = dict(model=plan['model_path'], adapter=ck['path'],
        adapter_sha256=ck['sha256']['adapter_model.safetensors'],
        input_sha256=digest([t['payload'] for t in tasks]), input_count=4000,
        phase=phase, learning_modality='VISION', max_tokens=1024, native_thinking=False,
        temperature=0, seed=42, batch_size=4, inference_image_max_pixels=None,
        parser=PARSER_VERSION, stop_sequences=ANSWER_STOP_SEQUENCES,
        recalled_rule=phase == 'apply', oracle_material=False, finalizer=False)
    configure_generation(output, config, records)
    lookup, seen = {(r['id'], r['modality']): r for r in tasks}, set()
    for r in records:
        key = r['id'], r['modality']
        assert key not in seen and key in lookup and r['arm'] == arm
        assert r['gold'] == lookup[key]['gold'] and r['input_sha256'] == digest(lookup[key]['payload'])
        seen.add(key)
    pending = [t for t in tasks if (t['id'], t['modality']) not in seen]
    if pending:
        engine = inference_engine(plan, arm)
        from swift import InferRequest, RequestConfig
        request = RequestConfig(max_tokens=1024, temperature=0, seed=42,
            return_details=True, stop=ANSWER_STOP_SEQUENCES)
        start_time = time.monotonic()
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
                prediction, reason = reparse(raw, task['options'])
                record = {k: v for k, v in task.items() if k not in {'payload', 'options'}}
                record.update(arm=arm, phase=phase, direction='V2T' if task['modality'] == 'T' else 'V2V',
                    raw_response=raw, raw_predicted=legacy, raw_parse_reason=legacy_reason,
                    predicted=prediction, parse_reason=reason, correct=prediction == task['gold'],
                    stop_reason=str(choice.finish_reason), generated_tokens=reply.usage.completion_tokens,
                    generation_max_tokens=1024, input_sha256=digest(task['payload']),
                    finalizer=None, saved_at=now())
                new.append(record)
            append(path, new)
            records.extend(new)
            groups = summaries(records)
            progress = dict(stage='generating', arm=arm, phase=phase, saved=len(records), total=4000,
                groups=groups, updated_at=now(), elapsed_seconds=round(time.monotonic() - start_time, 1))
            save(output / 'status.json', progress)
            save(output / 'summary.json', progress)
            print(json.dumps(dict(arm=arm, phase=phase, saved=len(records), total=4000,
                groups={mod: groups.get(mod + '/all') for mod in 'TV'})), flush=True)
        del engine
    assert len(records) == 4000
    paired, details = pair_counts(records)
    write_rows(output / 'paired_results.jsonl', details)
    save(output / 'summary.json', dict(status='complete', arm=arm, phase=phase, saved=4000,
        total=4000, groups=summaries(records), paired_outcomes=paired, completed_at=now()))
    save(output / 'status.json', dict(status='complete', saved=4000, total=4000, updated_at=now()))


def inference_engine(plan, arm=None):
    os.environ.update(CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1', MODELSCOPE_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', MAX_PIXELS='262144')
    import torch
    from swift import TransformersEngine
    torch.set_num_threads(4)
    engine = TransformersEngine(plan['model_path'], model_type='glm4v', template_type='glm4_5v',
        adapters=[plan['arms'][arm]['path']] if arm else None,
        torch_dtype=torch.bfloat16, attn_impl='sdpa', device_map='cuda:0', max_batch_size=4)
    engine.template.enable_thinking = False
    return engine


def judge_controls():
    cases = controls()
    cases.extend([
        dict(id='unresolved_visual_binding',
             reference='Zenta selects only protective covers and requires at least 4 objects.',
             candidate='Zenta selects only objects of the type shown in the reference image and requires at least 4 objects.',
             recording_required=False, expected_verdict='partial'),
        dict(id='explicit_visual_binding',
             reference='Zenta selects only protective covers and requires at least 4 objects.',
             candidate='Choose a batch of four or more protective covers; no other object types qualify.',
             recording_required=False, expected_verdict='correct'),
    ])
    expected_core = dict(equivalent_paraphrase=True, wrong_threshold=False,
        missing_condition=False, explicit_no_recall=False, contradiction=False,
        correct_recording=True, wrong_recording=True, missing_recording=True,
        unresolved_visual_binding=False, explicit_visual_binding=True)
    for case in cases:
        case['expected_core_correct'] = expected_core[case['id']]
    return cases


def control_passed(case, judgment):
    if judgment is None or judgment['verdict'] != case['expected_verdict']:
        return False
    if judgment['core_correct'] is not case['expected_core_correct']:
        return False
    if judgment['complete'] is not (case['expected_verdict'] == 'correct'):
        return False
    if case['recording_required']:
        return judgment['recording_correct'] is (case['id'] == 'correct_recording')
    return judgment['recording_correct'] is None


def judge_payload(case, retry=False):
    system = JUDGE_SYSTEM + (
        '\nNo reference image is available to the recalling model. A phrase such as "the pictured type", '
        '"as shown in the image", or "the image mapping" does not specify the actual object/category/mapping. '
        'Treat missing concrete visual bindings as missing core facts, even if the rest of the description matches. '
        'For partial recall core_correct=false whenever any core fact is omitted. '
        'For recording_required=true, recording_correct MUST be a JSON boolean, never null. '
        'Keep total output short: at most 2 brief points per list, reason under 15 words.')
    if retry:
        system += '\nReturn one short, complete JSON object with all required keys and consistent verdict and details.'
    return dict(messages=[dict(role='system', content=system), dict(role='user', content=json.dumps(dict(
        reference_rule=case['reference'], candidate_answer=case['candidate'],
        recording_required=case['recording_required']), ensure_ascii=False))])


def grade_batch(engine, cases, attempt_file):
    from swift import InferRequest, RequestConfig
    attempts, resolved, pending = Counter(), {}, list(cases)
    while pending:
        payloads = [judge_payload(c, attempts[c['id']] > 0) for c in pending]
        replies = engine.infer([InferRequest(**p) for p in payloads],
            RequestConfig(max_tokens=256, temperature=0, seed=42, return_details=True), use_tqdm=False)
        assert len(replies) == len(pending)
        again = []
        for case, payload, reply in zip(pending, payloads, replies):
            if isinstance(reply, Exception):
                raise reply
            attempts[case['id']] += 1
            choice = reply.choices[0]
            raw = choice.message.content or ''
            judgment, reason = parse_case_judgment(raw, case)
            original = judgment
            check = dict(applied=False)
            if judgment is not None:
                judgment, check = enforce_recording_evidence(judgment, case['reference'],
                    case['candidate'], case['recording_required'])
            result = dict(id=case['id'], attempt=attempts[case['id']], raw_response=raw,
                model_judgment=original, judgment=judgment, deterministic_check=check,
                parse_reason=reason, input_sha256=digest(payload), stop_reason=str(choice.finish_reason),
                generated_tokens=reply.usage.completion_tokens, saved_at=now())
            append(attempt_file, [result])
            if judgment is None and attempts[case['id']] < 2:
                again.append(case)
            else:
                resolved[case['id']] = result
        pending = again
    return resolved


def judge(work):
    plan = json.loads((work / 'experiment.json').read_text())
    verify(work, plan)
    output = work / 'judging'
    output.mkdir(exist_ok=True)
    path = output / 'judgments.jsonl'
    records = rows(path) if path.exists() else []
    engine = inference_engine(plan)
    calibration = rows(work / 'data/judge_controls.jsonl')
    results = {}
    for start in range(0, len(calibration), 4):
        results.update(grade_batch(engine, calibration[start:start + 4], output / 'calibration_attempts.jsonl'))
    failures = []
    for case in calibration:
        result = results[case['id']]['judgment']
        if not control_passed(case, result):
            failures.append(case['id'])
    calibration_path = output / 'calibration.json'
    if calibration_path.exists():
        archive = output / f'calibration_previous_{time.time_ns()}.json'
        freeze(calibration_path, archive)
    save(calibration_path, dict(status='pass' if not failures else 'failed', total=len(calibration),
        failures=failures, results=results, checked_at=now(), protocol=JUDGE_VERSION,
        scope='Basic controls including missing visual bindings; not independent human validation.'))
    if failures:
        save(output / 'status.json', dict(status='calibration_failed', failures=failures,
            updated_at=now(), recall_accuracy_published=False))
        print('Judge calibration failed; raw responses retained; proceeding to application stages.', flush=True)
        del engine
        return
    tasks = {r['qid']: r for r in rows(work / 'data/tasks.jsonl')}
    cases, metadata = [], {}
    for arm in ARMS:
        memories = rows(work / 'recall' / arm / 'responses.jsonl')
        assert len(memories) == 1000
        for r in memories:
            t = tasks[r['qid']]
            blind_id = digest(dict(arm=arm, qid=r['qid']))[:24]
            cases.append(dict(id=blind_id, reference=t['reference'], candidate=r['answer_for_judge'],
                recording_required=t['recording_required']))
            metadata[blind_id] = dict(arm=arm, id=r['id'], qid=r['qid'],
                response_sha256=digest(r['raw_response']), recording_required=t['recording_required'])
    assert len(cases) == len(metadata) == 2000
    store_rows(output / 'blind_case_manifest.jsonl', cases)
    seen = set()
    for r in records:
        assert r['blind_id'] not in seen and all(r[k] == v for k, v in metadata[r['blind_id']].items())
        seen.add(r['blind_id'])
    random.Random(42).shuffle(cases)
    pending = [c for c in cases if c['id'] not in seen]
    for start in range(0, len(pending), 4):
        batch = pending[start:start + 4]
        results = grade_batch(engine, batch, output / 'attempts.jsonl')
        new = []
        for case in batch:
            r = results[case['id']]
            new.append(dict(metadata[case['id']], blind_id=case['id'], judgment=r['judgment'],
                judge_parse_reason=r['parse_reason'], judge_attempts=r['attempt'],
                judge_input_sha256=r['input_sha256'], saved_at=now()))
        append(path, new)
        records.extend(new)
        save(output / 'status.json', dict(stage='judging', saved=len(records), total=2000,
            unresolved=sum(r['judgment'] is None or r['judgment']['verdict'] == 'uncertain' for r in records), updated_at=now()))
        print('Judged', len(records), '/2000', flush=True)
    assert len(records) == 2000
    save(output / 'status.json', dict(status='complete', saved=2000, total=2000, updated_at=now()))
    del engine


def recall_summary(work):
    output = work / 'judging'
    calibration = json.loads((output / 'calibration.json').read_text())
    if calibration['status'] != 'pass':
        return dict(status='calibration_failed', failures=calibration['failures'], groups=None)
    judged = rows(output / 'judgments.jsonl')
    assert len(judged) == len({r['blind_id'] for r in judged}) == 2000
    tasks = {t['qid']: t for t in rows(work / 'data/tasks.jsonl')}
    groups, review = {}, []
    for arm in ARMS:
        selected = [r for r in judged if r['arm'] == arm]
        assert len(selected) == 1000
        resolved = [r for r in selected if r['judgment'] is not None and r['judgment']['verdict'] != 'uncertain']
        correct = sum(r['judgment']['verdict'] == 'correct' for r in resolved)
        memories = rows(work / 'recall' / arm / 'responses.jsonl')
        convention = [r for r in resolved if r['recording_required']]
        groups[arm] = dict(n=1000, resolved=len(resolved), unresolved=1000 - len(resolved),
            full_correct=correct, full_recall_rate_all=correct / 1000 if len(resolved) == 1000 else None,
            full_recall_lower_bound=correct / 1000, full_recall_upper_bound=(correct + 1000 - len(resolved)) / 1000,
            core_correct=sum(r['judgment']['core_correct'] for r in resolved),
            verdict_counts=dict(Counter(r['judgment']['verdict'] if r['judgment'] else 'judge_format_failure' for r in selected)),
            recording=dict(n=104, resolved=len(convention), correct=sum(r['judgment']['recording_correct'] for r in convention)),
            exact_explicit_reference=sum(normalized(r['answer_for_judge']) == normalized(tasks[r['qid']]['reference']) for r in memories),
            exact_visual_training_target=sum(normalized(r['answer_for_judge']) == normalized(tasks[r['qid']]['visual_training_target']) for r in memories),
            length_limited=sum(r['stop_reason'] == 'length' for r in memories))
        lookup = {r['qid']: r for r in memories}
        for r in selected:
            if r['judgment'] is None or r['judgment']['verdict'] != 'correct':
                review.append(dict(r, raw_recall=lookup[r['qid']], reference=tasks[r['qid']]['reference'],
                    visual_training_target=tasks[r['qid']]['visual_training_target']))
    write_rows(work / 'recall_needs_review.jsonl', review)
    return dict(status='complete', calibration_passed=True, groups=groups, protocol=JUDGE_VERSION,
        judge='clean GLM-4.6V-Flash base; same family; no LoRA', no_learning_image_supplied=True)


def summarize(work):
    plan = json.loads((work / 'experiment.json').read_text())
    verify(work, plan)
    conditions, changes = {}, {}
    for arm in ARMS:
        records = {}
        for phase, folder in [('closed', 'evaluation_closed'), ('apply', 'evaluation_apply')]:
            summary = json.loads((work / folder / arm / 'summary.json').read_text())
            assert summary['status'] == 'complete' and summary['saved'] == 4000
            conditions[arm + '/' + phase] = summary
            records[phase] = rows(work / folder / arm / 'predictions.jsonl')
            assert len(records[phase]) == len({(r['id'], r['modality']) for r in records[phase]}) == 4000
        changes[arm] = {mod: matched_change([r for r in records['closed'] if r['modality'] == mod],
            [r for r in records['apply'] if r['modality'] == mod]) for mod in 'TV'}
    memory = recall_summary(work)
    save(work / 'recall_summary.json', memory)
    save(work / 'comparison.json', dict(status='complete', completed_at=now(),
        learning_modality='VISION', conditions=conditions, recall=memory,
        paired_change_from_same_adapter_closed_book=changes))
    lines = ['# GLM规则视觉学习：重复1次与10次', '',
        '| 每轮重复 | 测试方式 | V2T准确率 | V2T无效率 | V2V准确率 | V2V无效率 |',
        '|---|---|---:|---:|---:|---:|']
    for arm in ARMS:
        for phase in ['closed', 'apply']:
            g = conditions[arm + '/' + phase]['groups']
            t, v = g['T/all'], g['V/all']
            label = '直接闭卷' if phase == 'closed' else '回忆规则＋题目'
            lines.append(f'| {plan["repetitions"][arm]}次 | {label} | {t["accuracy"]:.2%} | {t["invalid_rate"]:.2%} | {v["accuracy"]:.2%} | {v["invalid_rate"]:.2%} |')
    lines += ['', '两组从干净基座各训练3轮；每种答题方式文字、图片各2000题。',
        '回忆只给规则名，不给学习图，评分参考为完整显式规则；只背出依赖图片的说明不等于记住图中绑定。',
        '训练与推理使用ms-swift，视觉编码器和对齐层冻结；原生thinking关闭，简短解释提示，输出1024 token。', '']
    if memory['status'] == 'complete':
        lines += ['| 每轮重复 | 规则回忆完整正确 | 完整正确率范围 | 待判定 |', '|---|---:|---:|---:|']
        for arm in ARMS:
            g = memory['groups'][arm]
            lines.append(f'| {plan["repetitions"][arm]}次 | {g["full_correct"]}/1000 | {g["full_recall_lower_bound"]:.2%}–{g["full_recall_upper_bound"]:.2%} | {g["unresolved"]} |')
    else:
        lines += ['AI裁判校准未通过，回忆准确率未发布；原始回忆保留，答题成绩独立报告。',
                  '失败控制：' + ', '.join(memory['failures'])]
    lines += ['', '回忆由同一LoRA生成并原样用于答题，不按正确性筛选或替换。'
        '回忆裁判为同家族干净基座，评分不等同于人工真值。',
        '详细分组和配对变化见comparison.json；回忆缺漏、错记及原答见recall_needs_review.jsonl。', '']
    (work / 'RESULTS_ZH.md').write_text('\n'.join(lines))
    plan.update(status='complete', completed_at=now(), recall_scoring_status=memory['status'])
    save(work / 'experiment.json', plan)
    save(work / 'status.json', dict(stage='complete', updated_at=now(),
        recall_scoring_status=memory['status'], results=str(work / 'RESULTS_ZH.md')))


def controller(root, work):
    assert socket.gethostname() == 'softengine-KVM'
    work.mkdir(parents=True, exist_ok=True)
    with (work / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = prepare(root, work)
        def status(stage, **extra):
            save(work / 'status.json', dict(stage=stage, controller_pid=os.getpid(),
                host='softengine-KVM', gpu='0', updated_at=now(), **extra))
        dependency = Path(plan['queue_dependency'])
        while True:
            previous = json.loads((dependency / 'status.json').read_text())
            if previous.get('stage') == 'complete':
                assert json.loads((dependency / 'comparison.json').read_text())['status'] == 'complete'
                break
            if previous.get('stage') == 'failed':
                status('dependency_failed', dependency=str(dependency), previous=previous,
                    training_started=False)
            else:
                progress_path = dependency / 'evaluation/repeat1/status.json'
                progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
                status('queued_after_text_recall_apply', dependency=str(dependency),
                    dependency_stage=previous.get('stage'), dependency_completed=progress.get('saved'),
                    dependency_total=progress.get('total'), training_started=False)
            time.sleep(30)
        tmp = work / 'tmp'
        tmp.mkdir(exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1', MODELSCOPE_OFFLINE='1',
            TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', MAX_PIXELS='262144',
            TMPDIR=str(tmp), PYTHONUNBUFFERED='1', HF_DATASETS_CACHE=str(work / 'dataset_cache'),
            NPROC_PER_NODE='1', MASTER_ADDR='127.0.0.1', MASTER_PORT='30391',
            PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        def execute(command, stage):
            verify(work, plan)
            while True:
                try:
                    idle('0')
                    break
                except RuntimeError as error:
                    status('waiting_for_free_gpu', next_stage=stage, reason=str(error))
                    time.sleep(30)
            save(work / (stage + '_command.json'), command)
            log = work / (stage + '.log')
            with log.open('a') as stream:
                child = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            status(stage, child_pid=child.pid, log=str(log))
            return child.wait()
        # A separate two-step run validates actual multimodal forward/backward on the GPU.
        smoke_ck = work / 'smoke/VISION/checkpoint-2'
        if not smoke_ck.exists():
            folder = work / 'smoke/VISION'
            if folder.exists() and any(folder.iterdir()):
                raise RuntimeError('Preserving interrupted smoke training; inspect smoke_training.log')
            code = execute(training_cmd(root, work, plan, 'repeat1', smoke=True), 'smoke_training')
            if code:
                log = work / 'smoke_training.log'
                failed_text = log.read_text(errors='replace')
                if not re.search(r'CUDA out of memory|CUDAOutOfMemoryError|torch\.OutOfMemoryError', failed_text):
                    raise RuntimeError('Visual smoke training failed; inspect smoke_training.log; no formal run started')
                revision = work / 'preflight_failures' / ('smoke_oom_' + str(time.time_ns()))
                revision.mkdir(parents=True)
                freeze(work / 'experiment.json', revision / 'experiment_before.json')
                for filename in ['smoke_training.log', 'smoke_training_command.json']:
                    freeze(work / filename, revision / filename)
                if (work / 'smoke').exists():
                    shutil.move(str(work / 'smoke'), str(revision / 'smoke'))
                plan['training'].update(micro_batch=1, gradient_accumulation=16)
                plan['training']['memory_revision'] = dict(reason='Actual CUDA OOM in separate two-step smoke run',
                    effective_batch_unchanged=16, applied_to_both_arms=True,
                    evidence=str(revision), updated_at=now())
                save(work / 'experiment.json', plan)
                if execute(training_cmd(root, work, plan, 'repeat1', smoke=True), 'smoke_training_retry'):
                    raise RuntimeError('Visual smoke retry failed; outputs retained; no formal run started')
        validate_training(work, plan, 'repeat1', smoke=True)
        for arm in ARMS:
            expected = work / f'train/{arm}/VISION' / ('checkpoint-' + str(plan['training']['optimizer_steps'][arm]))
            if not expected.exists():
                folder = expected.parent
                if folder.exists() and any(folder.iterdir()):
                    raise RuntimeError(f'Preserving interrupted training at {folder}')
                if execute(training_cmd(root, work, plan, arm), 'training_' + arm):
                    raise RuntimeError(f'{arm} training failed; outputs retained')
            validate_training(work, plan, arm)
        for stage in STAGES[3:-1]:
            if stage.startswith('closed_') or stage.startswith('apply_'):
                phase, arm = stage.split('_', 1)
                folder = 'evaluation_closed' if phase == 'closed' else 'evaluation_apply'
                completed = work / folder / arm / 'status.json'
            elif stage.startswith('recall_'):
                arm = stage.split('_', 1)[1]
                completed = work / 'recall' / arm / 'status.json'
            else:
                completed = work / 'judging/status.json'
            if completed.exists() and json.loads(completed.read_text()).get('status') == 'complete':
                continue
            cmd = [str(root / '.venv-glm/bin/python'), '-u',
                str(work / 'source_snapshots/run_rules_glm46_vision_suite.py'),
                '--root', str(root), '--work', str(work), '--stage', stage]
            if execute(cmd, stage):
                raise RuntimeError(f'{stage} failed; outputs retained; inspect {stage}.log')
        summarize(work)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--stage', choices=STAGES[3:])
    args = parser.parse_args()
    root, work = args.root.resolve(), args.work.resolve()
    try:
        if args.prepare_only:
            prepare(root, work)
        elif args.stage == 'summarize':
            summarize(work)
        elif args.stage == 'judge':
            judge(work)
        elif args.stage and args.stage.startswith('recall_'):
            plan = json.loads((work / 'experiment.json').read_text())
            verify(work, plan)
            original_recall(work, args.stage.split('_', 1)[1])
        elif args.stage:
            phase, arm = args.stage.split('_', 1)
            answer_questions(work, arm, phase)
        else:
            controller(root, work)
    except Exception as error:
        save(work / (f'failure_{args.stage}.json' if args.stage else 'status.json'),
            dict(stage='failed', error=repr(error), updated_at=now()))
        raise
