#!/usr/bin/env python3
"""Freeze and queue the 32B open-book transformation diagnostic after the VISION suite."""
import argparse
from collections import Counter
import copy
import fcntl
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

from answer_readout import options_from_question
from audit_rules_glm46_readout import VERSION as PARSER_VERSION, reparse, scores
from entityqa_common import digest, now, rows, save, sha, write_rows
from evaluate_entity_glm46_full import ANSWER_STOP_SEQUENCES, glm_readout
from run_entityqa_lora import idle

VERSION = 'rules120-qwen3vl32b-nf4-rule-given-v1'
SOURCES = ['run_rules_qwen32b_rule_given.py', 'entityqa_common.py', 'run_entityqa_lora.py',
           'answer_readout.py', 'audit_rules_glm46_readout.py', 'evaluate_entity_glm46_full.py',
           'entityqa_prompt_neutral_v5.py', 'answer_readout_entity_v2.py',
           'answer_readout_entity_v3.py', 'answer_readout_entity_v4.py']
PIXEL_ENV = ['MAX_PIXELS', 'MIN_PIXELS', 'IMAGE_MAX_TOKEN_NUM', 'IMAGE_MIN_TOKEN_NUM',
             'SPATIAL_MERGE_SIZE', 'MAX_RATIO']


def environment(cpu=False):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if cpu else '0',
               HF_HUB_OFFLINE='1', MODELSCOPE_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    for name in PIXEL_ENV:
        env.pop(name, None)
    return env


def freeze(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert sha(source) == sha(target), str(target)
    else:
        shutil.copy2(source, target)


def select_tasks(manifest, payloads):
    assert len(manifest) == len(payloads)
    selected, inputs = [], []
    seen = set()
    for m, p in zip(manifest, payloads):
        if m['family'] != 'transformation_procedure':
            continue
        key = m['id'], m['modality']
        assert key not in seen
        seen.add(key)
        assert set(p) <= {'messages', 'images'} and [x['role'] for x in p['messages']] == ['user']
        question = p['messages'][0]['content']
        assert question.startswith('Reference rule for this question (use the supplied rule):\n')
        assert options_from_question(question) == m['options']
        assert question.count('<image>') == len(p.get('images', [])) == (m['modality'] == 'V')
        assert all(Path(path).name.startswith('test_visual_') for path in p.get('images', []))
        selected.append(copy.deepcopy(m))
        inputs.append(copy.deepcopy(p))
    assert len(selected) == 480 and len({m['qid'] for m in selected}) == 120
    for mod in 'TV':
        subset = [r for r in selected if r['modality'] == mod]
        assert Counter(r['gold'] for r in subset) == dict.fromkeys('ABCD', 60)
        assert Counter(r['question_type'] for r in subset) == dict(direct=120, diagnosis=120)
    for i in range(0, len(selected), 4):
        group = selected[i:i + 4]
        assert len({m['qid'] for m in group}) == 1
        for j, kind in [(0, 'direct'), (2, 'diagnosis')]:
            t, v = group[j:j + 2]
            assert (t['modality'], v['modality']) == ('T', 'V')
            assert t['question_type'] == v['question_type'] == kind
            for field in ['id', 'qid', 'gold', 'options', 'rule_sha256']:
                assert t[field] == v[field], field
            rule = inputs[i + j]['messages'][0]['content'].split('\n\nQuestion:\n', 1)[0]
            visual_rule = inputs[i + j + 1]['messages'][0]['content'].split('\n\nQuestion:\n', 1)[0]
            assert rule == visual_rule
    return selected, inputs


def verify(work, plan):
    for rel, expected in plan['immutable_sha256'].items():
        assert sha(work / rel) == expected, rel
    for name, expected in plan['source_sha256'].items():
        assert sha(work / 'source_snapshots' / name) == expected, name
    for image in rows(work / 'data/image_hashes.jsonl'):
        assert sha(image['path']) == image['sha256'], image['path']


def prepare(root, work):
    work.mkdir(parents=True, exist_ok=True)
    if (work / 'experiment.json').exists():
        plan = json.loads((work / 'experiment.json').read_text())
        assert plan['protocol'] == VERSION and plan['project_root'] == str(root)
        verify(work, plan)
        return plan
    source = root / 'runs/rules1000_glm46_rule_given_20261108'
    assert json.loads((source / 'status.json').read_text())['stage'] == 'complete'
    manifest, inputs = select_tasks(rows(source / 'data/task_manifest.jsonl'),
                                    rows(source / 'data/inference_inputs.jsonl'))
    write_rows(work / 'data/task_manifest.jsonl', manifest)
    write_rows(work / 'data/inference_inputs.jsonl', inputs)
    image_paths = sorted({p for payload in inputs for p in payload.get('images', [])})
    assert len(image_paths) == 240
    write_rows(work / 'data/image_hashes.jsonl', [dict(path=p, sha256=sha(p)) for p in image_paths])
    original = rows(source / 'evaluation/base_rule_given/predictions.jsonl')
    indexed = {(r['id'], r['modality']): r for r in original}
    baseline = []
    for m, p in zip(manifest, inputs):
        record = copy.deepcopy(indexed[m['id'], m['modality']])
        assert record['gold'] == m['gold'] and record['input_sha256'] == digest(p)
        pred, reason = reparse(record['raw_response'], m['options'])
        record.update(predicted=pred, parse_reason=reason, correct=pred == m['gold'])
        baseline.append(record)
    write_rows(work / 'reference/glm_same_questions.jsonl', baseline)
    save(work / 'reference/glm_summary.json', summaries(baseline))
    metadata = work / 'model_metadata'
    listed = json.loads((metadata / 'source_manifest.json').read_text())
    # Preserve the exact upstream model revisions, even before the weights finish downloading.
    model = root / 'model_cache/Qwen3-VL-32B-Instruct'
    freeze(model / 'download_manifest.json', work / 'reference/model_download_manifest.json')
    for name in SOURCES:
        freeze(root / 'scripts' / name, work / 'source_snapshots' / name)
    immutable = [p for folder in ['data', 'reference', 'model_metadata']
                 for p in (work / folder).iterdir() if p.is_file()]
    plan = dict(protocol=VERSION, status='prepared', created_at=now(), project_root=str(root),
        host=socket.gethostname(), gpu='0', environment=str(root / '.venv'),
        model='Qwen/Qwen3-VL-32B-Instruct', model_path=str(model), training=False, adapters=[],
        framework='ms-swift TransformersEngine', batch_size=1,
        queue_dependency=str(root / 'runs/rules1000_glm46_vision_repeat1_repeat10_ep3_20261109'),
        source_rule_given_run=str(source), total=480, rules=120, questions_per_modality=240,
        family='transformation_procedure', prompt='verbatim original GLM rule-given inputs',
        max_tokens=1024, temperature=0, seed=42, native_thinking=False,
        brief_prompted_reasoning=True, parser=PARSER_VERSION, finalizer=False,
        stop_sequences=ANSWER_STOP_SEQUENCES,
        image_policy='Original test images; model-default Qwen3-VL/ms-swift preprocessing; no pixel overrides',
        quantization=dict(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype='bfloat16',
            llm_int8_skip_modules=['visual', 'lm_head']),
        metadata_manifest_sha256=digest(listed),
        packages={name: importlib.metadata.version(name) for name in
                  ['torch', 'transformers', 'ms-swift', 'bitsandbytes', 'accelerate', 'qwen-vl-utils']},
        immutable_sha256={str(p.relative_to(work)): sha(p) for p in immutable},
        source_sha256={name: sha(work / 'source_snapshots' / name) for name in SOURCES},
        limitations=[
            'Open-book rule application only; not a learned cross-modal transfer evaluation.',
            'Model family and quantization differ from GLM; differences cannot be attributed to size alone.',
            'Same source images, but model-specific vision preprocessing is not an equal pixel budget.',
            'A 1024-token response budget can produce length truncation; report invalid and length rates.',
        ])
    save(work / 'experiment.json', plan)
    (work / 'README_ZH.md').write_text(
        '# 更大模型规则开卷对照\n\n'
        'Qwen3-VL-32B-Instruct，4bit NF4、双重量化、BF16计算；视觉层与lm_head保持BF16。'
        '使用项目.venv和ms-swift，不训练、不加载LoRA。\n\n'
        '复用GLM开卷实验全部120条多步变换规则，文字/图片各240题，逐题配对；'
        '每种模态direct/diagnosis各120题，ABCD各60题。规则、问题、选项、原图均原样保留。'
        '模型输入与含答案的审计清单分开保存。\n\n'
        '沿用简短解释及boxed提示，原生thinking关闭，输出1024 token，temperature=0，seed=42，batch=1。'
        '保留原始输出，用同一个保守解析器；无效计错，不补选。图片使用Qwen默认预处理，记录实际图像grid和输入长度。\n\n'
        '等当前视觉repeat1/repeat10实验全部完成、模型下载校验完成、GPU真正空闲后，自动加载模型，'
        '先运行独立文字/图片控制题，再测正式480题。控制题失败则保留记录并停止本实验。'
        '不停止其他进程。状态见status.json。\n\n'
        '这组是开卷规则应用诊断。模型家族、量化精度和视觉预处理也有变化，结果不能单独归因于参数量。\n')
    return plan


def cpu_audit(work, plan):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only subprocess required'
    assert not any(os.environ.get(k) for k in PIXEL_ENV)
    verify(work, plan)
    import torch
    from swift.model import get_model_processor
    from swift.template import get_template
    torch.set_num_threads(4)
    _, processor = get_model_processor(str(work / 'model_metadata'), model_type='qwen3_vl', load_model=False)
    template = get_template(processor, template_type='qwen3_vl', max_length=16384,
                            truncation_strategy='raise', enable_thinking=False)
    template.set_mode('transformers')
    audited = []
    inputs = rows(work / 'data/inference_inputs.jsonl')
    for i, payload in enumerate(inputs):
        encoded = template.encode(copy.deepcopy(payload))
        ids = encoded['input_ids']
        decoded = processor.tokenizer.decode(ids)
        # Check full prompt text survives actual template processing, including image expansion.
        for piece in payload['messages'][0]['content'].split('<image>'):
            assert piece in decoded, (i, 'changed or truncated prompt')
        grid = encoded.get('image_grid_thw')
        assert ('<|image_pad|>' in decoded) == bool(payload.get('images'))
        assert not decoded.endswith('<think>\n')
        assert len(ids) + plan['max_tokens'] < 16384
        audited.append(dict(index=i, input_sha256=digest(payload), formatted_tokens=len(ids),
            image_grid_thw=grid.tolist() if grid is not None else None))
        if (i + 1) % 40 == 0:
            print('CPU template audit', i + 1, '/480', flush=True)
    write_rows(work / 'template_audit_rows.jsonl', audited)
    save(work / 'template_audit.json', dict(status='pass', checked_at=now(), tasks=480,
        actual_ms_swift_template_encoded=True, cpu_only=True, labels_injected=False,
        prompt_text_preserved=True, truncation_count=0,
        max_formatted_tokens=max(r['formatted_tokens'] for r in audited),
        image_processor=processor.image_processor.to_dict(),
        input_sha256=sha(work / 'data/inference_inputs.jsonl'),
        audit_rows_sha256=sha(work / 'template_audit_rows.jsonl'),
        pixel_env={k: os.environ.get(k) for k in PIXEL_ENV} ))


def summaries(records):
    groups = {}
    for mod in 'TV':
        subset = [r for r in records if r['modality'] == mod]
        for name, selected in [('all', subset)] + [(kind, [r for r in subset if r['question_type'] == kind])
                                                   for kind in ['direct', 'diagnosis']]:
            if selected:
                groups[mod + '/' + name] = dict(scores(selected, 'predicted'),
                    length=sum(r['stop_reason'] == 'length' for r in selected),
                    parse_reasons=dict(Counter(r['parse_reason'] for r in selected)),
                    answer_counts=dict(Counter(str(r['predicted']) for r in selected)))
    return groups


def append(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def pending_indices(manifest, inputs, records):
    """Validate saved answers by identity; permit gaps from archived length retries."""
    assert len(manifest) == len(inputs)
    lookup = {(m['id'], m['modality']): i for i, m in enumerate(manifest)}
    assert len(lookup) == len(manifest)
    seen = set()
    for r in records:
        key = r['id'], r['modality']
        assert key in lookup and key not in seen, 'Unknown or duplicate saved answer'
        i = lookup[key]
        assert r['gold'] == manifest[i]['gold'] and r['input_sha256'] == digest(inputs[i])
        seen.add(key)
    return [i for i, m in enumerate(manifest) if (m['id'], m['modality']) not in seen]


def generation_budget(plan, audit):
    cap = plan['max_tokens']
    assert isinstance(cap, int) and 0 < cap
    assert audit['max_formatted_tokens'] + cap < 16384, 'Generation budget exceeds audited context'
    return cap


def batch_budget(plan):
    size = plan.get('batch_size', 1)
    assert isinstance(size, int) and 1 <= size <= 8
    return size


def checked_replies(replies, expected):
    assert len(replies) == expected, 'Inference batch reply count changed'
    for reply in replies:
        if isinstance(reply, Exception):
            raise reply
    return replies


def model_file_stats(model, files):
    result = {}
    for item in files:
        path = model / item['Path']
        stat = path.stat()
        assert stat.st_size == item['Size'], str(path)
        result[item['Path']] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns,
            ctime_ns=stat.st_ctime_ns, inode=stat.st_ino, device=stat.st_dev)
    return result


def validate_download(work, plan):
    model = Path(plan['model_path'])
    completed = json.loads((model / 'download_complete.json').read_text())
    assert completed['model'] == plan['model'] and completed['sha256_verified'] is True
    frozen = json.loads((work / 'reference/model_download_manifest.json').read_text())
    assert json.loads((model / 'download_manifest.json').read_text()) == frozen
    files = [r for r in frozen['Data']['Files'] if r['Type'] == 'blob']
    current_stats = model_file_stats(model, files)
    cache_path = work / 'model_integrity_cache.json'
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
        proof_path = work / cache['sha_proof_relative_path']
        if (cache['manifest_sha256'] == sha(work / 'reference/model_download_manifest.json')
                and cache['file_stats'] == current_stats and sha(proof_path) == cache['sha_proof_sha256']):
            proof = json.loads(proof_path.read_text())
            assert proof['status'] == 'pass'
            proven = {item['path']: item['sha256'] for item in proof['files']}
            assert proven == {item['Path']: item['Sha256'] for item in files}
            save(work / 'model_integrity_resume.json', dict(status='pass', checked_at=now(),
                method='Reuse prior full SHA256 proof only while file size,mtime,ctime,inode,device and manifest are unchanged',
                proof=str(proof_path), proof_sha256=cache['sha_proof_sha256'], files=len(files)))
            return
    checked = []
    for item in files:
        path = model / item['Path']
        assert path.stat().st_size == item['Size'] and sha(path) == item['Sha256'], str(path)
        checked.append(dict(path=item['Path'], sha256=item['Sha256'], size=item['Size'], revision=item['Revision']))
    assert model_file_stats(model, files) == current_stats, 'Model changed while hashing'
    save(work / 'model_integrity.json', dict(status='pass', checked_at=now(), files=checked,
         total_bytes=sum(r['size'] for r in checked)))
    save(cache_path, dict(manifest_sha256=sha(work / 'reference/model_download_manifest.json'),
        file_stats=current_stats, sha_proof_relative_path='model_integrity.json',
        sha_proof_sha256=sha(work / 'model_integrity.json')))


def evaluate(work, plan):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    assert not any(os.environ.get(k) for k in PIXEL_ENV)
    verify(work, plan)
    audit = json.loads((work / 'template_audit.json').read_text())
    assert audit['status'] == 'pass' and audit['input_sha256'] == sha(work / 'data/inference_inputs.jsonl')
    max_tokens = generation_budget(plan, audit)
    batch_size = batch_budget(plan)
    validate_download(work, plan)
    idle('0')
    import torch
    import bitsandbytes as bnb
    from transformers import BitsAndBytesConfig
    from swift import TransformersEngine, InferRequest, RequestConfig
    torch.set_num_threads(4)
    torch.manual_seed(42)
    quant = dict(plan['quantization'], bnb_4bit_compute_dtype=torch.bfloat16)
    engine = TransformersEngine(plan['model_path'], model_type='qwen3_vl', template_type='qwen3_vl',
        torch_dtype=torch.bfloat16, quantization_config=BitsAndBytesConfig(**quant),
        attn_impl='sdpa', device_map='cuda:0', max_batch_size=batch_size)
    engine.template.enable_thinking = False
    engine.template.max_length = 16384
    engine.template.truncation_strategy = 'raise'
    quantized = [n for n, m in engine.model.named_modules() if isinstance(m, bnb.nn.Linear4bit)]
    assert quantized and not any('visual' in n.split('.') or n.endswith('lm_head') for n in quantized)
    quantized_compute_dtypes = sorted({str(m.compute_dtype) for m in engine.model.modules()
                                      if isinstance(m, bnb.nn.Linear4bit)})
    assert quantized_compute_dtypes == ['torch.bfloat16'], quantized_compute_dtypes
    unquantized_dtypes = sorted({str(p.dtype) for n, p in engine.model.named_parameters()
                                if 'visual' in n.split('.') or n.startswith('lm_head.')})
    assert unquantized_dtypes == ['torch.bfloat16'], unquantized_dtypes
    devices = sorted({str(p.device) for p in engine.model.parameters()})
    assert devices == ['cuda:0'], devices
    runtime = dict(loaded_at=now(), quantized_linear_count=len(quantized), parameter_devices=devices,
        visual_and_lm_head_not_quantized=True, gpu_name=torch.cuda.get_device_name(),
        unquantized_visual_head_dtypes=unquantized_dtypes, quantized_compute_dtypes=quantized_compute_dtypes,
        memory_allocated_bytes=torch.cuda.memory_allocated(), memory_reserved_bytes=torch.cuda.memory_reserved(),
        quantization_config=engine.model.config.quantization_config.to_dict(),
        image_processor=engine.processor.image_processor.to_dict(),
        native_thinking=False, batch_size=batch_size)
    save(work / 'runtime_model.json', runtime)
    # Independent controls are never included in formal task scores.
    from PIL import Image, ImageDraw
    image = Image.new('RGB', (256, 256), 'white')
    ImageDraw.Draw(image).rectangle((32, 32, 224, 224), fill='red')
    control_path = work / 'controls/red_square.png'
    control_path.parent.mkdir(exist_ok=True)
    image.save(control_path)
    suffix = '\nExplain briefly and do not overthink. End with the final option letter in \\boxed{}.'
    controls = [
        (dict(messages=[dict(role='user', content='What is 2 + 3?\nA. 2\nB. 5\nC. 7\nD. 9' + suffix)]), 'B'),
        (dict(messages=[dict(role='user', content='<image>What color is the large square?\nA. Blue\nB. Green\nC. Red\nD. Yellow' + suffix)],
              images=[str(control_path)]), 'C')]
    request = RequestConfig(max_tokens=max_tokens, temperature=0, seed=42, return_details=True,
                            stop=plan['stop_sequences'])
    actual = engine._prepare_generation_config(request)
    assert actual.max_new_tokens == max_tokens and not actual.do_sample
    save(work / 'effective_generation_config.json', dict(max_tokens=max_tokens,
        generation_config=actual.to_dict(), recorded_at=now()))
    control_records = []
    for i, (payload, expected) in enumerate(controls):
        reply = engine.infer([InferRequest(**payload)], request, use_tqdm=False)[0]
        if isinstance(reply, Exception):
            raise reply
        raw = reply.choices[0].message.content or ''
        pred, reason = reparse(raw, options_from_question(payload['messages'][0]['content']))
        control_records.append(dict(id=i, payload=payload, expected=expected, predicted=pred,
            raw_response=raw, parse_reason=reason, pass_control=pred == expected,
            finish_reason=str(reply.choices[0].finish_reason)))
    save(work / 'controls/results.json', control_records)
    assert all(r['pass_control'] for r in control_records), 'Independent control failed; formal evaluation not started'
    if batch_size > 1:
        # Same two independent controls in one mixed text/image batch. No formal labels enter inference.
        tick = time.monotonic()
        replies = checked_replies(engine.infer([InferRequest(**copy.deepcopy(p)) for p, _ in controls],
            request, use_tqdm=False), len(controls))
        batched_controls = []
        for i, ((payload, expected), reply) in enumerate(zip(controls, replies)):
            raw = reply.choices[0].message.content or ''
            pred, reason = reparse(raw, options_from_question(payload['messages'][0]['content']))
            batched_controls.append(dict(id=i, expected=expected, predicted=pred, pass_control=pred == expected,
                raw_response=raw, parse_reason=reason, generated_tokens=reply.usage.completion_tokens))
        save(work / 'controls/batched_results.json', dict(batch_size=len(controls), records=batched_controls,
            seconds=round(time.monotonic() - tick, 3), checked_at=now()))
        assert all(r['pass_control'] for r in batched_controls), 'Batched control failed; formal evaluation not started'
    manifest, inputs = rows(work / 'data/task_manifest.jsonl'), rows(work / 'data/inference_inputs.jsonl')
    output = work / 'evaluation'
    output.mkdir(exist_ok=True)
    path = output / 'predictions.jsonl'
    records = rows(path) if path.exists() else []
    pending = pending_indices(manifest, inputs, records)
    order = {(m['id'], m['modality']): i for i, m in enumerate(manifest)}
    save(output / 'config.json', dict(protocol=VERSION, max_tokens=max_tokens, temperature=0, seed=42,
        batch_size=batch_size, input_sha256=sha(work / 'data/inference_inputs.jsonl'),
        parser=PARSER_VERSION, native_thinking=False, stop_sequences=plan['stop_sequences'],
        quantization=plan['quantization'], finalizer=False,
        budget_revision=plan.get('budget_revision'),
        retained_previous_non_length_replies=plan.get('retained_previous_non_length_replies', 0),
        throughput_revision=plan.get('throughput_revision')))
    started = time.monotonic()
    cursor, effective_batch_size = 0, batch_size
    while cursor < len(pending):
        indices = pending[cursor:cursor + effective_batch_size]
        tick = time.monotonic()
        save(output / 'inflight_batch.json', dict(status='generating', started_at=now(),
            batch_size=len(indices), configured_batch_size=batch_size, indices=indices,
            ids=[dict(id=manifest[i]['id'], modality=manifest[i]['modality']) for i in indices],
            max_tokens=max_tokens))
        torch.cuda.reset_peak_memory_stats()
        try:
            replies = checked_replies(engine.infer(
                [InferRequest(**copy.deepcopy(inputs[i])) for i in indices], request, use_tqdm=False), len(indices))
        except torch.OutOfMemoryError:
            if effective_batch_size == 1:
                raise
            effective_batch_size = max(1, effective_batch_size // 2)
            gc.collect()
            torch.cuda.empty_cache()
            append(work / 'batch_size_fallbacks.jsonl', dict(reason='cuda_oom', checked_at=now(),
                attempted_batch_size=len(indices), new_batch_size=effective_batch_size,
                pending_indices_unchanged=True))
            continue
        batch_seconds = round(time.monotonic() - tick, 3)
        new = []
        for i, reply in zip(indices, replies):
            m, payload = manifest[i], inputs[i]
            choice = reply.choices[0]
            raw = choice.message.content or ''
            legacy, legacy_reason = glm_readout(raw, m['options'])
            pred, reason = reparse(raw, m['options'])
            r = {k: v for k, v in m.items() if k != 'options'}
            r.update(raw_response=raw, raw_predicted=legacy, raw_parse_reason=legacy_reason,
                predicted=pred, parse_reason=reason, correct=pred == m['gold'],
                stop_reason=str(choice.finish_reason), generated_tokens=reply.usage.completion_tokens,
                prompt_tokens=reply.usage.prompt_tokens, generation_max_tokens=max_tokens,
                input_sha256=digest(payload), seconds=round(batch_seconds / len(indices), 3),
                seconds_semantics='amortized batch wall time; not individual latency',
                batch_seconds=batch_seconds, inference_batch_size=len(indices), saved_at=now())
            new.append(r)
        records.extend(new)
        records.sort(key=lambda item: order[item['id'], item['modality']])
        write_rows(path, records)
        batch_info = dict(status='complete', saved=len(records), total=480, indices=indices,
            batch_size=len(indices), seconds=batch_seconds,
            generated_tokens=sum(r['generated_tokens'] for r in new),
            max_reply_tokens=max(r['generated_tokens'] for r in new),
            aggregate_tokens_per_second=sum(r['generated_tokens'] for r in new) / batch_seconds,
            amortized_seconds_per_question=batch_seconds / len(indices),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), updated_at=now())
        append(work / 'throughput_batches.jsonl', batch_info)
        save(output / 'inflight_batch.json', batch_info)
        save(output / 'status.json', dict(status='generating', saved=len(records), total=480,
            groups=summaries(records), current_max_tokens=max_tokens,
            configured_batch_size=batch_size, effective_batch_size=effective_batch_size,
            last_batch=batch_info,
            generation_cap_counts=dict(Counter(str(item['generation_max_tokens']) for item in records)),
            elapsed_seconds=round(time.monotonic() - started, 1), updated_at=now()))
        print(json.dumps(batch_info), flush=True)
        cursor += len(indices)
    save(output / 'status.json', dict(status='complete', saved=480, total=480, groups=summaries(records),
        current_max_tokens=max_tokens,
        generation_cap_counts=dict(Counter(str(item['generation_max_tokens']) for item in records)), updated_at=now()))


def summarize(work, plan):
    records = rows(work / 'evaluation/predictions.jsonl')
    assert len(records) == 480 and len({(r['id'], r['modality']) for r in records}) == 480
    baseline = rows(work / 'reference/glm_same_questions.jsonl')
    groups = dict(qwen32b_nf4=summaries(records), glm46_bf16=summaries(baseline))
    paired, differences = [], []
    lookup = {(r['id'], r['modality']): r for r in records}
    for r, old in zip(records, baseline):
        assert (r['id'], r['modality'], r['gold'], r['input_sha256']) == (
            old['id'], old['modality'], old['gold'], old['input_sha256'])
        differences.append(dict(id=r['id'], modality=r['modality'], gold=r['gold'],
            qwen_predicted=r['predicted'], glm_predicted=old['predicted'],
            qwen_correct=r['correct'], glm_correct=old['correct']))
        if r['modality'] == 'T':
            v = lookup[r['id'], 'V']
            state = 'both_correct' if r['correct'] and v['correct'] else (
                'text_only_correct' if r['correct'] else 'vision_only_correct' if v['correct'] else 'both_wrong')
            paired.append(dict(id=r['id'], qid=r['qid'], outcome=state))
    write_rows(work / 'paired_results.jsonl', paired)
    write_rows(work / 'comparison_details.jsonl', differences)
    save(work / 'comparison.json', dict(status='complete', groups=groups,
        paired_outcomes=dict(Counter(r['outcome'] for r in paired)), completed_at=now(), limitations=plan['limitations'],
        current_qwen_max_tokens=plan['max_tokens'],
        qwen_generation_cap_counts=dict(Counter(str(r['generation_max_tokens']) for r in records)),
        glm_generation_cap_counts=dict(Counter(str(r['generation_max_tokens']) for r in baseline)),
        qwen_inference_batch_counts=dict(Counter(str(r.get('inference_batch_size', 1)) for r in records)),
        throughput_revision=plan.get('throughput_revision'),
        budget_revision=plan.get('budget_revision')))
    lines = ['# 多步变换：给定原规则的模型对照', '',
             '| 模型 | 测试输入 | 题数 | 准确率 | 无效率 | 截断题数 |', '|---|---|---:|---:|---:|---:|']
    for arm, name in [('glm46_bf16', 'GLM-4.6V-Flash BF16'), ('qwen32b_nf4', 'Qwen3-VL-32B NF4')]:
        for mod, label in [('T', '规则文本＋文字题'), ('V', '规则文本＋图片题')]:
            g = groups[arm][mod + '/all']
            lines.append(f'| {name} | {label} | {g["n"]} | {g["accuracy"]:.2%} | {g["invalid_rate"]:.2%} | {g["length"]} |')
    lines += ['', '120条规则，文字/图片各240题，完全同题配对；无效计错。使用相同简短推理提示与保守解析器。',
        f'Qwen当前输出上限{plan["max_tokens"]} token；逐条实际上限计数：'
        + json.dumps(dict(Counter(str(r['generation_max_tokens']) for r in records))) + '。GLM基线沿用原输出上限；提高预算后的结果不是同预算模型对照。',
        '若存在输出上限调整，原短预算回复已归档；只保留其正常结束回复，全部触及旧上限的题均用新上限重测，不按正误挑选。',
        '本实验没有训练或加载LoRA，测量提供规则后的应用能力。原生thinking关闭，按提示给简短解释。',
        '推理批量按用户提速要求调整，既有回复保留；逐题批量及批次耗时均记录，浮点计算路径可能因批量变化产生差异。',
        '模型家族、精度和视觉预处理均有差异，不能将提升单独归因为参数量。',
        '详细direct/diagnosis结果见comparison.json；原回复及tokens见evaluation/predictions.jsonl。', '']
    (work / 'RESULTS_ZH.md').write_text('\n'.join(lines))


def readiness(dependency, model):
    previous = json.loads((dependency / 'status.json').read_text())
    if previous.get('stage') != 'complete':
        return 'queued_after_vision_suite', dict(dependency_stage=previous.get('stage'))
    for name in ['experiment.json', 'comparison.json']:
        assert json.loads((dependency / name).read_text())['status'] == 'complete', name
    if not (model / 'download_complete.json').exists():
        return 'waiting_for_model_download', {}
    return 'ready', {}


def controller(root, work):
    assert socket.gethostname() == 'softengine-KVM'
    work.mkdir(parents=True, exist_ok=True)
    with (work / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = prepare(root, work)
        if plan['status'] == 'complete':
            return
        def status(stage, **extra):
            save(work / 'status.json', dict(stage=stage, controller_pid=os.getpid(),
                updated_at=now(), host=socket.gethostname(), gpu='0', **extra))
        while True:
            state, detail = readiness(Path(plan['queue_dependency']), Path(plan['model_path']))
            if state == 'ready':
                try:
                    idle('0')
                    break
                except RuntimeError as error:
                    state, detail = 'waiting_for_free_gpu', dict(reason=str(error))
            status(state, dependency=plan['queue_dependency'], inference_started=False, **detail)
            time.sleep(30)
        verify(work, plan)
        audit = json.loads((work / 'template_audit.json').read_text())
        assert audit['status'] == 'pass' and audit['audit_rows_sha256'] == sha(work / 'template_audit_rows.jsonl')
        cmd = [str(root / '.venv/bin/python'), '-u', str(work / 'source_snapshots/run_rules_qwen32b_rule_given.py'),
               '--root', str(root), '--work', str(work), '--stage', 'evaluate']
        save(work / 'evaluation_command.json', cmd)
        with (work / 'evaluation.log').open('a') as stream:
            child = subprocess.Popen(cmd, cwd=root, env=environment(), stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        status('evaluating', child_pid=child.pid, inference_started=True)
        if child.wait():
            raise RuntimeError('32B evaluation failed; inspect evaluation.log; all outputs preserved')
        summarize(work, plan)
        plan.update(status='complete', completed_at=now())
        save(work / 'experiment.json', plan)
        status('complete', results=str(work / 'RESULTS_ZH.md'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--stage', choices=['prepare', 'audit', 'evaluate'])
    args = parser.parse_args()
    root, work = args.root.resolve(), args.work.resolve()
    try:
        if args.stage == 'prepare':
            prepare(root, work)
        elif args.stage in {'audit', 'evaluate'}:
            plan = json.loads((work / 'experiment.json').read_text())
            (cpu_audit if args.stage == 'audit' else evaluate)(work, plan)
        else:
            controller(root, work)
    except Exception as error:
        save(work / (f'failure_{args.stage}.json' if args.stage else 'status.json'),
             dict(stage='failed', error=repr(error), updated_at=now()))
        raise
