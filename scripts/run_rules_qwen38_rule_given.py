#!/usr/bin/env python3
"""Queue Qwen3.8-27B on the same paired open-book tasks after the 32B diagnostic."""
import argparse
from collections import Counter
import copy
from contextlib import nullcontext
import fcntl
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import time

from run_rules_qwen32b_rule_given import (
    SOURCES as SHARED_SOURCES, PIXEL_ENV, append, environment, freeze, select_tasks,
    summaries, validate_download, verify, generation_budget, batch_budget, checked_replies,
)
from answer_readout import options_from_question
from audit_rules_glm46_readout import VERSION as PARSER_VERSION, reparse
from entityqa_common import digest, now, rows, save, sha, write_rows
from evaluate_entity_glm46_full import ANSWER_STOP_SEQUENCES, glm_readout
from run_entityqa_lora import idle

VERSION = 'rules120-qwen38-27b-nf4-rule-given-v1'
PACKAGES = ['torch', 'transformers', 'ms-swift', 'bitsandbytes', 'accelerate', 'qwen-vl-utils']
SOURCES = list(dict.fromkeys(['run_rules_qwen38_rule_given.py', 'verified_ram_weights.py'] + SHARED_SOURCES))


def check_environment(plan):
    for name, expected in plan['packages'].items():
        assert importlib.metadata.version(name) == expected, f'Dependency changed: {name}'
    assert str(Path(os.sys.prefix).resolve()) == plan['environment']


def prepare(root, work):
    work.mkdir(parents=True, exist_ok=True)
    if (work / 'experiment.json').exists():
        plan = json.loads((work / 'experiment.json').read_text())
        assert plan['protocol'] == VERSION and plan['project_root'] == str(root)
        verify(work, plan)
        check_environment(plan)
        return plan
    previous = root / 'runs/rules120_qwen3vl32b_nf4_rule_given_20261109'
    parent = json.loads((previous / 'experiment.json').read_text())
    verify(previous, parent)
    manifest, inputs = select_tasks(rows(previous / 'data/task_manifest.jsonl'),
                                    rows(previous / 'data/inference_inputs.jsonl'))
    for rel in ['data/task_manifest.jsonl', 'data/inference_inputs.jsonl', 'data/image_hashes.jsonl',
                'reference/glm_same_questions.jsonl', 'reference/glm_summary.json']:
        freeze(previous / rel, work / rel)
    assert rows(work / 'data/task_manifest.jsonl') == manifest
    assert rows(work / 'data/inference_inputs.jsonl') == inputs
    model = root / 'model_cache/Qwen3.8-27B'
    freeze(model / 'download_manifest.json', work / 'reference/model_download_manifest.json')
    metadata = work / 'model_metadata'
    listed = json.loads((metadata / 'source_manifest.json').read_text())
    for item in listed['files']:
        assert sha(metadata / item['Path']) == item['Sha256'], item['Path']
    config = json.loads((metadata / 'config.json').read_text())
    assert config['model_type'] == 'qwen3_5' and config['vision_config']
    for name in SOURCES:
        freeze(root / 'scripts' / name, work / 'source_snapshots' / name)
    immutable = [p for folder in ['data', 'reference', 'model_metadata']
                 for p in (work / folder).iterdir() if p.is_file()]
    plan = dict(protocol=VERSION, status='prepared', created_at=now(), project_root=str(root),
        environment=str(root / '.venv-qwen38'), model='Qwen/Qwen3.8-27B', model_path=str(model),
        model_type='qwen3_5', template_type='qwen3_8', framework='ms-swift TransformersEngine',
        host=socket.gethostname(), gpu='0', training=False, adapters=[], batch_size=1,
        queue_dependency=str(previous), queue_policy='Wait for completed 32B run and free GPU; never stop another task',
        source_rule_given_run=parent['source_rule_given_run'], total=480, rules=120,
        questions_per_modality=240, family='transformation_procedure',
        max_tokens=1024, temperature=0, seed=42, native_thinking=False, brief_prompted_reasoning=True,
        prompt='Verbatim same rule-given inputs as GLM and Qwen3-VL-32B; no additional system instruction',
        stop_sequences=ANSWER_STOP_SEQUENCES, parser=PARSER_VERSION, finalizer=False,
        quantization=copy.deepcopy(parent['quantization']),
        image_policy='Original test images; model-default Qwen3.8/ms-swift preprocessing; no pixel overrides',
        packages={name: importlib.metadata.version(name) for name in PACKAGES},
        immutable_sha256={str(p.relative_to(work)): sha(p) for p in immutable},
        source_sha256={name: sha(work / 'source_snapshots' / name) for name in SOURCES},
        limitations=[
            'Open-book rule application diagnostic; not a trained cross-modal transfer score.',
            'Model generation, architecture, size, quantization and vision preprocessing can affect results.',
            'Native thinking is explicitly disabled for the same 1024-token brief-reasoning comparison.',
            'GPU quantized loading and actual inference remain pending until earlier experiments finish.',
        ])
    check_environment(plan)
    save(work / 'experiment.json', plan)
    (work / 'README_ZH.md').write_text(
        '# Qwen3.8-27B：规则开卷对照\n\n'
        '使用Qwen/Qwen3.8-27B干净模型，不训练、不加LoRA。模型架构qwen3_5，对话模板qwen3_8；'
        '使用项目.venv-qwen38，通过只读复用已安装依赖，保持旧环境的包不变。\n\n'
        '与GLM、Qwen3-VL-32B保持同一批120条多步变换规则、480题：文字与图片各240题，'
        'direct/diagnosis各120题、ABCD答案各60题。逐题给完整原规则，保留原题、选项和图片。'
        '含gold的评分清单独立于模型输入。\n\n'
        '与32B对照一致：4bit NF4双重量化、BF16计算，视觉编码器与lm_head不量化，batch1，'
        'temperature0，seed42，1024输出token。显式关闭原生thinking，保留原题的简短解释与boxed提示，'
        '不额外加入默认xhigh推理指令；不强制补选。图片用模型默认预处理。\n\n'
        '在32B实验全部完成且GPU空闲后自动运行；不会抢占当前视觉实验。正式开始前核对权重SHA256、'
        '检查实际量化和显存，再用独立文字/图片控制题验证推理链路。控制失败保留输出、停止本次评测。\n\n'
        '分别报告文字/图片准确率、无效率、截断、direct/diagnosis及同题配对。'
        '保留原始回复，与GLM及32B逐题对照；这里评估提供规则后的应用能力。\n')
    return plan


def cpu_audit(work, plan):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not any(os.environ.get(k) for k in PIXEL_ENV)
    check_environment(plan)
    verify(work, plan)
    import torch
    from swift.model import get_model_processor
    from swift.template import get_template
    torch.set_num_threads(4)
    _, processor = get_model_processor(str(work / 'model_metadata'),
        model_type=plan['model_type'], load_model=False, torch_dtype=torch.bfloat16)
    template = get_template(processor, template_type=plan['template_type'],
        enable_thinking=False, max_length=16384, truncation_strategy='raise')
    template.set_mode('transformers')
    audit_rows = []
    inputs = rows(work / 'data/inference_inputs.jsonl')
    from PIL import Image
    for i, payload in enumerate(inputs):
        encoded = template.encode(copy.deepcopy(payload))
        ids = encoded['input_ids']
        decoded = processor.tokenizer.decode(ids)
        for piece in payload['messages'][0]['content'].split('<image>'):
            assert piece in decoded, (i, 'prompt changed or truncated')
        assert decoded.endswith('<think>\n\n</think>\n\n'), (i, 'native thinking not disabled')
        assert 'Reasoning effort is set to' not in decoded, (i, 'extra reasoning instruction')
        assert ('<|image_pad|>' in decoded) == bool(payload.get('images'))
        assert len(ids) + plan['max_tokens'] < 16384
        grid = encoded.get('image_grid_thw')
        sizes = []
        for path in payload.get('images', []):
            with Image.open(path) as image:
                sizes.append(list(image.size))
        audit_rows.append(dict(index=i, input_sha256=digest(payload), encoded_ids_sha256=digest(ids),
            formatted_tokens=len(ids), image_grid_thw=None if grid is None else grid.tolist(),
            original_image_sizes=sizes))
        if (i + 1) % 40 == 0:
            print('CPU Qwen3.8 template audit', i + 1, '/480', flush=True)
    write_rows(work / 'template_audit_rows.jsonl', audit_rows)
    save(work / 'template_audit.json', dict(status='pass', checked_at=now(), tasks=480,
        cpu_only=True, actual_ms_swift_template_encoded=True, labels_injected=False,
        native_thinking_disabled_verified=True, no_default_reasoning_instruction=True,
        truncation_count=0, prompt_text_preserved=True, model_type=plan['model_type'],
        template_type=plan['template_type'], max_formatted_tokens=max(r['formatted_tokens'] for r in audit_rows),
        image_processor=processor.image_processor.to_dict(),
        input_sha256=sha(work / 'data/inference_inputs.jsonl'),
        audit_rows_sha256=sha(work / 'template_audit_rows.jsonl')))


def make_record(task, payload, reply, seconds, max_tokens=1024, inference_batch_size=1):
    choice = reply.choices[0]
    raw = choice.message.content or ''
    legacy, legacy_reason = glm_readout(raw, task['options'])
    predicted, reason = reparse(raw, task['options'])
    record = {k: v for k, v in task.items() if k != 'options'}
    record.update(raw_response=raw, raw_reasoning_content=getattr(choice.message, 'reasoning_content', None),
        raw_predicted=legacy, raw_parse_reason=legacy_reason, predicted=predicted, parse_reason=reason,
        correct=predicted == task['gold'], stop_reason=str(choice.finish_reason),
        generated_tokens=reply.usage.completion_tokens, prompt_tokens=reply.usage.prompt_tokens,
        generation_max_tokens=max_tokens, input_sha256=digest(payload), seconds=round(seconds, 3),
        inference_batch_size=inference_batch_size, saved_at=now())
    return record


def check_saved(records, manifest, inputs):
    assert len(records) <= len(manifest)
    for i, r in enumerate(records):
        assert (r['id'], r['modality'], r['gold'], r['input_sha256']) == (
            manifest[i]['id'], manifest[i]['modality'], manifest[i]['gold'], digest(inputs[i]))


def gpu_admission(work, plan):
    required = plan.get('required_free_gpu_mib')
    if required is None:
        idle('0')
        return
    device = Path('/dev/nvidia0')
    if device.exists() and not os.access(device, os.R_OK | os.W_OK):
        raise RuntimeError('GPU0 device permission denied')
    result = subprocess.check_output(['nvidia-smi', '-i', '0',
        '--query-gpu=memory.free,memory.used', '--format=csv,noheader,nounits'], text=True, timeout=15)
    free, used = [int(v.strip()) for v in result.strip().split(',')]
    if free < required:
        raise RuntimeError(f'GPU0 free {free} MiB; need at least {required} MiB')
    save(work / 'gpu_admission.json', dict(checked_at=now(), free_mib=free, used_mib=used,
        required_free_mib=required, policy='Use available memory; never signal any other GPU task',
        existing_processes=subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
            '--format=csv,noheader'], text=True, timeout=15)))


def check_quantization_policy(plan):
    """Transformers 5 skip patterns must name Qwen's nested visual subtree."""
    from transformers.quantizers.quantizers_utils import should_convert_module
    patterns = plan['quantization']['llm_int8_skip_modules']
    for module in ['model.visual', 'model.visual.blocks.0.attn.qkv',
                   'model.visual.merger.linear_fc1', 'lm_head']:
        assert not should_convert_module(module, patterns), f'Unprotected BF16 module: {module}'
    assert should_convert_module('model.language_model.layers.0.mlp.down_proj', patterns), (
        'Language model projection must retain NF4 quantization')


def evaluate(work, plan):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    assert not any(os.environ.get(k) for k in PIXEL_ENV)
    check_environment(plan)
    verify(work, plan)
    audit = json.loads((work / 'template_audit.json').read_text())
    assert audit['status'] == 'pass' and audit['input_sha256'] == sha(work / 'data/inference_inputs.jsonl')
    assert audit['audit_rows_sha256'] == sha(work / 'template_audit_rows.jsonl')
    max_tokens, batch_size = generation_budget(plan, audit), batch_budget(plan)
    validate_download(work, plan)
    gpu_admission(work, plan)
    import torch
    import bitsandbytes as bnb
    from transformers import BitsAndBytesConfig
    from swift import TransformersEngine, InferRequest, RequestConfig
    torch.set_num_threads(4)
    torch.manual_seed(42)
    check_quantization_policy(plan)
    quant = dict(plan['quantization'], bnb_4bit_compute_dtype=torch.bfloat16)
    loader = nullcontext()
    if plan.get('weight_loading', {}).get('method') == 'verified_official_tmpfs_cache':
        from verified_ram_weights import use_verified_ram_weights
        loading = plan['weight_loading']
        def report_loading(event):
            append(work / 'weight_loading_events.jsonl', event)
            save(work / 'model_loading_status.json', event)
            print(json.dumps(event), flush=True)
        loader = use_verified_ram_weights(plan['model_path'],
            json.loads((work / 'reference/model_download_manifest.json').read_text()),
            loading['directory'], max_bytes=loading['max_bytes'],
            rate_bytes=loading['rate_bytes'], reserve_bytes=loading['reserve_bytes'],
            reporter=report_loading)
    else:
        save(work / 'model_loading_status.json', dict(stage='loading_local_weights',
            recorded_at=now(), method=plan.get('weight_loading', {}).get('method', 'local_memory_map')))
    loading_tick = time.monotonic()
    with loader as cache:
        engine = TransformersEngine(plan['model_path'], model_type=plan['model_type'],
            template_type=plan['template_type'], torch_dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(**quant), attn_impl='sdpa',
            device_map='cuda:0', max_batch_size=batch_size)
        if cache is not None:
            save(work / 'weight_loading_summary.json', dict(status='complete',
                method=plan['weight_loading'], stats=cache.stats, completed_at=now()))
        else:
            save(work / 'weight_loading_summary.json', dict(status='complete',
                method=plan.get('weight_loading', {}), seconds=round(time.monotonic() - loading_tick, 2),
                completed_at=now()))
    gc.collect()
    engine.template.enable_thinking = False
    engine.template.max_length = 16384
    engine.template.truncation_strategy = 'raise'
    quantized = [n for n, m in engine.model.named_modules() if isinstance(m, bnb.nn.Linear4bit)]
    protected_quantized = [n for n in quantized if 'visual' in n.split('.') or n.endswith('lm_head')]
    save(work / 'runtime_quantization_modules.json', dict(checked_at=now(),
        quantized=quantized, protected_quantized=protected_quantized,
        requested_skip_modules=plan['quantization']['llm_int8_skip_modules']))
    assert quantized and not protected_quantized, (
        f'Incorrect NF4 module selection: count={len(quantized)}, protected={protected_quantized}')
    compute = sorted({str(m.compute_dtype) for m in engine.model.modules() if isinstance(m, bnb.nn.Linear4bit)})
    assert compute == ['torch.bfloat16'], compute
    unquantized = sorted({str(p.dtype) for n, p in engine.model.named_parameters()
                         if 'visual' in n.split('.') or n.startswith('lm_head.')})
    assert unquantized == ['torch.bfloat16'], unquantized
    devices = sorted({str(p.device) for p in engine.model.parameters()})
    assert devices == ['cuda:0'], devices
    # Audit and runtime must produce the same full input IDs and image grid.
    inputs = rows(work / 'data/inference_inputs.jsonl')
    audits = rows(work / 'template_audit_rows.jsonl')
    engine.template.set_mode('transformers')
    for i in range(4):
        encoded = engine.template.encode(copy.deepcopy(inputs[i]))
        assert digest(encoded['input_ids']) == audits[i]['encoded_ids_sha256']
    save(work / 'runtime_model.json', dict(loaded_at=now(), model=plan['model'],
        model_type=plan['model_type'], template_type=plan['template_type'],
        quantized_linear_count=len(quantized), quantized_compute_dtypes=compute,
        unquantized_visual_head_dtypes=unquantized, parameter_devices=devices,
        native_thinking=False, audit_runtime_input_ids_match=True, batch_size=batch_size,
        memory_allocated_bytes=torch.cuda.memory_allocated(), memory_reserved_bytes=torch.cuda.memory_reserved(),
        gpu_name=torch.cuda.get_device_name(), image_processor=engine.processor.image_processor.to_dict(),
        quantization_config=engine.model.config.quantization_config.to_dict()))
    request = RequestConfig(max_tokens=max_tokens, temperature=0, seed=42, return_details=True,
                            stop=plan['stop_sequences'])
    actual = engine._prepare_generation_config(request)
    assert actual.max_new_tokens == max_tokens and not actual.do_sample
    save(work / 'effective_generation_config.json', dict(max_tokens=max_tokens,
        generation_config=actual.to_dict(), recorded_at=now()))
    from PIL import Image, ImageDraw
    control_image = work / 'controls/red_square.png'
    control_image.parent.mkdir(exist_ok=True)
    image = Image.new('RGB', (256, 256), 'white')
    ImageDraw.Draw(image).rectangle((32, 32, 224, 224), fill='red')
    image.save(control_image)
    suffix = '\nExplain briefly and do not overthink. End with the final option letter in \\boxed{}.'
    controls = [(dict(messages=[dict(role='user', content='What is 2 + 3?\nA. 2\nB. 5\nC. 7\nD. 9' + suffix)]), 'B'),
        (dict(messages=[dict(role='user', content='<image>What color is the large square?\nA. Blue\nB. Green\nC. Red\nD. Yellow' + suffix)],
              images=[str(control_image)]), 'C')]
    control_results = []
    for i, (payload, expected) in enumerate(controls):
        reply = engine.infer([InferRequest(**copy.deepcopy(payload))], request, use_tqdm=False)[0]
        if isinstance(reply, Exception):
            raise reply
        raw = reply.choices[0].message.content or ''
        pred, reason = reparse(raw, options_from_question(payload['messages'][0]['content']))
        control_results.append(dict(id=i, payload=payload, expected=expected, predicted=pred,
            raw_response=raw, parse_reason=reason, pass_control=pred == expected,
            finish_reason=str(reply.choices[0].finish_reason)))
    save(work / 'controls/results.json', control_results)
    assert all(r['pass_control'] for r in control_results), 'Control failed; formal tasks not started'
    if batch_size > 1:
        tick = time.monotonic()
        # Validate mixed text/image batching at the configured size before formal questions.
        grouped_controls = [controls[i % len(controls)] for i in range(batch_size)]
        replies = checked_replies(engine.infer([InferRequest(**copy.deepcopy(p)) for p, _ in grouped_controls],
            request, use_tqdm=False), len(grouped_controls))
        results = []
        for i, ((payload, expected), reply) in enumerate(zip(grouped_controls, replies)):
            raw = reply.choices[0].message.content or ''
            pred, reason = reparse(raw, options_from_question(payload['messages'][0]['content']))
            results.append(dict(id=i, expected=expected, predicted=pred, pass_control=pred == expected,
                raw_response=raw, parse_reason=reason))
        save(work / 'controls/batched_results.json', dict(batch_size=batch_size,
            records=results, seconds=round(time.monotonic() - tick, 3), checked_at=now()))
        assert all(r['pass_control'] for r in results), 'Mixed batched control failed; formal tasks not started'
    manifest = rows(work / 'data/task_manifest.jsonl')
    output = work / 'evaluation'
    output.mkdir(exist_ok=True)
    path = output / 'predictions.jsonl'
    records = rows(path) if path.exists() else []
    check_saved(records, manifest, inputs)
    config = dict(protocol=VERSION, model=plan['model'], quantization=plan['quantization'],
        max_tokens=max_tokens, temperature=0, seed=42, batch_size=batch_size, native_thinking=False,
        stop_sequences=plan['stop_sequences'], parser=PARSER_VERSION, finalizer=False,
        input_sha256=sha(work / 'data/inference_inputs.jsonl'))
    config_path = output / 'config.json'
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config, 'Resume protocol changed'
    else:
        assert not records, 'Missing configuration for existing predictions'
        save(config_path, config)
    started = time.monotonic()
    effective_batch_size = batch_size
    while len(records) < len(manifest):
        indices = list(range(len(records), min(len(records) + effective_batch_size, len(manifest))))
        tick = time.monotonic()
        save(output / 'inflight_batch.json', dict(status='generating', indices=indices,
            batch_size=len(indices), max_tokens=max_tokens, started_at=now()))
        torch.cuda.reset_peak_memory_stats()
        try:
            replies = checked_replies(engine.infer([InferRequest(**copy.deepcopy(inputs[i])) for i in indices],
                request, use_tqdm=False), len(indices))
        except torch.OutOfMemoryError:
            if effective_batch_size == 1:
                raise
            effective_batch_size = max(1, effective_batch_size // 2)
            gc.collect()
            torch.cuda.empty_cache()
            append(work / 'batch_size_fallbacks.jsonl', dict(reason='cuda_oom',
                attempted_batch_size=len(indices), new_batch_size=effective_batch_size, updated_at=now()))
            continue
        elapsed = time.monotonic() - tick
        new = [make_record(manifest[i], inputs[i], reply, elapsed / len(indices), max_tokens,
            inference_batch_size=len(indices)) for i, reply in zip(indices, replies)]
        for record in new:
            record.update(batch_seconds=round(elapsed, 3),
                seconds_semantics='amortized batch wall time; not individual latency')
        records.extend(new)
        write_rows(path, records)
        batch_info = dict(status='complete', saved=len(records), total=480, indices=indices,
            batch_size=len(indices), seconds=round(elapsed, 3),
            generated_tokens=sum(r['generated_tokens'] for r in new),
            aggregate_tokens_per_second=sum(r['generated_tokens'] for r in new) / elapsed,
            amortized_seconds_per_question=elapsed / len(indices),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), updated_at=now())
        append(work / 'throughput_batches.jsonl', batch_info)
        save(output / 'inflight_batch.json', batch_info)
        save(output / 'status.json', dict(status='generating', saved=len(records), total=480,
            max_tokens=max_tokens, configured_batch_size=batch_size, effective_batch_size=effective_batch_size,
            last_batch=batch_info,
            groups=summaries(records), elapsed_seconds=round(time.monotonic() - started, 1), updated_at=now()))
        print(json.dumps(batch_info), flush=True)
    save(output / 'status.json', dict(status='complete', saved=480, total=480,
                                    groups=summaries(records), updated_at=now()))


def summarize(work, plan):
    manifest, inputs = rows(work / 'data/task_manifest.jsonl'), rows(work / 'data/inference_inputs.jsonl')
    ours = rows(work / 'evaluation/predictions.jsonl')
    baseline = rows(work / 'reference/glm_same_questions.jsonl')
    previous = rows(work / 'reference/qwen32_partial_predictions.jsonl') if (work / 'reference/qwen32_partial_predictions.jsonl').exists() else rows(Path(plan['queue_dependency']) / 'evaluation/predictions.jsonl')
    models = dict(glm46_bf16=baseline, qwen38_27b_nf4=ours)
    assert all(len(records) == 480 for records in models.values())
    for records in models.values():
        check_saved(records, manifest, inputs)
    lookup = {(r['id'], r['modality']): r for r in ours}
    pairs = []
    for r in ours:
        if r['modality'] == 'T':
            v = lookup[r['id'], 'V']
            state = 'both_correct' if r['correct'] and v['correct'] else (
                'text_only_correct' if r['correct'] else 'vision_only_correct' if v['correct'] else 'both_wrong')
            pairs.append(dict(id=r['id'], qid=r['qid'], outcome=state))
    write_rows(work / 'paired_results.jsonl', pairs)
    details = [dict(id=m['id'], modality=m['modality'], gold=m['gold'],
                    model_results={name: dict(predicted=records[i]['predicted'], correct=records[i]['correct'])
                                   for name, records in models.items()}) for i, m in enumerate(manifest)]
    write_rows(work / 'comparison_details.jsonl', details)
    groups = {name: summaries(records) for name, records in models.items()}
    # The earlier 32B experiment was stopped. Compare it only on its existing identical inputs.
    old_lookup = {(r['id'], r['modality']): r for r in previous}
    assert len(old_lookup) == len(previous)
    overlap = {name: [r for r in records if (r['id'], r['modality']) in old_lookup]
               for name, records in models.items()}
    overlap['qwen3vl32b_nf4'] = previous
    for records in overlap.values():
        assert len(records) == len(previous)
        for r in records:
            old = old_lookup[r['id'], r['modality']]
            assert (r['gold'], r['input_sha256']) == (old['gold'], old['input_sha256'])
    save(work / 'comparison.json', dict(status='complete', groups=groups,
        qwen32_partial_overlap=dict(tasks=len(previous), groups={name: summaries(rs) for name, rs in overlap.items()}),
        generation_cap_counts={name: dict(Counter(str(r['generation_max_tokens']) for r in rs)) for name, rs in {**models, 'qwen3vl32b_nf4_partial': previous}.items()},
        paired_outcomes=dict(Counter(r['outcome'] for r in pairs)), completed_at=now(), limitations=plan['limitations']))
    lines = ['# 多步变换规则开卷：GLM与两组Qwen对照', '',
        '| 模型 | 题目模态 | 题数 | 准确率 | 无效率 | 截断数 |', '|---|---|---:|---:|---:|---:|']
    for name, label in [('glm46_bf16', 'GLM-4.6V-Flash BF16'),
                        ('qwen38_27b_nf4', 'Qwen3.8-27B NF4')]:
        for mod, mod_label in [('T', '文字'), ('V', '图片')]:
            g = groups[name][mod + '/all']
            lines.append(f'| {label} | {mod_label} | {g["n"]} | {g["accuracy"]:.2%} | {g["invalid_rate"]:.2%} | {g["length"]} |')
    lines += ['', f'两组均给完整原规则，同题同选项原图；简短解释提示，原生thinking关闭。Qwen3.8输出上限{plan["max_tokens"]} token，GLM保持原预算，无效计错。',
              f'Qwen3-VL-32B仅完成{len(previous)}题；三模型在这些相同题上的对照单独列于comparison.json的qwen32_partial_overlap，不能把该部分结果当作全量480题。',
              '模型架构、精度和视觉预处理有差异，不能只归因于参数量。这里是开卷应用能力，不是训练后的迁移成绩。', '']
    (work / 'RESULTS_ZH.md').write_text('\n'.join(lines))


def readiness(dependency, model, allow_stopped_dependency=False):
    state = json.loads((dependency / 'status.json').read_text())
    if state.get('stage') == 'stopped_by_user' and allow_stopped_dependency:
        plan = json.loads((dependency / 'experiment.json').read_text())
        comparison = json.loads((dependency / 'comparison_partial.json').read_text())
        assert plan['status'] == 'stopped_by_user' and comparison['status'] == 'partial_stopped_by_user'
        assert state['saved'] == comparison['saved'] > 0
        assert sha(dependency / 'evaluation/predictions.jsonl') == comparison['predictions_sha256']
        if not (model / 'download_complete.json').exists():
            return 'waiting_for_model_download', {}
        return 'ready', dict(previous_run_stopped=True, previous_saved=state['saved'])
    if state.get('stage') != 'complete':
        return 'queued_after_qwen32b', dict(dependency_stage=state.get('stage'))
    for name in ['experiment.json', 'comparison.json']:
        assert json.loads((dependency / name).read_text())['status'] == 'complete', name
    if not (model / 'download_complete.json').exists():
        return 'waiting_for_model_download', {}
    return 'ready', {}


def controller(root, work):
    assert socket.gethostname() == 'softengine-KVM'
    with (work / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = prepare(root, work)
        if plan['status'] == 'complete':
            return
        def status(stage, **extra):
            save(work / 'status.json', dict(stage=stage, controller_pid=os.getpid(),
                updated_at=now(), host=socket.gethostname(), gpu='0', **extra))
        while True:
            stage, detail = readiness(Path(plan['queue_dependency']), Path(plan['model_path']),
                allow_stopped_dependency=plan.get('explicit_start_after_stopped_dependency', False))
            if stage == 'ready':
                try:
                    gpu_admission(work, plan)
                    break
                except RuntimeError as error:
                    stage, detail = 'waiting_for_free_gpu', dict(reason=str(error))
            status(stage, dependency=plan['queue_dependency'], inference_started=False, **detail)
            time.sleep(30)
        audit = json.loads((work / 'template_audit.json').read_text())
        assert audit['status'] == 'pass' and audit['audit_rows_sha256'] == sha(work / 'template_audit_rows.jsonl')
        cmd = [str(root / '.venv-qwen38/bin/python'), '-u',
            str(work / 'source_snapshots/run_rules_qwen38_rule_given.py'),
            '--root', str(root), '--work', str(work), '--stage', 'evaluate']
        save(work / 'evaluation_command.json', cmd)
        with (work / 'evaluation.log').open('a') as stream:
            child = subprocess.Popen(cmd, cwd=root, env=environment(), stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        status('evaluating', child_pid=child.pid, inference_started=True)
        if child.wait():
            raise RuntimeError('Qwen3.8 evaluation failed; inspect evaluation.log; artifacts preserved')
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
