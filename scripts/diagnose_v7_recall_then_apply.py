#!/usr/bin/env python3
"""Paired retrieval/application development diagnostic; frozen v7 is read-only."""
import argparse
import copy
import dataclasses
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path

import run_v7_evaluation as ev
from run_v7_full_experiment import ROOT, MODELS, environment, read, rows, save, sha
from v7_gpu_resources import reserve

DATA = ROOT / 'rules_repaired_single_image_v7'
WORK = ROOT / 'runs/v7_full_experiment'
PROTOCOLS = ('direct', 'self_recalled_material', 'correct_material')


def normalized(text):
    return ' '.join(re.findall(r'\w+|[^\w\s]', text.lower()))


def immutable_rows(path, content):
    if path.exists():
        assert rows(path) == content, f'Changed inputs: {path}'
    else:
        ev.save_rows(path, content)


def build(out):
    knowledge = {r['content']['id']: r['content'] for r in rows(DATA / 'audit/rule_knowledge.jsonl')}
    train = {r['id']: r for r in rows(WORK / 'data/sft_text.jsonl')}
    labels = {r['id']: r for r in rows(DATA / 'audit/eval_labels.jsonl')}
    groups = defaultdict(list)
    for qid, k in knowledge.items():
        groups[k['fact']['relation']].append(qid)
    # New fixed selection, independent of old probe results and recall success.
    chosen = []
    for relation, ids in sorted(groups.items()):
        chosen.extend(sorted(ids, key=lambda q: hashlib.sha256(
            ('v7-recall-then-apply:42:' + q).encode()).hexdigest())[:4])
    recall = []
    for qid in chosen:
        k, t = knowledge[qid], train[qid]
        assert t['messages'][-1]['role'] == 'assistant'
        assert t['messages'][-1]['content'] == k['material']['learning']['text']
        recall.append({'id': qid, 'qid': qid, 'relation': k['fact']['relation'],
                       'messages': copy.deepcopy(t['messages'][:-1]),
                       'reference': t['messages'][-1]['content'],
                       'structured_rule': json.loads(k['fact']['quote']),
                       'recording_protocol': k.get('inspection_recording_protocol')})
    probes = []
    for row in rows(DATA / 'T2T/eval.jsonl'):
        if row['qid'] not in chosen:
            continue
        ready = ev.prepare_row(row, DATA, 'T2T')
        label = labels[row['id']]
        probes.append({'id': row['id'], 'qid': row['qid'], 'messages': ready['messages'],
                       'gold': label['answer'], 'relation': knowledge[row['qid']]['fact']['relation'],
                       'question_type': label['question_type'],
                       'evaluation_group': label['evaluation_group']})
    assert len(chosen) == 36 and len(probes) == 72
    assert Counter(p['qid'] for p in probes) == {qid: 2 for qid in chosen}
    immutable_rows(out / 'recall_probes.jsonl', recall)
    immutable_rows(out / 'application_probes.jsonl', probes)
    return recall, probes


def application_messages(probe, protocol, recalled, reference):
    messages = copy.deepcopy(probe['messages'])
    if protocol != 'direct':
        material = recalled if protocol == 'self_recalled_material' else reference
        messages[-1]['content'] = 'Learning material for this question:\n' + material + '\n\n' + messages[-1]['content']
    return messages


def metrics(group):
    n = len(group)
    valid = [r for r in group if r['predicted'] is not None]
    return {'n': n, 'correct': sum(r['correct'] for r in group),
            'accuracy': sum(r['correct'] for r in group) / n if n else None,
            'valid_n': len(valid),
            'valid_accuracy': sum(r['correct'] for r in valid) / len(valid) if valid else None,
            'format_errors': n - len(valid), 'empty': sum(not r['response'].strip() for r in group),
            'truncated': sum(r['finish_reason'] == 'length' for r in group)}


def summarize(out, recall_probes, probes, records):
    applications = [r for r in records if r['stage'] == 'application']
    recalls = [r for r in records if r['stage'] == 'recall']
    index = {(r['mode'], r['protocol'], r['id']): r for r in applications}
    result = {}
    paired = {}
    for mode in ('base', 'adapter'):
        for protocol in PROTOCOLS:
            group = [r for r in applications if r['mode'] == mode and r['protocol'] == protocol]
            result[mode + ':' + protocol] = {**metrics(group),
                'by_question_type': {kind: metrics([r for r in group if r['question_type'] == kind])
                                     for kind in sorted({r['question_type'] for r in probes})},
                'by_relation': {kind: metrics([r for r in group if r['relation'] == kind])
                                for kind in sorted({r['relation'] for r in probes})},
                'by_evaluation_group': {kind: metrics([r for r in group if r['evaluation_group'] == kind])
                                        for kind in sorted({r['evaluation_group'] for r in probes})}}
        transitions = {}
        for other in PROTOCOLS[1:]:
            counts = Counter((index[mode, 'direct', p['id']]['correct'],
                              index[mode, other, p['id']]['correct']) for p in probes)
            transitions[other] = {'both_correct': counts[True, True], 'both_wrong': counts[False, False],
                                  'fixed': counts[False, True], 'broken': counts[True, False]}
        paired[mode] = transitions
    reference = {r['qid']: r['reference'] for r in recall_probes}
    recall_summary = {}
    for mode in ('base', 'adapter'):
        group = [r for r in recalls if r['mode'] == mode]
        recall_summary[mode] = {'n': len(group),
            'exact_full_card': sum(r['response'] == reference[r['qid']] for r in group),
            'normalized_full_card': sum(normalized(r['response']) == normalized(reference[r['qid']]) for r in group),
            'empty': sum(not r['response'].strip() for r in group),
            'truncated': sum(r['finish_reason'] == 'length' for r in group),
            'semantic_factual_review': 'pending; literal equality is not semantic correctness'}
    report = {'status': 'complete', 'metrics': result, 'paired_vs_direct': paired,
              'recall': recall_summary, 'requests': len(records), 'rules': len(recall_probes),
              'questions_per_cell': len(probes), 'formal_benchmark': False,
              'scope': 'TEXT Qwen3 SFT final five-epoch adapter; developer diagnostic on original T2T questions. '
                       'Self material contains only that mode\'s generated recall; correct material is an external-material control.',
              'finished_at': ev.now()}
    save(out / 'report.json', report)
    lines = ['# 先回忆规则再应用：五epoch适配器诊断', '',
             '36条规则，9类各4条；72道原始T2T选择题。固定哈希抽样，不按回忆成功或答题结果挑选。', '',
             '回忆使用原训练user提示，不提供答案、题目或正确规则。随后使用该模型自己生成的完整规则作为材料，'
             '独立请求回答原题。正确材料是单独的辅助对照；这些结果不计入正式盲测。基础模型使用相同权重关闭LoRA。', '',
             '| 模型 | 答题条件 | 正确/总数 | 准确率 | 格式错误 |', '|---|---|---:|---:|---:|']
    for mode in ('base', 'adapter'):
        for protocol in PROTOCOLS:
            m = result[mode + ':' + protocol]
            lines.append(f"| {mode} | {protocol} | {m['correct']}/{m['n']} | {100*m['accuracy']:.2f}% | {m['format_errors']} |")
    lines += ['', '同一规则的两题有关联，样本量小，不能据此推断全量或其他模型、PT、VISION的效果。'
              '逐字复现率仅是保守指标，语义事实需另行核验。', '',
              '完整提示见 application_inputs.jsonl；完整原始输出见 predictions.jsonl；回忆及参考原文对照见 recall_review_zh.md。']
    (out / 'report_zh.md').write_text('\n'.join(lines) + '\n')
    recall_index = {(r['mode'], r['qid']): r for r in recalls}
    lines = ['# 规则回忆与参考材料逐条对照', '']
    review = []
    for p in recall_probes:
        lines += [f"## {p['qid']} · {p['relation']}", '', '**原训练提示**', '',
                  p['messages'][-1]['content'], '', '**正确材料**', '', p['reference'], '']
        for mode in ('base', 'adapter'):
            r = recall_index[mode, p['qid']]
            lines += [f'**{mode} 回忆**', '', r['response'], '']
            review.append({'mode': mode, 'qid': p['qid'], 'relation': p['relation'],
                           'exact_card': r['response'] == p['reference'],
                           'normalized_card': normalized(r['response']) == normalized(p['reference']),
                           'factual_status': 'pending', 'notes': ''})
    (out / 'recall_review_zh.md').write_text('\n'.join(lines) + '\n')
    immutable_rows(out / 'recall_review_template.jsonl', review)
    return report


def run(out, gpu, prepare_only=False):
    out.mkdir(parents=True, exist_ok=True)
    recall_probes, probes = build(out)
    checkpoint_info = read(WORK / 'jobs/qwen3_sft_text/checkpoint_manifest.json')
    cp = Path(checkpoint_info['path'])
    trainer = read(cp / 'trainer_state.json')
    assert trainer['epoch'] == 5.0 and trainer['global_step'] == 315
    assert sha(cp / 'adapter_model.safetensors') == checkpoint_info['sha256']['adapter_model.safetensors']
    tracked = [DATA / p for p in ('audit/rule_knowledge.jsonl', 'audit/eval_labels.jsonl',
                                 'image_hashes.jsonl', 'T2T/eval.jsonl')]
    tracked.append(WORK / 'data/sft_text.jsonl')
    config = {'model': str(MODELS['qwen3']), 'adapter': ev.checkpoint(cp), 'epochs': 5,
              'selection': 'SHA256(v7-recall-then-apply:42: + qid), first four per relation',
              'rules': [p['qid'] for p in recall_probes], 'protocols': list(PROTOCOLS),
              'modes': ['base', 'adapter'], 'generation': {'temperature': 0, 'seed': 42,
              'recall_max_tokens': 768, 'answer_max_tokens': 16, 'batch_size': 8},
              'formal_benchmark': False, 'training': False, 'native_thinking': False,
              'sources': {str(p): sha(p) for p in tracked}, 'script_sha256': sha(Path(__file__))}
    if (out / 'manifest.json').exists():
        assert read(out / 'manifest.json')['config'] == config, 'Cannot resume changed config'
    else:
        save(out / 'manifest.json', {'config': config, 'created_at': ev.now()})
    if prepare_only:
        print('prepared', len(recall_probes), 'rules', len(probes), 'questions', flush=True)
        return
    raw = out / 'predictions.jsonl'
    records, notes = ev.read_rows(raw, allow_partial_tail=True) if raw.exists() else ([], [])
    if notes:
        ev.save_rows(raw, records)
    seen = {(r['mode'], r['stage'], r['protocol'], r['id']) for r in records}
    assert len(seen) == len(records), 'Duplicate responses'
    expected = 2 * (len(recall_probes) + len(probes) * 3)
    save(out / 'status.json', {'status': 'waiting_for_gpu', 'saved': len(records), 'expected': expected,
                              'pid': os.getpid(), 'gpu': gpu, 'updated_at': ev.now()})
    if len(records) < expected:
        with reserve(WORK, gpu, str(out) + ':recall_then_apply', 22000) as reservation:
            reservation.bind(os.getpid())
            os.environ.update(environment(gpu, DATA))
            import torch
            from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig
            from swift.utils import seed_everything
            seed_everything(42)
            save(out / 'status.json', {'status': 'loading_model', 'pid': os.getpid(), 'gpu': gpu,
                                      'saved': len(records), 'expected': expected, 'updated_at': ev.now()})
            engine = TransformersEngine(str(MODELS['qwen3']), adapters=[str(cp)],
                       torch_dtype=torch.bfloat16, device_map='cuda:0', max_batch_size=8, attn_impl='sdpa')
            engine.template.enable_thinking = False
            params = [p for n, p in engine.model.named_parameters() if 'lora_B' in n]
            verification = {'active_adapters': str(engine.model.active_adapters), 'lora_B_tensors': len(params),
                            'nonzero_lora_B_tensors': sum(bool(torch.count_nonzero(p)) for p in params),
                            'base_control': 'same loaded model with disable_adapter()'}
            assert verification['nonzero_lora_B_tensors'] == 252
            save(out / 'adapter_verification.json', verification)

            def generate(mode, stage, protocol, pending, max_tokens):
                for start in range(0, len(pending), 8):
                    batch = pending[start:start + 8]
                    outputs = engine.infer([InferRequest(messages=copy.deepcopy(p['messages'])) for p in batch],
                        RequestConfig(max_tokens=max_tokens, temperature=0, seed=42, return_details=True), use_tqdm=False)
                    assert len(outputs) == len(batch)
                    for p, output in zip(batch, outputs):
                        if isinstance(output, Exception):
                            raise output
                        c = output.choices[0]
                        response = c.message.content or ''
                        record = {k: p[k] for k in ('id', 'qid', 'relation')}
                        record.update(mode=mode, stage=stage, protocol=protocol, response=response,
                            finish_reason=c.finish_reason, prompt_sha256=ev.digest(p['messages']),
                            raw_response=dataclasses.asdict(output), saved_at=ev.now())
                        if stage == 'application':
                            record.update({k: p[k] for k in ('gold', 'question_type', 'evaluation_group')})
                            record.update(predicted=ev.letter(response), correct=ev.letter(response) == p['gold'])
                        with raw.open('a') as f:
                            f.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
                            f.flush()
                            os.fsync(f.fileno())
                        records.append(record)
                        seen.add((mode, stage, protocol, p['id']))
                    print(mode, stage, protocol, start + len(batch), '/', len(pending), flush=True)
                    save(out / 'status.json', {'status': 'running', 'mode': mode, 'stage': stage,
                        'protocol': protocol, 'saved': len(records), 'expected': expected, 'pid': os.getpid(),
                        'gpu': gpu, 'updated_at': ev.now()})

            reference = {p['qid']: p['reference'] for p in recall_probes}
            for mode in ('base', 'adapter'):
                with engine.model.disable_adapter() if mode == 'base' else nullcontext():
                    pending = [p for p in recall_probes if (mode, 'recall', 'training_prompt', p['id']) not in seen]
                    generate(mode, 'recall', 'training_prompt', pending, 768)
                    recalled = {r['qid']: r['response'] for r in records if r['mode'] == mode and r['stage'] == 'recall'}
                    inputs = []
                    for protocol in PROTOCOLS:
                        prepared = []
                        for p in probes:
                            z = copy.deepcopy(p)
                            z['messages'] = application_messages(p, protocol, recalled[p['qid']], reference[p['qid']])
                            z.update(mode=mode, protocol=protocol)
                            inputs.append(z)
                            if (mode, 'application', protocol, p['id']) not in seen:
                                prepared.append(z)
                        generate(mode, 'application', protocol, prepared, 16)
                    immutable_rows(out / f'application_inputs_{mode}.jsonl', inputs)
    all_inputs = [p for mode in ('base', 'adapter') for p in rows(out / f'application_inputs_{mode}.jsonl')]
    immutable_rows(out / 'application_inputs.jsonl', all_inputs)
    assert len(records) == expected
    assert len(seen) == expected
    for p in tracked:
        assert sha(p) == config['sources'][str(p)], f'Source changed during run: {p}'
    report = summarize(out, recall_probes, probes, records)
    save(out / 'status.json', {'status': 'complete_pending_factual_review', 'saved': len(records),
                              'expected': expected, 'updated_at': ev.now()})
    print(json.dumps(report['metrics'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--prepare-only', action='store_true')
    a = ap.parse_args()
    run(a.out.resolve(), a.gpu, a.prepare_only)
