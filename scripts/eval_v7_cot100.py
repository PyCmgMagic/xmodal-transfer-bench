"""Frozen 100-question-per-modality CoT diagnostic; durable matched arms."""
import argparse
import copy
import dataclasses
import hashlib
import json
import os
import random
import subprocess
import traceback
from collections import Counter
from pathlib import Path

import run_v7_evaluation as ev
from analyze_v7_atomic_cot_diagnostic import final_letter
from build_v7_experiment_data import name
from run_v7_full_experiment import ROOT, MODELS, rows, read, save, sha, environment
from run_v7_fullparam_ep10 import dense_identity
from v7_gpu_resources import reserve, can_admit

OUT = ROOT / 'runs/v7_cot100_20261021'
SOURCE = ROOT / 'runs/v7_process_scale64_20261021'
DATA = ROOT / 'rules_repaired_single_image_v7'
ADAPTER = SOURCE / 'seed42/definitions/checkpoint/checkpoint-3840'
FAMILIES = ('disjunctive_filter', 'interval_filter', 'conjunctive_filter', 'include_with_exclusion')
SYSTEM = ('Solve the multiple-choice question using the learned rule. Briefly state the rule, '
          'check each observation, and select the matching option. Keep the explanation within '
          '200 words. End with FINAL: followed by exactly one uppercase letter A, B, C, or D.')
SUFFIX = ('\n\nGive a concise explanation (at most 200 words). End with FINAL: followed by '
          'exactly one uppercase letter A, B, C, or D.')
RECALL_SYSTEM = ('Recall the learned named rule. State its exact conditions and logical connective '
                 'in at most 120 words. Do not invent an example or solve a test case. '
                 'If you do not know the rule, say so briefly.')
LEDGER = ROOT / 'runs/v7_full_experiment'


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'manifest.json').exists():
        return validate()
    source = read(SOURCE / 'manifest.json')
    assert read(SOURCE / 'seed42/definitions/state.json')['status'] == 'complete'
    trained_ids = {r['qid'] for r in rows(SOURCE / 'definitions_training.jsonl')}
    assert trained_ids == set(source['ids']) and len(trained_ids) == 64
    knowledge = rows(SOURCE / 'knowledge.jsonl')
    selected = []
    for family, count in zip(FAMILIES, (13, 13, 12, 12)):
        eligible = [k for k in knowledge if json.loads(k['fact']['quote'])['selection_topology'] == family]
        eligible.sort(key=lambda k: hashlib.sha256(('cot100-frozen-20261021:' + k['id']).encode()).hexdigest())
        selected += eligible[:count]
    ids = {k['id'] for k in selected}
    assert len(ids) == 50 and ids <= trained_ids
    original = [p for p in rows(SOURCE / 'probes.jsonl')
                if p['condition'] == 'original' and p['protocol'] == 'short' and p['qid'] in ids]
    assert len(original) == 200
    labels = {p['id']: p for p in rows(DATA / 'audit/eval_labels.jsonl')}
    questions = []
    for p in original:
        z = copy.deepcopy(p)
        assert z['messages'][-1]['content'].endswith(ev.SUFFIX)
        z['messages'][-1]['content'] = z['messages'][-1]['content'][:-len(ev.SUFFIX)] + SUFFIX
        z['messages'][0]['content'] = SYSTEM
        z.update(id='direct_cot:' + p['id'], protocol='direct_cot', modality=p['input_mod'])
        assert z['gold'] == labels[z['source_id']]['answer']
        questions.append(z)
    for mod in ('T', 'V'):
        pp = [p for p in questions if p['modality'] == mod]
        assert len(pp) == len({p['source_id'] for p in pp}) == 100
        assert Counter(p['task'] for p in pp) == {'direct': 50, 'diagnosis': 50}
    assert {p['source_id'] for p in questions if p['modality'] == 'T'} == {
        p['source_id'] for p in questions if p['modality'] == 'V'}
    recalls = []
    for k in sorted(selected, key=lambda k: k['id']):
        recalls.append(dict(id='recall:' + k['id'], qid=k['id'], protocol='recall',
            messages=[dict(role='system', content=RECALL_SYSTEM),
                      dict(role='user', content='Recall the complete stored rule for ' + name(k) +
                           '. Include all exact conditions, comparisons, values and output conventions.')]))
    ev.save_rows(OUT / 'knowledge.jsonl', selected)
    ev.save_rows(OUT / 'questions.jsonl', questions)
    ev.save_rows(OUT / 'recall_probes.jsonl', recalls)
    images = {p: sha(Path(p)) for z in questions for p in z.get('images', [])}
    source_images = {str(DATA / r['path']): r for r in rows(DATA / 'image_hashes.jsonl')}
    for p, h in images.items():
        assert h == source_images[p]['source_sha256'] == source_images[p]['repaired_sha256']
    paths = [SOURCE / 'manifest.json', SOURCE / 'knowledge.jsonl', SOURCE / 'probes.jsonl',
             SOURCE / 'definitions_training.jsonl', SOURCE / 'seed42/definitions/checkpoint_manifest.json',
             DATA / 'audit/eval_labels.jsonl', Path(__file__),
             ROOT / 'scripts/analyze_v7_atomic_cot_diagnostic.py']
    model = dense_identity(MODELS['qwen3'])
    assert model == source['base']
    config = dict(created_at=ev.now(), model=model, adapter=ev.checkpoint(ADAPTER), ids=sorted(ids),
        selection='Fixed SHA256 order, 13/13/12/12 rules across four families; no evaluation-score selection.',
        sources={str(p): sha(p) for p in paths}, files={p.name: sha(p) for p in OUT.glob('*.jsonl')}, images=images,
        n_rules=50, questions_per_modality=100, requests_per_arm=450, total_requests=900,
        generation=dict(batch_size=16, temperature=0, seed=42, recall_tokens=768, cot_tokens=3072,
                        native_thinking_cot=False, native_thinking_recall=False, explicit_reasoning_cot=True),
        training=source['training'], checkpoint_selection='seed42 definitions, fixed final epoch3, step3840; clean-base initialization.',
        protocols=['direct_cot', 'self_recall_cot'],
        scoring='Unique terminal FINAL: A/B/C/D, or a bare letter; unclosed think markup flagged separately. Invalid finals count as incorrect.',
        limits=['Different checkpoint and rule scope from the earlier eight-rule atomic pilot; no direct sample-size-only comparison.',
                '50 independent rule clusters; 100 questions per modality, two tasks per rule, matched modalities.',
                'Four set-selection families only; this is not the entire 1000-rule benchmark.',
                'Self recall includes only the same arm own uncorrected generated text, and is an auxiliary text-assisted condition.',
                'Explicit CoT uses the Instruct model ordinary assistant template, without a forced <think> prefix; matched across current arms.',
                'Generation budget/instructions differ from earlier truncated diagnostic; matched across current arms.',
                'Model explanations are generated text, not verified internal computation.'])
    save(OUT / 'manifest.json', config)
    return validate()


def validate():
    m = read(OUT / 'manifest.json')
    for p, h in m['sources'].items():
        assert sha(Path(p)) == h, p
    for p, h in m['files'].items():
        assert sha(OUT / p) == h, p
    for p, h in m['images'].items():
        assert sha(Path(p)) == h, p
    assert ev.checkpoint(ADAPTER) == m['adapter']
    return m


def applications(records):
    memories = {r['qid']: r for r in records if r['protocol'] == 'recall'}
    assert len(memories) == 50
    result = []
    for p in rows(OUT / 'questions.jsonl'):
        z = copy.deepcopy(p)
        z.update(id=p['id'].replace('direct_cot:', 'self_recall_cot:', 1), protocol='self_recall_cot')
        text = memories[z['qid']]['response']
        z['messages'][-1]['content'] = 'Your own prior rule recall (uncorrected):\n' + text + '\n\n' + z['messages'][-1]['content']
        z['recall_response_sha256'] = ev.digest(text)
        z['recall_truncated'] = memories[z['qid']]['finish_reason'] == 'length'
        result.append(z)
    return result


def summarize(arm):
    folder = OUT / arm
    raw = folder / 'predictions.jsonl'
    rr, _ = ev.read_rows(raw, allow_partial_tail=True) if raw.exists() else ([], [])
    metrics = {}
    for protocol in ('direct_cot', 'self_recall_cot'):
        for mod in ('T', 'V'):
            pp = [r for r in rr if r['protocol'] == protocol and r.get('modality') == mod]
            n = len(pp); correct = sum(r['correct'] for r in pp); valid = sum(r['predicted'] is not None for r in pp)
            metrics[protocol + '_' + mod] = dict(n=n, expected=100, correct=correct,
                accuracy=correct / n if n else None, valid=valid, valid_accuracy=correct / valid if valid else None,
                invalid_final=n-valid, truncated=sum(r['finish_reason'] == 'length' for r in pp),
                empty=sum(not r['response'].strip() for r in pp),
                malformed_think_markup=sum(r['response'].count('<think>') != r['response'].count('</think>') for r in pp),
                by_task={t: dict(n=sum(r['task'] == t for r in pp), correct=sum(r['correct'] for r in pp if r['task'] == t))
                         for t in ('direct', 'diagnosis')},
                by_family={f: dict(n=sum(r['family'] == f for r in pp), correct=sum(r['correct'] for r in pp if r['family'] == f))
                           for f in FAMILIES})
    recall = [r for r in rr if r['protocol'] == 'recall']
    report = dict(status='complete' if len(rr) == 450 else 'partial', saved=len(rr), expected=450,
        metrics=metrics, recall=dict(n=len(recall), truncated=sum(r['finish_reason'] == 'length' for r in recall)),
        raw_sha256=sha(raw) if raw.exists() else None)
    save(folder / 'report.json', report)
    return report


def evaluate(arm, gpu):
    m = validate(); folder = OUT / arm; folder.mkdir(exist_ok=True)
    raw = folder / 'predictions.jsonl'
    records, notes = ev.read_rows(raw, allow_partial_tail=True) if raw.exists() else ([], [])
    if notes:
        # Preserve incomplete bytes before repairing an interrupted JSONL tail.
        import shutil
        shutil.copy2(raw, folder / ('predictions_interrupted_' + str(os.getpid()) + '.jsonl'))
        ev.save_rows(raw, records)
    seen = {r['id'] for r in records}; assert len(seen) == len(records)
    if len(records) < 450:
        save(folder / 'state.json', dict(status='waiting_for_gpu', gpu=gpu, pid=os.getpid(), saved=len(records), total=450))
        with reserve(LEDGER, gpu, str(folder) + ':cot100', 26000) as reservation:
            reservation.bind(os.getpid())
            os.environ.update(environment(gpu, DATA))
            import torch
            from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig
            from swift.utils import seed_everything
            torch.set_num_threads(4); seed_everything(42)
            save(folder / 'state.json', dict(status='loading', gpu=gpu, pid=os.getpid(), saved=len(records), total=450))
            engine = TransformersEngine(str(MODELS['qwen3']), adapters=[str(ADAPTER)] if arm == 'trained' else [],
                torch_dtype=torch.bfloat16, device_map='cuda:0', max_batch_size=16, attn_impl='sdpa')
            engine.model.eval()
            loaded = bool(getattr(engine.model, 'peft_config', None))
            assert loaded == (arm == 'trained')
            save(folder / 'adapter_verification.json', dict(adapter_loaded=loaded,
                active_adapters=str(getattr(engine.model, 'active_adapters', None)),
                nonzero_lora_B=sum(bool(torch.count_nonzero(p)) for n, p in engine.model.named_parameters() if 'lora_B' in n)))
            for protocol in ('direct_cot', 'recall', 'self_recall_cot'):
                probes = rows(OUT / 'questions.jsonl') if protocol == 'direct_cot' else (
                    rows(OUT / 'recall_probes.jsonl') if protocol == 'recall' else applications(records))
                if protocol == 'self_recall_cot':
                    ev.save_rows(folder / 'application_probes.jsonl', probes)
                # Instruct is not a Thinking checkpoint; explain through the explicit user instruction.
                engine.template.enable_thinking = False
                pending = [p for p in probes if p['id'] not in seen]
                save(folder / 'state.json', dict(status='running', gpu=gpu, pid=os.getpid(), protocol=protocol,
                    saved=len(records), total=450, updated_at=ev.now()))
                for start in range(0, len(pending), 16):
                    batch = pending[start:start+16]
                    replies = engine.infer([InferRequest(messages=copy.deepcopy(p['messages']), images=p.get('images', [])) for p in batch],
                        RequestConfig(max_tokens=768 if protocol == 'recall' else 3072, temperature=0, seed=42, return_details=True), use_tqdm=False)
                    assert len(replies) == len(batch)
                    for p, reply in zip(batch, replies):
                        if isinstance(reply, Exception):
                            raise reply
                        choice = reply.choices[0]; answer = choice.message.content or ''
                        reasoning = getattr(choice.message, 'reasoning_content', None) or ''
                        response = (f'<think>{reasoning}</think>\n' if reasoning else '') + answer
                        pred = final_letter(response) if protocol != 'recall' else None
                        r = {k: p[k] for k in ('id', 'qid', 'source_id', 'protocol', 'modality', 'gold', 'task', 'family', 'evaluation_group',
                                                'recall_response_sha256', 'recall_truncated') if k in p}
                        r.update(arm=arm, response=response, answer_content=answer, reasoning_content=reasoning,
                            predicted=pred, correct=pred == p['gold'] if protocol != 'recall' else None,
                            finish_reason=choice.finish_reason, batch_size=len(batch),
                            input_sha256=ev.digest(dict(messages=p['messages'], images=p.get('images', []))),
                            raw_response=dataclasses.asdict(reply))
                        with raw.open('a') as h:
                            h.write(json.dumps(r, ensure_ascii=False, default=str) + '\n'); h.flush(); os.fsync(h.fileno())
                        records.append(r); seen.add(r['id'])
                    save(folder / 'state.json', dict(status='running', gpu=gpu, pid=os.getpid(), protocol=protocol,
                        saved=len(records), total=450, updated_at=ev.now()))
                    summarize(arm)
                    print(arm, protocol, len(records), '/450', flush=True)
    lookup = {p['id']: p for p in rows(OUT / 'questions.jsonl') + rows(OUT / 'recall_probes.jsonl') + applications(records)}
    assert len(records) == len(seen) == len(lookup) == 450
    for r in records:
        p = lookup[r['id']]
        assert r['input_sha256'] == ev.digest(dict(messages=p['messages'], images=p.get('images', [])))
        if r['protocol'] != 'recall':
            assert r['gold'] == p['gold'] and r['predicted'] == final_letter(r['response'])
        raw_message = r['raw_response']['choices'][0]['message']
        assert r['answer_content'] == (raw_message['content'] or '')
        assert r['reasoning_content'] == (raw_message.get('reasoning_content') or '')
    validate()
    assert dense_identity(MODELS['qwen3']) == m['model']
    summarize(arm)
    save(folder / 'state.json', dict(status='complete', saved=450, total=450, ended_at=ev.now()))


def report():
    reports = {arm: summarize(arm) for arm in ('base', 'trained')}
    complete = all(r['status'] == 'complete' for r in reports.values())
    save(OUT / 'summary.json', dict(status='complete' if complete else 'partial', arms=reports))
    lines = ['# 100题/模态 CoT诊断', '',
        'Qwen3-VL-4B，50条已学规则、每规则direct/diagnosis各1题，每模态100题。基座不挂适配器；训练组为64规则定义SFT、seed42、最终3epoch检查点，和此前8规则atomic模型不同。', '',
        '每个条件固定100题；分母包含格式错误与截断。回忆后作答使用同一模型自己生成、未经修正的规则文字，单列为辅助条件。', '',
        '| 模型 | 条件 | 文字题 | 图片题 | 文字截断/格式错误 | 图片截断/格式错误 |',
        '|---|---|---:|---:|---:|---:|']
    for arm in ('base', 'trained'):
        for protocol in ('direct_cot', 'self_recall_cot'):
            t, v = [reports[arm]['metrics'][protocol + '_' + mod] for mod in ('T', 'V')]
            def value(r):
                return f"{r['correct']}/100（{r['accuracy']:.1%}）" if r['n'] == 100 else f"未完成 {r['n']}/100"
            lines.append(f"| {arm} | {protocol} | {value(t)} | {value(v)} | {t['truncated']}/{t['invalid_final']} | {v['truncated']}/{v['invalid_final']} |")
    lines += ['', '生成：temperature=0，seed42，batch16，CoT最多3072 tokens，要求200词以内解释；回忆最多768 tokens。采用Instruct正常对话模板，显式要求解释，不强制插入<think>。原图、正式题和标签不变。早期启动均没有产生已评分回答，完整结果统一使用batch16和修正后的模板。', '',
              '100题来自50条规则，题目并非完全独立；不能视为100条独立规则。没有视觉学习模型，不能报告V2T/V2V。不得与之前不同检查点/预算的16题结果直接归因比较。', '',
              '以下为自动抽取的原始回答样例，尚需人工核查规则事实、视觉读取及应用步骤；不根据生成解释推断内部计算。']
    examples = []
    for arm in ('base', 'trained'):
        path = OUT / arm / 'predictions.jsonl'
        rr = rows(path) if path.exists() else []
        for protocol in ('direct_cot', 'self_recall_cot'):
            for mod in ('T', 'V'):
                for category in ('wrong_final', 'invalid_final', 'correct'):
                    pool = [r for r in rr if r['protocol'] == protocol and r.get('modality') == mod and
                            ('correct' if r['correct'] else 'invalid_final' if r['predicted'] is None else 'wrong_final') == category]
                    for r in sorted(pool, key=lambda r: r['id'])[:2]:
                        examples.append(dict(category=category, **r))
    ev.save_rows(OUT / 'explanation_samples.jsonl', examples)
    if complete:
        allrr = {a: rows(OUT / a / 'predictions.jsonl') for a in ('base', 'trained')}
        delta = {}
        for protocol in ('direct_cot', 'self_recall_cot'):
            for mod in ('T', 'V'):
                pairs = [{r['id']: r for r in allrr[a] if r['protocol'] == protocol and r.get('modality') == mod} for a in ('base', 'trained')]
                assert set(pairs[0]) == set(pairs[1]) and len(pairs[0]) == 100
                by_rule = {}
                for key in pairs[0]:
                    r0, r1 = pairs[0][key], pairs[1][key]
                    by_rule.setdefault(r0['qid'], []).append(int(r1['correct'])-int(r0['correct']))
                values = [sum(x)/len(x) for _, x in sorted(by_rule.items())]
                assert len(values) == 50
                rng = random.Random(42)
                samples = sorted(sum(rng.choice(values) for _ in values)/len(values) for _ in range(10000))
                delta[protocol + '_' + mod] = dict(trained_minus_base=sum(values)/len(values),
                    cluster_bootstrap_95_interval=[samples[249], samples[9749]], rule_clusters=50)
        save(OUT / 'paired_rule_differences.json', delta)
        lines += ['', '训练后减基座差值（按规则配对bootstrap，百分点）：']
        for key, v in delta.items():
            lines.append(f"- {key}: {100*v['trained_minus_base']:+.1f}；95%区间 [{100*v['cluster_bootstrap_95_interval'][0]:+.1f}, {100*v['cluster_bootstrap_95_interval'][1]:+.1f}]。")
    (OUT / 'report_zh.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(status='complete' if complete else 'partial', arms=reports), ensure_ascii=False), flush=True)


def launch():
    prepare()
    children = []
    for arm, gpu in (('trained', '0'), ('base', '1')):
        folder = OUT / arm; folder.mkdir(exist_ok=True)
        with (folder / 'worker.log').open('a') as log:
            p = subprocess.Popen([str(ROOT / '.venv/bin/python'), '-u', str(Path(__file__).resolve()),
                                  '--stage', 'evaluate', '--arm', arm, '--gpu', gpu],
                                 cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        children.append((arm, p))
    save(OUT / 'launcher_state.json', dict(status='running', pid=os.getpid(),
        children={a: p.pid for a, p in children}, started_at=ev.now()))
    codes = {a: p.wait() for a, p in children}
    report()
    save(OUT / 'launcher_state.json', dict(status='complete' if not any(codes.values()) else 'failed',
        returncodes=codes, ended_at=ev.now()))
    if any(codes.values()):
        raise RuntimeError(codes)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage', choices=['prepare', 'evaluate', 'report', 'launch'], required=True)
    ap.add_argument('--arm', choices=['base', 'trained'])
    ap.add_argument('--gpu', choices=['0', '1'])
    args = ap.parse_args()
    if args.stage == 'prepare':
        m = prepare(); print(json.dumps({k: m[k] for k in ('n_rules', 'questions_per_modality', 'total_requests')}))
    elif args.stage == 'evaluate':
        assert args.arm and args.gpu
        try:
            with ev.run_lock(OUT / (args.arm + '.lock')):
                evaluate(args.arm, args.gpu)
        except Exception:
            save(OUT / args.arm / 'state.json', dict(status='failed', pid=os.getpid(), error=traceback.format_exc(), ended_at=ev.now()))
            raise
    elif args.stage == 'report':
        report()
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        with ev.run_lock(OUT / 'launcher.lock'):
            launch()
