#!/usr/bin/env python3
"""Independent rule-only supervision repair; immutable originals, resumable pilots."""
import argparse
import copy
import dataclasses
import itertools
import json
import math
import os
import random
import re
import subprocess
from collections import Counter
from pathlib import Path

import run_v7_evaluation as ev
from build_v7_experiment_data import name
from pilot_v7_task_then_transfer import DATA, LEDGER, CONVENTION, OP, card
from run_v7_full_experiment import ROOT, MODELS, environment, final_checkpoint, read, rows, save, sha, train_command
from v7_gpu_resources import reserve

OUT = ROOT / 'runs/v7_atomic_supervision_20261015'
PARENT = ROOT / 'runs/v7_task_then_transfer_20261015/pilot_v2/warmup/checkpoint_manifest.json'
ARMS = ('original_text', 'atomic_text', 'original_vision', 'atomic_vision', 'warmup')
SYSTEM = CONVENTION + ' Answer the requested fact or complete rule directly.'


def msg(q, a=None):
    result = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': q}]
    if a is not None:
        result.append({'role': 'assistant', 'content': a})
    return result


def aliases(k):
    r = json.loads(k['fact']['quote'])
    if 'category_labels' not in r:
        return [name(k)]
    return [separator.join(p) for p in itertools.permutations(r['category_labels']) for separator in (', ', ' / ')]


def atoms(k):
    r = json.loads(k['fact']['quote'])
    if 'conditions' in r:
        result = [(f'condition_{i}', f'the full condition numbered {i + 1}, including its field, comparison and value',
                   f"{c['field']} {OP[c['operator']]} {c['value']}") for i, c in enumerate(r['conditions'])]
        combination = f"at least {r['required_count']} of {len(r['conditions'])} conditions" if 'required_count' in r else 'all conditions' if r['mode'] == 'all' else 'at least one condition'
        return result + [('combination', 'how many of the listed conditions must hold for KEEP', combination)]
    lower, upper = r['thresholds']['lower'], r['thresholds']['upper']
    lo, mid, hi = r['category_labels']
    ratio = r['ratio']['numerator_field'] + '/' + r['ratio']['denominator_field']
    if r.get('joint_attributes'):
        ratio += ' (' + ', '.join(r['joint_attributes']) + ')'
    return [('lower', 'the lower threshold as a decimal number', str(lower)),
            ('upper', 'the upper threshold as a decimal number', str(upper)),
            ('below', 'the category strictly below the lower threshold', lo),
            ('middle', 'the category strictly between the two thresholds', mid),
            ('above', 'the category strictly above the upper threshold', hi),
            ('ratio', 'the numerator/denominator definition and any required target attributes', ratio),
            ('at_lower', 'the category exactly at the lower threshold', mid),
            ('at_upper', 'the category exactly at the upper threshold', mid)]


def validate(out):
    m = read(out / 'manifest.json')
    assert m['script_sha256'] == sha(Path(__file__))
    for p, h in m['sources'].items():
        assert sha(Path(p)) == h, p
    for p, h in m['files'].items():
        assert sha(out / p) == h, p
    for r in rows(DATA / 'image_hashes.jsonl'):
        assert sha(DATA / r['path']) == r['source_sha256'] == r['repaired_sha256'], r['path']
    assert ev.checkpoint(Path(read(PARENT)['path'])) == m['parent']
    save(out / 'source_validation.json', {'status': 'pass', 'original_images_unchanged': 3000, 'checked_at': ev.now()})


def prepare(out):
    if (out / 'manifest.json').exists():
        validate(out)
        return
    selected = rows(out / 'target_knowledge.jsonl'); assert len(selected) == 8
    selection = read(out / 'selection.json'); assert [k['id'] for k in selected] == selection['target_ids']
    assert not set(selection['excluded_prior_ids']) & set(selection['target_ids'])
    assert all(r['eligible'] for r in rows(out / 'image_review.jsonl'))
    full = ROOT / 'runs/v7_full_experiment/data'
    historical = {mod: {r['id']: r for r in rows(full / f'sft_{mod}.jsonl')} for mod in ('text', 'vision')}
    source_files = [out / 'selection.json', out / 'target_knowledge.jsonl', out / 'image_review.jsonl',
                    PARENT, DATA / 'audit/rule_knowledge.jsonl', DATA / 'audit/eval_labels.jsonl', DATA / 'image_hashes.jsonl']
    source_files += [full / f'sft_{mod}.jsonl' for mod in ('text', 'vision')]
    # Construct every training target BEFORE reading any original target test inputs.
    for arm in ARMS[:-1]:
        vision = arm.endswith('vision'); repaired = arm.startswith('atomic'); training = []
        for k in selected:
            for i in range(64):
                label = aliases(k)[i % len(aliases(k))]
                aa = atoms(k); full_answer = not repaired or i < 16
                if full_answer:
                    q = ('Explain the complete rule named {}.', 'State every condition and category mapping for {}.',
                         'Recall the exact rule for {}.', 'Describe the full decision rule for {}.')[i % 4].format(label)
                    a = card(k) if repaired else historical['vision' if vision else 'text'][k['id']]['messages'][-1]['content']
                    task = 'full_rule'
                else:
                    key, what, a = aa[(i - 16) % len(aa)]
                    q = ('For {}, state {}.', 'Recall {}: return {}.', 'Under {}, what is {}?')[(i // len(aa)) % 3].format(label, what)
                    task = key
                if vision:
                    q = '<image>Using only this original learning reference, ' + q[0].lower() + q[1:]
                r = {'id': f'{k["id"]}:{i}', 'qid': k['id'], 'channel': 'rule_knowledge', 'task': task,
                     'messages': msg(q, a)}
                if vision:
                    r['images'] = [str(DATA / 'images' / k['id'] / 'learning_visual.png')]
                training.append(r)
        random.Random(42).shuffle(training)
        assert len(training) == 512
        assert all(sum(m['content'].count('<image>') for m in r['messages']) == len(r.get('images', [])) for r in training)
        ev.save_rows(out / f'{arm}.jsonl', training)
    for mod in ('text', 'vision'):
        # PT candidates retain raw-material pretraining semantics. Not trained in this SFT-only pilot.
        pt = []
        for k in selected:
            r = {'id': k['id'], 'qid': k['id'], 'messages': [{'role': 'assistant', 'content': ('<image>' if mod == 'vision' else '') + card(k)}]}
            if mod == 'vision':
                r['images'] = [str(DATA / 'images' / k['id'] / 'learning_visual.png')]
            pt.append(r)
        ev.save_rows(out / f'pt_explicit_{mod}_candidate.jsonl', pt)
    # Same ID/row order/budget in all four training arms. Tokens differ intentionally.
    id_orders = [[r['id'] for r in rows(out / f'{arm}.jsonl')] for arm in ARMS[:-1]]
    assert all(v == id_orders[0] for v in id_orders)
    labels = {r['id']: r for r in rows(DATA / 'audit/eval_labels.jsonl')}
    targets = {k['id']: k for k in selected}; probes = []; heading = {}
    for direction in ('T2T', 'T2V'):
        source_files.append(DATA / direction / 'eval.jsonl')
        for row in rows(DATA / direction / 'eval.jsonl'):
            if row['qid'] not in targets:
                continue
            p = ev.prepare_row(row, DATA, direction); k = targets[row['qid']]
            if direction == 'T2T' and labels[row['id']]['question_type'] == 'direct':
                heading[k['id']] = p['messages'][-1]['content'].split('\n\nTest input:', 1)[0]
            base = dict(p, id='direct:' + direction + ':' + row['id'], condition='direct_' + direction[-1],
                        fmt='mcq', gold=labels[row['id']]['answer'], task=labels[row['id']]['question_type'],
                        relation=k['fact']['relation'], evaluation_group=labels[row['id']]['evaluation_group'])
            probes.append(base)
            oracle = copy.deepcopy(base); oracle.update(id='oracle:' + base['id'], condition='oracle_' + direction[-1])
            oracle['messages'][-1]['content'] = 'Correct learning reference:\n' + card(k) + '\n\n' + oracle['messages'][-1]['content']
            probes.append(oracle)
    assert len(heading) == 8
    for k in selected:
        image = str(DATA / 'images' / k['id'] / 'learning_visual.png')
        for condition, q, images in (
            ('recall_canonical', f'Recall the complete stored rule for {name(k)}, including all values, comparisons and category mappings.', []),
            ('recall_task', 'Recall the complete stored rule identified by this question heading. Do not answer a test case. Include every condition and mapping.\nHeading: ' + heading[k['id']], []),
            ('recall_image', '<image>State the complete rule shown in this original learning reference for ' + name(k) + '.', [image])):
            p = {'id': condition + ':' + k['id'], 'qid': k['id'], 'condition': condition, 'fmt': 'card',
                 'gold': card(k), 'task': 'recall', 'relation': k['fact']['relation'], 'messages': msg(q)}
            if images:
                p['images'] = images
            probes.append(p)
        for key, what, a in atoms(k):
            q = f'From memory, give only {what} for the rule identified below.\nRule heading: ' + heading[k['id']]
            for condition in ('fact_noimage', 'fact_image'):
                z = {'id': f'{condition}:{k["id"]}:{key}', 'qid': k['id'], 'condition': condition,
                     'fmt': 'fact', 'gold': a, 'fact_key': key, 'task': 'fact', 'relation': k['fact']['relation'],
                     'messages': msg(q if condition == 'fact_noimage' else '<image>Use this original learning reference.\n' + q.replace('From memory, ', ''))}
                if condition == 'fact_image':
                    z['images'] = [image]
                probes.append(z)
    ev.save_rows(out / 'probes.jsonl', probes)
    m = {'script_sha256': sha(Path(__file__)), 'target_rules': selection['target_ids'],
         'prior_rules_excluded': len(selection['excluded_prior_ids']), 'parent': ev.checkpoint(Path(read(PARENT)['path'])),
         'arms': ARMS, 'epochs': 1, 'rows_per_arm': 512, 'steps': 128, 'batch_size': 4, 'gradient_accumulation': 1,
         'lr': 1e-4, 'rank': 8, 'alpha': 32, 'dropout': .05, 'seed': 42, 'max_length': 8192,
         'freeze_vision_aligner': True, 'source_model': str(MODELS['qwen3']),
         'repair': '16 compact complete-rule rows +48 short factual rows/rule. Both modalities supervise identical facts. Input VISION contains only image+name+generic query. No target test observations, choices, labels in training.',
         'control': '64 full original-material explanations/rule with same name aliases; repeated historical answers. Same parent, IDs, row order, exposures, steps, optimizer. Prompt/answer token counts not equal.',
         'limits': ['8 independent-from-prior-development rules,2families,1seed; development evidence only',
                    'Bundled repair: explicit grounded answers+short fact QA vs historical explanations; cannot isolate these components',
                    'Original captions may contain semantic conventions not visible; two source images excluded before evaluation',
                    'VISION SFT learns image-to-text with factual targets; this is not unsupervised visual learning',
                    'PT explicit candidates prepared as material-only files but NOT trained or evaluated in this SFT pilot'],
         'evaluation': 'Same original MCQs; direct and two-stage self-recall separated; oracle external-reference controls separated; no native thinking. Blind recall gets names/headings only, no test observations or original learning image.',
         'directions': {'text': {'T2T': 'T', 'T2V': 'V'}, 'vision': {'V2T': 'T', 'V2V': 'V'}},
         'sources': {str(p): sha(p) for p in source_files},
         'files': {p.name: sha(p) for p in out.glob('*.jsonl')}, 'created_at': ev.now()}
    save(out / 'manifest.json', m)
    snapshots = out / 'source_snapshots'; snapshots.mkdir(exist_ok=True)
    import shutil
    for p in (Path(__file__), ROOT / 'scripts/pilot_v7_task_then_transfer.py', ROOT / 'scripts/run_v7_full_experiment.py', ROOT / 'scripts/run_v7_evaluation.py'):
        shutil.copy2(p, snapshots / p.name)
    validate(out)
    print('prepared', len(probes), 'initial probes/arm', flush=True)


def train(out, arm, gpu):
    parent = Path(read(PARENT)['path'])
    if arm == 'warmup':
        return parent
    folder = out / arm; folder.mkdir(exist_ok=True); cpdir = folder / 'checkpoint'
    cp = final_checkpoint(cpdir, 1)
    if cp is None:
        recover = [p for p in cpdir.glob('checkpoint-*') if all((p / f).exists() for f in ('optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'trainer_state.json', 'adapter_model.safetensors'))]
        resume = max(recover, key=lambda p: read(p / 'trainer_state.json')['global_step']) if recover else None
        if cpdir.exists() and not resume:
            cpdir.rename(folder / ('interrupted_' + str(int(__import__('time').time()))))
        cmd = train_command('qwen3', 'sft', out / (arm + '.jsonl'), cpdir, 1, resume=resume)
        for flag, value in (('--per_device_train_batch_size', '4'), ('--gradient_accumulation_steps', '1'), ('--save_total_limit', '3')):
            cmd[cmd.index(flag) + 1] = value
        cmd += ['--enable_channel_loss', 'true', '--save_strategy', 'steps', '--save_steps', '32']
        if not resume:
            cmd += ['--adapters', str(parent)]
        save(folder / 'train_state.json', {'status': 'queued', 'command': cmd})
        with reserve(LEDGER, gpu, str(folder) + ':train', 29000 if arm.endswith('vision') else 22000) as res:
            with (folder / 'train.log').open('a') as log:
                proc = subprocess.Popen(cmd, cwd=ROOT, env=environment(gpu, DATA), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                res.bind(proc.pid); save(folder / 'train_state.json', {'status': 'running', 'pid': proc.pid, 'command': cmd, 'started_at': ev.now()})
                code = proc.wait()
            assert code == 0, f'{arm} train failed ({code}); see train.log'
        cp = final_checkpoint(cpdir, 1); assert cp
    st = read(cp / 'trainer_state.json'); args = read(cp / 'args.json')
    assert st['global_step'] == 128 and abs(st['epoch'] - 1) < 1e-6
    assert args['adapters'] == [str(parent)] and args['dataset'] == [str(out / (arm + '.jsonl'))]
    losses = [r['loss'] for r in rows(cpdir / 'logging.jsonl') if 'loss' in r]
    assert losses and all(math.isfinite(x) for x in losses)
    from safetensors import safe_open
    with safe_open(str(cp / 'adapter_model.safetensors'), framework='pt', device='cpu') as f:
        keys = list(f.keys()); bs = [k for k in keys if 'lora_B' in k]
        assert len(bs) == 252 and all(bool(f.get_tensor(k).count_nonzero()) for k in bs)
        assert not any('.visual.' in k or '.merger.' in k for k in keys)
    save(folder / 'checkpoint_manifest.json', {**ev.checkpoint(cp), 'parent': ev.checkpoint(parent), 'epoch': 1,
                                             'global_step': 128, 'final_loss': losses[-1], 'nonzero_lora_B': len(bs),
                                             'selection': 'fixed final epoch1; no score selection'})
    save(folder / 'train_state.json', {'status': 'complete', 'checkpoint': str(cp)})
    return cp


def norm(s):
    return re.sub(r'\s+', ' ', s.strip().strip('`').rstrip('.')).casefold()


def short_equal(text, gold, key):
    if key in ('lower', 'upper'):
        try:
            return math.isclose(float(text.strip().rstrip('.')), float(gold), abs_tol=1e-8)
        except ValueError:
            return False
    return norm(text) == norm(gold)


def evaluate(out, arm, gpu, cp):
    folder = out / arm; folder.mkdir(exist_ok=True)
    config = {'adapter': ev.checkpoint(cp), 'probes_sha256': sha(out / 'probes.jsonl'), 'seed': 42,
              'temperature': 0, 'max_tokens': {'mcq': 16, 'fact': 96, 'card': 384}, 'batch_size': 4, 'thinking': False}
    if (folder / 'eval_manifest.json').exists():
        assert read(folder / 'eval_manifest.json') == config
    else:
        save(folder / 'eval_manifest.json', config)
    raw = folder / 'predictions.jsonl'
    records, notes = ev.read_rows(raw, allow_partial_tail=True) if raw.exists() else ([], [])
    if notes:
        ev.save_rows(raw, records)
    assert len(records) == len({r['id'] for r in records})
    probes = rows(out / 'probes.jsonl'); seen = {r['id'] for r in records}

    def two_stage():
        recalled = {r['qid']: r for r in records if r['condition'] == 'recall_task'}
        assert len(recalled) == 8
        result = []
        for p in probes:
            if not p['condition'].startswith('direct_'):
                continue
            z = copy.deepcopy(p); z.update(id='self:' + p['id'], condition='self_' + p['condition'][-1])
            z['messages'][-1]['content'] = 'Your own prior rule recall (uncorrected):\n' + recalled[p['qid']]['response'] + '\n\n' + z['messages'][-1]['content']
            z['recall_response_sha256'] = ev.digest(recalled[p['qid']]['response'])
            result.append(z)
        return result

    def infer(engine, group):
        from swift.infer_engine import InferRequest, RequestConfig
        for start in range(0, len(group), 4):
            batch = group[start:start + 4]
            replies = engine.infer([InferRequest(messages=copy.deepcopy(p['messages']), images=p.get('images', [])) for p in batch],
                                   RequestConfig(max_tokens=config['max_tokens'][batch[0]['fmt']], temperature=0, seed=42, return_details=True), use_tqdm=False)
            assert len(replies) == len(batch)
            for p, reply in zip(batch, replies):
                if isinstance(reply, Exception):
                    raise reply
                c = reply.choices[0]; text = c.message.content or ''; predicted = ev.letter(text) if p['fmt'] == 'mcq' else None
                r = {key: p[key] for key in ('id', 'qid', 'condition', 'fmt', 'gold', 'task', 'relation')}
                for key in ('fact_key', 'evaluation_group', 'recall_response_sha256'):
                    if key in p:
                        r[key] = p[key]
                r.update(response=text, predicted=predicted, correct=predicted == p['gold'] if p['fmt'] == 'mcq' else
                         short_equal(text, p['gold'], p.get('fact_key')) if p['fmt'] == 'fact' else text.strip() == p['gold'],
                         format_error=p['fmt'] == 'mcq' and predicted is None, finish_reason=c.finish_reason,
                         input_sha256=ev.digest({'messages': p['messages'], 'images': p.get('images', [])}),
                         raw_response=dataclasses.asdict(reply), saved_at=ev.now())
                records.append(r); seen.add(r['id'])
                with raw.open('a') as f:
                    f.write(json.dumps(r, ensure_ascii=False, default=str) + '\n'); f.flush(); os.fsync(f.fileno())
            save(folder / 'state.json', {'status': 'running', 'saved': len(records), 'stage': batch[0]['condition'], 'pid': os.getpid()})
            print(arm, batch[0]['condition'], start + len(batch), '/', len(group), flush=True)

    initial_pending = [p for p in probes if p['id'] not in seen]
    self_complete = sum(r['condition'].startswith('self_') for r in records) == 32
    if initial_pending or not self_complete:
        save(folder / 'state.json', {'status': 'queued_eval', 'saved': len(records)})
        with reserve(LEDGER, gpu, str(folder) + ':eval', 24000) as res:
            res.bind(os.getpid()); os.environ.update(environment(gpu, DATA))
            import torch
            from swift.infer_engine import TransformersEngine
            from swift.utils import seed_everything
            seed_everything(42)
            engine = TransformersEngine(str(MODELS['qwen3']), adapters=[str(cp)], torch_dtype=torch.bfloat16,
                                        device_map='cuda:0', max_batch_size=4, attn_impl='sdpa')
            engine.template.enable_thinking = False
            for fmt in ('card', 'fact', 'mcq'):
                infer(engine, [p for p in initial_pending if p['fmt'] == fmt])
            extra = two_stage(); ev.save_rows(folder / 'self_probes.jsonl', extra)
            infer(engine, [p for p in extra if p['id'] not in seen])
    else:
        assert rows(folder / 'self_probes.jsonl') == two_stage()
    allprobes = probes + rows(folder / 'self_probes.jsonl'); lookup = {p['id']: p for p in allprobes}
    assert len(records) == len(seen) == len(lookup)
    for r in records:
        p = lookup[r['id']]
        assert r['input_sha256'] == ev.digest({'messages': p['messages'], 'images': p.get('images', [])})
        assert r['response'] == (r['raw_response']['choices'][0]['message']['content'] or '')
    metrics = {}
    for condition in sorted({r['condition'] for r in records}):
        rr = [r for r in records if r['condition'] == condition]
        metrics[condition] = {'n': len(rr), 'correct': sum(r['correct'] for r in rr),
                              'format_errors': sum(r['format_error'] for r in rr),
                              'truncated': sum(r['finish_reason'] == 'length' for r in rr),
                              'empty': sum(not r['response'].strip() for r in rr)}
    save(folder / 'report.json', {'status': 'complete', 'arm': arm, 'metrics': metrics,
                                 'note': 'Fact/card correct counts use conservative matching, not semantic accuracy; audit nonmatching content separately. Oracle has external reference. Self protocol uses uncorrected generated recall only.', 'finished_at': ev.now()})
    save(folder / 'state.json', {'status': 'complete', 'saved': len(records)})
    assert ev.checkpoint(cp) == config['adapter']


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--prepare', action='store_true'); ap.add_argument('--arms', nargs='+', choices=ARMS)
    ap.add_argument('--gpu', default='0'); a = ap.parse_args(); out = a.out.resolve()
    if a.prepare:
        prepare(out)
    else:
        validate(out)
        for arm in a.arms or []:
            with ev.run_lock(out / (arm + '.lock')):
                cp = train(out, arm, a.gpu); evaluate(out, arm, a.gpu, cp)
        validate(out)


if __name__ == '__main__':
    main()
