#!/usr/bin/env python3
"""Independent two-family, one-epoch development pilot; frozen v7 is read-only."""
import argparse
import copy
import hashlib
import itertools
import json
import math
import os
import random
import re
import subprocess
from collections import Counter
from contextlib import nullcontext
from fractions import Fraction
from pathlib import Path

import run_v7_evaluation as ev
from build_v7_experiment_data import name
from run_v7_full_experiment import (ROOT, MODELS, environment, final_checkpoint,
                                    read, rows, save, sha, train_command)
from v7_gpu_resources import reserve

DATA = ROOT / 'rules_repaired_single_image_v7'
WORK = ROOT / 'runs/v7_full_experiment'
RELATIONS = ('set_selection_rule', 'three_way_category')
VARIANTS = ('text_explicit', 'vision_explicit', 'vision_generic_cards')


def msg(question, answer=None, mcq=False):
    result = [{'role': 'system', 'content': ev.SYSTEM if mcq else
               'Answer directly with only the requested fact or result. Do not give reasoning.'},
              {'role': 'user', 'content': question}]
    if answer is not None:
        result.append({'role': 'assistant', 'content': str(answer)})
    return result


def solve(rule, observation):
    if 'conditions' in rule:
        comparisons = {'eq': lambda a, b: a == b, 'ge': lambda a, b: a >= b,
                       'le': lambda a, b: a <= b}
        flags = [comparisons[c['operator']](observation[c['field']], c['value']) for c in rule['conditions']]
        return 'KEEP' if (all(flags) if rule['mode'] == 'all' else any(flags)) else 'OMIT'
    n, d = observation
    assert 0 <= n <= d and d > 0
    ratio = Fraction(n, d)
    lower, upper = (Fraction(str(rule['thresholds'][f])) for f in ('lower', 'upper'))
    return rule['category_labels'][0 if ratio < lower else 2 if ratio > upper else 1]


def check_independently(rule, observation):
    """Second implementation: set constraints / integer cross-products."""
    if 'conditions' in rule:
        failed = 0
        for c in rule['conditions']:
            actual, expected = observation[c['field']], c['value']
            failed += not (actual == expected if c['operator'] == 'eq' else
                           actual >= expected if c['operator'] == 'ge' else actual <= expected)
        kept = failed == 0 if rule['mode'] == 'all' else failed < len(rule['conditions'])
        return 'KEEP' if kept else 'OMIT'
    n, d = observation
    lo, hi = [Fraction(str(rule['thresholds'][f])) for f in ('lower', 'upper')]
    index = 1
    if n * lo.denominator < d * lo.numerator:
        index = 0
    elif n * hi.denominator > d * hi.numerator:
        index = 2
    return rule['category_labels'][index]


def atoms(rule):
    if 'conditions' in rule:
        result = [('combination mode (all or any)', rule['mode'])]
        result += [(f"{c['field']} condition with operator {c['operator']} (give only the required value)", str(c['value']))
                   for c in rule['conditions']]
        if len(result) < 4:
            if all(c['field'] == 'value' for c in rule['conditions']):
                result.append(('whether the interval endpoints are included (yes or no)', 'yes'))
            else:
                result.append(('number of listed conditions', str(len(rule['conditions']))))
        return result[:4]
    return [('lower threshold', str(rule['thresholds']['lower'])),
            ('upper threshold', str(rule['thresholds']['upper'])),
            ('label below the lower threshold', rule['category_labels'][0]),
            ('label above the upper threshold', rule['category_labels'][2])]


def pool(rule, split):
    if 'conditions' not in rule:
        ds = (20, 40) if split == 'train' else (60, 80)
        return [(n, d) for d in ds for n in range(d + 1)]
    if rule['conditions'][0]['field'] == 'value':
        lo, hi = [c['value'] for c in rule['conditions']]
        return [{'value': i} for i in range(max(0, lo - 3), hi + 5)]
    alternatives = {
        'train': {'category': ['bottle', 'storage box', 'folder'], 'material': ['wood', 'glass', 'fabric'],
                  'color': ['red', 'green', 'yellow']},
        'valid': {'category': ['toolbox', 'basket', 'cup'], 'material': ['paper', 'leather', 'ceramic'],
                  'color': ['blue', 'orange', 'purple']}}
    fields = [c['field'] for c in rule['conditions']]
    values = [[c['value']] + [x for x in alternatives[split][c['field']] if x != c['value']]
              for c in rule['conditions']]
    return [dict(zip(fields, values_)) for values_ in itertools.product(*values)]


def describe(rule, value):
    if isinstance(value, dict):
        return ', '.join(f'{k}: {v}' for k, v in value.items())
    return f"{rule['ratio']['numerator_field']}: {value[0]}, {rule['ratio']['denominator_field']}: {value[1]}"


def assignment(values):
    return '; '.join(f'{letter}={value}' for letter, value in zip('ABCD', values))


def choices(question, contents, correct_content, position, rng, required_wrong=None):
    wrong = list(dict.fromkeys(x for x in contents if x != correct_content))
    assert len(wrong) >= 3, (contents, correct_content)
    rng.shuffle(wrong)
    options = wrong[:3]
    if required_wrong is not None:
        assert required_wrong in wrong
        if required_wrong not in options:
            options[-1] = required_wrong
        rng.shuffle(options)
    options.insert(position, correct_content)
    prompt = question + '\n\nOptions:\n' + '\n'.join(f'{l}. {v}' for l, v in zip('ABCD', options)) + ev.SUFFIX
    return prompt, 'ABCD'[position], options


def make_examples(k, split):
    rule, identifier = json.loads(k['fact']['quote']), name(k)
    rng = random.Random(ev.digest([k['id'], split]))
    observations = pool(rule, split)
    labels = ['KEEP', 'OMIT'] if 'conditions' in rule else rule['category_labels']
    by_answer = {label: [o for o in observations if solve(rule, o) == label] for label in labels}
    assert all(by_answer.values())
    for o in observations:
        assert solve(rule, o) == check_independently(rule, o)
    examples, audit = [], []

    def add(task, question, answer, fmt='semantic', **detail):
        sid = f"{k['id']}:{split}:{task}:{len(examples)}"
        examples.append({'id': sid, 'qid': k['id'], 'relation': k['fact']['relation'],
                         'task': task, 'answer_format': fmt, 'messages': msg(question, answer, fmt == 'mcq')})
        audit.append({'id': sid, 'task': task, 'answer': str(answer), **detail})

    if split == 'train':
        for i in range(32):
            question = ('Explain', 'Describe', 'State', 'Recall')[i % 4] + f' the complete rule named {identifier}.'
            question += (' Include its exact conditions and boundaries.', ' Include how to apply it.',
                         ' Give all concrete rule values.', ' Include the meanings of its results.')[i // 8]
            add('rule_card', question, k['material']['learning']['text'])
        for i in range(8):
            focus, answer = atoms(rule)[i % 4]
            question = (f'For {identifier}, what is the {focus}?' if i < 4 else
                        f'State the {focus} used by {identifier}.')
            add('atomic_fact', question, answer)
    else:
        for focus, answer in atoms(rule):
            add('atomic_fact', f'Retrieve from memory: the {focus} for {identifier}. Return only the value.', answer)

    # Four single observations for training, eight new-count/new-wording probes.
    single_count = 8 if split == 'train' else 8
    for i in range(single_count):
        label = labels[i % len(labels)]
        # Include exact lower/upper boundary cases as well as all result classes.
        options_for_label = by_answer[label]
        o = options_for_label[(i * 7 + 1) % len(options_for_label)]
        question = (f'Apply {identifier} to one observation: {describe(rule, o)}. Return only its result.' if split == 'train' else
                    f'New case for {identifier}: {describe(rule, o)}. Which result applies?')
        if i < 4 and split == 'train':
            add('single_application', question, label, observation=o, result=label)
        else:
            # Four distinct options even for a two/three-result single observation.
            contents = labels + (['EMPTY', 'ALL'] if 'conditions' in rule else ['Outside all three categories'])
            q, gold, opts = choices(question, contents, label, i % 4, rng)
            add('single_mcq', q, gold, 'mcq', observation=o, result=label, options=opts)

    seen_vectors = set()
    accepted_indices = set(rng.sample(list(range(8, 16)), 2))
    for i in range(16):
        # Whole four-observation combinations are split without using any v7 eval.
        while True:
            observed = [rng.choice(observations) for _ in range(4)]
            signature = ev.digest(observed)
            if signature not in seen_vectors:
                seen_vectors.add(signature)
                break
        values = [solve(rule, o) for o in observed]
        correct = assignment(values)
        all_results = [assignment(v) for v in itertools.product(labels, repeat=4)]
        question = (f'Apply {identifier}. Report the result for every candidate.\n' if split == 'train' else
                    f'Using {identifier}, choose the complete result for these observations.\n')
        question += '\n'.join(f'Candidate {l}: {describe(rule, o)}.' for l, o in zip('ABCD', observed))
        diagnosis = i >= 8
        detail = {'observations': observed, 'results': values}
        if diagnosis:
            # A report-copy strategy has the same 25% baseline as guessing
            # one of four options. The submitted report is always listed.
            accepted = i in accepted_indices
            submitted = correct if accepted else rng.choice([v for v in all_results if v != correct])
            question += '\nSubmitted report: ' + submitted + '. Check the report against every observation and choose the verified complete result.'
            # Uniform option wording prevents a lone "Accept" option from
            # identifying the submitted report and creating a 50% shortcut.
            contents = ['Result: ' + v + '.' for v in all_results]
            target = 'Result: ' + correct + '.'
            detail.update(submitted=submitted, accepted=accepted)
        else:
            contents, target = ['Result: ' + v + '.' for v in all_results], 'Result: ' + correct + '.'
        required = 'Result: ' + submitted + '.' if diagnosis and not accepted else None
        q, gold, opts = choices(question, contents, target, i % 4, rng, required_wrong=required)
        add('diagnosis_mcq' if diagnosis else 'direct_mcq', q, gold, 'mcq', options=opts, **detail)
    assert len(examples) == (64 if split == 'train' else 28)
    return examples, audit


def prepare(out):
    if (out / 'manifest.json').exists():
        m = read(out / 'manifest.json')
        for f, h in m['files'].items():
            assert sha(out / f) == h, f
        return
    out.mkdir(parents=True, exist_ok=True)
    all_knowledge = [r['content'] for r in rows(DATA / 'audit/rule_knowledge.jsonl')]
    selected = []
    for relation in RELATIONS:
        selected += sorted([k for k in all_knowledge if k['fact']['relation'] == relation],
                           key=lambda k: hashlib.sha256(
                               ('grounded-pilot-v1:42:' + k['id']).encode()).hexdigest())[:4]
    # Training is built and saved before original evaluation questions or labels are read.
    training, checks, heldout, validation_checks = [], [], [], []
    train_signatures = set()
    for k in selected:
        tr, audit = make_examples(k, 'train')
        va, va_audit = make_examples(k, 'valid')
        training += tr
        checks += audit
        heldout += va
        validation_checks += va_audit
        signatures = {ev.digest(a['observations']) for a in audit if 'observations' in a}
        assert not signatures.intersection(ev.digest(a['observations']) for a in va_audit if 'observations' in a)
        train_signatures.update(ev.digest(r['messages'][:-1]) for r in tr)
    random.Random(42).shuffle(training)
    by_id = {k['id']: k for k in selected}
    for variant in VARIANTS:
        result = copy.deepcopy(training)
        if variant.startswith('vision'):
            for r in result:
                r['images'] = [str(DATA / 'images' / r['qid'] / 'learning_visual.png')]
                r['messages'][1]['content'] = '<image>Using this learning reference image, ' + r['messages'][1]['content']
                if variant == 'vision_generic_cards' and r['task'] == 'rule_card':
                    r['messages'][-1]['content'] = by_id[r['qid']]['material']['learning']['visual_text']
        ev.save_rows(out / f'{variant}.jsonl', result)
    ev.save_rows(out / 'training_solver_checks.jsonl', checks)
    ev.save_rows(out / 'validation_solver_checks.jsonl', validation_checks)
    probes = []

    def add_probe(r, condition, material=False):
        p = copy.deepcopy(r)
        p['gold'] = p['messages'].pop()['content']
        p['condition'] = condition
        if material:
            p['messages'][1]['content'] = 'Learning material:\n' + by_id[p['qid']]['material']['learning']['text'] + '\n\n' + p['messages'][1]['content']
        p['id'] += ':' + condition
        probes.append(p)

    for r in heldout:
        assert ev.digest(r['messages'][:-1]) not in train_signatures
        add_probe(r, 'new_' + r['task'])
        if r['answer_format'] == 'mcq':
            add_probe(r, 'oracle_' + r['task'], True)
    # Exact seen factual/application prompts are capacity controls, not validation.
    for k in selected:
        for task in ('atomic_fact', 'single_application', 'direct_mcq', 'diagnosis_mcq'):
            subset = [r for r in training if r['qid'] == k['id'] and r['task'] == task]
            for r in subset[:2]:
                add_probe(r, 'seen_' + task)
    labels = {r['id']: r for r in rows(DATA / 'audit/eval_labels.jsonl')}
    for direction in ('T2T', 'T2V'):
        for r in rows(DATA / direction / 'eval.jsonl'):
            if r['qid'] not in by_id:
                continue
            ready = ev.prepare_row(r, DATA, direction)
            p = {'id': direction + ':' + r['id'], 'qid': r['qid'], 'relation': by_id[r['qid']]['fact']['relation'],
                 'task': labels[r['id']]['question_type'], 'answer_format': 'mcq', 'gold': labels[r['id']]['answer'],
                 'messages': ready['messages'], 'images': ready.get('images', []), 'condition': 'original_' + direction,
                 'evaluation_group': labels[r['id']]['evaluation_group']}
            probes.append(p)
            p = copy.deepcopy(p)
            p['id'] += ':material'
            p['condition'] = 'oracle_original_' + direction
            p['messages'][1]['content'] = 'Learning material:\n' + by_id[p['qid']]['material']['learning']['text'] + '\n\n' + p['messages'][1]['content']
            probes.append(p)
    ev.save_rows(out / 'probes.jsonl', probes)
    assert len({p['id'] for p in probes}) == len(probes)
    hashes = rows(DATA / 'image_hashes.jsonl')
    for item in hashes:
        assert sha(DATA / item['path']) == item['source_sha256'] == item['repaired_sha256']
    inputs = [DATA / 'audit/rule_knowledge.jsonl', DATA / 'audit/eval_labels.jsonl', DATA / 'image_hashes.jsonl']
    inputs += [DATA / d / 'eval.jsonl' for d in ('T2T', 'T2V', 'V2T', 'V2V')]
    save(out / 'manifest.json', {'status': 'prepared', 'rules': [k['id'] for k in selected],
        'selection': 'SHA256(grounded-pilot-v1:42:ID), first 4 per family; no outcome filtering',
        'relations': dict(Counter(k['fact']['relation'] for k in selected)), 'model': str(MODELS['qwen3']),
        'epochs': 1, 'batch_size': 4, 'gradient_accumulation_steps': 1, 'learning_rate': 1e-4,
        'lora_rank': 8, 'lora_alpha': 32, 'lora_dropout': .05, 'seed': 42,
        'rows_per_training': len(training), 'tasks': dict(Counter(r['task'] for r in training)),
        'diagnosis_generation': 'v2: 25% correct submissions; submitted report always included as an option',
        'probes': dict(Counter(p['condition'] for p in probes)), 'images_verified': len(hashes),
        'image_spotcheck': 'All eight learning references visually checked against structured rules before training; thresholds and named objects match.',
        'files': {p.name: sha(p) for p in out.glob('*.jsonl')}, 'sources': {str(p): sha(p) for p in inputs},
        'script_sha256': sha(Path(__file__)),
        'limits': ['Development pilot on only eight rules, not a new independent benchmark.',
                   '64 rows/rule include intentional repeated rule cards; one epoch is over this expanded corpus.',
                   'Train/validation question wording and four-observation combinations differ; truth patterns and positive values can overlap.',
                   'Category train denominator 20/40; validation 60/80. Frozen original questions are unchanged.',
                   'Generic VISION control changes only the 32/64 full rule-card targets, keeping atomic/application supervision identical.',
                   'TEXT vs VISION differs in input modality; not an isolated target-effect comparison.',
                   'Oracle material controls are separately labeled and never counted as blind adapter performance.',
                   'SFT pilot only. Historical raw-material PT is not relabeled or changed.'], 'created_at': ev.now()})
    print('Prepared:', len(training), 'rows per variant;', len(probes), 'probes', flush=True)


def semantic_correct(text, gold):
    def normalize(s):
        return re.sub(r'\s+', ' ', s.strip().rstrip('.').lower())
    return normalize(text) == normalize(gold)


def summarize(out, variant, cp):
    folder = out / variant
    records = rows(folder / 'predictions.jsonl')
    metrics = {}
    for mode in ('base', 'trained'):
        metrics[mode] = {}
        for condition in sorted({p['condition'] for p in records}):
            group = [p for p in records if p['model_condition'] == mode and p['condition'] == condition]
            if not group:
                continue
            def counts(g):
                valid = [p for p in g if not p['format_error']]
                return {'n': len(g), 'correct': sum(p['correct'] for p in g),
                        'accuracy': sum(p['correct'] for p in g) / len(g),
                        'format_errors': sum(p['format_error'] for p in g),
                        'empty': sum(not p['response'].strip() for p in g),
                        'valid_accuracy': sum(p['correct'] for p in valid) / len(valid) if valid else None}
            metrics[mode][condition] = {**counts(group),
                'by_relation': {rel: counts([p for p in group if p['relation'] == rel]) for rel in RELATIONS},
                'by_task': {t: counts([p for p in group if p['task'] == t]) for t in sorted({p['task'] for p in group})}}
    save(folder / 'report.json', {'status': 'complete', 'variant': variant, 'checkpoint': ev.checkpoint(cp),
                                'metrics': metrics, 'finished_at': ev.now()})
    save(folder / 'state.json', {'status': 'complete', 'finished_at': ev.now()})


def evaluate(out, variant, cp, gpu, base_only=False):
    folder = out / variant
    folder.mkdir(parents=True, exist_ok=True)
    raw = folder / 'predictions.jsonl'
    saved, notes = ev.read_rows(raw, allow_partial_tail=True) if raw.exists() else ([], [])
    if notes:
        ev.save_rows(raw, saved)
    # Reuse the identical, material-controlled base run across all adapters.
    # This avoids spending GPU time re-evaluating the same base questions.
    oracle = out / 'oracle'
    if not base_only and not any(r['model_condition'] == 'base' for r in saved):
        if read(oracle / 'state.json').get('status') == 'complete':
            base_rows = rows(oracle / 'predictions.jsonl')
            assert len(base_rows) == len(rows(out / 'probes.jsonl'))
            saved = base_rows + saved
            ev.save_rows(raw, saved)
            save(folder / 'base_control_provenance.json', {'path': str(oracle / 'predictions.jsonl'),
                 'sha256': sha(oracle / 'predictions.jsonl'), 'identical_probes': True})
    seen = {(p['model_condition'], p['id']) for p in saved}
    assert len(seen) == len(saved)
    probes = rows(out / 'probes.jsonl')
    modes = ('base',) if base_only else ('base', 'trained')
    if not any((m, p['id']) not in seen for m in modes for p in probes):
        if not base_only:
            summarize(out, variant, cp)
        return
    with reserve(WORK, gpu, str(folder) + ':evaluation', 30000) as reservation:
        reservation.bind(os.getpid())
        os.environ.update(environment(gpu, DATA))
        import torch
        from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig
        from swift.utils import seed_everything
        seed_everything(42)
        engine = TransformersEngine(str(MODELS['qwen3']), adapters=[str(cp)] if cp else None,
                                    torch_dtype=torch.bfloat16, device_map='cuda:0', max_batch_size=8, attn_impl='sdpa')
        engine.template.enable_thinking = False
        save(folder / 'state.json', {'status': 'evaluating', 'pid': os.getpid(), 'checkpoint': str(cp)})
        for mode in modes:
            context = engine.model.disable_adapter() if mode == 'base' and cp else nullcontext()
            with context:
                for fmt in ('semantic', 'mcq'):
                    pending = [p for p in probes if p['answer_format'] == fmt and (mode, p['id']) not in seen]
                    for start in range(0, len(pending), 8):
                        batch = pending[start:start + 8]
                        replies = engine.infer([InferRequest(messages=copy.deepcopy(p['messages']), images=p.get('images', [])) for p in batch],
                            RequestConfig(max_tokens=48 if fmt == 'semantic' else 16, temperature=0, seed=42), use_tqdm=False)
                        assert len(replies) == len(batch)
                        for p, reply in zip(batch, replies):
                            if isinstance(reply, Exception):
                                raise reply
                            text = reply.choices[0].message.content or ''
                            parsed = ev.letter(text) if fmt == 'mcq' else None
                            r = {k: p[k] for k in ('id', 'qid', 'condition', 'relation', 'task', 'gold', 'answer_format')}
                            r.update(model_condition=mode, response=text, predicted=parsed,
                                     correct=(parsed == p['gold']) if fmt == 'mcq' else semantic_correct(text, p['gold']),
                                     format_error=fmt == 'mcq' and parsed is None, prompt_sha256=ev.digest(p['messages']),
                                     saved_at=ev.now())
                            with raw.open('a') as f:
                                f.write(json.dumps(r, ensure_ascii=False) + '\n')
                                f.flush()
                                os.fsync(f.fileno())
                            saved.append(r)
                        print(variant, mode, fmt, start + len(batch), '/', len(pending), flush=True)
    if base_only:
        save(folder / 'state.json', {'status': 'complete', 'mode': 'base_only', 'finished_at': ev.now()})
    else:
        summarize(out, variant, cp)


def run(out, variant, gpu):
    folder = out / variant
    folder.mkdir(parents=True, exist_ok=True)
    with ev.run_lock(folder / '.lock'):
        if read(folder / 'report.json').get('status') == 'complete':
            return
        manifest = read(out / 'manifest.json')
        assert all(sha(Path(p)) == h for p, h in manifest['sources'].items())
        save(folder / 'execution_manifest.json', {'script_sha256': sha(Path(__file__)),
             'training_sha256': sha(out / f'{variant}.jsonl'), 'probes_sha256': sha(out / 'probes.jsonl'),
             'created_at': ev.now()})
        cp = final_checkpoint(folder / 'checkpoint', 1)
        if not cp:
            resume_candidates = [p for p in (folder / 'checkpoint').glob('checkpoint-*') if (p / 'optimizer.pt').exists()]
            resume = max(resume_candidates, key=lambda p: int(p.name.split('-')[-1])) if resume_candidates else None
            with reserve(WORK, gpu, str(folder) + ':training', 30000) as reservation:
                cmd = train_command('qwen3', 'sft', out / f'{variant}.jsonl', folder / 'checkpoint', 1, resume=resume)
                for flag, value in [('--gradient_accumulation_steps', '1'), ('--per_device_train_batch_size', '4')]:
                    cmd[cmd.index(flag) + 1] = value
                with (folder / 'train.log').open('a') as log:
                    child = subprocess.Popen(cmd, cwd=ROOT, env=environment(gpu, DATA), stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    reservation.bind(child.pid)
                    save(folder / 'state.json', {'status': 'training', 'pid': child.pid, 'command': cmd, 'started_at': ev.now()})
                    code = child.wait()
                if code:
                    save(folder / 'state.json', {'status': 'failed', 'code': code})
                    raise RuntimeError(f'{variant}: training failed {code}')
            cp = final_checkpoint(folder / 'checkpoint', 1)
            assert cp
        trainer = read(cp / 'trainer_state.json')
        logs = rows(folder / 'checkpoint/logging.jsonl')
        losses = [float(r['loss']) for r in logs if 'loss' in r]
        assert losses and all(math.isfinite(x) for x in losses)
        assert trainer['global_step'] == 128 and abs(trainer['epoch'] - 1) < 1e-6
        from safetensors import safe_open
        with safe_open(str(cp / 'adapter_model.safetensors'), framework='pt', device='cpu') as weights:
            keys = list(weights.keys())
            assert not any(any(s in key for s in ('.visual.', '.vision_tower.', '.merger.', '.mm_projector.')) for key in keys)
            b_keys = [key for key in keys if 'lora_B' in key]
            nonzero = sum(bool(weights.get_tensor(key).count_nonzero()) for key in b_keys)
            assert b_keys and nonzero == len(b_keys)
        save(folder / 'checkpoint_manifest.json', {**ev.checkpoint(cp), 'epoch': trainer['epoch'],
             'global_step': trainer['global_step'], 'losses_finite': True, 'final_logged_loss': losses[-1],
             'lora_B_tensors': len(b_keys), 'nonzero_lora_B_tensors': nonzero, 'vision_aligner_frozen': True})
        evaluate(out, variant, cp, gpu)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--variant', choices=(*VARIANTS, 'oracle'), default='text_explicit')
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--prepare-only', action='store_true')
    a = ap.parse_args()
    out = a.out.resolve()
    prepare(out)
    if not a.prepare_only:
        if a.variant == 'oracle':
            evaluate(out, 'oracle', None, a.gpu, True)
        else:
            run(out, a.variant, a.gpu)


if __name__ == '__main__':
    main()
