#!/usr/bin/env python3
"""Separate visual observation errors and image-derived text replay effects."""
import argparse
import copy
import dataclasses
import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

import pilot_v7_counterfactual_names as previous
import run_v7_evaluation as ev
from build_v7_experiment_data import name
from pilot_v7_atomic_supervision import msg
from run_v7_full_experiment import ROOT, MODELS, rows, read, save, sha, environment, train_command, final_checkpoint
from summarize_v7_counterfactual_names import metric
from v7_gpu_resources import reserve

OUT = ROOT / 'runs/v7_visual_replay_20261016'
SOURCE = previous.OUT
ARMS = ('vision_control', 'text_replay')


def validate(out):
    m = read(out / 'manifest.json')
    assert m['script_sha256'] == sha(Path(__file__))
    for path, digest in m['sources'].items():
        assert sha(Path(path)) == digest, path
    for path, digest in m['files'].items():
        assert sha(out / path) == digest, path
    for image in rows(previous.DATA / 'image_hashes.jsonl'):
        assert sha(previous.DATA / image['path']) == image['source_sha256'] == image['repaired_sha256']
    assert ev.checkpoint(Path(m['parent']['path'])) == m['parent']


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        validate(out)
        return
    previous.validate(SOURCE)
    assert read(SOURCE / 'completion_audit.json')['status'] == 'complete'
    prior = rows(SOURCE / 'vision/predictions.jsonl')
    teacher = {r['qid']: r for r in prior if r['condition'] == 'recall_image'}
    reviews = {r['qid']: r for r in rows(SOURCE / 'semantic_reviews.jsonl')
               if r['arm'] == 'vision' and r['condition'] == 'recall_image'}
    assert len(teacher) == len(reviews) == 8
    assert all(r['core_facts_correct'] and r['whole_response_consistent'] for r in reviews.values())
    assert all(reviews[q]['response'] == r['response'] for q, r in teacher.items())
    original = rows(SOURCE / 'vision.jsonl')
    text_prompts = {r['id']: r for r in rows(SOURCE / 'text.jsonl')}
    # Use only saved vision outputs from LEARNING images. Never use tests to build training answers.
    for arm in ARMS:
        train = []
        for r in original:
            z = copy.deepcopy(r if arm == 'vision_control' else text_prompts[r['id']])
            z['messages'][-1]['content'] = teacher[r['qid']]['response']
            z['teacher_response_sha256'] = ev.digest(teacher[r['qid']]['response'])
            z['teacher_source_id'] = teacher[r['qid']]['id']
            assert ('images' in z) == (arm == 'vision_control')
            assert 'Options:' not in str(z) and 'Test input:' not in str(z)
            train.append(z)
        assert len(train) == 512 and len({r['qid'] for r in train}) == 8
        ev.save_rows(out / (arm + '.jsonl'), train)
    assert [r['id'] for r in rows(out / 'vision_control.jsonl')] == [r['id'] for r in rows(out / 'text_replay.jsonl')]
    # Only after both training datasets are fixed, construct held-constant test diagnostics.
    blocks = rows(SOURCE / 'blocks.jsonl')
    scenes = {b['source_id']: b for b in blocks}
    extract = []
    for sid, b in scenes.items():
        fields = 'shape and color' if b['group'] == 0 else 'value (the number of visible items in that candidate)'
        schema = '{"A":{"shape":"...","color":"..."},"B":{...},"C":{...},"D":{...}}' if b['group'] == 0 else '{"A":{"value":0},"B":{"value":0},"C":{"value":0},"D":{"value":0}}'
        q = ('<image>Transcribe the visible observations for candidates A, B, C, D. '
             'For each candidate give only ' + fields + '. Count separately for each candidate. '
             'Do not apply a rule or decide KEEP/OMIT. Return one JSON object and no other text. '
             'Use lowercase English property names and values. Required schema: ' + schema)
        extract.append({'id': 'extract:' + sid, 'source_id': sid, 'group': b['group'], 'phase': 'extract',
                        'messages': [{'role': 'user', 'content': q}], 'images': [b['image']],
                        'truth': {c: x for c, x in zip('ABCD', b['structured_observations'])}})
    assert len(extract) == 5
    ev.save_rows(out / 'extraction_probes.jsonl', extract)
    source_files = [SOURCE / f for f in ('manifest.json', 'completion_audit.json', 'vision.jsonl', 'text.jsonl',
        'probes.jsonl', 'blocks.jsonl', 'target_knowledge.jsonl', 'semantic_reviews.jsonl',
        'vision/predictions.jsonl', 'vision/checkpoint_manifest.json', 'warmup/eval_manifest.json',
        'text/eval_manifest.json', 'model_hash_validation.json', 'original_data_hash_validation.json')]
    source_files += [ROOT / 'scripts' / f for f in ('pilot_v7_counterfactual_names.py',
        'run_v7_evaluation.py', 'run_v7_full_experiment.py', 'v7_gpu_resources.py')]
    for arm in ('warmup', 'text'):
        source_files.append(SOURCE / arm / 'predictions.jsonl')
    save(out / 'manifest.json', {
        'created_at': ev.now(), 'script_sha256': sha(Path(__file__)),
        'sources': {str(p): sha(p) for p in source_files},
        'files': {p.name: sha(p) for p in out.glob('*.jsonl')},
        'parent': ev.checkpoint(Path(read(SOURCE / 'vision/checkpoint_manifest.json')['path'])),
        'arms': ARMS, 'training_rows_per_arm': 512, 'new_epochs': 1, 'new_steps': 128,
        'batch': 4, 'accumulation': 1, 'learning_rate': 1e-4, 'seed': 42,
        'teacher': 'All eight previous VISION recall_image outputs, verbatim, from original learning images. All eight manually reviewed; no score-dependent dropping or correction.',
        'comparison': 'Both start from same previous VISION checkpoint, new optimizer, same teacher targets, row order, repetitions and128steps. Control retains original learning images/prompts; replay removes learning images and uses generic name-only TEXT prompts.',
        'classification': 'text_replay is image-derived textual bridge supervision, NOT pure visual learning or a pure V2T/V2V result. Control remains image-conditioned supervision. Both inherit previous VISION training.',
        'diagnostic': 'Three existing arms each generate5rule-free observation transcripts and160MCQs: self/oracle rule crossed with own/true observations, shift0only. Prior direct and oracle scores are reused references, not new requests.',
        'new_generations_planned': {'diagnostic': 495, 'new_training_arms': 1632, 'total': 2127},
        'limits': ['Same8development rules/5images; one seed; no independent confirmation.',
                   'Tests never enter training. Structured true observations and correct rules appear only in separately labelled oracle diagnostics.',
                   'No native CoT; observation transcriptions are a separate generated intermediate.',
                   'Replay changes both image presence and prompt; comparison does not isolate either factor.',
                   'Teacher already passed source review; no claim that blind self-distillation fixes incorrect visual knowledge.',
                   'No PT/FT training, no original data or image changes.']})
    snap = out / 'source_snapshots'
    snap.mkdir(exist_ok=True)
    for p in [Path(__file__)] + [p for p in source_files if p.suffix == '.py']:
        shutil.copy2(p, snap / p.name)
    validate(out)
    print('Prepared: two512row training sets; five original observation scenes.', flush=True)


class Runner:
    def __init__(self, folder, config, gpu):
        self.folder, self.config, self.gpu = folder, config, gpu
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / 'eval_manifest.json').exists():
            assert read(folder / 'eval_manifest.json') == config
        else:
            save(folder / 'eval_manifest.json', config)
        self.raw = folder / 'predictions.jsonl'
        self.rr, notes = ev.read_rows(self.raw, allow_partial_tail=True) if self.raw.exists() else ([], [])
        if notes:
            ev.save_rows(self.raw, self.rr)
        self.seen = {r['id'] for r in self.rr}
        assert len(self.seen) == len(self.rr)
        self.engine = None

    def infer(self, probes):
        from swift.infer_engine import InferRequest, RequestConfig
        pending = [p for p in probes if p['id'] not in self.seen]
        for start in range(0, len(pending), 4):
            batch = pending[start:start + 4]
            fmt = batch[0]['fmt']
            assert all(p['fmt'] == fmt for p in batch)
            replies = self.engine.infer([
                InferRequest(messages=copy.deepcopy(p['messages']), images=p.get('images', [])) for p in batch],
                RequestConfig(max_tokens=self.config['max_tokens'][fmt], temperature=0, seed=42, return_details=True), use_tqdm=False)
            assert len(replies) == len(batch)
            for p, reply in zip(batch, replies):
                if isinstance(reply, Exception):
                    raise reply
                c = reply.choices[0]
                text = c.message.content or ''
                pred = ev.letter(text) if fmt == 'mcq' else None
                r = {k: v for k, v in p.items() if k not in ('messages', 'images', 'truth')}
                r.update(response=text, predicted=pred, correct=pred == p['gold'] if p.get('gold') in list('ABCD') else None,
                         format_error=fmt == 'mcq' and pred is None, finish_reason=c.finish_reason,
                         input_sha256=ev.digest({'messages': p['messages'], 'images': p.get('images', [])}),
                         raw_response=dataclasses.asdict(reply))
                self.rr.append(r)
                self.seen.add(r['id'])
                with self.raw.open('a') as f:
                    f.write(json.dumps(r, ensure_ascii=False, default=str) + '\n')
                    f.flush()
                    os.fsync(f.fileno())
            save(self.folder / 'eval_state.json', {'status': 'running', 'saved': len(self.rr), 'pid': os.getpid()})
            print(self.folder.name, len(self.rr), flush=True)

    def load(self):
        os.environ.update(environment(self.gpu, previous.DATA))
        import torch
        from swift.infer_engine import TransformersEngine
        from swift.utils import seed_everything
        seed_everything(42)
        self.engine = TransformersEngine(str(MODELS['qwen3']), adapters=[self.config['adapter']['path']],
            torch_dtype=torch.bfloat16, device_map='cuda:0', max_batch_size=4, attn_impl='sdpa')
        self.engine.template.enable_thinking = False

    def audit(self, probes, expected):
        assert len(self.rr) == len(self.seen) == expected
        lookup = {p['id']: p for p in probes}
        assert set(lookup) == self.seen
        for r in self.rr:
            p = lookup[r['id']]
            assert r['input_sha256'] == ev.digest({'messages': p['messages'], 'images': p.get('images', [])})
            assert r['response'] == (r['raw_response']['choices'][0]['message']['content'] or '')
        assert ev.checkpoint(Path(self.config['adapter']['path'])) == self.config['adapter']
        save(self.folder / 'eval_state.json', {'status': 'complete', 'saved': expected})


def config(cp, tokens):
    return {'adapter': ev.checkpoint(cp), 'temperature': 0, 'seed': 42, 'batch': 4,
            'thinking': False, 'max_tokens': tokens}


def observation_json(response):
    # Preserve raw transcript downstream; normalisation is for scoring only.
    t = response.strip()
    if t.startswith('```'):
        t = re.sub(r'^```(?:json)?\s*|\s*```$', '', t)
    try:
        parsed = json.loads(t)
        if not isinstance(parsed, dict) or set(parsed) != set('ABCD'):
            return None
        for value in parsed.values():
            if not isinstance(value, dict):
                return None
        return {k: {f.lower(): v.lower().strip() if isinstance(v, str) else v for f, v in d.items()} for k, d in parsed.items()}
    except (ValueError, TypeError):
        return None


def diagnose(out, arm, gpu):
    folder = out / 'observation_diagnostic' / arm
    cp = Path(read(SOURCE / arm / 'eval_manifest.json')['adapter']['path'])
    run = Runner(folder, config(cp, {'extract': 384, 'mcq': 16}), gpu)
    extract = [dict(p, fmt='extract', condition='observe_only') for p in rows(out / 'extraction_probes.jsonl')]
    original = [p for p in rows(SOURCE / 'probes.jsonl') if p['condition'] == 'named_V' and p['shift'] == 0]
    byscene = {p['source_id']: p for p in extract}
    recall = {r['qid']: r for r in rows(SOURCE / arm / 'predictions.jsonl') if r['condition'] == 'recall'}
    knowledge = {k['id']: k for k in rows(SOURCE / 'target_knowledge.jsonl')}

    def applications():
        transcribed = {r['source_id']: r for r in run.rr if r.get('phase') == 'extract'}
        assert len(transcribed) == 5
        pp = []
        for rc in ('self', 'oracle'):
            for ob in ('own', 'truth'):
                for p in original:
                    sourceid = p['original_source_id']
                    response = transcribed[sourceid]['response']
                    obs = response if ob == 'own' else json.dumps(byscene[sourceid]['truth'], ensure_ascii=False)
                    rule = recall[p['qid']]['response'] if rc == 'self' else previous.card(knowledge[p['qid']])
                    z = copy.deepcopy(p)
                    z.update(id=f'{rc}_{ob}:' + p['id'], condition=rc + '_' + ob, phase='apply',
                             rule_context_sha256=ev.digest(rule), observation_sha256=ev.digest(obs))
                    z.pop('images', None)
                    body = z['messages'][-1]['content'].replace('<image>', 'Observation records:\n' + obs)
                    z['messages'][-1]['content'] = 'Rule context:\n' + rule + '\n\n' + body
                    pp.append(z)
        assert len(pp) == 160
        return pp

    if len(run.rr) < 165:
        with reserve(previous.LEDGER, gpu, str(folder) + ':eval', 24000) as resource:
            resource.bind(os.getpid())
            run.load()
            run.infer(extract)
            apply = applications()
            ev.save_rows(folder / 'application_probes.jsonl', apply)
            run.infer(apply)
    apply = applications()
    run.audit(extract + apply, 165)
    metrics = {c: metric([r for r in run.rr if r['condition'] == c]) for c in ('self_own', 'self_truth', 'oracle_own', 'oracle_truth')}
    extraction = []
    for p in extract:
        r = next(r for r in run.rr if r['id'] == p['id'])
        parsed = observation_json(r['response'])
        field_checks = []
        for candidate, facts in p['truth'].items():
            for field, value in facts.items():
                observed = parsed.get(candidate, {}).get(field) if parsed else None
                if isinstance(value, int) and isinstance(observed, str):
                    try:
                        observed = float(observed)
                    except ValueError:
                        pass
                field_checks.append({'candidate': candidate, 'field': field, 'expected': value,
                                     'observed': observed, 'correct': observed == value})
        extraction.append({'id': p['id'], 'group': p['group'], 'raw_response': r['response'],
            'parsed': parsed, 'json_valid': parsed is not None, 'scene_correct': all(x['correct'] for x in field_checks),
            'field_checks': field_checks, 'finish_reason': r['finish_reason']})
    save(folder / 'report.json', {'status': 'complete', 'metrics': metrics, 'extraction': extraction,
        'note': 'New diagnostics use structured JSON observations; previous T/V direct scores use original prompts. Reused source scores are not new generations.'})
    ev.save_rows(folder / 'extraction_reviews.jsonl', extraction)


def train(out, arm, gpu):
    folder = out / arm
    folder.mkdir(exist_ok=True)
    cpdir = folder / 'checkpoint'
    cp = final_checkpoint(cpdir, 1)
    if cp is None:
        recover = [p for p in cpdir.glob('checkpoint-*') if all((p / f).exists() for f in
            ('optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'trainer_state.json', 'adapter_model.safetensors'))]
        resume = max(recover, key=lambda p: read(p / 'trainer_state.json')['global_step']) if recover else None
        if cpdir.exists() and not resume:
            cpdir.rename(folder / ('interrupted_' + str(int(__import__('time').time()))))
        cmd = train_command('qwen3', 'sft', out / (arm + '.jsonl'), cpdir, 1, resume=resume)
        for flag, value in (('--per_device_train_batch_size', '4'), ('--gradient_accumulation_steps', '1'),
                            ('--save_total_limit', '3'), ('--save_strategy', 'steps')):
            cmd[cmd.index(flag) + 1] = value
        cmd += ['--save_steps', '32']
        if not resume:
            cmd += ['--adapters', read(out / 'manifest.json')['parent']['path']]
        save(folder / 'train_state.json', {'status': 'queued', 'command': cmd})
        with reserve(previous.LEDGER, gpu, str(folder) + ':train', 29000 if arm == 'vision_control' else 22000) as resource:
            with (folder / 'train.log').open('a') as log:
                process = subprocess.Popen(cmd, cwd=ROOT, env=environment(gpu, previous.DATA),
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                resource.bind(process.pid)
                save(folder / 'train_state.json', {'status': 'running', 'pid': process.pid, 'command': cmd})
                code = process.wait()
            assert code == 0, (arm, code)
        cp = final_checkpoint(cpdir, 1)
    assert cp is not None
    state = read(cp / 'trainer_state.json')
    args = read(cp / 'args.json')
    assert state['global_step'] == 128 and abs(state['epoch'] - 1) < 1e-6
    assert args['dataset'] == [str(out / (arm + '.jsonl'))]
    assert args['adapters'] == [read(out / 'manifest.json')['parent']['path']]
    losses = [r['loss'] for r in rows(cpdir / 'logging.jsonl') if 'loss' in r]
    assert losses and all(math.isfinite(x) for x in losses)
    from safetensors import safe_open
    with safe_open(str(cp / 'adapter_model.safetensors'), framework='pt', device='cpu') as f:
        keys = list(f.keys())
        bkeys = [k for k in keys if 'lora_B' in k]
        assert len(bkeys) == 252 and all(bool(f.get_tensor(k).count_nonzero()) for k in bkeys)
        assert not any('.visual.' in k or '.merger.' in k for k in keys)
    save(folder / 'checkpoint_manifest.json', {**ev.checkpoint(cp), 'parent': read(out / 'manifest.json')['parent'],
        'epoch': 1, 'new_global_steps': 128, 'last_logged_loss': losses[-1], 'nonzero_lora_B': 252,
        'route': 'pure_image_continuation' if arm == 'vision_control' else 'visual_origin_text_replay_bridge',
        'selection': 'fixed final checkpoint; no evaluation selection'})
    save(folder / 'train_state.json', {'status': 'complete', 'checkpoint': str(cp)})


def evaluate(out, arm, gpu):
    folder = out / arm
    cp = Path(read(folder / 'checkpoint_manifest.json')['path'])
    run = Runner(folder, config(cp, {'card': 384, 'mcq': 16}), gpu)
    pp = rows(SOURCE / 'probes.jsonl')

    def own_recall_probes():
        recall = {r['qid']: r for r in run.rr if r['condition'] == 'recall'}
        assert len(recall) == 8
        probes = []
        for p in pp:
            if not p['condition'].startswith('named_'):
                continue
            z = copy.deepcopy(p)
            z.update(id='self:' + p['id'], condition='self_' + p['modality'])
            z['messages'][-1]['content'] = 'Your own prior rule recall (uncorrected):\n' + recall[p['qid']]['response'] + '\n\n' + z['messages'][-1]['content']
            z['recall_response_sha256'] = ev.digest(recall[p['qid']]['response'])
            probes.append(z)
        return probes

    if len(run.rr) < 816:
        with reserve(previous.LEDGER, gpu, str(folder) + ':eval', 24000) as resource:
            resource.bind(os.getpid())
            run.load()
            run.infer([p for p in pp if p['fmt'] == 'card'])
            run.infer([p for p in pp if p['fmt'] == 'mcq'])
            own = own_recall_probes()
            ev.save_rows(folder / 'self_probes.jsonl', own)
            run.infer(own)
    run.audit(pp + own_recall_probes(), 816)
    result = {}
    for c in ('named_T', 'named_V', 'self_T', 'self_V', 'oracle_T', 'oracle_V'):
        group = [r for r in run.rr if r['condition'] == c]
        result[c] = metric(group)
        result[c]['shift0'] = metric([r for r in group if r['shift'] == 0])
        result[c]['by_task'] = {t: metric([r for r in group if r['task'] == t]) for t in ('direct', 'diagnosis')}
        result[c]['by_group'] = {str(g): metric([r for r in group if r['group'] == g]) for g in (0, 1)}
        blocks = [[r for r in group if (r['block'], r['shift']) == k] for k in sorted({(r['block'], r['shift']) for r in group})]
        result[c]['paired'] = {'n': len(blocks), 'all4_correct': sum(all(r['correct'] for r in b) for b in blocks),
            'all4_same_answer': sum(len({r['predicted'] for r in b}) == 1 for b in blocks)}
    result['recall_outputs'] = [r for r in run.rr if r['fmt'] == 'card']
    save(folder / 'report.json', {'status': 'complete', 'metrics': result})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--stage', choices=['prepare', 'diagnose', 'train', 'eval'], required=True)
    parser.add_argument('--arm', choices=list(ARMS) + ['warmup', 'text', 'vision'])
    parser.add_argument('--gpu', default='0')
    a = parser.parse_args()
    out = a.out.resolve()
    if a.stage == 'prepare':
        prepare(out)
    else:
        validate(out)
        with ev.run_lock(out / (a.arm + '_' + a.stage + '.lock')):
            {'diagnose': diagnose, 'train': train, 'eval': evaluate}[a.stage](out, a.arm, a.gpu)
