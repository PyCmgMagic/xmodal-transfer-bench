#!/usr/bin/env python3
"""Original 1000-rule SFT, dense TEXT/VISION, ten epochs, then four directions."""
import argparse
import copy
import csv
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import run_v7_evaluation as ev
from run_v7_full_experiment import ROOT, MODELS, environment, rows, read, save, sha, train_command, alive
from v7_gpu_resources import reserve, ledger
from explore_v7_visual_replay import Runner as SavedRunner
from v7_dense_checkpoint_storage import checkpoint_lock

DATA = ROOT / 'rules_repaired_single_image_v7'
ORIGINAL = ROOT / 'runs/v7_full_experiment/data'
DEFAULT = ROOT / 'runs/v7_fullparam_sft_ep10_20261019'
MODEL = MODELS['qwen3']
AUDIT = ROOT / 'scripts/v7_full_parameter_audit.py'
PLUGIN = ROOT / 'scripts/v7_ep10_checkpoint_plugin.py'
STORAGE = ROOT / 'scripts/v7_dense_checkpoint_storage.py'
LEDGER = ROOT / 'runs/v7_full_experiment'
DIRECTIONS = {'text': ('T2T', 'T2V'), 'vision': ('V2T', 'V2V'), 'baseline': ('T2T', 'T2V')}


def dense_identity(path):
    path = Path(path)
    assert (path / 'config.json').is_file()
    weights = sorted(path.glob('model*.safetensors'))
    assert weights
    names = ['config.json', 'model.safetensors.index.json', 'generation_config.json',
             'tokenizer.json', 'tokenizer_config.json', 'preprocessor_config.json',
             'processor_config.json', 'chat_template.jinja']
    files = weights + [path / n for n in names if (path / n).is_file()]
    return {'path': str(path), 'kind': 'dense', 'sha256': {p.name: sha(p) for p in files}}


def scratch_root(work):
    return Path(read(work / 'manifest.json')['scratch_root'])


def setup_scratch(work):
    path = Path('/dev/shm') / f'xmodal-v7-ep10-{os.getuid()}-{work.name}'
    path.mkdir(exist_ok=True)
    owner = path / 'owner.json'
    expected = {'work': str(work), 'purpose': 'This run only: temporary full optimizer recovery checkpoints.'}
    if owner.exists():
        assert read(owner) == expected
    else:
        assert not list(path.iterdir()), path
        save(owner, expected)
    return path


def prepare(work):
    work.mkdir(parents=True, exist_ok=True)
    if (work / 'manifest.json').exists():
        validate(work)
        return
    old = read(ORIGINAL / 'manifest.json')
    assert old['rows'] == 1000 and Path(old['dataset']) == DATA
    for item in old['sources'].values():
        assert sha(Path(item['path'])) == item['sha256'], item['path']
    train_ids = None
    for mod in ('text', 'vision'):
        source = ORIGINAL / f'sft_{mod}.jsonl'
        assert sha(source) == old['outputs'][source.name]['sha256']
        rr = rows(source)
        assert len(rr) == len({r['id'] for r in rr}) == 1000
        ids = {r['qid'] for r in rr}
        if train_ids is None:
            train_ids = ids
        assert train_ids == ids
        for r in rr:
            assert [m['role'] for m in r['messages']] == ['user', 'assistant']
            assert r['messages'][-1]['content'].strip()
            assert len(r.get('images', [])) == (1 if mod == 'vision' else 0)
            for name in r.get('images', []):
                assert Path(name).is_file() and Path(name).name == 'learning_visual.png'
        dest = work / 'data' / source.name
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(source, dest)
        # Fixed four longest assistant targets: computational preflight, not score selection.
        selected = sorted(rr, key=lambda r: (-len(r['messages'][-1]['content']), r['id']))[:4]
        ev.save_rows(work / 'data' / f'smoke_{mod}.jsonl', selected)
    labels = ev.id_map(rows(DATA / 'audit/eval_labels.jsonl'), DATA)
    assert len(labels) == 2000 and {r['qid'] for r in labels.values()} == train_ids
    for d in ev.DIRECTIONS:
        raw = rows(DATA / d / 'eval.jsonl')
        assert {r['id'] for r in raw} == set(labels)
        prepared = [ev.prepare_row(r, DATA, d) for r in raw]
        assert all(len(r.get('images', [])) == int(d.endswith('V')) for r in prepared)
        ev.save_rows(work / 'data' / f'eval_{d}.jsonl', prepared)
    for a, b in (('T2T', 'V2T'), ('T2V', 'V2V')):
        assert sha(work / 'data' / f'eval_{a}.jsonl') == sha(work / 'data' / f'eval_{b}.jsonl')
    images = rows(DATA / 'image_hashes.jsonl')
    assert len(images) == len(list((DATA / 'images').rglob('*.png'))) == 3000
    for r in images:
        assert sha(DATA / r['path']) == r['source_sha256'] == r['repaired_sha256']
    ds = read(ROOT / 'runs/v7_full_parameter_20261018/zero2_cpu.json')
    assert 'optimizer' not in ds and ds['zero_optimization']['stage'] == 2
    save(work / 'zero2_cpu.json', ds)
    scratch = setup_scratch(work)
    assert shutil.disk_usage(scratch).free > 215 * 1024**3, 'Scratch must fit two retained states plus one serialized write.'
    assert shutil.disk_usage(work).free > 75 * 1024**3, 'Need room for bounded dense model exports.'
    extensions = work / 'torch_extensions'
    cached = ROOT / 'runs/v7_full_parameter_20261018/torch_extensions'
    if cached.exists() and not extensions.exists():
        shutil.copytree(cached, extensions)
    source_files = [Path(__file__), AUDIT, PLUGIN, STORAGE, ROOT / 'scripts/run_v7_evaluation.py',
                    ROOT / 'scripts/run_v7_full_experiment.py', ROOT / 'scripts/v7_gpu_resources.py',
                    ROOT / 'scripts/explore_v7_visual_replay.py']
    model = dense_identity(MODEL)
    manifest = {'created_at': ev.now(), 'dataset': str(DATA), 'model': model,
        'user_choice': 'Original 1000-rule explanation SFT; no augmented application MCQs.',
        'modalities': ['text', 'vision'], 'epochs': 10, 'steps_each': 630,
        'batch_size': 4, 'gradient_accumulation': 4, 'effective_batch': 16,
        'learning_rate': 1e-5, 'schedule': 'cosine', 'warmup_ratio': .05,
        'seed': 42, 'max_length': 8192, 'max_pixels': 1003520,
        'optimizer': 'DeepSpeed ZeRO2 CPU AdamW: FP32 master weights and states; BF16 forward/backward.',
        'optimizer_config': {'beta1': .9, 'beta2': .95, 'epsilon': 1e-8, 'weight_decay': .1, 'grad_clip': 1.},
        'loss': 'Default causal SFT token loss on assistant replies only; no custom equal-example loss.',
        'scopes': {'text': 'All language weights; image tower/aligner frozen because no image input.',
                   'vision': 'All VLM weights including vision and aligner.'},
        'initialization': 'Both start from original pretrained Qwen3-VL-4B-Instruct, not the eight-rule warmup or pilot checkpoints.',
        'selection': 'Fixed final epoch10, step630. No evaluation-based checkpoint or epoch selection.',
        'evaluation': {'protocol': 'strict', 'temperature': 0, 'max_new_tokens': 32, 'thinking': False,
                       'directions': DIRECTIONS, 'rows_each': 2000, 'trained_requests': 8000,
                       'fresh_baseline_unique_requests': 4000,
                       'no_learning_material_at_test': True, 'cot_deferred': True},
        'scratch_root': str(scratch),
        'checkpoint_storage': {'full_optimizer_state': 'Shared-memory scratch, latest one per arm, serialized writes.',
            'persistent': 'Latest two dense model-only exports per arm, atomically verified every epoch.',
            'recovery_limit': 'Exact optimizer resume survives process restarts, not host reboot/shared-memory loss. Disk exports retain weights and logs; no silent optimizer-reset continuation.',
            'cleanup': 'Only this run scratch is removed after final model/update checks; historical runs are untouched.'},
        'original_data_manifest': {'path': str(ORIGINAL / 'manifest.json'), 'sha256': sha(ORIGINAL / 'manifest.json')},
        'original_sources': old['sources'], 'all3000_images_checked': True,
        'files': {str(f.relative_to(work)): sha(f) for f in sorted((work / 'data').glob('*'))},
        'sources': {str(f.resolve()): sha(f) for f in source_files},
        'versions': {n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'ms-swift', 'deepspeed')}}
    manifest['files']['zero2_cpu.json'] = sha(work / 'zero2_cpu.json')
    save(work / 'manifest.json', manifest)
    snapshots = work / 'source_snapshots'
    snapshots.mkdir(exist_ok=True)
    for f in source_files:
        shutil.copy2(f, snapshots / f.name)
    save(work / 'source_validation.json', {'status': 'pass', 'training_rules': 1000,
        'original_sft_files_byte_identical': True, 'images_verified': 3000,
        'eval_rows_per_direction': 2000, 'same_ids_all_directions': True,
        'same_rule_learning_and_application': True, 'held_out_rule_split': False})
    print('PREPARED original1000SFT, two dense arms, epoch10/step630, four directions.', flush=True)


def validate(work):
    m = read(work / 'manifest.json')
    assert m['epochs'] == 10 and m['steps_each'] == 630
    for f, want in m['sources'].items():
        assert sha(Path(f)) == want, f
    for f, want in m['files'].items():
        assert sha(work / f) == want, f
    for item in m['original_sources'].values():
        assert sha(Path(item['path'])) == item['sha256']


def encode_training(work):
    if read(work / 'encoding_validation.json').get('status') == 'pass':
        return
    os.environ.update(job_env(work, '0'))
    from swift.model import get_model_processor
    from swift.template import get_template
    _, processor = get_model_processor(str(MODEL), model_type='qwen3_vl', load_model=False)
    template = get_template(processor, template_type='qwen3_vl', max_length=8192, enable_thinking=False)
    template.set_mode('train')
    records = []
    for mod in ('text', 'vision'):
        for r in rows(work / 'data' / f'sft_{mod}.jsonl'):
            encoded = template.encode(copy.deepcopy(r))
            tokens, labels = encoded['input_ids'], encoded['labels']
            assert len(tokens) == len(labels) < 8192, (mod, r['id'], len(tokens))
            text = processor.tokenizer.decode(tokens)
            question = r['messages'][0]['content'].replace('<image>', '').strip()
            target = r['messages'][-1]['content'].strip()
            decoded_target = processor.tokenizer.decode([x for x in labels if x != -100])
            assert question in text and target in decoded_target, (mod, r['id'])
            assert any(x == -100 for x in labels) and sum(x != -100 for x in labels) > 0
            records.append({'modality': mod, 'id': r['id'], 'input_tokens': len(tokens),
                            'supervised_tokens': sum(x != -100 for x in labels)})
        print('ENCODING_CHECKED', mod, len(records), flush=True)
    save(work / 'encoding_validation.json', {'status': 'pass', 'rows': 2000,
        'all_questions_and_targets_preserved': True, 'records': records})


def job_env(work, gpu, mod=None, smoke=False):
    env = environment(gpu, DATA)
    env.update(CXX='/mnt/yunpan/anaconda3/envs/tda/bin/x86_64-conda-linux-gnu-c++',
        CC='/mnt/yunpan/anaconda3/envs/tda/bin/x86_64-conda-linux-gnu-gcc',
        TORCH_EXTENSIONS_DIR=str(work / 'torch_extensions'), TMPDIR=str(work / 'tmp'),
        OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', MAX_JOBS='4', IMAGE_MAX_TOKEN_NUM='980',
        MASTER_ADDR='127.0.0.1', MASTER_PORT=str(30190 + int(gpu)))
    (work / 'tmp').mkdir(exist_ok=True)
    if mod:
        folder = work / ('smoke' if smoke else 'jobs') / mod
        folder.mkdir(parents=True, exist_ok=True)
        env.update(NPROC_PER_NODE='1', V7_PARAMETER_AUDIT=str(folder / 'parameter_audit.jsonl'),
            V7_EP10_SCRATCH_ROOT=str(scratch_root(work)),
            V7_EP10_CHECKPOINT_LOCK=str(work / '.checkpoint_storage.lock'),
            V7_EP10_PERSISTENT_CHECKPOINTS=str(folder / 'checkpoint'),
            V7_EP10_CHECKPOINT_EVENTS=str(folder / 'checkpoint_events.jsonl'))
    return env


def recovery_checkpoints(folder):
    return sorted((p for p in folder.glob('checkpoint-*') if (p / 'trainer_state.json').is_file()
                   and list(p.glob('global_step*/**/*optim_states.pt'))),
                  key=lambda p: read(p / 'trainer_state.json')['global_step'])


def training_command(work, mod, smoke=False, smoke_steps=1):
    folder = work / ('smoke' if smoke else 'jobs') / mod
    scratch = scratch_root(work) / ('smoke_' + mod if smoke else mod) / 'checkpoint'
    scratch.mkdir(parents=True, exist_ok=True)
    source = work / 'data' / (f'smoke_{mod}.jsonl' if smoke else f'sft_{mod}.jsonl')
    cmd = train_command('qwen3', 'sft', source, scratch, epochs=10)
    changes = {'--tuner_type': 'full', '--per_device_train_batch_size': '4',
               '--gradient_accumulation_steps': '1' if smoke else '4', '--learning_rate': '1e-5',
               '--save_total_limit': '1', '--freeze_vit': str(mod == 'text').lower(),
               '--freeze_aligner': str(mod == 'text').lower()}
    for flag, value in changes.items():
        cmd[cmd.index(flag) + 1] = value
    for flag in ('--target_modules', '--lora_rank', '--lora_alpha', '--lora_dropout'):
        i = cmd.index(flag)
        del cmd[i:i+2]
    cmd += ['--deepspeed', str(work / 'zero2_cpu.json'), '--model_type', 'qwen3_vl',
        '--external_plugins', str(AUDIT), str(PLUGIN), '--callbacks', 'v7_full_parameter_audit',
        '--packing', 'false', '--padding_free', 'false', '--attn_impl', 'sdpa',
        '--weight_decay', '.1', '--adam_beta1', '.9', '--adam_beta2', '.95',
        '--adam_epsilon', '1e-8', '--max_grad_norm', '1', '--optim', 'adamw_torch']
    if smoke:
        cmd += ['--max_steps', str(smoke_steps), '--logging_steps', '1', '--warmup_ratio', '0']
    recover = recovery_checkpoints(scratch)
    if recover:
        cmd += ['--resume_from_checkpoint', str(recover[-1]), '--load_args', 'false']
    elif (folder / 'checkpoint').exists() and list((folder / 'checkpoint').glob('checkpoint-*')):
        raise RuntimeError('Disk model export exists but scratch optimizer state is missing. Refusing a silent optimizer-reset restart.')
    return cmd


def final_checkpoint(work, mod, smoke=False):
    p = work / ('smoke' if smoke else 'jobs') / mod / 'checkpoint'
    candidates = sorted(p.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[1]))
    if not candidates:
        return None
    cp = candidates[-1]
    state = read(cp / 'trainer_state.json')
    if state.get('global_step') == (2 if smoke else 630):
        assert smoke or abs(state['epoch'] - 10) < 1e-6, state
        assert (cp / 'dense_export.json').is_file()
        return cp
    return None


def train(work, mod, gpu, smoke=False):
    # Full CPU optimizer + recovery-state RAM, rather than GPU VRAM, limits
    # simultaneous dense trainers. Evaluation does not take this lock.
    with checkpoint_lock(work / '.host_memory_training.lock'):
        return _train(work, mod, gpu, smoke)


def _train(work, mod, gpu, smoke=False):
    folder = work / ('smoke' if smoke else 'jobs') / mod
    folder.mkdir(parents=True, exist_ok=True)
    if read(folder / ('validated.json' if smoke else 'state.json')).get('status') in ('pass', 'complete'):
        assert final_checkpoint(work, mod, smoke)
        return
    old = read(folder / 'state.json')
    if old.get('status') == 'running' and alive(old.get('pid')):
        pid = old['pid']
        command = Path(f'/proc/{pid}/cmdline').read_bytes()
        marker = str(scratch_root(work) / ('smoke_' + mod if smoke else mod)).encode()
        assert marker in command, 'Refuse to adopt a reused PID belonging to another process.'
        with reserve(LEDGER, gpu, str(folder) + ':train', 30000, adopted_pid=pid):
            while alive(pid):
                time.sleep(10)
    assert read(work / 'encoding_validation.json').get('status') == 'pass'
    for target in ((1, 2) if smoke else (630,)):
        if final_checkpoint(work, mod, smoke):
            break
        if smoke:
            recover = recovery_checkpoints(scratch_root(work) / ('smoke_' + mod) / 'checkpoint')
            if recover and read(recover[-1] / 'trainer_state.json')['global_step'] >= target:
                continue
        cmd = training_command(work, mod, smoke, target)
        save(folder / 'state.json', {'status': 'queued', 'command': cmd, 'requested_epochs': 10, 'smoke': smoke})
        with reserve(LEDGER, gpu, str(folder) + ':train', 30000) as reservation:
            with (folder / 'train.log').open('a') as log:
                proc = subprocess.Popen(cmd, cwd=ROOT, env=job_env(work, gpu, mod, smoke),
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                reservation.bind(proc.pid)
                save(folder / 'state.json', {'status': 'running', 'pid': proc.pid, 'command': cmd,
                                            'target_steps': target, 'smoke': smoke, 'started_at': ev.now()})
                code = proc.wait()
            if code:
                save(folder / 'state.json', {'status': 'failed', 'returncode': code, 'ended_at': ev.now()})
                raise RuntimeError(f'Training failed ({code}): {folder / "train.log"}')
    cp = final_checkpoint(work, mod, smoke)
    assert cp, folder
    state = read(cp / 'trainer_state.json')
    loss = [float(x['loss']) for x in state['log_history'] if 'loss' in x]
    assert loss and all(math.isfinite(x) for x in loss)
    pa = rows(folder / 'parameter_audit.jsonl')
    start = next(x for x in pa if x['event'] == 'train_begin')
    assert start['groups']['language']['trainable'] == start['groups']['language']['parameters']
    assert start['groups']['vision']['trainable'] == (0 if mod == 'text' else start['groups']['vision']['parameters'])
    end = [x for x in pa if x['event'] == 'step_end'][-1]
    assert end['step'] == (2 if smoke else 630)
    assert end['master_weights'] and all(x['dtype'] == 'torch.float32' and x['device'] == 'cpu' for x in end['master_weights'])
    identity = dense_identity(cp)
    assert identity['sha256'] != read(work / 'manifest.json')['model']['sha256']
    save(folder / 'checkpoint_manifest.json', {'model': identity, 'epoch': state['epoch'],
        'step': state['global_step'], 'losses': loss, 'selection': 'smoke2steps' if smoke else 'fixed final epoch10',
        'fp32_cpu_master_weights_verified': True, 'trainable_groups': start['groups']})
    save(folder / 'state.json', {'status': 'complete', 'checkpoint': str(cp), 'epoch': state['epoch'],
                               'step': state['global_step'], 'ended_at': ev.now()})
    if smoke:
        # Compare actual common weight tensors, rather than relying on metadata hashes.
        from safetensors import safe_open
        import torch
        base_index = read(MODEL / 'model.safetensors.index.json')['weight_map']
        end_index = read(cp / 'model.safetensors.index.json')['weight_map']
        changed = {}
        for scope in (('language', 'vision') if mod == 'vision' else ('language',)):
            name = next(n for n in base_index if
                (n.endswith('layers.0.self_attn.q_proj.weight') if scope == 'language'
                 else n.endswith('visual.patch_embed.proj.weight')))
            with safe_open(str(MODEL / base_index[name]), framework='pt') as f:
                a = f.get_tensor(name)
            with safe_open(str(cp / end_index[name]), framework='pt') as f:
                b = f.get_tensor(name)
            assert torch.isfinite(b).all() and not torch.equal(a, b), ('smoke update absent', name)
            changed[scope] = name
        # The resumed step2 checkpoint must reload and accept both text and images.
        evaluate(work, mod, gpu, smoke=True)
        save(folder / 'validated.json', {'status': 'pass', 'resumed_from_step1_to_step2': True,
            'finite_loss': True, 'full_model_reload_text_and_image': True, 'actual_changed_weights': changed, 'model': identity})
        with checkpoint_lock(work / '.checkpoint_storage.lock'):
            shutil.rmtree(scratch_root(work) / ('smoke_' + mod))
        # Retain the final smoke model, remove its superseded own step1 export.
        for old in (folder / 'checkpoint').glob('checkpoint-*'):
            if old != cp:
                assert (old / 'dense_export.json').is_file()
                shutil.rmtree(old)


class DenseRunner(SavedRunner):
    def load(self):
        os.environ.update(job_env(Path(self.config['work']), self.gpu))
        import torch
        from swift.infer_engine import TransformersEngine
        from swift.utils import seed_everything
        seed_everything(42)
        self.engine = TransformersEngine(self.config['model']['path'], model_type='qwen3_vl',
            torch_dtype=torch.bfloat16, device_map='cuda:0', max_batch_size=4, attn_impl='sdpa')
        self.engine.template.enable_thinking = False

    def audit(self, probes):
        assert len(self.rr) == len(self.seen) == len(probes)
        lookup = {p['id']: p for p in probes}
        assert self.seen == set(lookup)
        for r in self.rr:
            p = lookup[r['id']]
            assert r['input_sha256'] == ev.digest({'messages': p['messages'], 'images': p.get('images', [])})
            assert r['response'] == (r['raw_response']['choices'][0]['message']['content'] or '')
            assert r['predicted'] == ev.letter(r['response'])
        save(self.folder / 'eval_state.json', {'status': 'complete', 'saved': len(probes)})


def evaluate(work, mod, gpu, smoke=False):
    folder = work / ('smoke' if smoke else 'jobs') / mod
    model = dense_identity(MODEL if mod == 'baseline' else final_checkpoint(work, mod, smoke))
    labels = ev.id_map(rows(DATA / 'audit/eval_labels.jsonl'), DATA)
    audit = {r['sample_id']: r for r in rows(DATA / 'audit/sample_audit.jsonl')}
    engine = None
    # No model is loaded if every direction was already completed.
    needed = []
    for d in (('T2T', 'T2V') if smoke else DIRECTIONS[mod]):
        probes = [{**r, 'fmt': 'mcq'} for r in rows(work / 'data' / f'eval_{d}.jsonl')]
        if smoke:
            probes = probes[:1]
        config = {'model': model, 'work': str(work), 'direction': d,
            'probes_sha256': ev.digest(probes), 'temperature': 0, 'seed': 42,
            'batch': 4, 'thinking': False, 'max_tokens': {'mcq': 32}}
        runner = DenseRunner(folder / 'evaluation' / d, config, gpu)
        needed.append((d, probes, runner))
    pending = any(len(r.rr) < len(p) for _, p, r in needed)
    def finish_all():
        nonlocal engine
        for d, probes, run in needed:
            if len(run.rr) < len(probes):
                if engine is None:
                    run.load()
                    engine = run.engine
                run.engine = engine
                run.infer(probes)
            run.audit(probes)
            report, records = ev.score(probes, {r['id']: r for r in run.rr}, labels, audit, d)
            report.update(model=model, epoch=0 if mod == 'baseline' else (2 if smoke else 10),
                          raw_outputs_preserved=True, no_learning_material=True)
            save(run.folder / 'report.json', report)
            ev.save_rows(run.folder / 'results.jsonl', records)
            ev.save_rows(run.folder / 'errors.jsonl', [r for r in records if not r['correct']])
    key = str(folder) + ':eval'
    try:
        if pending:
            with reserve(LEDGER, gpu, key, 24000) as res:
                res.bind(os.getpid())
                finish_all()
        else:
            finish_all()
    finally:
        for _, _, runner in needed:
            runner.engine = None
        engine = None
        if pending:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_initialized():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            # This process survives for its next stage; release only its own finished lease.
            with ledger(LEDGER) as entries:
                if entries.get(key, {}).get('pid') == os.getpid():
                    entries.pop(key)
    assert model == dense_identity(Path(model['path']))


def training_updates(work, mod):
    import torch
    from safetensors import safe_open
    torch.set_num_threads(4)
    cp = final_checkpoint(work, mod)
    indexes = []
    for root in (MODEL, cp):
        index = read(root / 'model.safetensors.index.json')['weight_map']
        indexes.append(index)
    a, b = indexes
    assert not set(a) - set(b)
    assert set(b) - set(a) in (set(), {'lm_head.weight'})
    stats = {'language': Counter(), 'vision': Counter()}
    for name in sorted(a):
        with safe_open(str(MODEL / a[name]), framework='pt') as f:
            x = f.get_tensor(name)
        with safe_open(str(cp / b[name]), framework='pt') as f:
            y = f.get_tensor(name)
        assert x.shape == y.shape and torch.isfinite(y).all()
        scope = 'vision' if name.startswith('model.visual.') else 'language'
        stats[scope]['tensors'] += 1
        stats[scope]['changed_tensors'] += int(not torch.equal(x, y))
    assert stats['language']['changed_tensors'] > 0
    assert (stats['vision']['changed_tensors'] > 0) == (mod == 'vision')
    save(work / 'jobs' / mod / 'actual_weight_updates.json', stats)
    return stats


def report(work, require_complete=False):
    records = []
    for mod in ('text', 'vision'):
        for d in DIRECTIONS[mod]:
            trained = read(work / 'jobs' / mod / 'evaluation' / d / 'report.json')
            base_d = 'T2' + d[-1]
            base = read(work / 'jobs/baseline/evaluation' / base_d / 'report.json')
            records.append({'direction': d, 'modality': mod, 'trained': trained or None, 'baseline': base or None,
                'baseline_input_alias': base_d, 'gain': trained['accuracy'] - base['accuracy'] if trained and base else None})
    complete = all(r['trained'] and r['baseline'] and r['trained']['matched'] == 2000
                   and r['baseline']['matched'] == 2000 for r in records)
    if require_complete:
        assert complete
        validate(work)
        assert dense_identity(MODEL) == read(work / 'manifest.json')['model']
        for image in rows(DATA / 'image_hashes.jsonl'):
            assert sha(DATA / image['path']) == image['source_sha256'] == image['repaired_sha256']
        for mod in ('text', 'vision'):
            assert read(work / 'jobs' / mod / 'state.json')['step'] == 630
            assert read(work / 'jobs' / mod / 'actual_weight_updates.json')
        with ledger(LEDGER) as entries:
            active = {k: v for k, v in entries.items() if str(work) in k}
        assert not active, active
    overall = None
    if complete:
        trained_n = sum(r['trained']['total'] for r in records)
        trained_c = sum(r['trained']['correct'] for r in records)
        base_n = sum(r['baseline']['total'] for r in records)
        base_c = sum(r['baseline']['correct'] for r in records)
        overall = {'trained_correct': trained_c, 'trained_total': trained_n,
                   'trained_accuracy': trained_c / trained_n, 'baseline_accuracy': base_c / base_n,
                   'gain': trained_c / trained_n - base_c / base_n,
                   'baseline_physical_unique_requests': 4000}
    save(work / 'summary.json', {'status': 'complete' if complete else 'running',
        'rows': records, 'overall': overall, 'baseline_aliases_are_not_additional_observations': True, 'updated_at': ev.now()})
    header = ['direction', 'accuracy', 'valid_accuracy', 'baseline_accuracy', 'gain', 'correct', 'total',
              'direct_accuracy', 'diagnosis_accuracy', 'missing', 'empty', 'format_errors']
    with (work / 'comparison.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in records:
            s, b = r['trained'], r['baseline']
            if s:
                writer.writerow({'direction': r['direction'], 'accuracy': s['accuracy'],
                    'valid_accuracy': s['accuracy_on_valid_only'], 'baseline_accuracy': b.get('accuracy') if b else None,
                    'gain': r['gain'], 'correct': s['correct'], 'total': s['total'],
                    'direct_accuracy': s['by_question_type']['direct']['accuracy'],
                    'diagnosis_accuracy': s['by_question_type']['diagnosis']['accuracy'],
                    'missing': s['missing_prediction'], 'empty': s['empty_prediction'], 'format_errors': s['format_violation']})
    lines = ['# 原始1000规则，全参数SFT，10 epoch', '',
        'Qwen3-VL-4B-Instruct原始基座。TEXT更新全部语言参数；VISION更新整个模型。原图与原SFT材料保持不变。', '',
        'batch4×accum4，LR1e-5，cosine，warmup5%，seed42，max8192，标准助手token SFT损失，CPU卸载FP32 AdamW；固定第10epoch/630步，不按评测选模型。', '',
        '评测只提供原题和测试图片，不额外给学习材料；温度0、短答案A/B/C/D、max32，不开COT。direct与diagnosis分别报告。', '',
        '| 方向 | 训练前 | 训练后 | 差值 | direct | diagnosis |', '|---|---:|---:|---:|---:|---:|']
    for r in records:
        s, b = r['trained'], r['baseline']
        def pct(x):
            return '待完成' if x is None else f'{x:.2%}'
        lines.append('| ' + r['direction'] + ' | ' + ' | '.join([
            pct(b['accuracy'] if b else None), pct(s['accuracy'] if s else None),
            '待完成' if r['gain'] is None else f"{100*r['gain']:+.2f}pp",
            pct(s['by_question_type']['direct']['accuracy'] if s else None),
            pct(s['by_question_type']['diagnosis']['accuracy'] if s else None)]) + ' |')
    lines += ['', '每方向2000题。训练和评测使用同一批1000条知识，评测是知识应用，不是held-out规则泛化。基座相同文字/图片输入分别复用为T2T/V2T与T2V/V2V基线，不能当独立样本。', '',
        '标准答案、有效输出准确率、缺失/空答/格式错误和两类任务分项见各方向report.json/results.jsonl，完整原答见predictions.jsonl。', '',
        '完整优化器恢复状态位于共享内存，可恢复训练进程中断；主机重启或共享内存丢失后不能精确恢复优化器，程序会拒绝静默重置优化器续训。每epoch验证后同步最近两份模型权重到磁盘；最终权重与日志永久保留，完成后只清理本次共享内存状态。']
    if overall:
        lines += ['', f"四方向总体：训练后{overall['trained_correct']}/{overall['trained_total']}（{overall['trained_accuracy']:.2%}），训练前{overall['baseline_accuracy']:.2%}，差值{100*overall['gain']:+.2f}pp。", '',
                  '![loss and accuracy](' + str(work / 'loss_and_accuracy.png') + ')']
    (work / 'report_zh.md').write_text('\n'.join(lines) + '\n')
    if complete:
        plot(work, records)
    if require_complete:
        save(work / 'completion_audit.json', {'status': 'complete', 'trained_arms': 2, 'epochs_each': 10,
            'training_rules_each': 1000, 'steps_each': 630, 'trained_eval_answers': 8000,
            'fresh_unique_baseline_answers': 4000, 'original3000images_unchanged': True,
            'actual_parameter_updates_verified': True, 'active_gpu_reservations': active})
        state = read(work / 'state.json')
        state.update(status='complete', ended_at=ev.now())
        save(work / 'state.json', state)
        save(work / 'final_artifact_manifest.json', {'status': 'complete', 'completed_at': ev.now(),
            'files': {str(f.relative_to(work)): sha(f) for f in sorted(work.rglob('*')) if f.is_file()
                and not {'checkpoint', 'tmp', 'torch_extensions'} & set(f.relative_to(work).parts)
                and f.name != 'final_artifact_manifest.json' and f.suffix in ('.json', '.jsonl', '.csv', '.md', '.py', '.png')}})


def plot(work, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout='constrained')
    for mod in ('text', 'vision'):
        cp = final_checkpoint(work, mod)
        losses = [r for r in read(cp / 'trainer_state.json')['log_history'] if 'loss' in r]
        axes[0].plot([r['epoch'] for r in losses], [r['loss'] for r in losses], label=mod.upper())
    axes[0].set(xlabel='Epoch', ylabel='Logged SFT loss', title='Original 1000 explanation examples')
    axes[0].legend()
    xs = list(range(4))
    axes[1].bar([x-.18 for x in xs], [100*r['baseline']['accuracy'] for r in result], .36, label='Pretrained')
    axes[1].bar([x+.18 for x in xs], [100*r['trained']['accuracy'] for r in result], .36, label='Full SFT ep10')
    axes[1].set_xticks(xs, [r['direction'] for r in result])
    axes[1].set(ylabel='Accuracy (%)', ylim=(0, 100), title='Closed-book, 2000 questions/direction')
    axes[1].legend()
    fig.savefig(work / 'loss_and_accuracy.png', dpi=180)
    plt.close(fig)


def worker(work, mod, gpu):
    validate(work)
    train(work, mod, gpu, smoke=True)
    train(work, mod, gpu)
    if not read(work / 'jobs' / mod / 'actual_weight_updates.json'):
        training_updates(work, mod)
    # Release this run's recovery states after verified dense final export; inference uses disk weights.
    own = scratch_root(work) / mod
    if own.exists():
        with checkpoint_lock(work / '.checkpoint_storage.lock'):
            shutil.rmtree(own)
    evaluate(work, mod, gpu)
    if mod == 'text':
        with checkpoint_lock(work / '.baseline.lock'):
            evaluate(work, 'baseline', gpu)
    with ev.run_lock(work / '.summary.lock'):
        report(work)


def run(work):
    validate(work)
    encode_training(work)
    children = []
    save(work / 'state.json', {'status': 'running', 'pid': os.getpid(), 'started_at': ev.now()})
    # The fresh baseline uses GPU0 while the first dense trainer uses GPU1.
    # The host-memory lock permits the next trainer to overlap with evaluation.
    for mod, gpu in (('baseline', '0'), ('vision', '1'), ('text', '0')):
        log = (work / f'{mod}_launcher.log').open('a')
        cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--work', str(work),
               '--stage', 'eval' if mod == 'baseline' else 'worker', '--modality', mod, '--gpu', gpu]
        child = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        children.append((mod, child, log))
    exits = {}
    while children:
        for item in list(children):
            mod, proc, log = item
            if proc.poll() is not None:
                exits[mod] = proc.returncode
                log.close()
                children.remove(item)
        time.sleep(5 if children else 0)
    if any(exits.values()):
        save(work / 'state.json', {'status': 'failed', 'worker_exit_codes': exits, 'ended_at': ev.now()})
        raise RuntimeError(exits)
    save(work / 'state.json', {'status': 'validating', 'worker_exit_codes': exits, 'updated_at': ev.now()})
    report(work, require_complete=True)
    print('COMPLETE: two epoch10 dense models,8000trained+4000baseline answers.', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', type=Path, default=DEFAULT)
    ap.add_argument('--stage', required=True, choices=('prepare', 'encode', 'smoke', 'train', 'eval', 'worker', 'run', 'report'))
    ap.add_argument('--modality', choices=('text', 'vision', 'baseline'))
    ap.add_argument('--gpu', default='0')
    a = ap.parse_args()
    work = a.work.resolve()
    if a.stage == 'prepare':
        prepare(work)
    elif a.stage == 'encode':
        validate(work)
        encode_training(work)
    elif a.stage == 'run':
        with ev.run_lock(work / '.run.lock'):
            run(work)
    elif a.stage == 'worker':
        with ev.run_lock(work / ('.' + a.modality + '.worker.lock')):
            worker(work, a.modality, a.gpu)
    elif a.stage == 'report':
        with ev.run_lock(work / '.summary.lock'):
            report(work)
    else:
        validate(work)
        if a.stage in ('smoke', 'train'):
            train(work, a.modality, a.gpu, smoke=a.stage == 'smoke')
        else:
            if a.modality == 'baseline':
                with checkpoint_lock(work / '.baseline.lock'):
                    evaluate(work, a.modality, a.gpu)
            else:
                evaluate(work, a.modality, a.gpu)
