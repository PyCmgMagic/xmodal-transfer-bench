#!/usr/bin/env python3
"""Matched one-epoch ablation: emphasize MCQ answer loss, suppress suffix loss."""
import argparse
import json
import subprocess
from pathlib import Path

import pilot_v7_rule_mastery as mastery
import run_v7_evaluation as ev
from run_v7_full_experiment import ROOT, environment, final_checkpoint, read, save, sha, train_command
from v7_gpu_resources import reserve


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--gpu', default='0')
    a = ap.parse_args()
    source, out = a.source.resolve(), a.out.resolve()
    plugin = ROOT / 'scripts/v7_answer_content_loss.py'
    expected = {'train_sha256': sha(source / 'train.jsonl'), 'probes_sha256': sha(source / 'probes.jsonl'),
                'epochs': 1, 'learning_rate': 1e-4, 'batch_size': 1, 'gradient_accumulation_steps': 1,
                'loss_scale': 'v7_mcq_content', 'plugin_sha256': sha(plugin), 'source': str(source)}
    out.mkdir(parents=True, exist_ok=True)
    if not (out / 'manifest.json').exists():
        for name in ('train.jsonl', 'probes.jsonl', 'generated_solver_checks.jsonl', 'mcq_checks.jsonl'):
            (out / name).write_bytes((source / name).read_bytes())
        save(out / 'manifest.json', {'configuration': expected, 'source_manifest': read(source / 'manifest.json'),
            'script_sha256': sha(Path(__file__)), 'change': 'Only loss weights: MCQ answer=3, suffix=0.001; retain all semantic supervision and labels.',
            'no_data_or_inference_change': True, 'created_at': ev.now()})
    assert read(out / 'manifest.json')['configuration'] == expected
    assert sha(out / 'train.jsonl') == expected['train_sha256']
    assert sha(out / 'probes.jsonl') == expected['probes_sha256']
    with ev.run_lock(out / '.lock'):
        if read(out / 'report.json').get('status') == 'complete':
            return
        cp = final_checkpoint(out / 'checkpoint', 1)
        if not cp:
            with reserve(mastery.WORK, a.gpu, str(out) + ':training', 16000) as reservation:
                cmd = train_command('qwen3', 'sft', out / 'train.jsonl', out / 'checkpoint', 1)
                for flag, value in [('--gradient_accumulation_steps', '1'), ('--loss_scale', 'v7_mcq_content')]:
                    cmd[cmd.index(flag) + 1] = value
                cmd += ['--external_plugins', str(plugin)]
                with (out / 'train.log').open('a') as log:
                    child = subprocess.Popen(cmd, cwd=ROOT, env=environment(a.gpu, mastery.DATA), stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    reservation.bind(child.pid)
                    save(out / 'state.json', {'status': 'training', 'pid': child.pid, 'command': cmd, 'started_at': ev.now()})
                    code = child.wait()
                if code:
                    save(out / 'state.json', {'status': 'failed', 'returncode': code})
                    raise RuntimeError(f'Loss ablation failed: {code}')
            cp = final_checkpoint(out / 'checkpoint', 1)
            assert cp
        mastery.evaluate(out, cp, a.gpu)


if __name__ == '__main__':
    main()
