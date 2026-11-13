#!/usr/bin/env python3
"""Watch an existing GLM training process, then resume checkpoint validation/evaluation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time

from entityqa_common import now, save

ROOT = Path(__file__).resolve().parents[1]


def process_identity(pid):
    proc = Path('/proc') / str(pid)
    try:
        stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] in {'Z', 'X'}:
            return None
        return proc.stat().st_uid, stat[19]
    except FileNotFoundError:
        return None


def run(args):
    work = args.work.resolve()
    identity = process_identity(args.training_pid)
    if identity is None:
        raise RuntimeError('The selected training process is no longer running')
    proc = Path('/proc') / str(args.training_pid)
    command = (proc / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
    assert identity[0] == os.getuid(), 'Training process belongs to another user'
    assert any(value.endswith('/swift/cli/sft.py') for value in command), command
    for option, expected in [('--output_dir', str(work / 'train/TEXT')),
                             ('--model_type', 'glm4v')]:
        assert command.count(option) == 1 and command[command.index(option) + 1] == expected, option
    if args.check_only:
        print(json.dumps(dict(pid=args.training_pid, identity=identity,
                              work=str(work), status='verified')))
        return

    lock = (work / 'continuation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    save(work / 'continuation.json', dict(status='watching_training', pid=os.getpid(),
        training_pid=args.training_pid, training_process_identity=identity, started_at=now(),
        action='Wait for the existing process; validate final checkpoint; run LoRA and base evaluations.'))
    while process_identity(args.training_pid) == identity:
        save(work / 'status.json', dict(stage='training_GLM_TEXT', controller_pid=os.getpid(),
            training_pid=args.training_pid, host=os.uname().nodename, gpu='0',
            log=str(work / 'training_GLM_TEXT.log'), adopted_existing_training=True,
            updated_at=now()))
        time.sleep(30)

    final_checkpoint = work / 'train/TEXT/checkpoint-8505'
    if not all((final_checkpoint / name).is_file() for name in
               ['trainer_state.json', 'adapter_model.safetensors', 'adapter_config.json']):
        raise RuntimeError('Training exited without the final checkpoint; preserving outputs')
    save(work / 'continuation.json', dict(status='resuming_controller', pid=os.getpid(),
        completed_training_pid=args.training_pid, updated_at=now()))
    python = ROOT / '.venv-glm/bin/python'
    os.execv(str(python), [str(python), '-u', str(ROOT / 'scripts/run_entity_glm46_full.py'),
                         '--work', str(work)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--training-pid', type=int, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        if not args.check_only:
            save(args.work / 'continuation.json', dict(status='failed', pid=os.getpid(),
                error=repr(error), updated_at=now()))
        raise
