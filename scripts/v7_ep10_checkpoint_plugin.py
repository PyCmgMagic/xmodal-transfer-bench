"""Serialize scratch checkpoint writes and publish model-only disk checkpoints."""
import json
import os
import shutil
import time
from pathlib import Path

from swift.trainers.mixin import SwiftMixin
from v7_dense_checkpoint_storage import checkpoint_lock, export_dense_checkpoint

_original = SwiftMixin._save_checkpoint


def _save_with_disk_export(self, *args, **kwargs):
    persistent = os.environ.get('V7_EP10_PERSISTENT_CHECKPOINTS')
    if not persistent:
        return _original(self, *args, **kwargs)
    assert self.args.world_size == 1, 'This storage plan is for single-GPU workers.'
    scratch = Path(self.args.output_dir).resolve()
    owner = Path(os.environ['V7_EP10_SCRATCH_ROOT']).resolve()
    assert owner in scratch.parents and str(owner).startswith('/dev/shm/xmodal-v7-ep10-')
    with checkpoint_lock(os.environ['V7_EP10_CHECKPOINT_LOCK']):
        current = scratch / f'checkpoint-{self.state.global_step}'
        published = Path(persistent) / current.name / 'dense_export.json'
        # Transformers can request the same final step from both step-end and
        # epoch-end. No optimizer update occurs between these requests.
        if published.exists() and (current / 'trainer_state.json').exists() \
                and list(current.glob('global_step*/**/*optim_states.pt')):
            self.state.last_model_checkpoint = str(current)
            print(f'XMODAL_CHECKPOINT_ALREADY_EXPORTED step={self.state.global_step}', flush=True)
            return
        # At most one worker writes a new ~68 GiB recovery checkpoint at a time.
        if shutil.disk_usage(scratch).free < 75 * 1024**3:
            raise RuntimeError('Less than 75 GiB scratch available; retaining the previous recoverable checkpoint.')
        mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        if int(mem['MemAvailable'].split()[0]) * 1024 < 90 * 1024**3:
            raise RuntimeError('Insufficient host memory for a complete checkpoint; previous checkpoint retained.')
        started = time.monotonic()
        result = _original(self, *args, **kwargs)
        source = Path(self.state.last_model_checkpoint)
        target = export_dense_checkpoint(source, persistent, keep=2)
        event = {'global_step': self.state.global_step, 'epoch': self.state.epoch,
                 'scratch': str(source), 'persistent': str(target),
                 'elapsed_seconds': time.monotonic() - started}
        with Path(os.environ['V7_EP10_CHECKPOINT_EVENTS']).open('a') as f:
            f.write(json.dumps(event) + '\n')
        print('XMODAL_CHECKPOINT_EXPORTED ' + json.dumps(event), flush=True)
        return result


SwiftMixin._save_checkpoint = _save_with_disk_export
