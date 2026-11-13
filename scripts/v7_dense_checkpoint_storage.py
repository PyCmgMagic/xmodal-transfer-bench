"""Bounded checkpoint storage: full recovery state in scratch, dense exports on disk."""
import fcntl
import hashlib
import json
import os
import re
import shutil
from contextlib import contextmanager
from pathlib import Path


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


@contextmanager
def checkpoint_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def export_dense_checkpoint(source, destination, keep=2):
    """Publish a verified complete dense model before removing older exports.

    DeepSpeed global_step*/ optimizer partitions stay in scratch. Published
    checkpoints support inference; exact optimizer recovery uses scratch.
    """
    source, destination = Path(source), Path(destination)
    assert re.fullmatch(r'checkpoint-\d+', source.name), source
    assert (source / 'config.json').is_file()
    assert (source / 'trainer_state.json').is_file()
    assert list(source.glob('model*.safetensors'))
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / source.name
    if target.exists():
        manifest = json.loads((target / 'dense_export.json').read_text())
        for name, want in manifest['sha256'].items():
            assert file_sha(source / name) == file_sha(target / name) == want
        return target
    partial = destination / ('.' + source.name + '.partial.' + str(os.getpid()))
    assert not partial.exists(), partial
    partial.mkdir()
    try:
        hashes = {}
        for path in sorted(source.iterdir()):
            if not path.is_file() or path.is_symlink() or path.name in ('optimizer.pt',):
                continue
            out = partial / path.name
            shutil.copy2(path, out)
            with out.open('rb') as f:
                os.fsync(f.fileno())
            want = file_sha(path)
            assert file_sha(out) == want, path.name
            hashes[path.name] = want
        manifest = {'source': str(source), 'sha256': hashes,
                    'exact_optimizer_resume': False,
                    'recovery': 'Full DeepSpeed/RNG/scheduler state remains in scratch until training is verified complete.'}
        (partial / 'dense_export.json').write_text(json.dumps(manifest, indent=2) + '\n')
        with (partial / 'dense_export.json').open('rb') as f:
            os.fsync(f.fileno())
        partial.rename(target)
        fd = os.open(destination, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        old = sorted((x for x in destination.iterdir() if x.is_dir() and not x.is_symlink()
                      and re.fullmatch(r'checkpoint-\d+', x.name)),
                     key=lambda x: int(x.name.split('-')[1]))
        for path in old[:-keep]:
            assert (path / 'dense_export.json').is_file()
            shutil.rmtree(path)
        return target
    finally:
        if partial.exists():
            shutil.rmtree(partial)
