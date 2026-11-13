"""Read cold model shards through a bounded, SHA256-verified tmpfs cache.

Only the calling process's Transformers safetensors opener is changed. Existing
resident local tensors remain usable; cold shards come from the pinned official
ModelScope revision. No package, original weight, or system setting is changed.
"""
import argparse
from collections import OrderedDict
from contextlib import contextmanager
import ctypes
import hashlib
import json
import mmap
import os
from pathlib import Path
import shutil
import struct
import threading
import time
import urllib.parse
import urllib.request

MIB = 2 ** 20
GIB = 2 ** 30


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(MIB), b''):
            h.update(chunk)
    return h.hexdigest()


def memory_available():
    values = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    return int(values['MemAvailable'].split()[0]) * 1024


def header(path):
    with Path(path).open('rb') as f:
        prefix = f.read(8)
        size = struct.unpack('<Q', prefix)[0]
        if not 0 < size < 16 * MIB:
            raise ValueError('Unexpected safetensors header size')
        raw = f.read(size)
        if len(raw) != size:
            raise ValueError('Incomplete safetensors header')
        return json.loads(raw), size + 8


def resident_range(path, start, end):
    """mincore checks residency without faulting the file's data pages in."""
    if end == start:
        return True
    page = os.sysconf('SC_PAGE_SIZE')
    aligned = start // page * page
    length = end - aligned
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)]
    with Path(path).open('rb') as f:
        with mmap.mmap(f.fileno(), length, access=mmap.ACCESS_COPY, offset=aligned) as mapping:
            addr = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
            vector = (ctypes.c_ubyte * ((length + page - 1) // page))()
            if libc.mincore(addr, length, vector):
                raise OSError(ctypes.get_errno(), 'mincore failed')
            return all(value & 1 for value in vector)


class VerifiedRamCache:
    def __init__(self, manifest, model_path, directory, *, max_bytes=8 * GIB,
                 rate_bytes=16 * MIB, reserve_bytes=16 * GIB, reporter=None,
                 opener=None, allow_non_tmpfs=False):
        self.manifest = manifest
        self.model_path = Path(model_path).resolve()
        self.directory = Path(directory).resolve()
        if not allow_non_tmpfs and not self.directory.is_relative_to(Path('/dev/shm')):
            raise ValueError('Weight cache must be below /dev/shm')
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.directory.stat().st_uid != os.getuid():
            raise ValueError('Cache directory must belong to the current user')
        self.items = {x['Path']: x for x in manifest['Data']['Files'] if x['Type'] == 'blob'}
        self.max_bytes = max_bytes
        self.rate_bytes = rate_bytes
        self.reserve_bytes = reserve_bytes
        if max_bytes <= 0 or rate_bytes <= 0 or reserve_bytes < 0:
            raise ValueError('Invalid resource limit')
        self.reporter = reporter or (lambda value: print(json.dumps(value), flush=True))
        self.opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.entries = OrderedDict()
        self.lock = threading.RLock()
        self.stats = dict(local_resident_tensors=0, ram_tensors=0, downloads=0,
                          downloaded_bytes=0, reused_verified_shards=0, peak_cached_bytes=0)
        self.original_open = None

    def emit(self, stage, **values):
        self.reporter(dict(component='verified_ram_weights', stage=stage,
                           recorded_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), **values))

    def _evict(self, name):
        entry = self.entries.pop(name)
        if entry.get('pointer') is not None:
            entry['pointer'].__exit__(None, None, None)
        entry['path'].unlink(missing_ok=True)
        self.emit('evicted', file=name)

    def _make_room(self, name):
        size = self.items[name]['Size']
        if size > self.max_bytes:
            raise MemoryError('Single shard exceeds configured RAM cache')
        while self.entries and sum(x['size'] for x in self.entries.values()) + size > self.max_bytes:
            self._evict(next(iter(self.entries)))
        # A completed shard left by a preflight is kept only if it fits this cache.
        for other in self.directory.glob('*.safetensors'):
            if other.name != name and other.name not in self.entries:
                other.unlink()
        if memory_available() < self.reserve_bytes + size:
            raise MemoryError('Insufficient available memory to preserve the configured reserve')
        if shutil.disk_usage(self.directory).free < size + GIB:
            raise MemoryError('Insufficient free tmpfs capacity')

    def stage(self, name):
        """Return a complete verified file; partial or wrong bytes are never exposed."""
        with self.lock:
            if name in self.entries:
                self.entries.move_to_end(name)
                return self.entries[name]['path']
            item = self.items[name]
            if Path(name).name != name or not name.endswith('.safetensors'):
                raise ValueError('Only flat safetensors shard paths are supported')
            target = self.directory / name
            if target.exists() and target.stat().st_size == item['Size'] and file_sha(target) == item['Sha256']:
                self.stats['reused_verified_shards'] += 1
                self.emit('reused_verified', file=name, size=item['Size'], sha256=item['Sha256'])
            else:
                target.unlink(missing_ok=True)
                self._make_room(name)
                partial = target.with_suffix('.safetensors.partial')
                partial.unlink(missing_ok=True)
                url = ('https://modelscope.cn/models/Qwen/Qwen3.8-27B/resolve/'
                       + item['Revision'] + '/' + urllib.parse.quote(name))
                self.emit('download_started', file=name, size=item['Size'],
                          rate_limit_MiB_s=self.rate_bytes / MIB)
                started = time.monotonic()
                count, last = 0, 0
                digest = hashlib.sha256()
                try:
                    with self.opener.open(url, timeout=45) as response, partial.open('xb', buffering=0) as out:
                        if response.status != 200:
                            raise ValueError('Expected a complete shard HTTP response')
                        declared = response.headers.get('Content-Length')
                        if declared is not None and int(declared) != item['Size']:
                            raise ValueError('Upstream shard size changed')
                        while True:
                            chunk = response.read(MIB)
                            if not chunk:
                                break
                            count += len(chunk)
                            if count > item['Size']:
                                raise ValueError('Upstream shard exceeds manifest size')
                            digest.update(chunk)
                            if out.write(chunk) != len(chunk):
                                raise OSError('Short write to our tmpfs cache')
                            self.stats['downloaded_bytes'] += len(chunk)
                            delay = count / self.rate_bytes - (time.monotonic() - started)
                            if delay > 0:
                                time.sleep(delay)
                            if time.monotonic() - last >= 20:
                                if memory_available() < self.reserve_bytes:
                                    raise MemoryError('Memory reserve crossed; aborting our RAM download')
                                self.emit('downloading', file=name, bytes=count, total=item['Size'],
                                          seconds=round(time.monotonic() - started, 1))
                                last = time.monotonic()
                    if (count != item['Size'] or partial.stat().st_size != item['Size']
                            or digest.hexdigest() != item['Sha256']):
                        raise ValueError('Downloaded shard failed manifest size/SHA256 verification')
                    # Metadata must also match the already-frozen local model.
                    if header(partial) != header(self.model_path / name):
                        raise ValueError('Verified shard metadata differs from local model')
                    partial.replace(target)
                    self.stats['downloads'] += 1
                    self.emit('verified', file=name, size=count, sha256=digest.hexdigest(),
                              seconds=round(time.monotonic() - started, 2))
                except BaseException:
                    partial.unlink(missing_ok=True)
                    raise
            self.entries[name] = dict(path=target, size=item['Size'], pointer=None)
            self.stats['peak_cached_bytes'] = max(self.stats['peak_cached_bytes'],
                sum(x['size'] for x in self.entries.values()))
            if self.stats['peak_cached_bytes'] > self.max_bytes:
                self._evict(name)
                raise MemoryError('RAM cache limit exceeded')
            return target

    def tensor(self, source, name, key):
        with self.lock:
            item = source.header[key]
            a, b = (x + source.data_start for x in item['data_offsets'])
            if name not in self.entries and resident_range(self.model_path / name, a, b):
                self.stats['local_resident_tensors'] += 1
                return source.native.get_tensor(key)
            self.stage(name)
            entry = self.entries[name]
            if entry['pointer'] is None:
                entry['pointer'] = self.original_open(str(entry['path']), framework='pt', device='cpu')
            self.stats['ram_tensors'] += 1
            return entry['pointer'].get_tensor(key)

    def close(self, keep_files=False):
        with self.lock:
            for name in list(self.entries):
                if keep_files:
                    entry = self.entries.pop(name)
                    if entry['pointer'] is not None:
                        entry['pointer'].__exit__(None, None, None)
                else:
                    self._evict(name)
            self.emit('closed', **self.stats)


class CachedSlice:
    def __init__(self, owner, key):
        self.owner, self.key = owner, key
        self.native = owner.native.get_slice(key)

    def __getattr__(self, name):
        return getattr(self.native, name)

    def __getitem__(self, index):
        return self.owner.cache.tensor(self.owner, self.owner.name, self.key)[index]


class CachedOpen:
    def __init__(self, cache, path, *args, **kwargs):
        self.cache = cache
        self.name = Path(path).name
        self.native = cache.original_open(path, *args, **kwargs)
        self.header, self.data_start = header(path)

    def __getattr__(self, name):
        return getattr(self.native, name)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self.native.__exit__(*args)

    def get_slice(self, key):
        return CachedSlice(self, key)

    def get_tensor(self, key):
        return self.cache.tensor(self, self.name, key)


@contextmanager
def use_verified_ram_weights(model_path, manifest, directory, **kwargs):
    from transformers import modeling_utils
    original = modeling_utils.safe_open
    cache = VerifiedRamCache(manifest, model_path, directory, **kwargs)
    cache.original_open = original

    def open_weights(path, *args, **options):
        p = Path(path).resolve()
        if p.parent == cache.model_path and p.name.endswith('.safetensors'):
            return CachedOpen(cache, str(p), *args, **options)
        return original(path, *args, **options)

    previous_async = os.environ.get('HF_DEACTIVATE_ASYNC_LOAD')
    os.environ['HF_DEACTIVATE_ASYNC_LOAD'] = '1'
    modeling_utils.safe_open = open_weights
    try:
        yield cache
    finally:
        modeling_utils.safe_open = original
        if previous_async is None:
            os.environ.pop('HF_DEACTIVATE_ASYNC_LOAD', None)
        else:
            os.environ['HF_DEACTIVATE_ASYNC_LOAD'] = previous_async
        cache.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--stage-shard', required=True)
    args = parser.parse_args()
    cache = VerifiedRamCache(json.loads(args.manifest.read_text()), args.model_path, args.directory)
    cache.stage(args.stage_shard)
    cache.close(keep_files=True)
