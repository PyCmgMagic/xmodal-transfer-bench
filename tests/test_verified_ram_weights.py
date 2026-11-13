import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from verified_ram_weights import GIB, VerifiedRamCache, resident_range, use_verified_ram_weights


def checkpoint(value):
    head = json.dumps(dict(weight=dict(dtype='F32', shape=[2], data_offsets=[0, 8]))).encode()
    head += b' ' * (-len(head) % 8)
    return struct.pack('<Q', len(head)) + head + struct.pack('<2f', value, value + 1)


class FakeResponse(io.BytesIO):
    status = 200

    def __init__(self, value):
        super().__init__(value)
        self.headers = {'Content-Length': str(len(value))}


class FakeOpener:
    def __init__(self, blobs):
        self.blobs, self.calls = blobs, []

    def open(self, url, timeout):
        name = url.rsplit('/', 1)[1]
        self.calls.append(name)
        return FakeResponse(self.blobs[name])


class VerifiedRamWeightsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = self.root / 'model'
        self.model.mkdir()
        self.blobs = {f'model-{i}.safetensors': checkpoint(i) for i in range(2)}
        self.manifest = dict(Data=dict(Files=[dict(Type='blob', Path=name, Size=len(data),
            Sha256=hashlib.sha256(data).hexdigest(), Revision='pinned-revision')
            for name, data in self.blobs.items()]))
        for name, data in self.blobs.items():
            (self.model / name).write_bytes(data)
        self.events = []
        self.opener = FakeOpener(self.blobs)
        self.patch_memory = patch('verified_ram_weights.memory_available', return_value=100 * GIB)
        self.patch_disk = patch('verified_ram_weights.shutil.disk_usage',
                                return_value=SimpleNamespace(free=100 * GIB))
        self.patch_memory.start()
        self.patch_disk.start()
        self.addCleanup(self.patch_memory.stop)
        self.addCleanup(self.patch_disk.stop)

    def cache(self, **options):
        return VerifiedRamCache(self.manifest, self.model, self.root / 'cache',
            reporter=self.events.append, opener=self.opener, rate_bytes=100 * GIB,
            reserve_bytes=0, allow_non_tmpfs=True, **options)

    def test_complete_verified_bytes_only_and_same_local_weights(self):
        cache = self.cache()
        p = cache.stage('model-0.safetensors')
        self.assertEqual(p.read_bytes(), self.blobs[p.name])
        self.assertEqual((self.model / p.name).read_bytes(), self.blobs[p.name])
        self.assertEqual(cache.stats['downloads'], 1)
        self.assertEqual(self.opener.calls, [p.name])
        self.assertEqual(cache.stage(p.name), p)
        self.assertEqual(self.opener.calls, [p.name])
        cache.close()
        self.assertFalse(p.exists())

    def test_wrong_sha_never_exposed_or_kept(self):
        self.opener.blobs = dict(self.blobs)
        self.opener.blobs['model-0.safetensors'] = checkpoint(99)
        cache = self.cache()
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            cache.stage('model-0.safetensors')
        self.assertEqual(list(cache.directory.iterdir()), [])
        self.assertEqual(cache.entries, {})

    def test_eviction_keeps_cache_within_bound(self):
        cache = self.cache(max_bytes=len(checkpoint(0)))
        a = cache.stage('model-0.safetensors')
        b = cache.stage('model-1.safetensors')
        self.assertFalse(a.exists())
        self.assertTrue(b.exists())
        self.assertLessEqual(cache.stats['peak_cached_bytes'], cache.max_bytes)
        cache.close()

    def test_verified_preflight_is_reused_without_network(self):
        cache = self.cache()
        first = cache.stage('model-0.safetensors')
        cache.close(keep_files=True)
        other = self.cache()
        before = len(self.opener.calls)
        self.assertEqual(other.stage(first.name), first)
        self.assertEqual(len(self.opener.calls), before)
        self.assertEqual(other.stats['reused_verified_shards'], 1)
        other.close()

    def test_memory_reserve_stops_our_download(self):
        cache = self.cache()
        with patch('verified_ram_weights.memory_available', return_value=0):
            with self.assertRaises(MemoryError):
                cache.stage('model-0.safetensors')
        self.assertEqual(self.opener.calls, [])

    def test_only_tmpfs_allowed_by_default(self):
        with self.assertRaisesRegex(ValueError, '/dev/shm'):
            VerifiedRamCache(self.manifest, self.model, self.root / 'cache')

    def test_resident_local_tensor_does_not_download(self):
        cache = self.cache()
        source = SimpleNamespace(header={'weight': {'data_offsets': [0, 8]}}, data_start=0,
                                 native=SimpleNamespace(get_tensor=lambda key: ('original', key)))
        with patch('verified_ram_weights.resident_range', return_value=True):
            self.assertEqual(cache.tensor(source, 'model-0.safetensors', 'weight'), ('original', 'weight'))
        self.assertEqual(self.opener.calls, [])
        self.assertEqual(cache.stats['local_resident_tensors'], 1)

    def test_scoped_opener_and_async_setting_are_restored_on_error(self):
        import os
        original = lambda *a, **k: None
        modeling = SimpleNamespace(safe_open=original)
        old_async = os.environ.get('HF_DEACTIVATE_ASYNC_LOAD')
        with patch.dict(sys.modules, {'transformers': SimpleNamespace(modeling_utils=modeling)}):
            with self.assertRaisesRegex(RuntimeError, 'expected'):
                with use_verified_ram_weights(self.model, self.manifest, self.root / 'cache',
                    allow_non_tmpfs=True, opener=self.opener, reporter=self.events.append):
                    self.assertIsNot(modeling.safe_open, original)
                    self.assertEqual(os.environ['HF_DEACTIVATE_ASYNC_LOAD'], '1')
                    raise RuntimeError('expected')
        self.assertIs(modeling.safe_open, original)
        self.assertEqual(os.environ.get('HF_DEACTIVATE_ASYNC_LOAD'), old_async)


if __name__ == '__main__':
    unittest.main()
