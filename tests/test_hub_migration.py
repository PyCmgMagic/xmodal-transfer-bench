"""Portable release checks without GPU, network, or original project assets."""
import copy
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import fetch_hub_assets as fetch
import manage_hub_release as release
import train_rules_full_sft_ep10 as trainer
from hf_upload_transport import RequestGate, RateLimitedAdapter, retry_delay, is_hub_api


class MigrationTests(unittest.TestCase):
    def test_upload_workers_share_one_request_pacing_gate(self):
        now = [0.0]
        def sleep(seconds):
            now[0] += seconds
        gate = RequestGate(interval=0.5, clock=lambda: now[0], sleep=sleep)
        gate.wait()
        gate.wait()
        gate.wait()
        self.assertEqual(now[0], 1.0)

    def test_hf_429_waits_before_retrying(self):
        import requests
        now = [0.0]
        gate = RequestGate(0.5, clock=lambda: now[0],
                           sleep=lambda seconds: now.__setitem__(0, now[0] + seconds))
        responses = []
        for status in [429, 200]:
            response = requests.Response()
            response.status_code = status
            response.headers['Retry-After'] = '10'
            response._content = b''
            response._content_consumed = True
            responses.append(response)
        request = requests.Request('POST', 'https://huggingface.co/api/datasets/owner/data/preupload/main').prepare()
        with patch('requests.adapters.HTTPAdapter.send', side_effect=responses) as send:
            result = RateLimitedAdapter(gate).send(request)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(send.call_count, 2)
        self.assertGreaterEqual(now[0], 12)

    def test_rate_limit_reset_header_and_default(self):
        self.assertEqual(retry_delay({'RateLimit': '"api";r=0;t=123'}), 123)
        self.assertEqual(retry_delay({}), 300)

    def test_binary_storage_transfer_does_not_consume_api_pacing(self):
        self.assertFalse(is_hub_api('https://storage.example.com/blob'))
        self.assertTrue(is_hub_api('https://huggingface.co/datasets/owner/data.git/info/lfs/objects/batch'))

    def test_relative_project_root_is_supported(self):
        self.assertEqual(fetch.safe_target(Path('.'), 'README.md'), Path('README.md').resolve())

    def test_images_relocate_without_changing_messages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'unit').mkdir()
            (root / 'unit/a.png').write_bytes(b'unchanged image')
            sample = {'messages': [{'role': 'user', 'content': '<image>rule'}],
                      'images': ['images/unit/a.png']}
            expected = copy.deepcopy(sample['messages'])
            with patch.dict(os.environ, {'XMODAL_IMAGE_ROOT': tmp}):
                actual = trainer.relocate_images([sample])
            self.assertEqual(actual[0]['messages'], expected)
            self.assertEqual(actual[0]['images'], [str(root / 'unit/a.png')])

    def test_images_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {'XMODAL_IMAGE_ROOT': tmp}):
                with self.assertRaises(ValueError):
                    trainer.relocate_images([{'images': ['images/../../secret']}])
            with self.assertRaises(ValueError):
                fetch.safe_target(Path(tmp), '../secret')

    def test_manifest_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'sample.json').write_text('{"x":1}')
            release.write_manifest(root)
            self.assertEqual(fetch.verify_manifest(root)['status'], 'pass')
            (root / 'sample.json').write_text('{"x":2}')
            with self.assertRaises(ValueError):
                fetch.verify_manifest(root)

    def test_shard_index_detects_truncated_tensor_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            header = json.dumps({'weight': {
                'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}).encode()
            shard = root / 'model.safetensors'
            shard.write_bytes(struct.pack('<Q', len(header)) + header + b'\0' * 4)
            (root / 'model.safetensors.index.json').write_text(
                json.dumps({'weight_map': {'weight': shard.name}}))
            self.assertEqual(release.validate_shards(root)['indexed_tensors'], 1)
            shard.write_bytes(struct.pack('<Q', len(header)) + header + b'\0')
            with self.assertRaises(ValueError):
                release.validate_shards(root)

    def test_token_scanner_does_not_echo_secret(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            secret = 'hf_' + 'a' * 30
            (root / 'oops.txt').write_text(secret)
            result = release.scan_code(root)
            self.assertEqual(result[0]['kind'], 'credential_pattern')
            self.assertNotIn(secret, json.dumps(result))

    def test_download_uses_and_reuses_resolved_commit(self):
        from types import SimpleNamespace
        calls = []
        class Api:
            def repo_info(self, repo_id, **kwargs):
                calls.append(kwargs['revision'])
                return SimpleNamespace(sha='a' * 40)
        fake = SimpleNamespace(HfApi=Api, snapshot_download=lambda *a, **k: None)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {'huggingface_hub': fake}):
            root = Path(tmp)
            config = {'base': {'repo_id': 'owner/base', 'repo_type': 'model',
                               'revision': None, 'local_dir': 'models/base'}}
            lock = root / 'assets.lock.json'
            fetch.fetch(config, ['base'], root, lock, 1)
            fetch.fetch(config, ['base'], root, lock, 1)
            self.assertEqual(calls, ['main', 'a' * 40])
            self.assertEqual(json.loads(lock.read_text())['assets']['base']['status'], 'complete')


if __name__ == '__main__':
    unittest.main()
