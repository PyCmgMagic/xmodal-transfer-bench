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


class MigrationTests(unittest.TestCase):
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
