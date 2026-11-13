"""Check storage failure recovery without allocating real model/optimizer weights."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import v7_dense_checkpoint_storage as storage


class DenseStorageContract(unittest.TestCase):
    def source(self, root, step):
        p = root / f'checkpoint-{step}'
        p.mkdir()
        (p / 'config.json').write_text('{}')
        (p / 'trainer_state.json').write_text(json.dumps({'global_step': step}))
        (p / 'model.safetensors').write_bytes(f'fake-model-{step}'.encode())
        (p / f'global_step{step}').mkdir()
        (p / f'global_step{step}/optimizer.pt').write_bytes(b'recovery-state')
        return p

    def test_export_is_verified_bounded_and_leaves_optimizer_in_scratch(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            scratch, out = root / 'scratch', root / 'out'
            scratch.mkdir()
            sources = [self.source(scratch, i) for i in (1, 2, 3)]
            for s in sources:
                storage.export_dense_checkpoint(s, out)
            self.assertEqual({p.name for p in out.iterdir()}, {'checkpoint-2', 'checkpoint-3'})
            latest = out / 'checkpoint-3'
            self.assertFalse((latest / 'global_step3').exists())
            self.assertTrue((sources[2] / 'global_step3/optimizer.pt').exists())
            self.assertEqual(storage.export_dense_checkpoint(sources[2], out), latest)
            (latest / 'model.safetensors').write_bytes(b'corrupt')
            with self.assertRaises(AssertionError):
                storage.export_dense_checkpoint(sources[2], out)

    def test_failed_export_preserves_previous_published_checkpoint(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            scratch, out = root / 'scratch', root / 'out'
            scratch.mkdir()
            first = self.source(scratch, 1)
            second = self.source(scratch, 2)
            storage.export_dense_checkpoint(first, out)
            original = storage.shutil.copy2
            def fail_weights(a, b):
                if Path(a).suffix == '.safetensors':
                    raise OSError('simulated disk write failure')
                return original(a, b)
            with mock.patch.object(storage.shutil, 'copy2', side_effect=fail_weights):
                with self.assertRaises(OSError):
                    storage.export_dense_checkpoint(second, out)
            self.assertEqual({p.name for p in out.iterdir()}, {'checkpoint-1'})
            self.assertTrue((second / 'global_step2/optimizer.pt').exists())
            storage.export_dense_checkpoint(second, out)
            self.assertTrue((out / 'checkpoint-2/dense_export.json').is_file())


if __name__ == '__main__':
    unittest.main()
