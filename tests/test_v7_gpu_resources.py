"""Memory admission includes pending loads, external users and descendants."""
import sys
import tempfile
import unittest
import json
import os
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import v7_gpu_resources as gpu
import run_v7_evaluations as evaluations


class GPUAdmission(unittest.TestCase):
    def test_legacy_evaluation_child_is_discovered_without_restarting(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp)/'work';out=work/'evaluations';out.mkdir(parents=True)
            proc=Path(tmp)/'proc';child=proc/'123';child.mkdir(parents=True)
            cmd=['python','/scripts/run_v7_adapter_eval.py','--work',str(work),'--model','qwen3',
                 '--objective','sft','--modality','vision','--gpu','1','--protocols','strict']
            (child/'cmdline').write_bytes(b'\0'.join(x.encode() for x in cmd)+b'\0')
            with mock.patch.object(evaluations,'Path',side_effect=lambda p:proc if p=='/proc' else Path(p)):
                evaluations.discover_children(work,out)
            state=json.loads((out/'qwen3_sft_vision/launcher_state.json').read_text())
            self.assertEqual(state['pid'],123)
            self.assertEqual(state['command'],cmd)
            self.assertEqual(state['status'],'running')

    def test_unused_job_budget_and_external_memory_are_reserved(self):
        entries = {'train': {'peak_mib': 22000}, 'eval': {'peak_mib': 28000}}
        owned = {'train': 17000, 'eval': 24000}
        # Used includes another user's 10 GB. Only 17 GB is available.
        self.assertEqual(gpu.available_mib(81000, 51000, entries, owned), 17000)
        self.assertEqual(gpu.available_mib(81000, 51000, entries, {}), -24000)

    def test_child_cuda_process_is_charged_to_wrapper(self):
        entries = {'train': {'pid': 10}, 'eval': {'pid': 20}}
        with mock.patch.object(gpu, 'parent', side_effect=lambda p: {11:10, 21:20}.get(p, 1)):
            self.assertEqual(gpu.owned_memory(entries, [(11, 17000), (21, 24000), (30, 10000)]),
                             {'train':17000, 'eval':24000})

    def test_pending_model_load_blocks_second_admission(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(gpu, 'snapshot', return_value=(12000, 0, [])):
            with gpu.reserve(tmp, 0, 'pending', 8000, poll=0):
                self.assertFalse(gpu.can_admit(tmp, 0, 1))
                self.assertTrue(gpu.can_admit(tmp, 1, 8000))
                with self.assertRaisesRegex(RuntimeError, 'already active'):
                    with gpu.reserve(tmp, 0, 'pending', 1, adopted_pid=99999999):
                        pass
            self.assertTrue(gpu.can_admit(tmp, 0, 8000))

    def test_coordinator_lease_covers_sequential_workers_and_releases(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
                gpu, 'snapshot', return_value=(46068, 0, [])):
            with gpu.reserve(tmp, 0, 'comparison', 41000, poll=0) as lease:
                lease.bind(os.getpid())
                for arm in ('multiform10', 'repeat10'):
                    self.assertFalse(gpu.can_admit(tmp, 0, 18000), arm)
                    with gpu.ledger(tmp) as entries:
                        self.assertEqual(entries['comparison']['pid'], os.getpid())
            self.assertTrue(gpu.can_admit(tmp, 0, 41000))

if __name__ == '__main__':
    unittest.main()
