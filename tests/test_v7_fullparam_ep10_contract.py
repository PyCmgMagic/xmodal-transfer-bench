"""Verify the four-direction comparison and fixed final-checkpoint contract."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_v7_fullparam_ep10 as run


class FullDataContract(unittest.TestCase):
    def test_overall_uses_each_trained_direction_once_and_base_input_aliases(self):
        with tempfile.TemporaryDirectory() as name:
            work = Path(name)
            def result(accuracy):
                return {'accuracy': accuracy, 'accuracy_on_valid_only': accuracy,
                        'correct': round(accuracy * 2000), 'total': 2000, 'matched': 2000,
                        'missing_prediction': 0, 'empty_prediction': 0, 'format_violation': 0,
                        'by_question_type': {'direct': {'accuracy': accuracy},
                                             'diagnosis': {'accuracy': accuracy}}}
            for mod, ds in [('text', [('T2T', .4), ('T2V', .3)]),
                            ('vision', [('V2T', .2), ('V2V', .5)]),
                            ('baseline', [('T2T', .25), ('T2V', .35)])]:
                for d, accuracy in ds:
                    run.save(work / 'jobs' / mod / 'evaluation' / d / 'report.json', result(accuracy))
            with mock.patch.object(run, 'plot'):
                run.report(work)
            summary = run.read(work / 'summary.json')
            self.assertEqual(summary['status'], 'complete')
            self.assertEqual([r['direction'] for r in summary['rows']], ['T2T', 'T2V', 'V2T', 'V2V'])
            self.assertEqual(summary['overall']['trained_total'], 8000)
            self.assertEqual(summary['overall']['trained_correct'], 2800)
            self.assertAlmostEqual(summary['overall']['gain'], .05)
            self.assertEqual(summary['overall']['baseline_physical_unique_requests'], 4000)
            self.assertEqual(summary['rows'][2]['baseline_input_alias'], 'T2T')

    def test_final_epoch_is_required_even_if_step_number_matches(self):
        with tempfile.TemporaryDirectory() as name:
            work = Path(name)
            cp = work / 'jobs/text/checkpoint/checkpoint-630'
            run.save(cp / 'trainer_state.json', {'global_step': 630, 'epoch': 9.5})
            run.save(cp / 'dense_export.json', {})
            with self.assertRaises(AssertionError):
                run.final_checkpoint(work, 'text')
            run.save(cp / 'trainer_state.json', {'global_step': 630, 'epoch': 10.})
            self.assertEqual(run.final_checkpoint(work, 'text'), cp)


if __name__ == '__main__':
    unittest.main()
