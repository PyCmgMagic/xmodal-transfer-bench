"""Checks for answer permutations and interruption-safe prediction reuse."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import entity_v19_experiment_data as data
import run_entity_v19_full_experiment as run


class ProtocolTests(unittest.TestCase):
    def test_rotation_tracks_gold_and_preserves_stem_suffix(self):
        original = data.rows(data.DATA / 'T2T/eval.jsonl')[0]
        text = original['messages'][-1]['content']
        gold = data.rows(data.DATA / 'audit/eval_labels.jsonl')[0]['answer']
        options = [text.split(f'\n{x}. ', 1)[1].split('\n', 1)[0] for x in 'ABCD']
        for r in range(4):
            changed = data.rotate(text, r)
            new_gold = 'ABCD'[('ABCD'.index(gold) - r) % 4]
            new_text = changed.split(f'\n{new_gold}. ', 1)[1].split('\n', 1)[0]
            self.assertEqual(new_text, options['ABCD'.index(gold)])
            self.assertEqual(changed.split('\nA. ')[0], text.split('\nA. ')[0])
            self.assertEqual(changed.split('\n\nAnswer')[1], text.split('\n\nAnswer')[1])
        self.assertEqual(data.rotate(text, 0), text)

    def test_resume_keeps_invalid_output_and_deduplicates(self):
        ev = data.evaluator()
        a = ev.prepare_row(data.rows(data.DATA / 'T2T/eval.jsonl')[0], data.DATA, 'T2T')
        b = ev.prepare_row(data.rows(data.DATA / 'T2T/eval.jsonl')[1], data.DATA, 'T2T')
        duplicate = copy.deepcopy(a)
        duplicate['id'] += '_duplicate'
        calls = []

        def fake_process(cmd, *args):
            values = data.rows(cmd[cmd.index('--val_dataset') + 1])
            self.assertTrue(all(set(v) <= {'messages', 'images'} for v in values))
            calls.append(len(values))
            # First attempt is interrupted after one INVALID response. The
            # invalid observation must count and must never be re-answered.
            selected = values[:1] if len(calls) == 1 else values
            output = []
            for v in selected:
                response = 'Explanation A' if len(calls) == 1 else 'B'
                output.append({**v, 'messages': v['messages'] + [{'role': 'assistant', 'content': response}],
                               'response': response})
            data.write_rows(Path(cmd[cmd.index('--result_path') + 1]), output)
            return 1 if len(calls) == 1 else 0

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            with patch.object(run, 'process', side_effect=fake_process), patch.object(run.STOP, 'wait'):
                output = run.infer(folder, 'qwen3', '0', folder / 'inference', [a, duplicate, b])
            self.assertEqual(calls, [2, 1])
            self.assertEqual([p['format_valid'] for p in output], [False, False, True])
            self.assertEqual(output[0]['response'], output[1]['response'])
            with patch.object(run, 'process', side_effect=AssertionError('Completed rows re-run')):
                again = run.infer(folder, 'qwen3', '0', folder / 'inference', [a, duplicate, b])
            self.assertEqual(output, again)

    def test_parsers_do_not_extract_letters_from_explanations(self):
        ev = data.evaluator()
        self.assertIsNone(ev.letter('The answer is A', 'strict'))
        self.assertEqual(ev.letter('<think> </think><|begin_of_box|>B<|end_of_box|>', 'glm-box'), 'B')
        self.assertIsNone(ev.letter('<think>I think A</think><|begin_of_box|>B<|end_of_box|>', 'glm-box'))

    def test_frozen_controls_are_paired_and_gold_is_balanced(self):
        root = data.DEFAULT_WORK / 'data'
        if not (root / 'labels.jsonl').exists():
            self.skipTest('Frozen experiment data unavailable')
        labels = data.rows(root / 'labels.jsonl')
        self.assertEqual(len(labels), 23696)
        mapping = data.read(root / 'control_sample.json')['wrong_image_unit']
        self.assertEqual(set(mapping), set(mapping.values()))
        self.assertTrue(all(k != v for k, v in mapping.items()))
        from collections import defaultdict
        golds = defaultdict(list)
        for row in labels:
            golds[row['test'], row['source_id']].append(row['answer'])
        self.assertTrue(all(sorted(v) == list('ABCD') for v in golds.values()))

    def test_statistics_average_within_unit_and_correct_family(self):
        import numpy as np
        import analyze_entity_v19_full_experiment as analysis
        records = []
        for uid, direct, statement in (('u1', [1, 1, 0, 0], [1, 1, 1, 1]),
                                       ('u2', [0, 0, 0, 0], [0, 0, 1, 1])):
            for probe, answers in (('direct', direct), ('statement', statement)):
                for rotation, correct in enumerate(answers):
                    records.append(dict(unit_id=uid, probe_id=probe, rotation=rotation, correct=correct))
        np.testing.assert_allclose(analysis.vector(records, ['u1', 'u2'], 'overall'), [.75, .25])
        np.testing.assert_allclose(analysis.vector(records, ['u1', 'u2'], 'direct'), [.5, 0.])
        zero = analysis.contrast([0., 0.], np.random.default_rng(42))
        self.assertEqual(zero['difference_pp'], 0)
        self.assertEqual(zero['ci95_pp'], [0., 0.])
        self.assertEqual(zero['p_bootstrap_centered'], 1.)
        family = {'a': {'p_bootstrap_centered': .01}, 'b': {'p_bootstrap_centered': .04},
                  'c': {'p_bootstrap_centered': .03}}
        analysis.holm(family)
        self.assertEqual([family[k]['p_holm'] for k in 'abc'], [.03, .06, .06])

    def test_single_seed_scope_excludes_existing_other_seed_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            data.save(work / 'evaluation_scope.json', {'evaluation_seeds': [42]})
            scope = data.evaluation_scope(work)
            self.assertEqual(scope['expected_predictions']['total'], 220576)
            self.assertEqual(scope['expected_model_states'], 22)
            self.assertTrue(data.in_evaluation_scope(work / 'base/qwen3/progress.json', work))
            self.assertTrue(data.in_evaluation_scope(work / 'evaluation/qwen3/seed_42/TEXT/epoch_5/progress.json', work))
            self.assertFalse(data.in_evaluation_scope(work / 'evaluation/qwen3/seed_43/TEXT/epoch_5/progress.json', work))
            self.assertFalse(data.in_evaluation_scope(work / 'evaluation/qwen3/seed_44/TEXT/epoch_5/progress.json', work))
            self.assertFalse(data.in_evaluation_scope(work / 'preflight/qwen3/TEXT/reload/progress.json', work))

    def test_single_seed_queue_keeps_base_curves_endpoint_controls(self):
        if not (data.DEFAULT_WORK / 'data/eval_controls.jsonl').exists():
            self.skipTest('Frozen experiment data unavailable')
        scope = {'evaluation_seeds': [42]}
        with patch.object(run, 'evaluation_scope', return_value=scope), \
             patch.object(run, 'infer') as infer, patch.object(run, 'score_predictions'), \
             patch.object(run, 'update_report'):
            run.evaluate_model(data.DEFAULT_WORK, 'qwen3', '0')
        self.assertEqual(len(infer.call_args_list), 11)
        self.assertEqual(sum(len(c.args[4]) for c in infer.call_args_list), 110288)
        for c in infer.call_args_list:
            self.assertNotIn('seed_43', c.args[3].parts)
            self.assertNotIn('seed_44', c.args[3].parts)


if __name__ == '__main__':
    unittest.main()
