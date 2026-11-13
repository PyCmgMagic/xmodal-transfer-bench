import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from entityqa_common import digest, save, sha, write_rows
from run_rules_qwen38_rule_given import check_saved, make_record, readiness, summarize, check_quantization_policy


class Qwen38DiagnosticTests(unittest.TestCase):
    def test_quantization_policy_protects_nested_vision_before_loading(self):
        check_quantization_policy(dict(quantization=dict(llm_int8_skip_modules=['model.visual', 'lm_head'])))

    def test_old_vision_skip_is_rejected_before_weight_loading(self):
        with self.assertRaisesRegex(AssertionError, 'Unprotected BF16'):
            check_quantization_policy(dict(quantization=dict(llm_int8_skip_modules=['visual', 'lm_head'])))

    def reply(self, text):
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=text, reasoning_content=None), finish_reason='stop')],
            usage=SimpleNamespace(completion_tokens=8, prompt_tokens=30))

    def task(self):
        return dict(id='question-1', qid='rule-1', modality='T', gold='A',
                    options=dict(A='one', B='two', C='three', D='four'))

    def test_readout_independent_of_gold_and_unambiguous_content_preserved(self):
        payload = dict(messages=[dict(role='user', content='Question')])
        original = copy.deepcopy(payload)
        task = self.task()
        a = make_record(task, payload, self.reply(r'Brief reason. \boxed{B}'), 0.5)
        task['gold'] = 'B'
        b = make_record(task, payload, self.reply(r'Brief reason. \boxed{B}'), 0.5)
        self.assertEqual(a['predicted'], b['predicted'])
        self.assertEqual(a['predicted'], 'B')
        self.assertFalse(a['correct'])
        self.assertTrue(b['correct'])
        self.assertEqual(a['input_sha256'], b['input_sha256'])
        self.assertEqual(payload, original)
        self.assertNotIn('options', a)
        self.assertEqual(a['raw_response'], r'Brief reason. \boxed{B}')

    def test_conflicting_boxes_remain_invalid_without_finalization(self):
        record = make_record(self.task(), dict(messages=[]),
            self.reply(r'\boxed{A} \boxed{B}'), 0.5)
        self.assertIsNone(record['predicted'])
        self.assertFalse(record['correct'])

    def test_record_keeps_actual_raised_cap_and_batch_without_input_changes(self):
        payload = dict(messages=[dict(role='user', content='Question')])
        record = make_record(self.task(), payload, self.reply(r'\boxed{A}'), 0.5,
                             max_tokens=4096, inference_batch_size=4)
        self.assertEqual(record['generation_max_tokens'], 4096)
        self.assertEqual(record['inference_batch_size'], 4)
        self.assertEqual(record['input_sha256'], digest(payload))

    def test_saved_records_require_same_order_gold_and_model_input(self):
        task = self.task()
        payload = dict(messages=[dict(role='user', content='Question')])
        record = make_record(task, payload, self.reply(r'\boxed{A}'), 0.5)
        check_saved([record], [task], [payload])
        for field, value in [('id', 'different'), ('gold', 'B'), ('input_sha256', digest('different'))]:
            changed = dict(record, **{field: value})
            with self.subTest(field=field), self.assertRaises(AssertionError):
                check_saved([changed], [task], [payload])

    def test_queue_requires_complete_32b_not_merely_idle_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            dep, model = Path(tmp) / 'dependency', Path(tmp) / 'model'
            dep.mkdir()
            model.mkdir()
            (model / 'download_complete.json').write_text('{}')
            for stage in ['queued_after_vision_suite', 'evaluating', 'failed']:
                (dep / 'status.json').write_text(json.dumps(dict(stage=stage)))
                state, detail = readiness(dep, model)
                self.assertEqual(state, 'queued_after_qwen32b')
                self.assertEqual(detail['dependency_stage'], stage)
            (dep / 'status.json').write_text(json.dumps(dict(stage='complete')))
            for name in ['experiment.json', 'comparison.json']:
                (dep / name).write_text(json.dumps(dict(status='complete')))
            self.assertEqual(readiness(dep, model)[0], 'ready')
            (model / 'download_complete.json').unlink()
            self.assertEqual(readiness(dep, model)[0], 'waiting_for_model_download')
            (dep / 'comparison.json').write_text(json.dumps(dict(status='partial')))
            with self.assertRaises(AssertionError):
                readiness(dep, model)

    def test_explicit_resume_after_stopped_dependency_requires_saved_hash_proof(self):
        with tempfile.TemporaryDirectory() as tmp:
            dep, model = Path(tmp) / 'dep', Path(tmp) / 'model'
            model.mkdir()
            save(model / 'download_complete.json', {})
            write_rows(dep / 'evaluation/predictions.jsonl', [dict(id='saved')])
            save(dep / 'status.json', dict(stage='stopped_by_user', saved=1))
            save(dep / 'experiment.json', dict(status='stopped_by_user'))
            save(dep / 'comparison_partial.json', dict(status='partial_stopped_by_user', saved=1,
                predictions_sha256=sha(dep / 'evaluation/predictions.jsonl')))
            self.assertEqual(readiness(dep, model)[0], 'queued_after_qwen32b')
            self.assertEqual(readiness(dep, model, allow_stopped_dependency=True)[0], 'ready')
            write_rows(dep / 'evaluation/predictions.jsonl', [dict(id='changed')])
            with self.assertRaises(AssertionError):
                readiness(dep, model, allow_stopped_dependency=True)

    def test_full_new_run_keeps_previous_partial_model_comparison_separate(self):
        from test_rules_qwen32b_rule_given import fixture
        manifest, inputs = fixture()
        records = [make_record(task, payload, self.reply(r'\boxed{A}'), 0.5,
                   max_tokens=4096, inference_batch_size=4) for task, payload in zip(manifest, inputs)]
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            write_rows(work / 'data/task_manifest.jsonl', manifest)
            write_rows(work / 'data/inference_inputs.jsonl', inputs)
            write_rows(work / 'evaluation/predictions.jsonl', records)
            write_rows(work / 'reference/glm_same_questions.jsonl', records)
            write_rows(work / 'reference/qwen32_partial_predictions.jsonl', records[:59])
            summarize(work, dict(max_tokens=4096, limitations=[]))
            report = json.loads((work / 'comparison.json').read_text())
            self.assertEqual(report['status'], 'complete')
            self.assertNotIn('qwen3vl32b_nf4', report['groups'])
            self.assertEqual(report['qwen32_partial_overlap']['tasks'], 59)
            prior = report['qwen32_partial_overlap']['groups']['qwen3vl32b_nf4']
            self.assertEqual(prior['T/all']['n'] + prior['V/all']['n'], 59)
            self.assertEqual(report['groups']['qwen38_27b_nf4']['T/all']['n'], 240)


if __name__ == '__main__':
    unittest.main()
