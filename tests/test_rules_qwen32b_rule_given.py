import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_rules_qwen32b_rule_given import (batch_budget, checked_replies, environment,
    generation_budget, pending_indices, readiness, select_tasks, summaries, validate_download)
from entityqa_common import digest, save, sha


def fixture():
    manifest, inputs = [], []
    options = dict(A='one', B='two', C='three', D='four')
    for i in range(120):
        for kind in ['direct', 'diagnosis']:
            for mod in 'TV':
                manifest.append(dict(id=f'{i}-{kind}', qid=str(i), modality=mod,
                    family='transformation_procedure', question_type=kind,
                    gold='ABCD'[i % 4], options=copy.deepcopy(options), rule_sha256=str(i)))
                text = (f'Reference rule for this question (use the supplied rule):\nRule {i}'
                        '\n\nQuestion:\n' + ('<image>' if mod == 'V' else '') +
                        'Question\nA. one\nB. two\nC. three\nD. four')
                p = dict(messages=[dict(role='user', content=text)])
                if mod == 'V':
                    p['images'] = [f'/images/test_visual_{i}_{kind}.png']
                inputs.append(p)
    return manifest, inputs


class LargerModelDiagnosticTests(unittest.TestCase):
    def test_preserves_original_inputs_and_order(self):
        manifest, inputs = fixture()
        old = copy.deepcopy(inputs)
        selected, payloads = select_tasks(manifest, inputs)
        self.assertEqual(selected, manifest)
        self.assertEqual(payloads, old)
        self.assertEqual(inputs, old)
        self.assertNotIn('gold', payloads[0])

    def test_rejects_mismatched_pair_labels_rules_or_options(self):
        for field, value in [('gold', 'D'), ('rule_sha256', 'other'),
                             ('options', dict(A='wrong', B='two', C='three', D='four'))]:
            manifest, inputs = fixture()
            manifest[1][field] = value
            with self.subTest(field=field), self.assertRaises(AssertionError):
                select_tasks(manifest, inputs)

    def test_rejects_training_image_and_label_in_model_payload(self):
        for alteration in ['image', 'gold']:
            manifest, inputs = fixture()
            if alteration == 'image':
                inputs[1]['images'] = ['/images/learning_visual.png']
            else:
                inputs[0]['gold'] = 'A'
            with self.subTest(alteration=alteration), self.assertRaises(AssertionError):
                select_tasks(manifest, inputs)

    def test_queue_never_runs_after_failed_or_partial_dependency(self):
        with tempfile.TemporaryDirectory() as tmp:
            dep, model = Path(tmp) / 'dep', Path(tmp) / 'model'
            dep.mkdir()
            model.mkdir()
            (model / 'download_complete.json').write_text('{}')
            for stage in ['failed', 'closed_repeat1', 'judge', 'apply_repeat10']:
                (dep / 'status.json').write_text(json.dumps(dict(stage=stage)))
                self.assertEqual(readiness(dep, model)[0], 'queued_after_vision_suite')
            (dep / 'status.json').write_text(json.dumps(dict(stage='complete')))
            for name in ['experiment.json', 'comparison.json']:
                (dep / name).write_text(json.dumps(dict(status='complete')))
            self.assertEqual(readiness(dep, model)[0], 'ready')
            (model / 'download_complete.json').unlink()
            self.assertEqual(readiness(dep, model)[0], 'waiting_for_model_download')
            (dep / 'comparison.json').write_text(json.dumps(dict(status='partial')))
            with self.assertRaises(AssertionError):
                readiness(dep, model)

    def test_invalid_answers_are_in_accuracy_denominator(self):
        records = [dict(modality='V', question_type='direct', predicted=p, gold='A',
                        stop_reason='length' if p is None else 'stop', parse_reason='test')
                   for p in ['A', 'B', None]]
        result = summaries(records)['V/all']
        self.assertEqual(result['accuracy'], 1 / 3)
        self.assertEqual(result['invalid_rate'], 1 / 3)
        self.assertEqual(result['length'], 1)

    def test_cpu_environment_clears_inherited_pixel_limits(self):
        before = dict(os.environ)
        try:
            os.environ.update(MAX_PIXELS='262144', IMAGE_MAX_TOKEN_NUM='256')
            child = environment(cpu=True)
            self.assertEqual(child['CUDA_VISIBLE_DEVICES'], '')
            self.assertNotIn('MAX_PIXELS', child)
            self.assertNotIn('IMAGE_MAX_TOKEN_NUM', child)
            self.assertEqual(os.environ['MAX_PIXELS'], '262144')
        finally:
            os.environ.clear()
            os.environ.update(before)

    def test_resume_with_archived_truncations_preserves_question_identity(self):
        manifest, inputs = fixture()
        # The middle answer was archived for retry; later completed answers remain valid.
        saved = [dict(id=manifest[i]['id'], modality=manifest[i]['modality'],
                      gold=manifest[i]['gold'], input_sha256=digest(inputs[i])) for i in [0, 2, 3]]
        expected = [1] + list(range(4, 480))
        self.assertEqual(pending_indices(manifest, inputs, saved), expected)
        self.assertEqual(pending_indices(manifest, inputs, list(reversed(saved))), expected)
        with self.assertRaises(AssertionError):
            pending_indices(manifest, inputs, saved + [saved[0]])
        tampered = copy.deepcopy(saved)
        tampered[0]['input_sha256'] = 'different input'
        with self.assertRaises(AssertionError):
            pending_indices(manifest, inputs, tampered)

    def test_raised_budget_fits_audited_full_prompt(self):
        self.assertEqual(generation_budget(dict(max_tokens=4096), dict(max_formatted_tokens=2092)), 4096)
        for cap in [0, -1, 16384, 1.5]:
            with self.subTest(cap=cap), self.assertRaises(AssertionError):
                generation_budget(dict(max_tokens=cap), dict(max_formatted_tokens=2092))

    def test_batching_preserves_reply_order_and_rejects_partial_success(self):
        replies = [object(), object()]
        self.assertIs(checked_replies(replies, 2), replies)
        with self.assertRaises(AssertionError):
            checked_replies(replies, 3)
        with self.assertRaisesRegex(RuntimeError, 'failed item'):
            checked_replies([replies[0], RuntimeError('failed item')], 2)
        self.assertEqual(batch_budget(dict(batch_size=8)), 8)
        self.assertEqual(batch_budget({}), 1)
        for value in [0, -1, 9, 1.5]:
            with self.subTest(value=value), self.assertRaises(AssertionError):
                batch_budget(dict(batch_size=value))

    def test_integrity_cache_rejects_changed_weights_even_with_same_size_and_mtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            work, model = Path(tmp) / 'work', Path(tmp) / 'model'
            model.mkdir()
            weights = model / 'model.safetensors'
            weights.write_bytes(b'original')
            item = dict(Path=weights.name, Type='blob', Size=8, Sha256=sha(weights), Revision='test')
            manifest = dict(Data=dict(Files=[item]))
            save(model / 'download_complete.json', dict(model='test/model', sha256_verified=True))
            save(model / 'download_manifest.json', manifest)
            save(work / 'reference/model_download_manifest.json', manifest)
            plan = dict(model='test/model', model_path=str(model))
            validate_download(work, plan)
            self.assertTrue((work / 'model_integrity_cache.json').exists())
            validate_download(work, plan)
            self.assertTrue((work / 'model_integrity_resume.json').exists())
            before = weights.stat()
            # Some filesystems update ctime at a clock tick rather than on each write.
            time.sleep(0.03)
            weights.write_bytes(b'modified')
            os.utime(weights, ns=(before.st_atime_ns, before.st_mtime_ns))
            with self.assertRaises(AssertionError):
                validate_download(work, plan)


if __name__ == '__main__':
    unittest.main()
