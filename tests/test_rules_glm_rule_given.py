import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_rules_glm46_rule_given import input_fingerprint, pair_counts, with_rule


class RuleGivenTests(unittest.TestCase):
    def test_exact_learning_material_and_test_payload_preserved(self):
        payload = {'messages': [{'role': 'user', 'content': '<image>\nQuestion\nA. one\nB. two\nC. three\nD. four'}],
                   'images': ['/tmp/test_visual_1.png']}
        original = copy.deepcopy(payload)
        material = 'A batch qualifies if it has at least 4 covers. Worked example: 4 covers qualify.'
        result = with_rule(payload, material)
        self.assertEqual(payload, original)
        self.assertTrue(result['messages'][0]['content'].endswith(payload['messages'][0]['content']))
        self.assertIn('\n' + material + '\n\nQuestion:\n', result['messages'][0]['content'])
        self.assertEqual(result['images'], payload['images'])
        self.assertEqual(set(result), {'messages', 'images'})

    def test_gold_is_not_part_of_model_input_or_fingerprint(self):
        payload = {'messages': [{'role': 'user', 'content': 'Question'}]}
        a = dict(id='q', qid='r', gold='A', payload=with_rule(payload, 'Rule'))
        b = dict(a, gold='D')
        self.assertEqual(input_fingerprint([a]), input_fingerprint([b]))
        self.assertNotIn('gold', a['payload'])

    def test_training_images_and_non_user_payload_rejected(self):
        with self.assertRaises(AssertionError):
            with_rule({'messages': [{'role': 'user', 'content': 'Question'}]}, 'Rule <image>')
        with self.assertRaises(AssertionError):
            with_rule({'messages': [{'role': 'assistant', 'content': 'Answer'}]}, 'Rule')

    def test_paired_outcomes_match_ids_not_list_positions(self):
        records = []
        for i, corrects in enumerate([(True, True), (True, False), (False, True), (False, False)]):
            for mod, correct in zip('TV', corrects):
                records.append(dict(id=str(i), qid='r', gold='A', rule_sha256='hash',
                    modality=mod, correct=correct, predicted='A' if correct else 'B',
                    family='f', question_type='direct'))
        records.reverse()
        counts, details = pair_counts(records)
        self.assertEqual(counts, dict(both_correct=1, text_only_correct=1, vision_only_correct=1, both_wrong=1))
        self.assertEqual(len(details), 4)


if __name__ == '__main__':
    unittest.main()
