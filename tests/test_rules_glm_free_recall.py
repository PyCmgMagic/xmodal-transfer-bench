import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_rules_glm46_free_recall import (answer_text, controls, enforce_recording_evidence,
    judge_payload, parse_case_judgment, parse_judgment, recall_payload, recording_codes)


class FreeRuleRecallTests(unittest.TestCase):
    def test_recall_contains_only_original_user_prompt(self):
        row = dict(messages=[dict(role='user', content='Explain rule Zenta.'),
                             dict(role='assistant', content='SECRET_REFERENCE')], gold='A')
        payload = recall_payload(row)
        self.assertEqual(payload, dict(messages=[dict(role='user', content='Explain rule Zenta.')]))
        self.assertNotIn('SECRET_REFERENCE', json.dumps(payload))
        payload['messages'][0]['content'] = 'mutated'
        self.assertEqual(row['messages'][0]['content'], 'Explain rule Zenta.')

    def test_judge_sees_reference_candidate_without_arm_identity(self):
        payload = judge_payload('reference', 'candidate', False)
        content = json.loads(payload['messages'][1]['content'])
        self.assertEqual(content, dict(reference_rule='reference', candidate_answer='candidate', recording_required=False))
        self.assertNotIn('repeat10', json.dumps(payload))

    def test_valid_semantic_judgment_and_glm_empty_prefix(self):
        obj = dict(verdict='correct', core_correct=True, complete=True, recording_correct=None,
                   missing_points=[], wrong_points=[], reason='Equivalent rule with reordered conditions.')
        result, reason = parse_judgment('<think></think>\n' + json.dumps(obj), False)
        self.assertEqual(result, obj)
        self.assertEqual(reason, 'valid')

    def test_parser_accepts_box_and_code_fence_wrappers(self):
        obj = dict(verdict='correct', core_correct=True, complete=True, recording_correct=None,
                   missing_points=[], wrong_points=[], reason='All conditions are preserved.')
        encoded = json.dumps(obj)
        raw = '<think></think>\n' + encoded + '</think>\n<|begin_of_box|>' + encoded + '<|end_of_box|>'
        self.assertEqual(parse_judgment(raw, False)[0], obj)
        self.assertEqual(parse_judgment('```json\n' + encoded + '\n```', False)[0], obj)

    def test_incorrect_threshold_can_be_described_as_a_missing_point(self):
        obj = dict(verdict='incorrect', core_correct=False, complete=False, recording_correct=None,
                   missing_points=['The minimum is 4 rather than 5.'], wrong_points=[], reason='Wrong threshold.')
        self.assertEqual(parse_judgment(json.dumps(obj), False)[0], obj)

    def test_malformed_or_inconsistent_judge_output_is_unresolved(self):
        obj = dict(verdict='correct', core_correct=True, complete=False, recording_correct=None,
                   missing_points=[], wrong_points=[], reason='reason')
        for raw in ['not json', '{}', json.dumps(obj), json.dumps(dict(obj, complete=True, core_correct='true'))]:
            self.assertIsNone(parse_judgment(raw, False)[0])
        self.assertIsNone(parse_judgment(json.dumps(dict(obj, complete=True)), True)[0])

    def test_uncertainty_is_distinct_from_wrong_recall(self):
        obj = dict(verdict='uncertain', core_correct=False, complete=False, recording_correct=None,
                   missing_points=[], wrong_points=[], reason='Reference is ambiguous.')
        self.assertEqual(parse_judgment(json.dumps(obj), False)[0]['verdict'], 'uncertain')

    def test_unfinished_thought_is_not_a_completed_recall(self):
        self.assertEqual(answer_text('<think>possibly something')[0], '')
        self.assertEqual(answer_text('<think></think>real recalled rule')[0], 'real recalled rule')

    def test_calibration_includes_paraphrase_threshold_and_code_errors(self):
        cases = controls()
        self.assertEqual(len(cases), 8)
        self.assertEqual(len({c['id'] for c in cases}), 8)
        self.assertEqual({c['expected_verdict'] for c in cases}, {'correct', 'partial', 'incorrect', 'no_recall'})
        self.assertIn('at least 5', next(c['candidate'] for c in cases if c['id'] == 'wrong_threshold'))

    def test_missing_recording_codes_cannot_be_marked_complete(self):
        case = next(c for c in controls() if c['id'] == 'missing_recording')
        self.assertEqual(set(recording_codes(case['reference'])), {'H7', 'J2', 'K9', 'M4'})
        model_judgment = dict(verdict='correct', core_correct=True, complete=True, recording_correct=True,
                              missing_points=[], wrong_points=[], reason='Looks complete.')
        judgment, check = enforce_recording_evidence(model_judgment, case['reference'], case['candidate'], True)
        self.assertEqual(judgment['verdict'], 'partial')
        self.assertFalse(judgment['complete'])
        self.assertFalse(judgment['recording_correct'])
        self.assertEqual(set(check['missing_codes']), {'H7', 'J2', 'K9', 'M4'})

    def test_missing_recording_null_is_resolved_by_exact_code_gate(self):
        case = next(c for c in controls() if c['id'] == 'missing_recording')
        obj = dict(verdict='correct', core_correct=True, complete=True, recording_correct=None,
                   missing_points=[], wrong_points=[], reason='Candidate matches the core rule.')
        result, reason = parse_case_judgment(json.dumps(obj), case)
        self.assertEqual(reason, 'missing_recording_codes_confirmed_deterministically')
        self.assertFalse(result['recording_correct'])


if __name__ == '__main__':
    unittest.main()
