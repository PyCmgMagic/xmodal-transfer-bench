import copy
from pathlib import Path
import tempfile
import unittest

from entityqa_common import STRICT_SUFFIX, eval_payload as legacy_payload
from entityqa_prompt_neutral_v5 import eval_payload, COT_SUFFIX


class NeutralPromptChecks(unittest.TestCase):
    def test_keeps_question_options_and_uses_no_answer_example(self):
        body = 'Which value matches?\nA. red\nB. blue\nC. green\nD. white'
        row = dict(gold='C', messages=[dict(role='user', content=body + STRICT_SUFFIX)])
        before = copy.deepcopy(row)
        payload = eval_payload(row, '.')
        self.assertEqual(payload, dict(messages=[dict(role='user', content=body + COT_SUFFIX)]))
        self.assertEqual(row, before)
        for letter in 'ABCD':
            self.assertNotIn('\\boxed{' + letter + '}', str(payload))
        self.assertIn('Explain briefly', str(payload))
        self.assertIn('\\boxed{A}', str(legacy_payload(row, '.')))

    def test_preserves_image_and_does_not_forward_audit_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, 'input.png').touch()
            body = '<image>\nWhich one?\nA. red\nB. blue\nC. green\nD. white'
            row = dict(messages=[dict(role='user', content=body + STRICT_SUFFIX)],
                       images=['input.png'], answer='D', gold='D', learning='hidden source')
            payload = eval_payload(row, folder)
            self.assertEqual(set(payload), {'messages', 'images'})
            self.assertEqual(payload['images'], [str(Path(folder, 'input.png').resolve())])
            self.assertEqual(payload['messages'][0]['content'], body + COT_SUFFIX)


if __name__ == '__main__':
    unittest.main()
