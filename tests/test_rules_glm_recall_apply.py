import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_rules_glm46_recall_apply import build_tasks, matched_change


def question(qid, modality='T'):
    text = 'Apply ' + qid + '.\nA. one\nB. two\nC. three\nD. four'
    payload = {'messages': [{'role': 'user', 'content': text}]}
    if modality == 'V':
        payload['messages'][0]['content'] = '<image>\n' + text
        payload['images'] = ['/tmp/test_visual_1.png']
    return dict(id=qid + '_question', qid=qid, modality=modality, gold='D', payload=payload)


def memory(qid, text, arm='repeat10'):
    return dict(id=qid, qid=qid, arm=arm, raw_response='<think></think>\n' + text,
                answer_for_judge=text)


class RecallApplyTests(unittest.TestCase):
    def test_uses_recalled_text_by_qid_even_when_it_is_wrong(self):
        questions = [question('r1'), question('r2'), question('r1', 'V')]
        before = copy.deepcopy(questions)
        recalls = [memory('r2', 'An incorrect remembered threshold of 123.'),
                   memory('r1', 'A remembered mapping, not a reference answer.')]
        tasks = build_tasks(questions, recalls, 'repeat10')
        self.assertEqual(questions, before)
        for task, original in zip(tasks, questions):
            text = task['payload']['messages'][0]['content']
            expected = recalls[1 if task['qid'] == 'r1' else 0]['answer_for_judge']
            self.assertIn('\n' + expected + '\n\nQuestion:\n', text)
            self.assertTrue(text.endswith(original['payload']['messages'][0]['content']))
            self.assertEqual(task['payload'].get('images'), original['payload'].get('images'))
            self.assertEqual(set(task['payload']), set(original['payload']))

    def test_no_recall_or_empty_recall_is_not_filtered_or_repaired(self):
        tasks = build_tasks([question('r1'), question('r2')],
                            [memory('r1', ''), memory('r2', 'I do not remember this rule.')], 'repeat10')
        self.assertEqual(len(tasks), 2)
        self.assertIn(':\n\n\nQuestion:\n', tasks[0]['payload']['messages'][0]['content'])
        self.assertIn('I do not remember this rule.', tasks[1]['payload']['messages'][0]['content'])

    def test_missing_duplicate_or_wrong_arm_memory_is_rejected(self):
        cases = [[], [memory('r1', 'one'), memory('r1', 'two')],
                 [memory('r1', 'one', 'repeat1')], [memory('r2', 'unrelated')]]
        for recalls in cases:
            with self.subTest(recalls=recalls), self.assertRaises(AssertionError):
                build_tasks([question('r1')], recalls, 'repeat10')

    def test_gold_and_audit_fields_never_enter_model_input(self):
        original = question('r1')
        original['reference_rule'] = 'DO NOT EXPOSE THE TRUE RULE'
        a = build_tasks([original], [memory('r1', 'Only the recalled rule.')], 'repeat10')[0]
        original['gold'] = 'A'
        b = build_tasks([original], [memory('r1', 'Only the recalled rule.')], 'repeat10')[0]
        self.assertEqual(a['payload'], b['payload'])
        self.assertNotIn('reference_rule', a['payload'])
        self.assertNotIn('DO NOT EXPOSE', a['payload']['messages'][0]['content'])

    def test_matched_changes_use_question_and_modality_not_row_order(self):
        old, new = [], []
        for i, (before, after) in enumerate([(True, True), (False, True), (True, False), (False, False)]):
            row = dict(id=str(i), qid='r', modality='T', gold='A')
            old.append(dict(row, correct=before))
            new.append(dict(row, correct=after))
        new.reverse()
        self.assertEqual(matched_change(old, new),
                         dict(both_correct=1, improved=1, worsened=1, both_wrong=1))


if __name__ == '__main__':
    unittest.main()
