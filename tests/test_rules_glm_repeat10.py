import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_rules_glm46_repeat10 import command, optimizer_steps, repeated_rows


class RuleRepeatTrainingTests(unittest.TestCase):
    def test_repetition_preserves_targets_and_unique_ids_without_aliasing(self):
        source = [dict(id='rule', qid='r', messages=[dict(role='user', content='Learn'),
                  dict(role='assistant', content='A rule, not a test answer')]),
                  dict(id='generic', answer='B', messages=[dict(role='user', content='Add'),
                  dict(role='assistant', content='2')])]
        result = repeated_rows(source)
        self.assertEqual(len(result), 20)
        self.assertEqual(len({r['id'] for r in result}), 20)
        self.assertEqual(sum(r['source_sample_id'] == 'rule' for r in result), 10)
        self.assertEqual(result[0]['messages'], source[0]['messages'])
        result[0]['messages'][1]['content'] = 'Changed'
        self.assertEqual(result[2]['messages'][1]['content'], source[0]['messages'][1]['content'])

    def test_training_schedule_and_clean_base(self):
        self.assertEqual(optimizer_steps(12520), 2349)
        cmd = command(Path('/tmp/rule-repeat-test'))
        self.assertEqual(cmd[cmd.index('--num_train_epochs') + 1], '3')
        self.assertEqual(cmd[cmd.index('--dataset') + 1], '/tmp/rule-repeat-test/data/train_text.jsonl')
        self.assertEqual(cmd[cmd.index('--output_dir') + 1], '/tmp/rule-repeat-test/train/TEXT')
        self.assertNotIn('--adapters', cmd)
        self.assertNotIn('--resume_from_checkpoint', cmd)


if __name__ == '__main__':
    unittest.main()
