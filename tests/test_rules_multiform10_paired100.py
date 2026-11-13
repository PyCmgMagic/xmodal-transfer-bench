"""Validate the paired sampling contract without loading a model."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_rules_multiform10_paired100 as runner


class PairedSampleContract(unittest.TestCase):
    def setUp(self):
        self.rows = {'T2T': [], 'T2V': []}
        self.labels = {}
        for index in range(75):
            qid = f'rule-{index:03}'
            for kind in ['direct', 'diagnosis']:
                sid = qid + '-' + kind
                self.labels[sid] = {'question_type': kind, 'answer': 'A'}
                for direction in self.rows:
                    self.rows[direction].append({'id': sid, 'qid': qid,
                                                 'messages': [{'role': 'user', 'content': direction + sid}]})

    def test_pairs_are_reproducible_across_input_order_and_gold_changes(self):
        selected, qids = runner.select_paired_sample(self.rows, self.labels)
        reordered = {d: list(reversed(rs)) for d, rs in self.rows.items()}
        other_labels = copy.deepcopy(self.labels)
        for label in other_labels.values():
            label['answer'] = 'D'
        other, other_qids = runner.select_paired_sample(reordered, other_labels)
        self.assertEqual(len(qids), 50)
        self.assertEqual(qids, other_qids)
        self.assertEqual(selected, other)
        self.assertEqual(len(selected['T2T']), 100)
        self.assertEqual([r['id'] for r in selected['T2T']],
                         [r['id'] for r in selected['T2V']])
        for qid in qids:
            kinds = [self.labels[r['id']]['question_type']
                     for r in selected['T2T'] if r['qid'] == qid]
            self.assertCountEqual(kinds, ['direct', 'diagnosis'])

    def test_missing_visual_counterpart_is_rejected(self):
        self.rows['T2V'].pop()
        with self.assertRaisesRegex(ValueError, 'source IDs differ'):
            runner.select_paired_sample(self.rows, self.labels)

    def test_duplicate_task_within_rule_is_rejected(self):
        self.labels['rule-000-diagnosis']['question_type'] = 'direct'
        with self.assertRaisesRegex(ValueError, 'Question-type pairing'):
            runner.select_paired_sample(self.rows, self.labels)


if __name__ == '__main__':
    unittest.main()
