import unittest
from collections import Counter
import json
from pathlib import Path
import tempfile

from prepare_entity_atomic_pilot import generic
from run_entity_all_repeat10_v5 import balanced_retention, swift_training_rows


class FullEntityRetentionChecks(unittest.TestCase):
    def test_removes_dev_overlap_and_replaces_with_same_label_balance(self):
        # Force a candidate from the training generator into dev to exercise exclusion.
        dev = [generic(0, 'train')] + [generic(i, 'dev') for i in range(12)]
        data, audit = balanced_retention(400, dev)
        prompts = [r['messages'][0]['content'] for r in data]
        self.assertEqual(len(prompts), len(set(prompts)))
        self.assertFalse(set(prompts) & {r['messages'][0]['content'] for r in dev})
        self.assertEqual(Counter(r['answer'] for r in data), dict.fromkeys('ABCD', 100))
        self.assertGreaterEqual(audit['rejected'].get('dev_prompt_overlap', 0), 1)

    def test_actual_json_loader_reads_multiple_chunks_with_mixed_audit_metadata(self):
        # Entity-only first chunks must not make later generic fields change the schema.
        from datasets import load_dataset
        entity = [dict(id=f'entity-{i}', knowledge_id=f'k-{i}', qid=f'q-{i}', variant='fact_0',
                       messages=[dict(role='user', content=f'Recall entity {i}.'),
                                 dict(role='assistant', content=f'The fact about entity {i}.')])
                  for i in range(40)]
        source = entity + [generic(i, 'train') for i in range(40)]
        training = swift_training_rows(source)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'training.jsonl'
            path.write_text(''.join(json.dumps(r) + '\n' for r in training))
            self.assertGreater(path.stat().st_size, 4096)
            loaded = load_dataset('json', data_files=str(path), chunksize=1024,
                                  cache_dir=str(Path(directory) / 'cache'), split='train')
            self.assertEqual(set(loaded.column_names), {'id', 'messages'})
            self.assertEqual(len(loaded), len(source))
            for old, actual in zip(source, loaded):
                self.assertEqual(actual['id'], old['id'])
                self.assertEqual(actual['messages'], old['messages'])


if __name__ == '__main__':
    unittest.main()
