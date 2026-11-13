import copy
import unittest
from collections import Counter
from pathlib import Path

from prepare_entity_atomic_pilot import generic
from run_entity_glm46_once import command, single_pass_data


class SinglePassDataChecks(unittest.TestCase):
    def core(self):
        return [dict(id=f'{i}-{variant}', knowledge_id=str(i), qid=f'Q{i}', variant=variant,
                     messages=[dict(role='user', content=f'{variant} question about {i}?'),
                               dict(role='assistant', content=f'Answer {i}.')])
                for i in range(9) for variant in ['fact_0', 'attribute_0']]

    def test_preserves_both_existing_qa_types_without_repetition(self):
        core = self.core()
        original = copy.deepcopy(core)
        training, metadata, retention = single_pass_data(core, [generic(i, 'train') for i in range(20)])
        self.assertEqual(core, original)
        self.assertEqual(metadata[:len(core)], core)
        self.assertEqual(Counter(r['variant'] for r in metadata if 'variant' in r),
                         {'fact_0': 9, 'attribute_0': 9})
        self.assertEqual(Counter(r['knowledge_id'] for r in metadata if 'knowledge_id' in r),
                         {str(i): 2 for i in range(9)})
        self.assertEqual(Counter(r['answer'] for r in retention), dict.fromkeys('ABCD', 2))
        self.assertTrue(all(set(r) == {'id', 'messages'} for r in training))
        self.assertEqual([r['messages'] for r in training[:len(core)]],
                         [r['messages'] for r in core])

    def test_rejects_a_repeated_qa_instead_of_silently_deduplicating(self):
        core = self.core()
        duplicate = copy.deepcopy(core[0])
        duplicate['id'] += '-repeat-1'
        with self.assertRaises(AssertionError):
            single_pass_data(core + [duplicate], [generic(i, 'train') for i in range(20)])

    def test_same_name_prompts_keep_distinct_existing_targets(self):
        core = self.core()
        core[2]['messages'][0]['content'] = core[0]['messages'][0]['content']
        training, _, _ = single_pass_data(core, [generic(i, 'train') for i in range(20)])
        self.assertEqual(training[0]['messages'], core[0]['messages'])
        self.assertEqual(training[2]['messages'], core[2]['messages'])

    def test_smoke_writes_to_a_separate_directory_and_full_training_starts_clean(self):
        work = Path('/tmp/glm-single-pass-test')
        smoke = command(work, smoke=True)
        full = command(work)
        self.assertEqual(smoke[smoke.index('--output_dir') + 1], str(work / 'smoke/TEXT'))
        self.assertEqual(full[full.index('--output_dir') + 1], str(work / 'train/TEXT'))
        self.assertEqual(full[full.index('--dataset') + 1], str(work / 'data/train_text.jsonl'))
        self.assertEqual(full[full.index('--num_train_epochs') + 1], '3')
        self.assertNotIn('--adapters', full)
        self.assertNotIn('--resume_from_checkpoint', full)


if __name__ == '__main__':
    unittest.main()
