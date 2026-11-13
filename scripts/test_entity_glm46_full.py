import unittest
import copy
import json
from pathlib import Path
import tempfile

from evaluate_entity_glm46_full import configure_generation, generation_details, glm_readout
from entityqa_common import rows, save, sha, write_rows


class GlmReadoutChecks(unittest.TestCase):
    def test_accepts_native_box_letter(self):
        self.assertEqual(glm_readout('<|begin_of_box|>C<|end_of_box|>', {})[0], 'C')

    def test_accepts_latex_box_inside_native_box(self):
        self.assertEqual(glm_readout('<|begin_of_box|>\\boxed{D}<|end_of_box|>', {})[0], 'D')

    def test_accepts_empty_thinking_and_bare_letter(self):
        self.assertEqual(glm_readout('<think></think>\nB', {})[0], 'B')

    def test_rejects_nonempty_reasoning_without_terminal_answer(self):
        answer, reason = glm_readout('<think>reason</think> The answer is probably B', {})
        self.assertIsNone(answer)
        self.assertEqual(reason, 'no_explicit_answer')

    def test_rejects_conflicting_native_boxes(self):
        answer, reason = glm_readout('<|begin_of_box|>A<|end_of_box|> then '
                                     '<|begin_of_box|>B<|end_of_box|>', {})
        self.assertIsNone(answer)
        self.assertEqual(reason, 'conflicting_answers')


class GenerationResumeChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.config = dict(max_tokens=6000, wall_seconds='6000 tokens', prompt='frozen')
        self.records = [dict(id='one', raw_response=r'Reasoning. \boxed{C}', generated_tokens=32)]
        save(self.output / 'config.json', self.config)
        write_rows(self.output / 'predictions.jsonl', self.records)
        self.original_hash = sha(self.output / 'predictions.jsonl')
        self.lower = dict(self.config, max_tokens=1024, wall_seconds='1024 tokens')

    def test_reduction_preserves_raw_output_and_archives_original_bytes(self):
        original = copy.deepcopy(self.records)
        configure_generation(self.output, self.lower, self.records, True)
        archive = self.output / 'configuration_revisions/cap_6000_to_1024'
        self.assertEqual(sha(archive / 'predictions.jsonl'), self.original_hash)
        self.assertEqual(rows(archive / 'predictions.jsonl'), original)
        self.assertEqual(self.records[0]['raw_response'], original[0]['raw_response'])
        self.assertEqual(rows(self.output / 'predictions.jsonl')[0]['generation_max_tokens'], 6000)
        self.assertEqual(json.loads((self.output / 'config.json').read_text()), self.lower)
        records = rows(self.output / 'predictions.jsonl')
        records.append(dict(id='two', generation_max_tokens=1024))
        write_rows(self.output / 'predictions.jsonl', records)
        configure_generation(self.output, self.lower, records)
        self.assertEqual(generation_details(self.output)['generation_cap_counts'], {'6000': 1, '1024': 1})

    def test_reduction_requires_explicit_flag_and_cannot_change_prompt(self):
        for config, allowed in [(self.lower, False), (dict(self.lower, prompt='changed'), True),
                                (dict(self.config, max_tokens=7000), True)]:
            with self.subTest(config=config, allowed=allowed):
                with self.assertRaises(AssertionError):
                    configure_generation(self.output, config, self.records, allowed)
                self.assertEqual(sha(self.output / 'predictions.jsonl'), self.original_hash)
                self.assertEqual(json.loads((self.output / 'config.json').read_text()), self.config)


if __name__ == '__main__':
    unittest.main()
