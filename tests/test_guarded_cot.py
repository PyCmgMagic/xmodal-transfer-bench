import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import guarded_cot as guard
import run_rules_guarded_cot as runner
import run_v7_evaluation as ev


class CharTokenizer:
    def decode(self, ids, **kwargs):
        return ''.join(chr(i) for i in ids)


class GuardedCotContract(unittest.TestCase):
    def test_checkpoint_hash_reuse_rejects_files_changed_after_audit(self):
        import os
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            cp = base / 'checkpoint'
            cp.mkdir()
            (cp / 'config.json').write_text('{}')
            weight = cp / 'model.safetensors'
            weight.write_bytes(b'weights')
            previous = base / 'evaluation_direct_answer_8tok_20261102/arm/manifest.json'
            ev.save_json(previous, {'config': {'adapters': {'T': {
                'path': str(cp.resolve()), 'kind': 'dense',
                'sha256': {p.name: ev.file_hash(p) for p in cp.iterdir()}}}}})
            with mock.patch.object(runner.queue, 'WORK', base), mock.patch.object(ev, 'checkpoint') as fresh:
                result = runner.checkpoint_identity('arm', cp)
                self.assertIn('hash_provenance', result)
                fresh.assert_not_called()
                os.utime(previous, ns=(1, 1))
                runner.checkpoint_identity('arm', cp)
                fresh.assert_called_once_with(cp)

    def test_prompt_boxes_never_stop_reasoning_but_generated_box_does(self):
        import torch
        prompt = r'Output one of \boxed{A}, \boxed{B}, \boxed{C}, \boxed{D}.'
        g = guard.GenerationGuard(CharTokenizer(), len(prompt), 0)
        scores = torch.zeros(1, 256)
        g(torch.tensor([[ord(c) for c in prompt]]), scores)
        self.assertIsNone(g.reason)
        self.assertTrue(torch.equal(scores, torch.zeros_like(scores)))
        reasoning = r'The first condition fails. Therefore \boxed{C}'
        g(torch.tensor([[ord(c) for c in prompt + reasoning]]), scores)
        self.assertEqual(g.reason, 'terminal_box')
        self.assertEqual(scores.argmax(-1).item(), 0)
        self.assertTrue(torch.isneginf(scores[0, 1:]).all())

    def test_deadline_and_four_block_repetition(self):
        import torch
        block = list(range(10, 18))
        self.assertIsNone(guard.repeated_suffix(block * 3))
        self.assertEqual(guard.repeated_suffix(block * 4), 8)
        g = guard.GenerationGuard(CharTokenizer(), 1, 0, max_seconds=1)
        g.started -= 2
        g(torch.tensor([[ord('Q'), ord('x')]]), torch.zeros(1, 256))
        self.assertEqual(g.reason, 'time_limit')

    def test_real_generate_stops_repetition_without_exhausting_budget(self):
        import torch
        from transformers import GPT2Config, GPT2LMHeadModel, LogitsProcessorList
        model = GPT2LMHeadModel(GPT2Config(vocab_size=128, n_positions=256,
            n_embd=16, n_layer=1, n_head=2, eos_token_id=0, pad_token_id=0)).eval()
        class RepeatingModelOutput:
            def __call__(self, input_ids, scores):
                token = 65 + (input_ids.shape[-1] - 1) % 8
                scores.fill_(-float('inf'))
                scores[:, token] = 0
                return scores
        g = guard.GenerationGuard(CharTokenizer(), 1, 0)
        with torch.no_grad():
            out = model.generate(torch.tensor([[1]]), attention_mask=torch.ones(1, 1, dtype=torch.long),
                                 do_sample=False, max_new_tokens=200,
                                 logits_processor=LogitsProcessorList([RepeatingModelOutput(), g]))
        self.assertEqual(g.reason, 'repeated_token_block')
        self.assertEqual(out[0, -1].item(), 0)
        self.assertLess(out.shape[-1] - 1, 200)

    def test_readout_keeps_test_image_and_own_reasoning_without_gold(self):
        prepared = {'id': 'q', 'gold': 'GOLD_SENTINEL', 'learning_material': 'ORACLE_SENTINEL',
                    'images': ['test.png'], 'messages': [
                        {'role': 'system', 'content': ev.SYSTEM},
                        {'role': 'user', 'content': '<image>\nQUESTION\nA. one\nB. two\nC. three\nD. four' + ev.SUFFIX}]}
        payload = guard.final_readout_payload(prepared, 'OWN_REASONING')
        text = json.dumps(payload)
        for item in ['test.png', 'QUESTION', 'OWN_REASONING']:
            self.assertIn(item, text)
        for item in ['GOLD_SENTINEL', 'ORACLE_SENTINEL', ev.SUFFIX]:
            self.assertNotIn(item, text)
        self.assertEqual(set(payload), {'messages', 'images'})

    def test_raw_cot_failure_and_constrained_answer_have_separate_scores(self):
        inputs = [{'id': sid, 'qid': sid} for sid in ['a', 'b']]
        labels = {sid: {'answer': 'A', 'evaluation_group': 'result_application',
                       'question_type': 'direct', 'bias_watchlist': False} for sid in ['a', 'b']}
        cot = {'a': {'response': r'Reason. \boxed{A}', 'guard_reason': 'terminal_box'},
               'b': {'response': 'Repeating reasoning.', 'guard_reason': 'repeated_sentence'}}
        finished = {sid: {'selected': 'A' if sid == 'a' else 'B',
                         'cot_response': cot[sid]['response'], 'guard_reason': cot[sid]['guard_reason'],
                         'finalizer_response': None if sid == 'a' else 'B',
                         'stage': 'cot' if sid == 'a' else 'constrained_readout'} for sid in ['a', 'b']}
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            r = runner.write_report(p, 'T2T', inputs, cot, finished, labels, {})
            self.assertEqual(r['cot_only']['valid'], 1)
            self.assertEqual(r['cot_only']['format_violation'], 1)
            self.assertEqual(r['valid'], 2)
            self.assertEqual(r['finalized'], 1)
            self.assertEqual(r['accuracy'], 0.5)
            results = ev.read_rows(p / 'results.jsonl')[0]
            self.assertEqual(results[1]['response'], 'Repeating reasoning.')
            self.assertEqual(results[1]['predicted'], 'B')
            self.assertFalse(results[1]['correct'])


if __name__ == '__main__':
    unittest.main()
