"""Safety checks for scoring and the frozen v7 experiment's input contract."""
import json
import sys
import tempfile
import subprocess
import unittest
import dataclasses
import types
import io
from contextlib import redirect_stdout
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import run_v7_evaluation as ev
import run_v7_adapter_eval as adapter_eval

class EvaluationContract(unittest.TestCase):
    def test_preflight_controller_log_does_not_block_fresh_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / '.run.lock').write_text('0')
            (output / 'controller.log').write_text('Preflight failed before inference.\n')
            self.assertFalse(ev.has_fresh_run_artifacts(output))
            (output / 'attempt_001.raw.jsonl').write_text('{"response":"A"}\n')
            self.assertTrue(ev.has_fresh_run_artifacts(output))

    def test_pt_inference_reconstructs_template_before_encoding_system(self):
        class Args:
            use_chat_template=False
            def get_template(self,processor):
                self.processor=processor
                return types.SimpleNamespace(use_chat_template=self.use_chat_template,
                    template_meta=types.SimpleNamespace(support_system=self.use_chat_template))
        args=Args();template=adapter_eval.evaluation_template(args,'processor')
        self.assertTrue(template.template_meta.support_system)
        self.assertEqual(args.processor,'processor')

    def test_batched_resume_preserves_request_images_and_ids(self):
        inputs = [{'id':str(i)} for i in range(7)]
        payload = {r['id']:{'messages':[{'role':'user','content':r['id']}],
            'images':[r['id']+'.png']} for r in inputs}
        seen = {'1':{}, '5':{}}
        batches = []
        def infer(requests, *_args, **_kwargs):
            batches.append([r['images'][0] for r in requests])
            result = [r['messages'][0]['content'] for r in requests]
            requests[0]['messages'][0]['content'] = 'engine mutation'
            return result
        engine = types.SimpleNamespace(infer=infer)
        result = list(adapter_eval.infer_pending(engine,dict,None,inputs,payload,seen,3))
        self.assertEqual([(r['id'],res) for r,res in result],[(str(i),str(i)) for i in (0,2,3,4,6)])
        self.assertEqual(batches,[['0.png','2.png','3.png'],['4.png','6.png']])
        self.assertEqual(payload['0']['messages'][0]['content'],'0')
        engine.infer = lambda *_a,**_k: []
        with self.assertRaises(RuntimeError):list(adapter_eval.infer_pending(engine,dict,None,inputs,payload,seen,3))

    def test_adapter_driver_runs_and_resumes_all_eight_cells(self):
        @dataclasses.dataclass
        class Usage:
            prompt_tokens:int = 1
            completion_tokens:int = 1
            total_tokens:int = 2
        calls = []
        class Engine:
            def __init__(self,*_a,**_k):self.template=types.SimpleNamespace()
            def infer(self, requests, config, **_kwargs):
                calls.extend(requests)
                response = 'FINAL: A' if config.max_tokens == 1536 else 'A'
                return [types.SimpleNamespace(choices=[types.SimpleNamespace(
                    message=types.SimpleNamespace(content=response),finish_reason='stop')],usage=Usage()) for _ in requests]
        modules = {'swift.infer_engine':types.SimpleNamespace(TransformersEngine=Engine,
            InferRequest=lambda **kw:kw,RequestConfig=lambda **kw:types.SimpleNamespace(**kw)),
            'swift.utils':types.SimpleNamespace(seed_everything=lambda _s:None),
            'torch':types.SimpleNamespace(bfloat16='fake')}
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);folder=work/'smoke/qwen3_sft_text';folder.mkdir(parents=True)
            (folder/'validated.json').write_text(json.dumps({'checkpoint':str(work/'checkpoint')}))
            argv=['eval','--root',str(ROOT/'data/rules_repaired_single_image_v7'),'--work',tmp,
                '--model','qwen3','--objective','sft','--modality','text','--gpu','0',
                '--preflight-only','--preflight-smoke-adapter','--protocols','strict','cot']
            with mock.patch.object(sys,'argv',argv),mock.patch.dict(sys.modules,modules),\
                    mock.patch.object(adapter_eval,'environment',return_value={}),\
                    mock.patch.object(ev,'checkpoint',return_value={'sha256':'fake'}),redirect_stdout(io.StringIO()):
                adapter_eval.main()
                self.assertEqual(len(calls),8)
                adapter_eval.main()
                self.assertEqual(len(calls),8,'Resuming must not regenerate any saved output')
                with mock.patch.object(sys,'argv',argv[:-3]):
                    adapter_eval.main()
                    status=json.loads((work/'eval_preflight/qwen3_sft_text/status.json').read_text())
                    self.assertEqual(status['protocols'],['strict'])
                    self.assertEqual(status['cells'],4)
                    self.assertEqual(len(calls),8,'Strict-only resume must reuse outputs')
                manifest=work/'eval_preflight/qwen3_sft_text/manifest.json'
                original=json.loads(manifest.read_text())
                original['config']['adapter_evaluator_sha256']='unknown-source'
                manifest.write_text(json.dumps(original))
                with self.assertRaisesRegex(AssertionError,'Unreviewed evaluator revision'):
                    adapter_eval.main()
            reports=list((work/'eval_preflight/qwen3_sft_text').glob('*/*/report.json'))
            self.assertEqual(len(reports),8)
            self.assertTrue(all(json.loads(p.read_text())['matched']==1 for p in reports))
            self.assertEqual(sum(bool(r.get('images')) for r in calls),4)

    def test_final_answer_does_not_extract_reasoning(self):
        cases = {
            'The tempting option is A.\nFINAL: B': 'B',
            'The tempting option is A. FINAL: B': 'B',
            'FINAL: A\nFINAL: B': None,
            'FINAL: B\nMore explanation': None,
            '<think>FINAL: A</think>\nFINAL: B': 'B',
            '<think>unfinished\nFINAL: A': None,
            'I think B is right': None,
            '<think>Reason</think>\nB': 'B',
            '<think>Reason</think><|begin_of_box|>C<|end_of_box|>': 'C',
            'FINAL: A\n<|begin_of_box|>B<|end_of_box|>': None,
            '<|begin_of_box|>A<|end_of_box|>\n<|begin_of_box|>B<|end_of_box|>': None,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(ev.letter(text, 'cot'), expected)
        self.assertIsNone(ev.letter('Answer: A', 'strict'))
        self.assertIsNone(ev.letter('<think>reason</think>A', 'glm-box'))
        self.assertEqual(ev.letter('<think></think><|begin_of_box|>D<|end_of_box|>', 'glm-box'), 'D')

    def test_cot_box_accepts_one_terminal_box_and_rejects_ambiguity(self):
        cases = {
            r'<think>Reason about option A.</think>\boxed{B}': 'B',
            r'\boxed{c}.': 'C',
            '<think>Reason</think><|begin_of_box|>D<|end_of_box|>': 'D',
            r'\boxed{A} \boxed{B}': None,
            r'\boxed{A} then more text': None,
            '<think>unfinished \\boxed{A}': None,
            'Reasoning only; answer is A': 'A',
            'The correct answer is C.': 'C',
            'The final answer: D!': 'D',
            'The answer is A, then reconsidered: B.': None,
            'A.': 'A',
            'Option A is plausible. Option B is correct.': None,
        }
        for response, expected in cases.items():
            with self.subTest(response=response):
                self.assertEqual(ev.letter(response, 'cot-box'), expected)

    def test_cot_box_prompt_keeps_image_and_drops_gold(self):
        data = ROOT/'data/rules_repaired_single_image_v7'
        for direction in ('T2T', 'T2V'):
            row = ev.read_rows(data/direction/'eval.jsonl')[0][0]
            prepared = ev.prepare_row({**row, 'answer': 'GOLD_SENTINEL'}, data, direction, 'cot-box')
            self.assertNotIn('GOLD_SENTINEL', json.dumps(prepared))
            self.assertIn('do not overthink', prepared['messages'][0]['content'].lower())
            self.assertIn('do not overthink', prepared['messages'][-1]['content'].lower())
            self.assertTrue(prepared['messages'][-1]['content'].endswith(
                r'\boxed{A}, \boxed{B}, \boxed{C}, or \boxed{D}. The box must contain only the answer option letter.'))
            self.assertEqual(len(prepared.get('images', [])), int(direction.endswith('V')))

    def test_cot_box_prompt_disallows_overthinking(self):
        data = ROOT/'data/rules_repaired_single_image_v7'
        row = ev.read_rows(data/'T2T/eval.jsonl')[0][0]
        prepared = ev.prepare_row(row, data, 'T2T', 'cot-box')
        self.assertIn('do not overthink', prepared['messages'][0]['content'].lower())
        self.assertIn('do not overthink', prepared['messages'][-1]['content'].lower())
        self.assertEqual(ev.letter(r'Reason: the rule selects this case. \boxed{C}', 'cot-box'), 'C')
        self.assertIsNone(ev.letter('reasoning repeated without an answer', 'cot-box'))

    def test_explicit_cot_does_not_force_native_thinking(self):
        for answer_format in ('cot', 'cot-box'):
            args = ev.parser().parse_args(['--answer-format', answer_format])
            cmd = ev.swift_command(args, ['swift'], None, Path('input.jsonl'), Path('out.jsonl'))
            self.assertEqual(cmd[cmd.index('--enable_thinking') + 1], 'false')
            args.native_thinking = 'true'
            cmd = ev.swift_command(args, ['swift'], None, Path('input.jsonl'), Path('out.jsonl'))
            self.assertEqual(cmd[cmd.index('--enable_thinking') + 1], 'true')

    def test_all_invalid_probe_halts_full_inference_and_retains_outputs(self):
        data = ROOT/'data/rules_repaired_single_image_v7'
        inputs = [ev.prepare_row(r, data, 'T2T', 'cot-box')
                  for r in ev.read_rows(data/'T2T/eval.jsonl')[0][:3]]
        labels = ev.id_map(ev.read_rows(data/'audit/eval_labels.jsonl')[0], 'gold')
        predictions = {r['id']: {**r, 'response': '<think>unfinished'} for r in inputs}
        args = ev.parser().parse_args(['--answer-format', 'cot-box', '--require-valid-probe'])
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(ev, 'collect_attempts', return_value=(predictions, [])), \
                mock.patch.object(ev, 'infer_attempt') as infer:
            report = ev.perform_direction(args, ['swift'], data, Path(tmp), 'T2T',
                                          inputs, None, labels, {})
            infer.assert_not_called()
            self.assertEqual(report['status'], 'blocked')
            self.assertIn('All format probes are invalid', report['error'])
            saved = ev.read_rows(Path(tmp)/'T2T/predictions.jsonl')[0]
            self.assertEqual([r['response'] for r in saved], ['<think>unfinished']*3)

    def test_all_prompts_and_image_ids(self):
        data = ROOT/'data/rules_repaired_single_image_v7'
        labels = ev.id_map(ev.read_rows(data/'audit/eval_labels.jsonl')[0], data)
        for d in ev.DIRECTIONS:
            source = ev.read_rows(data/d/'eval.jsonl')[0]
            self.assertEqual(len(source), 2000)
            self.assertEqual({r['id'] for r in source}, set(labels))
            for row in source:
                row_with_gold = {**row, 'answer': labels[row['id']]['answer'], 'learning_material': 'DO NOT LEAK'}
                for protocol in ('strict', 'cot'):
                    ready = ev.prepare_row(row_with_gold, data, d, protocol)
                    self.assertEqual(set(ready), {'id', 'qid', 'messages'} | ({'images'} if d.endswith('V') else set()))
                    self.assertNotIn('DO NOT LEAK', json.dumps(ready))
                    self.assertEqual([m['role'] for m in ready['messages']], ['system','user'])
                    if protocol == 'cot':
                        self.assertNotIn(ev.SUFFIX.strip(), ready['messages'][-1]['content'])
                        self.assertTrue(ready['messages'][-1]['content'].endswith('FINAL: D.'))

    def test_invalid_and_missing_outputs_are_counted_separately(self):
        ids = ['valid', 'wrong', 'empty', 'bad', 'missing']
        inputs = [{'id': sid, 'qid': sid} for sid in ids]
        labels = {sid: {'answer':'A','evaluation_group':'result_application','bias_watchlist':False,'question_type':'direct'} for sid in ids}
        predictions = {sid:{'response':text} for sid,text in zip(ids,['A','B','','answer is A'])}
        report, _ = ev.score(inputs,predictions,labels,{},'T2T')
        self.assertEqual(report['accuracy'],.2)
        self.assertEqual(report['accuracy_on_valid_only'],.5)
        self.assertEqual([report[k] for k in ('wrong_answer','empty_prediction','format_violation','missing_prediction')], [1,1,1,1])

    def test_partial_resume_does_not_hide_interior_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'raw.jsonl'
            p.write_bytes(b'{"id":"a"}\n{"id":')
            rows, notes=ev.read_rows(p, allow_partial_tail=True)
            self.assertEqual(rows,[{'id':'a'}])
            self.assertEqual(len(notes),1)
            p.write_bytes(b'bad\n{"id":"a"}\n')
            with self.assertRaises(ValueError):ev.read_rows(p,allow_partial_tail=True)

    def test_summary_keeps_all_adapters_and_directions(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([sys.executable,str(ROOT/'scripts/summarize_v7_experiment.py'),'--work',tmp],check=True,capture_output=True)
            report=json.loads((Path(tmp)/'summary/comparison.json').read_text())
            self.assertEqual(report['expected_requests'],192000)
            self.assertEqual(len(report['cells']),96)
            self.assertEqual(len(report['overall']),24)
            self.assertEqual(len({(r['model'],r['objective'],r['train_modality'],r['protocol'],r['direction']) for r in report['cells']}),96)
            self.assertEqual({r['direction'] for r in report['cells'] if r['train_modality']=='text'},set(ev.DIRECTIONS))
            self.assertEqual(report['complete_cells'],0)
            self.assertTrue(all(r['accuracy'] is None for r in report['cells']))

    def test_strict_only_summary_keeps_all_48_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'evaluations';folder.mkdir()
            (folder/'protocols.json').write_text(json.dumps({'protocols':['strict']}))
            subprocess.run([sys.executable,str(ROOT/'scripts/summarize_v7_experiment.py'),'--work',tmp],check=True,capture_output=True)
            report=json.loads((Path(tmp)/'summary/comparison.json').read_text())
            self.assertEqual(report['expected_requests'],96000)
            self.assertEqual(report['expected_cells'],48)
            self.assertEqual(len(report['cells']),48)
            self.assertEqual(len(report['overall']),12)
            self.assertEqual({r['protocol'] for r in report['cells']},{'strict'})

if __name__ == '__main__':
    unittest.main()
