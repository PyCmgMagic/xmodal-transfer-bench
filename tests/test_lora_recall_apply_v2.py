"""Guard against label leakage, ambiguous readout and unpaired scoring."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_lora_recall_apply_v2 as run


class RecallApplyContract(unittest.TestCase):
    def test_selected_recall_names_match_actual_training_prompts(self):
        path=run.DEFAULT_WORK/'manifest.json'
        if not path.exists():self.skipTest('Prepare v2 inputs first')
        manifest=json.loads(path.read_text())
        original=run.read(ROOT/'data/rules_repaired_single_image_v7/shared/train_text.jsonl')
        trained=run.read(ROOT/'runs/qwen3_lora_learning_ep5_20261024/data/rules_sft_text.jsonl')
        by_qid={a['id']:b['messages'][0]['content'] for a,b in zip(original,trained)}
        recalls={r['qid']:r for r in run.read(run.DEFAULT_WORK/'data/recall.jsonl')}
        self.assertEqual(len(manifest['selection']),90)
        for item in manifest['selection']:
            self.assertNotEqual(item['name'],item['qid'])
            expected=f'Explain the rule or codebook named {item["name"]}, including its conditions, mappings, and how to apply it.'
            self.assertEqual(by_qid[item['qid']],expected)
            self.assertTrue(recalls[item['qid']]['messages'][-1]['content'].startswith(expected))

    def test_answer_readout_tolerates_format_but_rejects_ambiguity(self):
        cases=[(r'Reasoning. \boxed{B}', 'B'),
               (r'Reasoning. \(\boxed{a}\).', 'A'),
               ('Reasoning. Final answer: C.', 'C'),
               ('Reasoning. Therefore, the correct answer is D.', 'D'),
               ('Reasoning. FINAL: A.', 'A'),
               ('B', 'B'),
               ('Reasoning. The final answer is B.', 'B'),
               ('Reasoning. The final result is A.', 'A'),
               ('Reasoning. This matches option C.', 'C'),
               ('Reasoning. Only D is the correct choice.', 'D'),
               ('Final selection: C and D. These candidates match option B. FINAL: B', 'B'),
               ('Final selection: B, C, D. This matches option A. FINAL: A', 'A'),
               ('Options A and B differ. I cannot decide.', None),
               (r'\boxed{A} \boxed{B}', None),
               (r'FINAL: A. \boxed{B}', None),
               (r'\boxed{B} The answer is A.', None),
               ('<think>unfinished FINAL: B', None)]
        for text,expected in cases:
            with self.subTest(text=text):self.assertEqual(run.parse_answer(text)[0],expected)

    def test_recall_never_sees_question_or_gold_and_answer_only_receives_self_recall(self):
        recall=run.recall_payload('Lenlar')
        self.assertEqual(set(recall),{'messages'})
        self.assertEqual([m['role'] for m in recall['messages']],['system','user'])
        self.assertIn('Lenlar',recall['messages'][1]['content'])
        row={'id':'id','qid':'qid','question':'<image>TEST_QUESTION',
             'images':['test.png'],'gold':'GOLD_SENTINEL','reference':'CORRECT_RULE_SENTINEL'}
        direct=run.application_payload(row)
        aided=run.application_payload(row,'SELF_RECALL_SENTINEL')
        for payload in (direct,aided):
            self.assertEqual(payload['images'],['test.png'])
            self.assertIn('TEST_QUESTION',payload['messages'][1]['content'])
            self.assertNotIn('GOLD_SENTINEL',json.dumps(payload))
            self.assertNotIn('CORRECT_RULE_SENTINEL',json.dumps(payload))
        self.assertNotIn('SELF_RECALL_SENTINEL',json.dumps(direct))
        self.assertIn('SELF_RECALL_SENTINEL',json.dumps(aided))
        self.assertEqual(direct['messages'][0],aided['messages'][0])

    def test_paired_scoring_keeps_invalid_in_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);(work/'data').mkdir();out=work/'jobs/example';out.mkdir(parents=True)
            labels=[{'id':'q1','qid':'r1','answer':'A'},{'id':'q2','qid':'r1','answer':'B'}]
            run.ev.save_rows(work/'data/eval_labels.jsonl',labels)
            records=[{'stage':'recall','id':'r1','qid':'r1','response':'self rule','finish_reason':'stop'}]
            for arm,answers in [('direct_boxed',['no final answer','FINAL: B']),
                                ('self_recall_boxed',[r'\boxed{A}',r'\boxed{C}'])]:
                for label,text in zip(labels,answers):
                    records.append({'stage':'application','id':label['id'],'qid':'r1',
                                    'direction':'T2T','arm':arm,'response':text,'finish_reason':'stop'})
            run.ev.save_rows(out/'raw.jsonl',records)
            manifest={'jobs':[{'name':'example','objective':'sft','learning_modality':'text','directions':['T2T']}],
                      'rules':1,'questions_per_cell':2,'rules_per_family':1,
                      'arms':['direct_boxed','self_recall_boxed'],'predictions_expected':5}
            s=run.collect(manifest,work);r=s['reports'][0]
            self.assertEqual(r['direct_boxed']['accuracy'],0.5)
            self.assertEqual(r['direct_boxed']['invalid'],1)
            self.assertEqual(r['self_recall_boxed']['accuracy'],0.5)
            self.assertEqual(r['self_recall_boxed']['invalid'],0)
            self.assertEqual(r['paired'],{'fixed':1,'broken':1,'both_correct':0,'both_wrong':0})
            self.assertEqual(r['delta_pp'],0)


if __name__=='__main__':unittest.main()
