import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import repair_answer_outputs as repair


class RepairOutputContract(unittest.TestCase):
    def test_parser_and_constrained_scores_are_separate_and_tasks_have_no_gold(self):
        meta={'experiment':'full_cot','job':'rules_sft_text','dataset':'rules',
              'direction':'T2T','arm':'direct','source':'raw.jsonl','source_report':'report.json'}
        payload={'messages':[{'role':'user','content':'QUESTION\nA. first\nB. second\nC. third\nD. fourth'}]}
        data=[({'id':'q1','qid':'r','response':'Option B is accurate.','finish_reason':'stop'},payload,'B',None),
              ({'id':'q2','qid':'r','response':'Some facts, no option.','finish_reason':'stop'},payload,'D',None)]
        with tempfile.TemporaryDirectory() as tmp, patch.object(repair,'cells',return_value=[(meta,data)]):
            work=Path(tmp);summary,pending=repair.refresh(work)
            r=summary['cells'][0]
            self.assertEqual(r['original_accuracy'],0)
            self.assertEqual(r['parsed_accuracy'],0.5)
            self.assertIsNone(r['combined_accuracy'])
            self.assertEqual(r['readout_pending'],1)
            task=pending['rules_sft_text'][0]
            self.assertNotIn('gold',task)
            self.assertNotIn('answer',task)
            self.assertIn('Some facts, no option.',task['payload']['messages'][-1]['content'])
            out=work/'jobs/rules_sft_text/readout.jsonl'
            repair.ev.save_rows(out,[{k:v for k,v in {**task,'selected':'D'}.items() if k!='payload'}])
            summary,pending=repair.refresh(work);r=summary['cells'][0]
            self.assertEqual(r['parsed_accuracy'],0.5)
            self.assertEqual(r['combined_accuracy'],1)
            self.assertEqual(r['finalized'],1)
            self.assertEqual(sum(map(len,pending.values())),0)
            result=repair.read(work/'results/full_cot/rules_sft_text/T2T/direct/results.jsonl')
            self.assertEqual([x['stage'] for x in result],['parsed','constrained_readout'])

    def test_real_transformers_generation_masks_every_non_option_token(self):
        import torch
        from transformers import GPT2Config,GPT2LMHeadModel
        from answer_readout import choice_hook
        class Tokenizer:
            def encode(self,letter,add_special_tokens=False):return [ord(letter)]
        torch.manual_seed(42)
        model=GPT2LMHeadModel(GPT2Config(vocab_size=128,n_positions=32,n_embd=16,n_layer=1,n_head=2,
                                        eos_token_id=0,pad_token_id=0)).eval()
        kw=choice_hook(Tokenizer())({'inputs':{}})['inputs']
        with torch.no_grad():
            output=model.generate(input_ids=torch.tensor([[1,2],[1,3]]),attention_mask=torch.ones((2,2),dtype=torch.long),
                                  max_new_tokens=1,do_sample=False,**kw)
        self.assertTrue(all(token in [65,66,67,68] for token in output[:,-1].tolist()))


if __name__=='__main__':unittest.main()
