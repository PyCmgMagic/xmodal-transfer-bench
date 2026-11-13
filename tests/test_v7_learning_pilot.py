"""Pilot inputs use learned semantic names and generated, solver-labelled practice."""
import json
import re
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import diagnose_v7_learning as diagnosis
import pilot_v7_application_sft as pilot


class LearningPilot(unittest.TestCase):
    def test_recall_uses_names_present_in_original_training(self):
        probes=diagnosis.build_probes(ROOT/'rules_repaired_single_image_v7',ROOT/'runs/v7_full_experiment')
        recalls=[r for r in probes if r['task']=='rule_recall']
        train={r['id']:r for r in diagnosis.rows(ROOT/'runs/v7_full_experiment/data/sft_text.jsonl')}
        self.assertEqual(len(recalls),24)
        self.assertEqual(sum(r['task']=='application' for r in probes),72)
        for r in recalls:
            text=r['blind']['messages'][-1]['content']
            identifier=text.split('rule named ',1)[1].split('?',1)[0]
            self.assertNotIn('scale1k_',identifier)
            self.assertIn(identifier,train[r['qid']]['messages'][-1]['content'])

    def test_generated_practice_is_balanced_and_independently_labelled(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp);pilot.build(ROOT/'rules_repaired_single_image_v7',ROOT/'runs/v7_full_experiment',dest)
            train=pilot.rows(dest/'train.jsonl');manifest=pilot.read(dest/'manifest.json')
            self.assertEqual(len(manifest['rules']),12)
            self.assertEqual(len(train),548)
            self.assertFalse(any(r.get('images') for r in train))
            self.assertTrue(all(':practice:' in r['id'] or r['id'].endswith(':explain') for r in train))
            choices=[];accept=[]
            for r in train:
                if r['messages'][0]['role']!='system':continue
                text=r['messages'][1]['content'];answer=r['messages'][-1]['content']
                options=dict(re.findall(r'(?m)^([A-D])\. (.*)$',text))
                self.assertEqual(set(options),set('ABCD'));self.assertIn(answer,options)
                choices.append(answer)
                if 'A technician submitted' in text:
                    accept.append(options[answer].startswith('Accept report.'))
            self.assertEqual(sum(accept),len(accept)//2)
            self.assertEqual(len(set(choices)),4)
            evaluation=pilot.rows(dest/'eval.jsonl')
            self.assertEqual(len(evaluation),24)
            self.assertFalse({r['id'] for r in train}&{r['id'] for r in evaluation})
            # Any source change must produce an explicit new experiment input.
            self.assertEqual(pilot.sha(dest/'train.jsonl'),manifest['train_sha256'])

if __name__=='__main__':unittest.main()
