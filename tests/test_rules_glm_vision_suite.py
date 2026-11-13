import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_rules_glm46_vision_suite import (
    ARMS, STAGES, control_passed, judge_controls, judge_payload, repeated_training, training_cmd, vision_core,
)


class VisionSuiteTests(unittest.TestCase):
    def test_visual_sft_uses_original_visual_description_and_user_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / 'images/r/learning_visual.png'
            image.parent.mkdir(parents=True)
            image.touch()
            source = [dict(id='r', qid='r', images=['images/r/learning_visual.png'],
                messages=[dict(role='assistant', content='<image>Use the pictured type, minimum 4.')])]
            text = [dict(id='r', qid='r', messages=[
                dict(role='user', content='Explain the rule or codebook named Zenta, including its conditions, mappings, and how to apply it.'),
                dict(role='assistant', content='SECRET TEXTUAL BINDING')])]
            original = copy.deepcopy(source)
            result = vision_core(source, text, root)[0]
            self.assertEqual(result['images'], [str(image)])
            self.assertEqual(result['messages'][-1]['content'], 'Use the pictured type, minimum 4.')
            self.assertTrue(result['messages'][0]['content'].startswith('<image>Using this learning reference image, explain'))
            self.assertNotIn('SECRET', json.dumps(result))
            self.assertEqual(source, original)

    def test_repetitions_preserve_images_targets_and_generic_ratio(self):
        unique = [dict(id='visual', qid='r', images=['learning_visual.png'],
                       messages=[dict(role='user', content='<image>Learn'), dict(role='assistant', content='Description')]),
                  dict(id='generic', images=[], answer='A',
                       messages=[dict(role='user', content='Add'), dict(role='assistant', content='2')])]
        output = repeated_training(unique, 10)
        self.assertEqual(len(output), 20)
        self.assertEqual(len({r['id'] for r in output}), 20)
        self.assertEqual(sum(bool(r['images']) for r in output), 10)
        output[0]['images'][0] = 'changed'
        self.assertEqual(output[2]['images'], ['learning_visual.png'])
        self.assertEqual(unique[0]['images'], ['learning_visual.png'])

    def test_both_training_arms_start_from_clean_base_with_three_epochs(self):
        plan = dict(model_path='/base', training=dict(micro_batch=4, gradient_accumulation=4, max_length=2048))
        for arm in ARMS:
            cmd = training_cmd(Path('/project'), Path('/work'), plan, arm)
            self.assertEqual(cmd[0], '/project/.venv-glm/bin/python')
            self.assertEqual(cmd[cmd.index('--model') + 1], '/base')
            self.assertEqual(cmd[cmd.index('--num_train_epochs') + 1], '3')
            self.assertEqual(cmd[cmd.index('--dataset') + 1], f'/work/data/train_{arm}.jsonl')
            self.assertEqual(cmd[cmd.index('--output_dir') + 1], f'/work/train/{arm}/VISION')
            self.assertNotIn('--adapters', cmd)
            self.assertNotIn('--resume_from_checkpoint', cmd)
            self.assertEqual(cmd[cmd.index('--freeze_vit') + 1], 'true')
            self.assertEqual(cmd[cmd.index('--freeze_aligner') + 1], 'true')

    def test_judge_includes_unresolved_visual_binding_controls_and_no_model_arm(self):
        cases = judge_controls()
        self.assertEqual(len(cases), 10)
        case = next(c for c in cases if c['id'] == 'unresolved_visual_binding')
        self.assertEqual(case['expected_verdict'], 'partial')
        payload = judge_payload(case)
        data = json.loads(payload['messages'][1]['content'])
        self.assertEqual(set(data), {'reference_rule', 'candidate_answer', 'recording_required'})
        self.assertNotIn('arm', data)
        self.assertNotIn('images', payload)
        self.assertIn('No reference image is available', payload['messages'][0]['content'])

    def test_stage_order_trains_both_before_evaluation_and_keeps_all_three_probes(self):
        self.assertLess(STAGES.index('train_repeat10'), STAGES.index('closed_repeat1'))
        for arm in ARMS:
            self.assertIn('closed_' + arm, STAGES)
            self.assertIn('recall_' + arm, STAGES)
            self.assertIn('apply_' + arm, STAGES)
            self.assertLess(STAGES.index('recall_' + arm), STAGES.index('apply_' + arm))

    def test_missing_core_condition_cannot_pass_calibration_as_complete_core_memory(self):
        case = next(c for c in judge_controls() if c['id'] == 'missing_condition')
        judgment = dict(verdict='partial', core_correct=True, complete=False, recording_correct=None)
        self.assertFalse(control_passed(case, judgment))
        judgment['core_correct'] = False
        self.assertTrue(control_passed(case, judgment))


if __name__ == '__main__':
    unittest.main()
