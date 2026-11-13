#!/usr/bin/env python3
"""Re-evaluate the fixed E23 models and paired questions under a neutral prompt."""
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

from entityqa_common import rows, save, sha, digest, now, STRICT_SUFFIX, COT_SUFFIX as OLD_SUFFIX, eval_payload as old_payload
from entityqa_prompt_neutral_v5 import eval_payload, COT_SUFFIX, PROMPT_VERSION
from run_entityqa_lora import idle


def metric(group):
    assert group
    return dict(n=len(group), correct=sum(r['raw_predicted'] == r['gold'] for r in group),
        accuracy=sum(r['raw_predicted'] == r['gold'] for r in group) / len(group),
        invalid=sum(r['raw_predicted'] is None for r in group),
        invalid_rate=sum(r['raw_predicted'] is None for r in group) / len(group),
        answer_counts=dict(Counter(str(r['raw_predicted']) for r in group)),
        stop_reasons=dict(Counter(r['stop_reason'] for r in group)),
        max_generated_tokens=max(r['generated_tokens'] for r in group))


def run():
    root = Path(__file__).resolve().parents[1]
    source = root / 'runs/entity50_repeat10_lora_ep3_20261104'
    output = source / 'evaluation_neutral_v5_20261104'
    output.mkdir(exist_ok=True)
    assert socket.gethostname() == 'softengine-KVM'
    lock = (output / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def status(stage, **extra):
        save(output / 'status.json', dict(stage=stage, controller_pid=os.getpid(),
            host=socket.gethostname(), gpu='0', updated_at=now(), **extra))
    data = source / 'evaluation_data'
    inventory = json.loads((data / 'package_inventory.json').read_text())
    for name, expected in inventory['sha256'].items():
        assert sha(data / name) == expected, name
    previous_plan = json.loads((source / 'experiment.json').read_text())
    assert sha(data / 'package_inventory.json') == previous_plan['evaluation_inventory_sha256']
    originals = {(r['id'], r['modality']): r for r in rows(source / 'evaluation/atomic_replay/predictions.jsonl')}
    labels = {r['id']: r for r in rows(data / 'audit/eval_labels.jsonl')}
    input_audit = []
    for mod, filename in [('T', 'eval_text.jsonl'), ('V', 'eval_multimodal.jsonl')]:
        selected = rows(data / 'shared' / filename)
        assert len(selected) == 132
        for row in selected:
            previous, current = old_payload(row, data), eval_payload(row, data)
            assert previous['messages'][-1]['content'][:-len(OLD_SUFFIX)] == current['messages'][-1]['content'][:-len(COT_SUFFIX)]
            assert previous.get('images') == current.get('images')
            assert digest(previous) == originals[row['id'], mod]['input_sha256']
            assert not any('\\boxed{' + c + '}' in current['messages'][-1]['content'] for c in 'ABCD')
            assert set(current) <= {'messages', 'images'}
            input_audit.append(dict(id=row['id'], qid=row['qid'], modality=mod,
                probe_id=labels[row['id']]['probe_id'], input_sha256=digest(current),
                previous_input_sha256=digest(previous), images=current.get('images', [])))
    assert len(input_audit) == 264
    save(output / 'input_audit.json', dict(created_at=now(), rows=input_audit,
        questions_options_images_unchanged=True, concrete_boxed_letter_examples_absent=True,
        audit_labels_not_in_inference_payload=True, inventory_sha256=sha(data / 'package_inventory.json')))
    snapshots = output / 'source_snapshots'
    names = ['run_entity50_neutral_eval_v5.py', 'evaluate_entityqa_neutral_v5.py',
             'evaluate_entityqa_fullfact_v2.py', 'entityqa_common.py', 'entityqa_prompt_neutral_v5.py',
             'answer_readout.py', 'answer_readout_entity_v2.py', 'answer_readout_entity_v3.py',
             'answer_readout_entity_v4.py', 'test_entity_neutral_prompt.py']
    if not snapshots.exists():
        snapshots.mkdir()
        for name in names:
            shutil.copy2(root / 'scripts' / name, snapshots / name)
    snapshot_hashes = {p.name: sha(p) for p in snapshots.glob('*.py')}
    adapter = source / 'atomic_replay/train/TEXT/checkpoint-237/adapter_model.safetensors'
    validation = json.loads((source / 'atomic_replay/train/TEXT/validation.json').read_text())
    assert sha(adapter) == validation['adapter_sha256']
    merge_manifest = source / 'atomic_replay/merged/merge_manifest.json'
    assert json.loads(merge_manifest.read_text())['adapter_sha256'] == validation['adapter_sha256']
    plan = dict(created_at=now(), status='running', purpose='Fix concrete boxed-letter anchoring; same fixed E23 weights/questions',
        prompt_version=PROMPT_VERSION, prompt_messages='one user message; no system message',
        cot_suffix=COT_SUFFIX, no_concrete_answer_examples=True, retraining=False,
        source=str(source), dataset_inventory_sha256=sha(data / 'package_inventory.json'),
        adapter_sha256=validation['adapter_sha256'], merged_manifest_sha256=sha(merge_manifest),
        original_questions_per_test_modality=100, rotation_questions_per_test_modality=32,
        knowledge_units=50, models=['atomic_replay', 'base'], new_generations_per_model=264,
        cot='brief prompted reasoning with explicit do not overthink; no native thinking prefix',
        max_tokens=65536, wall_seconds=180, temperature=0, seed=42,
        readout='entity-readout-v4-boxed-text-letter',
        primary='raw accuracy and invalid rate; constrained finalization supplementary only',
        old_generations_reused=False, checkpoint_selection='same pre-fixed final epoch 3',
        source_sha256=snapshot_hashes,
        limitations=['Development experiment; 4 entities were used for prompt diagnosis, all 50 are development entities.',
                     'Neutral prompt changes the instruction bundle including removing the system message, not only one phrase.',
                     'No VISION-trained model here; only base, T2T and T2V.',
                     'Source factual/temporal validity remains inherited from candidate data.'])
    if (output / 'experiment.json').exists():
        prior = json.loads((output / 'experiment.json').read_text())
        for key in ['prompt_version', 'cot_suffix', 'source_sha256', 'adapter_sha256', 'dataset_inventory_sha256']:
            assert prior[key] == plan[key], key
        plan = prior
    else:
        save(output / 'experiment.json', plan)
    tmp = Path('/tmp/xm-entity50-neutral-v5')
    tmp.mkdir(mode=0o700, exist_ok=True)
    assert tmp.stat().st_uid == os.getuid()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1', MODELSCOPE_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', MAX_PIXELS='262144',
        TMPDIR=str(tmp), VLLM_RPC_BASE_PATH=str(tmp), VLLM_WORKER_MULTIPROC_METHOD='spawn', PYTHONUNBUFFERED='1')
    env.pop('PYTORCH_CUDA_ALLOC_CONF', None)
    for arm in ['atomic_replay', 'base']:
        dest = output / arm
        summary = dest / 'summary.json'
        if summary.exists() and json.loads(summary.read_text())['status'] == 'complete':
            continue
        while True:
            try:
                idle('0')
                break
            except RuntimeError as error:
                status('waiting_for_free_gpu', next_arm=arm, reason=str(error))
                time.sleep(15)
        model = root / 'model_cache/Qwen3-VL-4B-Instruct' if arm == 'base' else source / 'atomic_replay/merged'
        cmd = [str(root / '.venv-vllm/bin/python'), '-u', str(snapshots / 'evaluate_entityqa_neutral_v5.py'),
               '--data', str(data), '--model', str(model), '--output', str(dest),
               '--arm', 'base' if arm == 'base' else 'TEXT', '--gpu', '0',
               '--max-tokens', '65536', '--wall-seconds', '180']
        save(output / (arm + '_command.json'), cmd)
        with (output / (arm + '.log')).open('a') as stream:
            proc = subprocess.Popen(cmd, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
        status('evaluating_' + arm, child_pid=proc.pid)
        if proc.wait():
            raise RuntimeError('Evaluation failed: ' + arm + '; inspect log')
        for _ in range(15):
            try:
                idle('0')
                break
            except RuntimeError:
                time.sleep(2)
    reports = {}
    tuned_qids = set(json.loads((source / 'option_bias_diagnostic/plan.json').read_text())['qids'])
    new_configs = []
    for arm in ['atomic_replay', 'base']:
        current = rows(output / arm / 'predictions.jsonl')
        previous = rows(source / 'evaluation' / arm / 'predictions.jsonl')
        new_configs.append(json.loads((output / arm / 'config.json').read_text()))
        assert len(current) == len(previous) == 264
        prior = {(r['id'], r['modality']): r for r in previous}
        assert all(r['gold'] == prior[r['id'], r['modality']]['gold'] for r in current)
        groups = {}
        for mod in ['T', 'V']:
            for kind in ['original', 'rotation']:
                group = [r for r in current if r['modality'] == mod and ((r['probe_id'] == 'rotation') == (kind == 'rotation'))]
                old = [prior[r['id'], r['modality']] for r in group]
                assert len(group) == (100 if kind == 'original' else 32)
                item = dict(current=metric(group), previous_prompt=metric(old))
                if kind == 'original':
                    item['prompt_diagnostic_4_entities'] = metric([r for r in group if r['qid'] in tuned_qids])
                    item['other_46_entities'] = metric([r for r in group if r['qid'] not in tuned_qids])
                    item['direct'] = metric([r for r in group if r['probe_id'] == 'direct'])
                    item['statement'] = metric([r for r in group if r['probe_id'] == 'statement'])
                groups[mod + '/' + kind] = item
        reports[arm] = groups
    assert new_configs[0]['input_sha256'] == new_configs[1]['input_sha256']
    current_a = {(r['id'], r['modality']): r for r in rows(output / 'atomic_replay/predictions.jsonl')}
    current_b = {(r['id'], r['modality']): r for r in rows(output / 'base/predictions.jsonl')}
    assert set(current_a) == set(current_b)
    assert all(current_a[k]['input_sha256'] == current_b[k]['input_sha256'] for k in current_a)
    assert all(sha(snapshots / name) == expected for name, expected in plan['source_sha256'].items())
    save(output / 'comparison.json', dict(status='complete', reports=reports, completed_at=now(),
        inference_inputs_match_between_models=True, primary='raw generations; no finalization in primary metrics'))
    plan.update(status='complete', completed_at=now())
    save(output / 'experiment.json', plan)
    status('complete', results='comparison.json')


if __name__ == '__main__':
    try:
        run()
    except Exception as error:
        save(Path(__file__).resolve().parents[1] / 'runs/entity50_repeat10_lora_ep3_20261104/evaluation_neutral_v5_20261104/status.json',
             dict(stage='failed', error=repr(error), controller_pid=os.getpid(), updated_at=now()))
        raise
