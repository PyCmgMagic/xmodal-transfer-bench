#!/usr/bin/env python3
"""Frozen 50-entity TEXT learning experiment: ten repeats per QA per epoch."""
import argparse
import copy
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
from types import SimpleNamespace

from answer_readout import options_from_question
from entityqa_common import rows, save, sha, digest, write_rows, now, eval_payload
from prepare_entity_atomic_pilot import SEED, entity_examples, generic, slots
from run_entityqa_lora import command, checkpoint, idle, merge


def prepare_learning(root, work):
    """This function reads learning sources only, never formal questions/golds."""
    source = root / 'runs/entity1025_fullfact_lora_ep3_20261104'
    qa_source = root / 'runs/entity1025_repaired_attrqa_lora_ep3_20261103'
    records = rows(source / 'data/learning_inputs.jsonl')
    qas = {r['id']: r for r in rows(qa_source / 'data/qa_pairs.jsonl')}
    selected = sorted(records, key=lambda r: hashlib.sha256((SEED + r['id']).encode()).hexdigest())[:50]
    assert len(selected) == len({r['qid'] for r in selected}) == 50
    core, recall = [], []
    for record in selected:
        qa = qas[record['id']]
        assert qa['source_sha256'] == digest(record)
        assert qa['answer'] in record['learning']
        examples = [r for r in entity_examples(record, qa) if r['variant'] in ['fact_0', 'attribute_0']]
        assert len(examples) == 2
        core.extend(examples)
        recall.extend([
            dict(id=record['id'] + '-attribute', qid=record['qid'], kind='trained_attribute_recall',
                 messages=[{'role': 'user', 'content': qa['question_text']}], expected=qa['answer']),
            dict(id=record['id'] + '-neutral', qid=record['qid'], kind='neutral_claim_recall',
                 messages=[{'role': 'user', 'content': f'What is the learned information concerning {record["name"]}? Answer concisely with the precise facts.'}],
                 expected=record['claim'], required_slots=[s[2] for s in slots(record['claim'])]),
            dict(id=record['id'] + '-exact', qid=record['qid'], kind='exact_training_claim_recall',
                 messages=[examples[0]['messages'][0]], expected=record['claim']),
        ])
    training = []
    for repeat in range(10):
        for row in core:
            item = copy.deepcopy(row)
            item['id'] += f'-repeat-{repeat}'
            training.append(item)
    retention = [generic(i, 'train') for i in range(252)]
    assert Counter(r['answer'] for r in retention) == dict.fromkeys('ABCD', 63)
    training.extend(retention)
    assert len(core) == 100 and len(training) == 1252
    assert len({r['id'] for r in training}) == 1252
    exposures = Counter((r['knowledge_id'], r['variant']) for r in training if 'knowledge_id' in r)
    assert len(exposures) == 100 and set(exposures.values()) == {10}
    assert not any(r.get('images') for r in training)
    write_rows(work / 'atomic_replay/data/train_text.jsonl', training)
    write_rows(work / 'data/entity_unique_training.jsonl', core)
    write_rows(work / 'data/learning_inputs.jsonl', selected)
    write_rows(work / 'data/recall_dev.jsonl', recall)
    write_rows(work / 'data/retention_dev.jsonl', [generic(i, 'dev') for i in range(96)])
    train_prompts = {r['messages'][0]['content'] for r in retention}
    assert train_prompts.isdisjoint(r['messages'][0]['content'] for r in rows(work / 'data/retention_dev.jsonl'))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(root / 'model_cache/Qwen3-VL-4B-Instruct'), local_files_only=True)
    lengths = [len(tokenizer.apply_chat_template(r['messages'], tokenize=True, enable_thinking=False)) for r in core + retention]
    assert max(lengths) <= 2048, max(lengths)
    save(work / 'data/token_audit.json', dict(unique_entity_QA=100, independent_instruction_QA=252,
        max_formatted_tokens=max(lengths), truncation_required=False,
        factual_target_tokens_per_epoch=10 * sum(len(tokenizer.encode(r['messages'][1]['content'], add_special_tokens=False)) for r in core),
        instruction_target_tokens_per_epoch=sum(len(tokenizer.encode(r['messages'][1]['content'], add_special_tokens=False)) for r in retention)))
    manifest = dict(version='entity50-repeat10-v1', created_at=now(), knowledge_units=50,
        selection='First 50 SHA256(' + SEED + ' + knowledge ID), no selection by score; includes previous 32 development entities',
        development_qids=[r['qid'] for r in selected], unique_factual_QA=100,
        repetitions_per_QA_per_epoch=10, exposures_per_QA_over_3_epochs=30,
        factual_rows_per_epoch=1000, independent_instruction_rows_per_epoch=252,
        total_rows_per_epoch=1252, test_data_used_for_training=False,
        training_sources=[str(source / 'data/learning_inputs.jsonl'), str(qa_source / 'data/qa_pairs.jsonl')],
        training_source_sha256={str(p): sha(p) for p in [source / 'data/learning_inputs.jsonl', qa_source / 'data/qa_pairs.jsonl']},
        files={str(p.relative_to(work)): sha(p) for p in sorted(work.rglob('*.jsonl'))},
        limitations=['Development experiment; these 50 entities are not future untouched confirmatory entities.',
                    'Repetitions are repeated exposure, not new independent knowledge.',
                    'Four-entity sanity and this 50-entity trial differ in scale and exposure; not a pure repetition ablation.',
                    'Source factual/temporal validity is inherited; matching source does not certify real-world truth.'])
    save(work / 'data/manifest.json', manifest)
    return manifest


def prepare_evaluation(root, work, manifest):
    """Run only after training files/hashes have been frozen."""
    for name, expected in manifest['files'].items():
        assert sha(work / name) == expected, name
    data = root / 'data/entity1025_repaired_20261103'
    dest = work / 'evaluation_data'
    qids = set(manifest['development_qids'])
    selected_ids = {r['id'] for r in rows(work / 'data/learning_inputs.jsonl')}
    # Keep exactly the same eight position-control entities as the 32-entity trial.
    prior_ids = sorted(r['id'] for r in rows(root / 'runs/entity_atomic_retention_pilot_20261104/data/learning_inputs.jsonl'))[:8]
    assert set(prior_ids) <= selected_ids
    labels = {r['id']: r for r in rows(data / 'audit/eval_labels.jsonl')}
    all_labels, outputs = {}, {}
    for modality, filename in [('T', 'eval_text.jsonl'), ('V', 'eval_multimodal.jsonl')]:
        original = [r for r in rows(data / 'shared' / filename) if r['qid'] in qids]
        assert len(original) == 100
        selected = []
        for row in original:
            item = copy.deepcopy(row)
            if item.get('images'):
                item['images'] = [str((data / p).resolve()) for p in item['images']]
            selected.append(item)
            all_labels[item['id']] = labels[item['id']]
        for row in original:
            if not row['id'].endswith('-direct') or row['id'].removesuffix('-direct') not in prior_ids:
                continue
            question = row['messages'][-1]['content']
            opts = options_from_question(question)
            assert len(opts) == 4
            before = question[:question.index('\nA. ')]
            suffix = question[question.index('\n\n', question.index('\nD. ')):]
            for shift in range(4):
                item = copy.deepcopy(row)
                item['id'] += f'-rotation-{shift}'
                letters = 'ABCD'
                reordered = letters[shift:] + letters[:shift]
                item['messages'][-1]['content'] = before + '\n' + '\n'.join(
                    f'{new}. {opts[old]}' for new, old in zip(letters, reordered)) + suffix
                if item.get('images'):
                    item['images'] = [str((data / p).resolve()) for p in item['images']]
                selected.append(item)
                label = copy.deepcopy(labels[row['id']])
                label.update(id=item['id'], probe_id='rotation', answer=letters[reordered.index(label['answer'])])
                all_labels[item['id']] = label
        assert len(selected) == 132
        outputs[modality] = selected
        write_rows(dest / 'shared' / filename, selected)
    assert {r['id'] for r in outputs['T']} == {r['id'] for r in outputs['V']}
    write_rows(dest / 'audit/eval_labels.jsonl', list(all_labels.values()))
    save(dest / 'package_inventory.json', dict(version='entity50-repeat10-dev-v1', knowledge_units=50,
        original_questions_per_modality=100, rotation_questions_per_modality=32,
        development_only=True, rotation_knowledge_ids=prior_ids,
        source_inventory_sha256=sha(data / 'package_inventory.json'),
        training_manifest_sha256=sha(work / 'data/manifest.json'),
        sha256={str(p.relative_to(dest)): sha(p) for p in sorted(dest.rglob('*.jsonl'))}))
    base_sources = [root / 'runs/entity1025_fullfact_lora_ep3_20261104/evaluation/base/predictions.jsonl',
                    root / 'runs/entity_atomic_retention_pilot_20261104/evaluation/base/predictions.jsonl']
    base = {}
    for source in base_sources:
        for r in rows(source):
            base.setdefault((r['id'], r['modality']), r)
    reusable = []
    for modality, selected in outputs.items():
        for r in selected:
            previous = base.get((r['id'], modality))
            if previous is not None:
                assert previous['input_sha256'] == digest(eval_payload(r, dest))
                assert previous['gold'] == all_labels[r['id']]['answer']
                reusable.append(previous)
    assert len(reusable) == 264
    write_rows(work / 'data/base_verified_predictions.jsonl', reusable)
    save(work / 'data/base_reuse_manifest.json', dict(created_at=now(), rows=len(reusable),
        identical_input_sha256_checked=True, raw_generation_unchanged=True,
        sources={str(p): sha(p) for p in base_sources}))


def summarize(work):
    reports = {}
    for arm in ['base', 'atomic_replay']:
        predictions = rows(work / f'evaluation/{arm}/predictions.jsonl')
        assert len(predictions) == 264
        groups = {}
        for modality in ['T', 'V']:
            for kind in ['original', 'rotation']:
                group = [r for r in predictions if r['modality'] == modality
                         and ((r['probe_id'] == 'rotation') == (kind == 'rotation'))]
                assert len(group) == (100 if kind == 'original' else 32)
                groups[modality + '/' + kind] = dict(n=len(group),
                    accuracy=sum(r['raw_predicted'] == r['gold'] for r in group) / len(group),
                    invalid_rate=sum(r['raw_predicted'] is None for r in group) / len(group),
                    answer_distribution=dict(Counter(str(r['raw_predicted']) for r in group)),
                    stop_reasons=dict(Counter(r['stop_reason'] for r in group)))
        recall = rows(work / f'diagnostics/{arm}/predictions.jsonl')
        assert len(recall) == 198
        diagnostics = {}
        for kind in ['trained_attribute_recall', 'exact_training_claim_recall', 'neutral_claim_recall', 'independent_instruction']:
            group = [r for r in recall if r['kind'] == kind]
            d = dict(n=len(group), length_limited=sum(r['finish_reason'] == 'length' for r in group))
            if kind == 'independent_instruction':
                d.update(accuracy=sum(r['correct'] for r in group) / len(group),
                         invalid_rate=sum(r['predicted'] is None for r in group) / len(group))
            else:
                d.update(strict_target_span_hit_rate=sum(r['target_span_hit'] for r in group) / len(group),
                         exact_string_match_rate=sum(r['response'] == r['expected'] for r in group) / len(group),
                         mean_token_f1=sum(r['token_f1'] for r in group) / len(group))
            diagnostics[kind] = d
        reports[arm] = dict(mcq=groups, diagnostics=diagnostics)
    save(work / 'comparison.json', dict(status='complete', reports=reports, completed_at=now(),
        primary_score='Raw generated answer, uniformly parsed with entity readout v4; constrained finalization is supplementary only.',
        interpretation='50-entity TEXT development experiment, not a confirmatory transfer estimate. Recall scores are lexical diagnostics.'))


def run(args):
    import torch
    torch.set_num_threads(4)
    root, work = args.root.resolve(), args.work.resolve()
    assert socket.gethostname() == 'softengine-KVM'
    work.mkdir(parents=True, exist_ok=True)
    lock = (work / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def status(stage, **extra):
        save(work / 'status.json', dict(stage=stage, controller_pid=os.getpid(), host=socket.gethostname(),
                                      gpu='0', updated_at=now(), **extra))
    status('preparing_learning')
    if (work / 'data/manifest.json').exists():
        manifest = json.loads((work / 'data/manifest.json').read_text())
        for name, expected in manifest['files'].items():
            assert sha(work / name) == expected, name
    else:
        manifest = prepare_learning(root, work)
    if not (work / 'evaluation_data/package_inventory.json').exists():
        prepare_evaluation(root, work, manifest)
    snapshots = work / 'source_snapshots'
    if not snapshots.exists():
        snapshots.mkdir()
        for name in ['run_entity_repeat10_50.py', 'prepare_entity_atomic_pilot.py', 'run_entityqa_lora.py',
                     'entityqa_common.py', 'answer_readout.py', 'answer_readout_entity_v2.py',
                     'answer_readout_entity_v3.py', 'answer_readout_entity_v4.py']:
            shutil.copy2(root / 'scripts' / name, snapshots / name)
        evaluator = (root / 'scripts/evaluate_entityqa_fullfact_v2.py').read_text().replace(
            'from answer_readout_entity_v2 import parse_answer, VERSION',
            'from answer_readout_entity_v4 import parse_answer, VERSION')
        (snapshots / 'evaluate_mcq.py').write_text(evaluator)
        evaluator = (root / 'scripts/evaluate_entity_atomic_recall.py').read_text().replace(
            'from answer_readout_entity_v3 import parse_answer',
            'from answer_readout_entity_v4 import parse_answer')
        (snapshots / 'evaluate_recall.py').write_text(evaluator)
    plan = dict(status='running', created_at=now(), host=socket.gethostname(), gpu='0',
        base=str(root / 'model_cache/Qwen3-VL-4B-Instruct'), training_framework='ms-swift swift sft',
        modality='TEXT only', knowledge_units=50, epochs=3, learning_rate=1e-4, rank=64, alpha=128,
        repetitions_per_QA_per_epoch=10, unique_factual_QA=100, factual_rows_per_epoch=1000,
        independent_instruction_rows_per_epoch=252, total_rows_per_epoch=1252,
        evaluation=dict(original_T2T_questions=100, original_T2V_questions=100,
            position_control_per_modality=32, original_adapter_recall_per_kind=50,
            independent_instruction_questions=48, brief_cot=True, max_tokens=65536, wall_seconds=180,
            primary='raw generations with uniform v4 readout', base_reuse='all 264 MCQ inputs verified by hash'),
        source_sha256={p.name: sha(p) for p in snapshots.glob('*.py')},
        manifest_sha256=sha(work / 'data/manifest.json'),
        evaluation_inventory_sha256=sha(work / 'evaluation_data/package_inventory.json'),
        development_qids=manifest['development_qids'], limitations=manifest['limitations'],
        automatic_full_retraining=False)
    save(work / 'experiment.json', plan)
    tmp = Path('/tmp/xm-entity50-repeat10')
    tmp.mkdir(mode=0o700, exist_ok=True)
    assert tmp.stat().st_uid == os.getuid()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1', MODELSCOPE_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', MAX_PIXELS='262144',
        NPROC_PER_NODE='1', MASTER_ADDR='127.0.0.1', MASTER_PORT='30377', TMPDIR=str(tmp),
        VLLM_RPC_BASE_PATH=str(tmp), VLLM_WORKER_MULTIPROC_METHOD='spawn', PYTHONUNBUFFERED='1',
        HF_DATASETS_CACHE=str(work / 'dataset_cache'), PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    def execute(cmd, stage):
        while True:
            try:
                idle('0')
                break
            except RuntimeError as error:
                status('waiting_for_free_gpu', next_stage=stage, reason=str(error))
                time.sleep(15)
        save(work / (stage + '_command.json'), cmd)
        with (work / (stage + '.log')).open('a') as stream:
            process = subprocess.Popen(cmd, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
        status(stage, child_pid=process.pid)
        if process.wait():
            raise RuntimeError(stage + ' failed; inspect its log')
        for _ in range(15):
            try:
                idle('0')
                break
            except RuntimeError:
                time.sleep(2)
    base = root / 'model_cache/Qwen3-VL-4B-Instruct'
    cmd, folder = command(SimpleNamespace(root=root, work=work / 'atomic_replay', model=base), 'TEXT', False)
    try:
        ck = checkpoint(folder)
    except (RuntimeError, FileNotFoundError, AssertionError):
        if folder.exists() and any(folder.iterdir()):
            raise RuntimeError('Incomplete existing checkpoint; inspect rather than overwrite')
        execute(cmd, 'training_TEXT')
        ck = checkpoint(folder)
    # Verify actual free generation from the original adapter before serving conversion.
    for arm in ['atomic_replay', 'base']:
        diag = work / f'diagnostics/{arm}/summary.json'
        if not diag.exists() or json.loads(diag.read_text())['status'] != 'complete':
            execute([str(root / '.venv/bin/python'), '-u', str(snapshots / 'evaluate_recall.py'),
                     '--work', str(work), '--model', str(base), '--arm', arm], 'diagnostics_' + arm)
    status('merging_for_batch_inference')
    merge(base, ck, work / 'atomic_replay/merged')
    for arm in ['base', 'atomic_replay']:
        output = work / 'evaluation' / arm
        if (output / 'summary.json').exists() and json.loads((output / 'summary.json').read_text())['status'] == 'complete':
            continue
        cmd = [str(root / '.venv-vllm/bin/python'), '-u', str(snapshots / 'evaluate_mcq.py'),
               '--data', str(work / 'evaluation_data'), '--model', str(base if arm == 'base' else work / 'atomic_replay/merged'),
               '--output', str(output), '--arm', 'base' if arm == 'base' else 'TEXT', '--gpu', '0']
        if arm == 'base':
            cmd += ['--import-predictions', str(work / 'data/base_verified_predictions.jsonl')]
        execute(cmd, 'evaluating_mcq_' + arm)
    summarize(work)
    plan.update(status='complete', completed_at=now())
    save(work / 'experiment.json', plan)
    status('complete', results='comparison.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        save(args.work / 'status.json', dict(stage='failed', error=repr(error), updated_at=now(), controller_pid=os.getpid()))
        raise
