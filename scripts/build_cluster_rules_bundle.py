#!/usr/bin/env python3
"""Build an immutable, portable rules1000 original/repeat10/multiform10 bundle."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import zipfile

from build_v7_experiment_data import name as canonical_name

TOOLS = Path(__file__).resolve().parent / 'cluster_rules'
sys.path.insert(0, str(TOOLS))
from common import ARMS, MODALITIES, DIRECTIONS, content_digest, digest, dump, inside, rows, write_rows
from validate import validate

FORMS = [
    'Explain the rule or codebook named {name}, including its conditions, mappings, and how to apply it.',
    'Describe the complete rule or codebook named {name} and explain how it is applied.',
    'State the full specification of {name}, including the details needed to use it correctly.',
    'Present the definition of {name} and its complete application procedure.',
    'Give a complete account of {name}, preserving all conditions and conventions in the learning material.',
    'What is the complete definition of {name}, and how should it be used?',
    'Explain how {name} works, including all supplied definitions and application details.',
    'Describe {name} in full, retaining the requirements and conventions given in its reference.',
    'Provide the full reference explanation for {name} and describe how to apply it correctly.',
    'State everything specified about {name} in the learning reference, including how to use it.',
]


def prompt(rule_name, variant, modality):
    result = FORMS[variant].format(name=rule_name)
    if modality == 'vision':
        result = '<image>Using this learning reference image, ' + result[0].lower() + result[1:]
    return result


def build(project, out, image_archive=None, resume_partial=False):
    source = project / 'data/rules_repaired_single_image_v7'
    baseline = project / 'runs/qwen3_lora_learning_ep5_20261024/data'
    if out.exists() and not (resume_partial and not (out / 'manifest.json').exists()):
        raise FileExistsError(f'Refusing to overwrite an existing bundle: {out}')
    out.mkdir(parents=True, exist_ok=True)
    knowledge_rows = list(rows(source / 'audit/rule_knowledge.jsonl'))
    knowledge = {r['content']['id']: r['content'] for r in knowledge_rows}
    if len(knowledge_rows) != len(knowledge) or len(knowledge) != 1000:
        raise ValueError('Expected exactly 1000 rules')
    qids = []
    targets, original_images = {}, {}
    source_files = {}

    def record_source(path):
        source_files[str(path.relative_to(project))] = digest(path)

    record_source(source / 'audit/rule_knowledge.jsonl')
    for modality, filename in [('text', 'train_text.jsonl'), ('vision', 'train_multimodal.jsonl')]:
        path = source / 'shared' / filename
        rr = list(rows(path))
        ids = [r['qid'] for r in rr]
        if len(ids) != len(set(ids)) or set(ids) != set(knowledge):
            raise ValueError('Training sources differ from rule inventory')
        if qids and ids != qids:
            raise ValueError('Text/vision source order differs')
        qids = ids
        targets[modality] = {}
        for r in rr:
            if [m['role'] for m in r['messages']] != ['assistant']:
                raise ValueError('Unexpected learning schema')
            answer = r['messages'][0]['content']
            if modality == 'vision':
                if not answer.startswith('<image>') or len(r.get('images', [])) != 1:
                    raise ValueError('Unexpected vision schema')
                answer = answer[len('<image>'):].strip()
                original_images[r['qid']] = r['images']
            learning = knowledge[r['qid']]['material']['learning']
            expected = learning['text' if modality == 'text' else 'visual_text']
            if answer != expected:
                raise ValueError('Source targets differ from knowledge learning material')
            targets[modality][r['qid']] = answer
        write_rows(out / f'audit/source_learning_{modality}.jsonl', rr)
        record_source(path)

    names = {qid: canonical_name(knowledge[qid]) for qid in qids}
    families = Counter(k['fact']['relation'] for k in knowledge.values())
    write_rows(out / 'audit/rule_index.jsonl', [
        {'qid': qid, 'name': names[qid], 'family': knowledge[qid]['fact']['relation'],
         'text_target_sha256': content_digest(targets['text'][qid]),
         'vision_target_sha256': content_digest(targets['vision'][qid])} for qid in qids])
    dump(out / 'augmentation.json', {'method': 'ten_full_definition_question_forms', 'forms': FORMS,
                                    'target_policy': 'original_modality_target_unchanged', 'seed': 42,
                                    'old_48full_24atomic_8logic_curriculum_reproduced': False})

    def make(qid, arm, variant, modality):
        z = {'id': qid if arm == 'original' else f'{qid}::{arm}:{variant:02d}', 'qid': qid,
             'messages': [{'role': 'user', 'content': prompt(names[qid], variant if arm == 'multiform10' else 0, modality)},
                          {'role': 'assistant', 'content': targets[modality][qid]}]}
        if modality == 'vision':
            z['images'] = original_images[qid]
        return z

    # Identical order and rule/variant allocation for both modalities and expanded arms.
    expanded = [(qid, variant) for qid in qids for variant in range(10)]
    random.Random(42).shuffle(expanded)
    for arm in ARMS:
        order = [(qid, 0) for qid in qids] if arm == 'original' else expanded
        for modality in MODALITIES:
            write_rows(out / f'train/{arm}/{modality}.jsonl',
                       (make(qid, arm, variant, modality) for qid, variant in order))
    # Verify complete baseline conversations against the exact previous experiment inputs.
    for modality in MODALITIES:
        old_path = baseline / f'rules_sft_{modality}.jsonl'
        old_rows = list(rows(old_path))
        new_rows = list(rows(out / f'train/original/{modality}.jsonl'))
        if len(old_rows) != len(new_rows):
            raise ValueError('Baseline row count differs')
        for old, new in zip(old_rows, new_rows):
            if old['messages'] != new['messages']:
                raise ValueError('Baseline messages changed')
            if modality == 'vision' and [Path(p).parts[-2:] for p in old['images']] != [Path(p).parts[-2:] for p in new['images']]:
                raise ValueError('Baseline image assignment changed')
        record_source(old_path)

    # Evaluation and gold are copied only AFTER all training records have been constructed.
    for direction in DIRECTIONS:
        src = source / direction / 'eval.jsonl'
        dst = out / 'eval' / f'{direction}.jsonl'
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        record_source(src)
    shutil.copyfile(source / 'audit/eval_labels.jsonl', out / 'eval/gold.jsonl')
    record_source(source / 'audit/eval_labels.jsonl')
    shutil.copyfile(source / 'image_hashes.jsonl', out / 'audit/image_hashes.jsonl')
    record_source(source / 'image_hashes.jsonl')
    for n in ('README.md', 'repair_report_zh.md', 'validation_report.json', 'build_manifest.json'):
        src = source / n
        if src.is_file():
            dst = out / 'audit' / f'source_{n}'
            shutil.copyfile(src, dst)
            record_source(src)

    image_bytes = 0
    images = list(rows(source / 'image_hashes.jsonl'))
    if len(images) != 3000 or len({r['path'] for r in images}) != 3000:
        raise ValueError('Unexpected image inventory')
    archive_stream = image_archive.open('rb', buffering=16 * 1024 * 1024) if image_archive else None
    if archive_stream and hasattr(os, 'posix_fadvise'):
        os.posix_fadvise(archive_stream.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
    archive = zipfile.ZipFile(archive_stream) if archive_stream else None
    try:
        prefix = 'xmodal/data/rules_repaired_single_image_v7/'
        if archive:
            available = {i.filename: i for i in archive.infolist()}
            if not all(prefix + r['path'] in available for r in images):
                raise ValueError('Archive is missing required original images')
            images.sort(key=lambda r: available[prefix + r['path']].header_offset)
        for i, r in enumerate(images, 1):
            dst = inside(out, r['path'])
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.is_file() and digest(dst) == r['repaired_sha256']:
                pass  # Reuse only byte-verified files from this interrupted build.
            elif archive:
                data = archive.read(prefix + r['path'])
                if hashlib.sha256(data).hexdigest() != r['repaired_sha256']:
                    raise ValueError(f'Archived image hash differs: {r["path"]}')
                dst.write_bytes(data)
            else:
                shutil.copyfile(inside(source, r['path']), dst)
                if digest(dst) != r['repaired_sha256']:
                    raise ValueError(f'Image content differs: {r["path"]}')
            image_bytes += dst.stat().st_size
            if i % 500 == 0:
                print(f'Copied and verified {i}/3000 images', flush=True)
    finally:
        if archive:
            archive.close()
        if archive_stream:
            archive_stream.close()

    scripts = [p for p in TOOLS.iterdir() if p.suffix in ('.py', '.slurm')]
    for src in scripts:
        dst = out / 'tools' / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        record_source(src)
    shutil.copyfile(project / 'scripts/rules_dense_training_audit.py', out / 'tools/rules_dense_training_audit.py')
    record_source(project / 'scripts/rules_dense_training_audit.py')
    dst = out / 'audit/builder_snapshot.py'
    shutil.copyfile(Path(__file__).resolve(), dst)
    shutil.copyfile(project / 'scripts/build_v7_experiment_data.py', out / 'audit/build_v7_experiment_data.py')
    record_source(Path(__file__).resolve())
    record_source(project / 'scripts/build_v7_experiment_data.py')
    frozen = json.loads((project / 'runs/rules_full_sft_ep10_20261026/manifest.json').read_text())
    record_source(project / 'runs/rules_full_sft_ep10_20261026/manifest.json')
    dump(out / 'model_identity.json', {'model': 'Qwen/Qwen3-VL-4B-Instruct', 'files': frozen['model']['files'],
                                      'hash_origin': 'recorded_full_sft_manifest_20261026',
                                      'weights_included': False})
    dump(out / 'environment_reference.json', {'versions': {**frozen['versions'], 'peft': '0.18.1', 'datasets': '3.3.2'},
                                             'python': '3.10', 'cuda_build': '12.4',
                                             'note': 'Reference versions, not an environment lock. Adapt the cluster runtime and verify compatibility.'})
    shutil.copyfile(TOOLS / 'README_ZH.md', out / 'README_ZH.md')
    record_source(TOOLS / 'README_ZH.md')
    manifest = {
        'schema_version': 1, 'bundle': out.name, 'created_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'rules_only', 'rules': 1000, 'families': dict(sorted(families.items())),
        'modalities': list(MODALITIES), 'arms': {'original': {'rows_per_rule': 1, 'rows_per_modality': 1000},
                                              'repeat10': {'rows_per_rule': 10, 'rows_per_modality': 10000},
                                              'multiform10': {'rows_per_rule': 10, 'rows_per_modality': 10000}},
        'evaluations': {d: {'rows': 2000, 'file': f'eval/{d}.jsonl'} for d in DIRECTIONS},
        'gold_file': 'eval/gold.jsonl', 'gold_rows': 2000, 'images': 3000, 'image_bytes': image_bytes,
        'default_training': {'framework': 'ms-swift', 'model': 'Qwen3-VL-4B-Instruct', 'tuner_type': 'full',
                             'epochs': 10, 'learning_rate': 1e-5, 'seed': 42, 'effective_batch_size': 16,
                             'compute_budget_policy': 'same_epochs_not_equal_steps'},
        'baseline_messages_equal_to_previous_inputs': True, 'training_targets_unchanged': True,
        'training_uses_evaluation_questions_or_gold': False, 'portable_image_paths': True,
        'model_weights_included': False, 'source_files_sha256': source_files,
        'limitations': ['Prompt augmentation only; not the old full/atomic/boundary curriculum.',
                        'VISION targets are the original captions and can include textual rule information.',
                        'Image hashes verify bytes, not human semantic correctness.',
                        'Same epochs give expanded arms about 10x the sample presentations of original.']}
    dump(out / 'manifest.json', manifest)
    report = validate(out, hashes=False)
    report['source_image_hashes_verified_during_copy'] = 3000
    dump(out / 'validation_report.json', report)
    files = sorted(p for p in out.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    with (out / 'SHA256SUMS').open('w') as stream:
        for path in files:
            stream.write(f'{digest(path)}  {path.relative_to(out)}\n')
    print(json.dumps({'bundle': str(out), 'status': 'built', 'files': len(files),
                      'train_rows_total': 42000, 'images': 3000, 'image_bytes': image_bytes}, indent=2), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--out', type=Path, default=Path('cluster_ready/rules1000_10x_20261027'))
    ap.add_argument('--image-archive', type=Path, help='Optional existing xmodal.zip; every image must match v7 hashes')
    ap.add_argument('--resume-partial', action='store_true', help='Resume an interrupted build lacking manifest.json')
    args = ap.parse_args()
    build(args.project.resolve(), args.out.resolve(), args.image_archive, args.resume_partial)
