#!/usr/bin/env python3
"""Build an immutable-source, exactly answer-balanced entity1020 derivative."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import os
import random
import re
import shutil
from collections import Counter
from pathlib import Path

from entity_v19_experiment_data import ROOT, now, read, rows, save, sha, write_rows

SOURCE = ROOT / 'entity1020/alternate_replaced_1572_cleaned'
DEST = ROOT / 'entity1020/balanced_1572'
LETTERS = 'ABCD'


def choices(text):
    marks = list(re.finditer(r'(?m)^([A-D])\.\s', text))
    assert [m.group(1) for m in marks] == list(LETTERS)
    end = text.index('\n\nAnswer with exactly one uppercase letter', marks[-1].end())
    options = [text[m.end():marks[i+1].start() if i < 3 else end].rstrip('\n')
               for i, m in enumerate(marks)]
    return text[:marks[0].start()], options, text[end:]


def reorder(text, permutation):
    prefix, options, suffix = choices(text)
    return prefix + '\n'.join(f'{letter}. {options[old]}'
        for letter, old in zip(LETTERS, permutation)) + suffix


def validate(dest=DEST):
    meta = read(dest / 'balance_manifest.json')
    assert meta and meta['seed'] == 42
    for rel, digest in meta['source_hashes'].items():
        assert sha(SOURCE / rel) == digest, rel
    for rel, digest in read(dest / 'package_inventory.json')['sha256'].items():
        assert sha(dest / rel) == digest, rel
    labels = {r['id']: r for r in rows(dest / 'audit/eval_labels.jsonl')}
    mapping = {r['id']: r for r in rows(dest / 'audit/answer_permutations.jsonl')}
    audit = {r['sample_id']: r for r in rows(dest / 'audit/sample_audit.jsonl')}
    answers = {r['id']: r for r in rows(dest / 'source/answers.jsonl')}
    source_answers = {r['id']: r for r in rows(SOURCE / 'source/answers.jsonl')}
    questions = {(r['id'], r['type']): r for r in rows(dest / 'source/questions.jsonl')}
    original_questions = {(r['id'], r['type']): r for r in rows(SOURCE / 'source/questions.jsonl')}
    for probe in ('direct', 'statement'):
        assert Counter(r['answer'] for r in labels.values() if r['probe_id'] == probe) == dict.fromkeys(LETTERS, 393)
    assert len(mapping) == len(labels) == 3144
    for direction in ('T2T', 'T2V', 'V2T', 'V2V'):
        assert sha(dest / direction / 'train.jsonl') == sha(SOURCE / direction / 'train.jsonl')
        originals = {r['id']: r for r in rows(SOURCE / direction / 'eval.jsonl')}
        for item in rows(dest / direction / 'eval.jsonl'):
            sid = item['id']; original = originals[sid]; record = mapping[sid]
            before = choices(original['messages'][-1]['content'])
            after = choices(item['messages'][-1]['content'])
            assert before[0] == after[0] and before[2] == after[2]
            assert Counter(before[1]) == Counter(after[1])
            assert before[1][LETTERS.index(record['old_answer'])] == after[1][LETTERS.index(record['new_answer'])]
            untouched = copy.deepcopy(item)
            untouched['messages'][-1]['content'] = original['messages'][-1]['content']
            assert untouched == original
            label = labels[sid]; a = audit[sid]; uid = a['unit_id']; probe = label['probe_id']
            assert label['answer'] == a['answer'] == answers[uid]['answers'][probe] == record['new_answer']
            q = questions[uid, probe]
            oldq = original_questions[uid, probe]
            assert q['options'][LETTERS.index(label['answer'])]['text'] == oldq['options'][LETTERS.index(source_answers[uid]['answers'][probe])]['text']
    assert rows(dest / 'T2T/eval.jsonl') == rows(dest / 'V2T/eval.jsonl') == rows(dest / 'shared/eval_text.jsonl')
    assert rows(dest / 'T2V/eval.jsonl') == rows(dest / 'V2V/eval.jsonl') == rows(dest / 'shared/eval_multimodal.jsonl')
    return dict(status='passed', units=1572, questions=3144, per_type=dict.fromkeys(LETTERS,393),
                pooled=dict.fromkeys(LETTERS,786), correct_option_text_preserved=True,
                all_four_directions_aligned=True, training_unchanged=True, checked_at=now())


_original_questions = None
def rows_cache_original_questions():
    global _original_questions
    if _original_questions is None:
        _original_questions = rows(SOURCE / 'source/questions.jsonl')
    return _original_questions


def build(dest=DEST):
    if (dest / 'balance_manifest.json').exists():
        result = validate(dest)
        print(result)
        return
    assert not dest.exists(), f'Partial destination requires inspection: {dest}'
    # Images are immutable hard links; text/metadata are independent copies.
    shutil.copytree(SOURCE, dest, ignore=shutil.ignore_patterns('images', 'runs', '__pycache__'))
    shutil.copytree(SOURCE / 'images', dest / 'images', copy_function=os.link)
    labels = rows(SOURCE / 'audit/eval_labels.jsonl')
    audit = {r['sample_id']: r for r in rows(SOURCE / 'audit/sample_audit.jsonl')}
    units = sorted({r['unit_id'] for r in audit.values()})
    assert len(units) == 1572
    rng = random.Random(42)
    # Balanced margins and almost uniform joint labels prevent a fixed relation
    # between the two question types for the same entity (16 cells: 98 or 99).
    pairs = [(a,b) for a in LETTERS for b in LETTERS for _ in range(98)] + [(a,a) for a in LETTERS]
    rng.shuffle(pairs)
    target = {uid: dict(zip(('direct','statement'), pair)) for uid,pair in zip(units,pairs)}
    mapping = {}
    for label in sorted(labels, key=lambda r:r['id']):
        sid = label['id']; uid = audit[sid]['unit_id']; probe = label['probe_id']
        new = target[uid][probe]; old = label['answer']
        incorrect = [i for i in range(4) if LETTERS[i] != old]
        rng.shuffle(incorrect)
        perm = incorrect.copy(); perm.insert(LETTERS.index(new),LETTERS.index(old))
        mapping[sid] = dict(id=sid,unit_id=uid,qid=label['qid'],probe_id=probe,
                           old_answer=old,new_answer=new,new_position_to_old_index=perm)
        label['answer'] = new
    write_rows(dest / 'audit/eval_labels.jsonl',labels)
    write_rows(dest / 'audit/answer_permutations.jsonl',mapping.values())
    for r in audit.values(): r['answer'] = mapping[r['sample_id']]['new_answer']
    write_rows(dest / 'audit/sample_audit.jsonl',audit.values())
    for rel in [f'{d}/eval.jsonl' for d in ('T2T','T2V','V2T','V2V')] + ['shared/eval_text.jsonl','shared/eval_multimodal.jsonl']:
        values = rows(SOURCE / rel)
        for r in values:
            r['messages'][-1]['content'] = reorder(r['messages'][-1]['content'], mapping[r['id']]['new_position_to_old_index'])
        write_rows(dest / rel,values)
    questions = rows(SOURCE / 'source/questions.jsonl')
    by_unit_probe = {(r['unit_id'],r['probe_id']):r for r in mapping.values()}
    for q in questions:
        perm = by_unit_probe[q['id'],q['type']]['new_position_to_old_index']
        old = q['options']; q['options'] = [{**old[i],'id':letter} for letter,i in zip(LETTERS,perm)]
    write_rows(dest / 'source/questions.jsonl',questions)
    answers = rows(SOURCE / 'source/answers.jsonl')
    for r in answers: r['answers'] = target[r['id']]
    write_rows(dest / 'source/answers.jsonl',answers)
    source_inventory = read(SOURCE / 'package_inventory.json')['sha256']
    save(dest / 'balance_manifest.json',dict(schema='entity1020-exact-answer-balance-v1',created_at=now(),
         source=str(SOURCE),seed=42,units=1572,questions=3144,
         per_type_before={p:dict(Counter(r['old_answer'] for r in mapping.values() if r['probe_id']==p)) for p in ('direct','statement')},
         per_type_after={p:dict.fromkeys(LETTERS,393) for p in ('direct','statement')},
         pooled_after=dict.fromkeys(LETTERS,786),
         joint_type_labels=dict(Counter(target[u]['direct']+target[u]['statement'] for u in units)),
         source_hashes=source_inventory,training_unchanged=True,images_unchanged=True,
         original_provenance_records='source/knowledge_samples.jsonl retains original generation options; use source/questions+answers and audit/eval_labels for balanced order'))
    for name in ('build_manifest.json','summary.json'):
        m = read(dest/name); m['answer_balance']='exact per question type'; m['option_order_changed']=True
        m['answers_and_question_text_unchanged']=False
        m['question_stems_and_option_texts_unchanged']=True
        save(dest/name,m)
    (dest/'README.md').write_text('# entity1020：答案位置均衡版\n\n'
        '1,572 个实体、3,144 道题。direct 和 statement 各 1,572 题，A/B/C/D 各 393 题；'
        '合计各 786 题。两类题联合答案的 16 种组合各 98 或 99 个实体。\n\n'
        '仅重排选项并同步答案，不改变题干、选项文本、训练材料或图片。四方向及 shared 文件一致。'
        '原始数据保持不变；图片使用硬链接，请勿直接编辑图片。\n\n'
        '每题映射见 audit/answer_permutations.jsonl；原始生成证据 knowledge_samples 保留历史顺序，'
        '均衡后题目与标签以 source/questions.jsonl、source/answers.jsonl、audit/eval_labels.jsonl 为准。\n\n'
        '2025-12-31 为旧知识分界，学习截点后的更新；尚未逐条核验事实首次公开日期。'
        '原包的图片身份认证限制继续适用。\n')
    # Run the upstream structural verifier before indexing local verification.
    spec = importlib.util.spec_from_file_location('entity1020_package_verifier', dest/'verify_package.py')
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    (dest/'package_inventory.json').unlink()
    save(dest/'verification.json',verifier.check(dest))
    inventory = {str(p.relative_to(dest)):sha(p) for p in sorted(dest.rglob('*'))
                 if p.is_file() and p.name!='package_inventory.json'}
    save(dest/'package_inventory.json',dict(sha256=inventory,file_count=len(inventory)+1,self_hash_excluded=True))
    result = validate(dest)
    # Store separate from the package inventory to keep it a stable derivative.
    save(dest.parent/'balanced_1572_validation.json',result)
    print(result)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--verify',action='store_true')
    args = ap.parse_args()
    print(validate()) if args.verify else build()
