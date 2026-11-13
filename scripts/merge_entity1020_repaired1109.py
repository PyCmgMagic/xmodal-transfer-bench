"""Merge the user-selected 21 repairs into 1088; preserve review status and lineage."""
from __future__ import annotations

import copy
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path

import repair_entity1020_pending402 as common

ROOT = common.ROOT
BASE = ROOT / 'entity1020/audit_filtered_1088'
REPAIRS = ROOT / 'entity1020/pending402_question_repair_v2/candidate'
DEST = ROOT / 'entity1020/merged_1109'
rows, write, save, sha = common.rows, common.write, common.save, common.sha
LETTERS, TYPES, DIRECTIONS = common.LETTERS, common.TYPES, common.DIRECTIONS


def validate():
    m = json.loads((DEST / 'build_manifest.json').read_text())
    for path, digest in m['input_hashes'].items():
        assert sha(Path(path)) == digest, path
    inventory = json.loads((DEST / 'package_inventory.json').read_text())
    for rel, digest in inventory['sha256'].items():
        assert sha(DEST / rel) == digest, rel
    answers = {r['id']: r for r in rows(DEST / 'source/answers.jsonl')}
    questions = {(r['id'], r['type']): r for r in rows(DEST / 'source/questions.jsonl')}
    labels = {r['id']: r for r in rows(DEST / 'audit/eval_labels.jsonl')}
    audits = {r['sample_id']: r for r in rows(DEST / 'audit/sample_audit.jsonl')}
    mappings = {r['id']: r for r in rows(DEST / 'audit/answer_permutations.jsonl')}
    members = {r['unit_id']: r for r in rows(DEST / 'audit/unit_membership.jsonl')}
    assert len(answers) == len(members) == 1109
    assert len(questions) == len(labels) == len(audits) == len(mappings) == 2218
    assert Counter(r['origin_group'] for r in members.values()) == {'base_filtered': 1088, 'repaired_candidate': 21}
    assert all(not r['temporal_novelty_approved'] for r in members.values())
    historical = {r['id'] for r in rows(REPAIRS.parent / 'historical_corrections/source/answers.jsonl')}
    assert not historical & set(answers)
    for name in ('learning', 'evidence'):
        rr = rows(DEST / f'source/{name}.jsonl')
        assert len(rr) == 1109 and {r['id'] for r in rr} == set(answers)
    assert len({r['qid'] for r in answers.values()}) == 1109
    balance = {}
    for typ in TYPES:
        cc = Counter(r['answers'][typ] for r in answers.values())
        assert sum(cc.values()) == 1109 and max(cc.values()) - min(cc.values()) <= 1
        balance[typ] = dict(cc)
    old_questions = {}
    for parent in (BASE, REPAIRS):
        old_questions.update({(r['id'], r['type']): r for r in rows(parent / 'source/questions.jsonl')})
    learning = {r['id']: r['text'] for r in rows(DEST / 'source/learning.jsonl')}
    image_paths = set()
    for direction in DIRECTIONS:
        train, ev = (rows(DEST / direction / f'{phase}.jsonl') for phase in ('train', 'eval'))
        assert len(train) == 1109 and {r['id'] for r in train} == set(answers)
        assert len(ev) == 2218 and {r['id'] for r in ev} == set(labels)
        for phase, rr in (('train', train), ('eval', ev)):
            visual = direction.startswith('V') if phase == 'train' else direction.endswith('V')
            for r in rr:
                assert not set(r) & {'answer', 'gold', 'labels', 'response'}
                assert sum(x['content'].count('<image>') for x in r['messages']) == int(visual)
                assert len(r.get('images', [])) == int(visual)
                for path in r.get('images', []):
                    assert not Path(path).is_absolute() and (DEST / path).is_file()
                    image_paths.add(path)
        for r in train:
            if direction.startswith('T'):
                assert r['messages'][0]['content'] == learning[r['id']]
        for r in ev:
            uid, typ = r['id'].rsplit('-', 1)
            q, before, mapping = questions[uid, typ], old_questions[uid, typ], mappings[r['id']]
            stem, options = common.split_prompt(r['messages'][-1]['content'])
            label = answers[uid]['answers'][typ]
            assert labels[r['id']]['answer'] == audits[r['id']]['answer'] == label == mapping['new_answer']
            assert not audits[r['id']]['automatic_passed']
            assert len(options) == len(set(options)) == 4
            if direction.endswith('T'):
                assert stem == q['question']
                assert options == [o['text'] for o in q['options']]
            assert [o['text'] for o in q['options']] == [before['options'][j]['text'] for j in mapping['new_position_to_previous_index']]
            assert q['options'][LETTERS.index(label)]['text'] == before['options'][LETTERS.index(mapping['old_answer'])]['text']
        if direction == 'V2V':
            train_images = {r['id']: r['images'][0] for r in train}
            for r in ev:
                assert train_images[r['id'].rsplit('-', 1)[0]] != r['images'][0]
    for phase, ds, shared in (
        ('train', ('T2T','T2V'), 'train_text'), ('train', ('V2T','V2V'), 'train_multimodal'),
        ('eval', ('T2T','V2T'), 'eval_text'), ('eval', ('T2V','V2V'), 'eval_multimodal')):
        assert rows(DEST / ds[0] / f'{phase}.jsonl') == rows(DEST / ds[1] / f'{phase}.jsonl') == rows(DEST / 'shared' / f'{shared}.jsonl')
    assert len(image_paths) == 2218
    result = dict(status='passed', knowledge_units=1109, questions_per_direction=2218,
        images=2218, answer_counts=balance, source_hashes_unchanged=True,
        correct_option_text_preserved=True, four_directions_aligned=True,
        train_eval_image_paths_distinct=True, historical_corrections_excluded=True,
        semantic_and_temporal_certification=False)
    save(DEST / 'verification.json', result)
    return result


def build():
    if DEST.exists():
        return validate()
    parents = (BASE, REPAIRS)
    original_answers = {p: {r['id']: r for r in rows(p / 'source/answers.jsonl')} for p in parents}
    assert [len(original_answers[p]) for p in parents] == [1088,21]
    assert not set(original_answers[BASE]) & set(original_answers[REPAIRS])
    frozen = {str(f): sha(f) for p in parents for f in p.rglob('*')
              if f.is_file() and f.suffix in ('.json', '.jsonl', '.md', '.py', '.txt')}
    DEST.mkdir(parents=True)
    # Preserve legacy source/audit files only as provenance: some contain obsolete labels.
    for parent, label in ((BASE,'base_1088'), (REPAIRS,'repairs_21')):
        for sub in ('source','audit','provenance'):
            if (parent / sub).exists():
                shutil.copytree(parent / sub, DEST / 'provenance' / label / sub)
    active = {}
    for name in ('entities','learning','evidence','questions','answers'):
        active[name] = [copy.deepcopy(r) for p in parents for r in rows(p / f'source/{name}.jsonl')]
    answers = {r['id']: r for r in active['answers']}
    origin = {uid: p for p in parents for uid in original_answers[p]}
    members = [dict(unit_id=uid, qid=answers[uid]['qid'], origin_path=str(origin[uid]),
                    origin_group='base_filtered' if origin[uid] == BASE else 'repaired_candidate',
                    review_status='no_specific_issue_identified_not_certified' if origin[uid] == BASE else 'question_repaired_temporal_review_pending',
                    included_by_user_request=True, temporal_novelty_approved=False)
               for uid in sorted(answers)]
    write(DEST / 'audit/unit_membership.jsonl', members)
    rng = random.Random(42)
    mappings = {}
    for ti, typ in enumerate(TYPES):
        targets = [LETTERS[(j+ti) % 4] for j in range(1109)]
        rng.shuffle(targets)
        for uid, target in zip(sorted(answers), targets):
            old = answers[uid]['answers'][typ]; gi = LETTERS.index(old)
            perm = [j for j in range(4) if j != gi]; rng.shuffle(perm)
            perm.insert(LETTERS.index(target), gi)
            mappings[f'{uid}-{typ}'] = dict(id=f'{uid}-{typ}', unit_id=uid, probe_id=typ,
                old_answer=old, new_answer=target, new_position_to_previous_index=perm,
                relative_to_parent=str(origin[uid]))
            answers[uid]['answers'][typ] = target
    for q in active['questions']:
        perm = mappings[f"{q['id']}-{q['type']}"]['new_position_to_previous_index']
        q['options'] = [dict(id=l,text=q['options'][j]['text']) for l,j in zip(LETTERS,perm)]
    for name, rr in active.items():
        write(DEST / f'source/{name}.jsonl', rr)
    # Active evidence paths must resolve to their verbatim parent provenance.
    evidence = rows(DEST / 'source/evidence.jsonl')
    for r in evidence:
        if 'original_record' in r:
            r['original_record'] = 'provenance/repairs_21/' + r['original_record']
    write(DEST / 'source/evidence.jsonl', evidence)
    for name in ('eval_labels','sample_audit'):
        rr = [copy.deepcopy(r) for p in parents for r in rows(p / f'audit/{name}.jsonl')]
        for r in rr:
            sid = r['id'] if name == 'eval_labels' else r['sample_id']
            r['answer'] = mappings[sid]['new_answer']
            if name == 'sample_audit':
                r['parent_automatic_passed'] = r.get('automatic_passed')
                r.update(automatic_passed=False, temporal_novelty_approved=False,
                    merged_origin='base_filtered' if origin[r['unit_id']] == BASE else 'repaired_candidate')
        write(DEST / f'audit/{name}.jsonl', rr)
    write(DEST / 'audit/answer_permutations.jsonl', list(mappings.values()))
    for d in DIRECTIONS:
        for phase in ('train','eval'):
            rr = []
            for p in parents:
                for original in rows(p / d / f'{phase}.jsonl'):
                    r = copy.deepcopy(original)
                    if phase == 'eval':
                        stem, options = common.split_prompt(r['messages'][-1]['content'])
                        perm = mappings[r['id']]['new_position_to_previous_index']
                        r['messages'][-1]['content'] = common.render(stem,[options[j] for j in perm])
                    if 'images' in r:
                        dest_paths = []
                        for path in r['images']:
                            source_image = Path(path) if Path(path).is_absolute() else p / path
                            relative = Path('images') / source_image.parent.name / source_image.name
                            target = DEST / relative; target.parent.mkdir(parents=True,exist_ok=True)
                            if target.exists():
                                assert os.path.samefile(source_image,target)
                            else:
                                os.link(source_image,target)
                            dest_paths.append(relative.as_posix())
                        r['images'] = dest_paths
                    rr.append(r)
            write(DEST / d / f'{phase}.jsonl',rr)
    for name,d,phase in [('train_text','T2T','train'),('train_multimodal','V2T','train'),
                         ('eval_text','T2T','eval'),('eval_multimodal','T2V','eval')]:
        write(DEST / 'shared' / f'{name}.jsonl', rows(DEST / d / f'{phase}.jsonl'))
    meta = dict(schema='entity1020-merged-repaired-1109-v1',knowledge_units=1109,
        base_units=1088,merged_repair_candidates=21,questions_per_direction=2218,
        training_rows_per_direction=1109,evaluation_rows_per_direction=2218,images=2218,
        cutoff='2025-12-31T23:59:59Z',seed=42,release_status='user_selected_merged_candidate_not_fully_certified',
        historical_corrections_included=0,selection_uses_predictions=False,evaluation_run=False,
        semantic_and_temporal_certification=False,source_packages=[str(p) for p in parents],
        input_hashes=frozen,answer_balance='Per type counts differ by at most one.',
        active_label_sources=['source/answers.jsonl','source/questions.jsonl','audit/eval_labels.jsonl'],
        provenance_note='Parent source and audit records retain obsolete positions; for traceability only.',
        images_note='Images are immutable hard links; never edit image bytes in place.')
    save(DEST / 'build_manifest.json',meta)
    save(DEST / 'summary.json',{k:v for k,v in meta.items() if k != 'input_hashes'})
    (DEST / 'README.md').write_text('''# entity1020：合并后的 1,109 组

用户指定合并：审查后保留的 1,088 组 + pending402_question_repair_v2/candidate 的 21 组修题候选。
共 1,109 组知识，每方向训练 1,109 条、评测 2,218 题（direct、statement 各 1,109 题），2,218 张图片。四方向共享知识，不计为四倍独立样本。

保留 2025-12-31 截止口径。21 组的加入不等于新颖性认证通过；1,088 组也只是此前未识别具体问题。逐单位状态见 audit/unit_membership.jsonl。4 组历史纠错未并入。

四方向文件位于 T2T/、T2V/、V2T/、V2V/。图片路径相对于本目录，使用时以本目录作为数据根目录解析。图片是硬链接，不能原地编辑。

统一以 seed=42 重排选项并同步标签；每种题型 ABCD 数量最多差 1。当前标签以 source/answers.jsonl、source/questions.jsonl 和 audit/eval_labels.jsonl 为准。
provenance/ 中保存父版本记录供追溯，包含旧选项次序和旧自动审查状态，不应作活动训练/评测数据。

原始包、1,088 组版和修复版均保持不变。本次仅合并、平衡和结构核验，没有启动训练或评测。核验结果见 verification.json。
''')
    files = {str(p.relative_to(DEST)):sha(p) for p in DEST.rglob('*')
             if p.is_file() and p.suffix != '.jpg' and 'images' not in p.relative_to(DEST).parts}
    save(DEST / 'package_inventory.json',dict(sha256=files))
    result = validate()
    (ROOT / 'entity1020/CURRENT_DATASET.md').write_text(
        '# 当前合并数据版本\n\n当前数据目录：`entity1020/merged_1109/`。\n\n'
        '1,109 组知识、每方向 2,218 道题；其中 21 组为用户指定并入的修复候选，尚未完成时间与语义认证。\n'
        '详见 merged_1109/README.md、audit/unit_membership.jsonl 和 verification.json。\n')
    return result


if __name__ == '__main__':
    print(json.dumps(build(),ensure_ascii=False,indent=2))
