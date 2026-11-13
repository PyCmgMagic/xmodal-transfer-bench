"""Build an independent entity1025 repair release from frozen review decisions.

This is a reviewed migration, not an automatic fact checker. Original packages
are read-only. Changes apply to whole paired units and never use model scores.
"""
import argparse
import copy
import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from verify_entity1025_repaired import read, sha, validate


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'data/entity_merged2022_position_annotated'
REVIEW = ROOT / 'reports/entity1025_review_20261103/full_audit'
PRIOR = ROOT / 'data/entity1020/merged_1109'
RELEASE = 'entity1025_repaired_20261103'
LETTERS = 'ABCD'
SUFFIX = ('Answer with exactly one uppercase letter (A, B, C, or D). '
          'Do not include an explanation or any other characters.')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def save_rows(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')


def dhash(im):
    pixels = im.convert('L').resize((9, 8), Image.Resampling.LANCZOS).tobytes()
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | (pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return f'{value:016x}'


def check_parent(inventory):
    for relative, expected in inventory['sha256'].items():
        assert sha(INPUT / relative) == expected, f'Original input changed: {relative}'


def main(output):
    frozen = {
        INPUT / 'package_inventory.json': 'e4424e5b78a91aef1bab8e4425275e62b65fc4ae6abcaa969e59dd6d1f8f8d1c',
        REVIEW / 'adjudications.jsonl': '532cb7226d2f526fa847ea7c474db6e33ea042266d263d3a16a75625e18f7713',
        REVIEW / 'history/repairs/dispositions_402.jsonl': '50c6cb056af79ad9f19c0edef91988f770252e7e7a62d7ca298ada01b9f0a9c8',
        REVIEW / 'history/knowledge/question_audit.jsonl': '11186483b0c6e33eaec0e4cc6f25ea9c49c7642d192d28c6acaa581465e2503d',
        PRIOR / 'source/learning.jsonl': 'c6c4081fcadd352dc6e5722b8c335b4e95721eaaeeaefee4dcf8884adb0a50ce',
        PRIOR / 'source/questions.jsonl': 'c02cb9e3d8445774d4045d865e9e5ac0ea68bb6ddf05afc041c8db888973a43f',
        PRIOR / 'source/answers.jsonl': 'a12c3fa029d1299ea2338eec45130b6d48199a40b5e1b38a59fb581ed303362f',
        PRIOR / 'source/evidence.jsonl': '5f7516545a3d785dd18beeeb5446b43c60550471c8f5b80f9b35d9b3749fe397',
    }
    for path, expected in frozen.items():
        assert sha(path) == expected, f'Re-review modified input: {path}'
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    inventory = json.loads((INPUT / 'package_inventory.json').read_text())
    print('Verifying original input inventory...', flush=True)
    check_parent(inventory)
    source = {name: read(INPUT / f'source/{name}.jsonl') for name in
              ['entities', 'learning', 'questions', 'answers', 'evidence', 'knowledge_samples']}
    entities = {x['qid']: x for x in source['entities']}
    learning = {x['qid']: x for x in source['learning']}
    questions = {(x['qid'], x['type']): x for x in source['questions']}
    answers = {x['qid']: x for x in source['answers']}
    evidence = {x['qid']: x for x in source['evidence']}
    original_entities = copy.deepcopy(entities)
    original_learning = copy.deepcopy(learning)
    original_questions = copy.deepcopy(questions)
    original_answers = copy.deepcopy(answers)
    roles = {x['qid']: x for x in read(INPUT / 'audit/image_roles.jsonl')}
    catalog = read(INPUT / 'audit/image_catalog.jsonl')
    adjudications = read(REVIEW / 'adjudications.jsonl')
    historical_qa = read(REVIEW / 'history/knowledge/question_audit.jsonl')
    historical_units = {x['qid']: x for x in read(REVIEW / 'history/knowledge/unit_audit.jsonl')}
    historical_dispositions = read(REVIEW / 'history/repairs/dispositions_402.jsonl')
    new_temporal = {x['qid']: x for x in read(REVIEW / 'new450_temporal_candidates.jsonl')}
    delete_reasons = defaultdict(list)
    changes = []

    def remove(qid, category, reason, reference=None):
        record = dict(category=category, reason_zh=reason)
        if reference:
            record['evidence_reference'] = reference
        if record not in delete_reasons[qid]:
            delete_reasons[qid].append(record)

    def change(qid, category, field, before, after, reason, **extra):
        if before == after:
            return
        changes.append(dict(qid=qid, name=entities[qid]['name'], category=category,
                            field=field, before=before, after=after, reason_zh=reason, **extra))

    for x in adjudications:
        if x['verdict'] == 'confirmed_content_issue':
            remove(x['qid'], x['category'], x['reason_zh'], x['adjudication_id'])
        if x['category'] in {'illustrated_portrait_requires_source', 'image_entity_timeline_conflict',
                             'same_photograph_different_qids'}:
            remove(x['qid'], x['category'], x['reason_zh'] + ' 未取得可靠独立替代图，整组移除。',
                   x['adjudication_id'])
    for x in historical_qa:
        # Event date alone is not publication evidence. Only the adjudicated,
        # supported pre-cutoff category is excluded here.
        if x['temporal_verdict'] == 'pre_cutoff_supported':
            remove(x['qid'], 'supported_pre_cutoff_publication', x['reason_zh'],
                   f"history/knowledge/question_audit.jsonl:{x['qid']}:{x['question_type']}")
        # Robert Dowd's old nonexclusive-options judgment was false. Tom
        # Courtenay's later partial-elimination qualification is retained.
        if x['structural_verdict'] == 'clear_unique' and x['qid'] != 'Q329798':
            remove(x['qid'], 'unreconstructed_question_shortcut', x['reason_zh'] +
                   ' 当前来源不足以安全重写区分事实及互斥干扰项，移除当前目标。',
                   f"history/knowledge/question_audit.jsonl:{x['qid']}:{x['question_type']}")
    for x in historical_dispositions:
        if x['action'] not in {'quarantine_pending_evidence', 'question_repaired_still_quarantined'}:
            remove(x['qid'], x['action'], x['reason_zh'],
                   f"history/repairs/dispositions_402.jsonl:{x['qid']}")
    remove('Q575733', 'unsupported_exact_job_title',
           '学习和所附new_evidence仅写new international role，不支持gold的具体职务；不按答案反向编造学习内容。')
    remove('Q30335340', 'conflicting_signature_filename',
           '精确签名文件名在证据与答案中冲突，且学习漏该属性；历史后续审查已停用此目标。')
    remove('Q42101', 'nonexclusive_occupations',
           '多种职业可同时成立，百科删减介绍不能使其他职业为假；且旧证据已有这些职业。')
    remove('Q5126233', 'explicit_pre_cutoff_announcement',
           '原new_evidence明确称Network 10于2025-12-23 announced奖章及首次颁奖场次；2026颁奖不等于2026首次公开。',
           'source/evidence.jsonl:Q5126233:new_knowledge.new_evidence')
    for qid, reason in {
        'Q3731697': '姓名卡内照片仅剩球衣及躯干，头部不在原图可用区域，裁切无法补回有效人物肖像。',
        'Q647405': '姓名卡内照片人物的上半脸已被圆形窗口裁掉，去掉姓名区后无法恢复完整可用肖像。',
        'Q7407943': '姓名卡内照片人物的上半脸在窗口外，原文件无可恢复的头部区域。',
        'Q51884025': '姓名卡内为拥挤骑行合照，前景骑手头部被窗口裁掉；无来源支持的可用完整目标肖像或替代图。',
    }.items():
        remove(qid, 'unusable_name_card_photo_after_text_removal', reason,
               'manual_full_size_card_followup_20261103')

    keep = set(entities).difference(delete_reasons)
    assert len(keep) > 0
    print(f'Whole-unit exclusions: {len(delete_reasons)}; retained: {len(keep)}', flush=True)
    output.mkdir(parents=True)
    # Historical copies are clearly isolated and never used as active views.
    for path in INPUT.rglob('*'):
        if path.is_file() and not path.is_relative_to(INPUT / 'images'):
            target = output / 'provenance/input_package' / path.relative_to(INPUT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    for relative in ['adjudications.jsonl', 'adjudication_summary.json',
                     'history/knowledge/unit_audit.jsonl', 'history/knowledge/question_audit.jsonl',
                     'history/repairs/dispositions_402.jsonl',
                     'history/followup/reconstruction_21_decisions.jsonl',
                     'new450_temporal_candidates.jsonl']:
        target = output / 'provenance/review' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REVIEW / relative, target)

    visual_train = {x['qid']: x for x in read(INPUT / 'shared/train_multimodal.jsonl')}
    visual_eval = {x['id']: x for x in read(INPUT / 'shared/eval_multimodal.jsonl')}
    text_eval = {x['id']: x for x in read(INPUT / 'shared/eval_text.jsonl')}

    def ref(qid, role):
        asset = roles[qid][role]
        if 'annotation' in asset:
            return asset['annotation']['phrase']
        kind = entities[qid]['kind']
        noun = {'person': 'person', 'natural_feature': 'place', 'building': 'landmark',
                'spacecraft': 'spacecraft', 'ship': 'ship'}.get(kind, 'entity')
        return f'the {noun} shown in the image'

    def prefix(qid, role):
        return '<image>\n' + ('Focus on the person marked 1 with a red box.\n'
                              if 'annotation' in roles[qid][role] else '')

    def render(qid, probe, stem, options, vision):
        unit_id = learning[qid]['id']
        row = copy.deepcopy((visual_eval if vision else text_eval)[f'{unit_id}-{probe}'])
        row['messages'][-1]['content'] = (prefix(qid, 'evaluation') if vision else '') + stem + '\n' + '\n'.join(
            f'{letter}. {text}' for letter, text in zip(LETTERS, options)) + '\n\n' + SUFFIX
        return row

    def set_visual_train(qid, body, reason, category='visual_proper_name_repair'):
        row = visual_train[qid]
        before = row['messages'][0]['content']
        after = prefix(qid, 'learning') + body
        change(qid, category, 'shared/train_multimodal:content', before, after, reason)
        row['messages'][0]['content'] = after

    def set_visual_eval(qid, probe, stem, options, reason, category='visual_proper_name_repair'):
        unit_id = learning[qid]['id']
        key = f'{unit_id}-{probe}'
        row = render(qid, probe, stem, options, True)
        change(qid, category, f'shared/eval_multimodal:{probe}',
               visual_eval[key]['messages'][-1]['content'], row['messages'][-1]['content'], reason)
        visual_eval[key] = row

    def replace_subject(text, qid, role, names):
        pattern = r'(?<!\w)(?:' + '|'.join(re.escape(s) for s in sorted(names, key=len, reverse=True)) + r')(?!\w)'
        return re.sub(pattern, lambda _: ref(qid, role), text)

    # Earlier repairs have reviewed, grounded question constructions, but no
    # first-publication approval. Reuse content, keeping that limitation explicit.
    prior_learning = {x['qid']: x for x in read(PRIOR / 'source/learning.jsonl')}
    prior_questions = {(x['qid'], x['type']): x for x in read(PRIOR / 'source/questions.jsonl')}
    prior_answers = {x['qid']: x for x in read(PRIOR / 'source/answers.jsonl')}
    prior_evidence = {x['qid']: x for x in read(PRIOR / 'source/evidence.jsonl')}
    candidate_qids = {x['qid'] for x in historical_dispositions
                      if x['action'] == 'question_repaired_still_quarantined'} & keep
    subject_aliases = {'Q201842': ['Adam Scott', 'Scott'], 'Q7821917': ['Antony Blinken', 'Blinken'],
                       'Q694759': ['Siobhain McDonagh', 'McDonagh'],
                       'Q7375304': ['Roz Savage', 'Savage']}
    for qid in sorted(candidate_qids):
        reason = next(x['reason_zh'] for x in historical_dispositions if x['qid'] == qid)
        names = subject_aliases.get(qid, [entities[qid]['name']])
        before = copy.deepcopy(learning[qid])
        learning[qid] = copy.deepcopy(prior_learning[qid])
        change(qid, 'historical_question_reconstruction', 'source/learning', before,
               learning[qid], reason)
        if qid in {'Q235219', 'Q2117578', 'Q468021'}:
            evidence[qid] = copy.deepcopy(prior_evidence[qid])
        # Keep the original correct letter, rather than importing another
        # package's random option order. Distractors remain symmetric.
        for probe in ['direct', 'statement']:
            old_q = copy.deepcopy(questions[qid, probe])
            replacement = copy.deepcopy(prior_questions[qid, probe])
            prior_gold = prior_answers[qid]['answers'][probe]
            gold_text = next(o['text'] for o in replacement['options'] if o['id'] == prior_gold)
            distractors = iter(o['text'] for o in replacement['options'] if o['id'] != prior_gold)
            gold = answers[qid]['answers'][probe]
            replacement['options'] = [dict(id=letter, text=gold_text if letter == gold else next(distractors))
                                      for letter in LETTERS]
            questions[qid, probe] = replacement
            change(qid, 'historical_question_reconstruction', f'source/questions:{probe}',
                   old_q, replacement, reason)
            set_visual_eval(qid, probe, replace_subject(replacement['question'], qid, 'evaluation', names),
                            [replace_subject(x['text'], qid, 'evaluation', names) for x in replacement['options']],
                            reason, 'historical_question_reconstruction')
        set_visual_train(qid, replace_subject(learning[qid]['text'], qid, 'learning', names),
                         reason, 'historical_question_reconstruction')
    save_rows(output / 'provenance/prior_repair_candidates/learning.jsonl',
              [prior_learning[q] for q in sorted(candidate_qids)])
    save_rows(output / 'provenance/prior_repair_candidates/questions.jsonl',
              [prior_questions[q, t] for q in sorted(candidate_qids) for t in ['direct', 'statement']])
    save_rows(output / 'provenance/prior_repair_candidates/answers.jsonl',
              [prior_answers[q] for q in sorted(candidate_qids)])
    save_rows(output / 'provenance/prior_repair_candidates/evidence.jsonl',
              [prior_evidence[q] for q in sorted(candidate_qids)])

    # Recover all protected direct option strings without substituting aliases
    # inside full names, work titles, file names, or relatives' names.
    bad_names = {x['qid'] for x in adjudications if x['category'] == 'visual_proper_name_corruption'} & keep
    for qid in sorted(bad_names):
        row = visual_eval[f"{learning[qid]['id']}-direct"]
        stem = row['messages'][-1]['content'].split('\nA. ', 1)[0]
        stem = stem.removeprefix(prefix(qid, 'evaluation'))
        set_visual_eval(qid, 'direct', stem, [x['text'] for x in questions[qid, 'direct']['options']],
                        '恢复source/questions中的四个原始选项；被考专名不再进行实体别名替换。')

    for qid in ['Q16232342', 'Q3372924']:
        if qid not in keep:
            continue
        name = entities[qid]['name']
        body = learning[qid]['text'].replace(name, ref(qid, 'learning'), 1)
        set_visual_train(qid, body, '只转换句首主体，保留节目名或被考绰号。')
        q = questions[qid, 'statement']
        set_visual_eval(qid, 'statement', replace_subject(q['question'], qid, 'evaluation', [name]),
                        [o['text'].replace(name, ref(qid, 'evaluation'), 1)
                         if o['text'].startswith(name) else o['text'] for o in q['options']],
                        '保留节目名或引号内绰号，只将陈述主语转换为图片指代。')
    if 'Q442547' in keep:
        qid = 'Q442547'
        full = 'Stephen Graham Kelly'
        body = learning[qid]['text'].replace(full, f"{ref(qid, 'learning')}, whose full name is {full},", 1)
        set_visual_train(qid, body, '将完整姓名保留为学习属性，消除图片指代后残留Kelly的半替换。')
    if 'Q7683243' in keep:
        qid = 'Q7683243'
        set_visual_train(qid, learning[qid]['text'].replace('Tania Rachel James', ref(qid, 'learning'), 1),
                         '只替换女儿这个主语，完整保留父亲Koduvathara L. James及母亲姓名。')
        q = questions[qid, 'statement']
        set_visual_eval(qid, 'statement', q['question'],
                        [x['text'].replace('Tania James', ref(qid, 'evaluation'), 1) for x in q['options']],
                        '仅转换句首女儿的指代，不把父亲的James替换为图片中的女儿。')
    if 'Q495501' in keep:
        qid = 'Q495501'
        set_visual_train(qid, learning[qid]['text'].replace('Mount Kumgang', ref(qid, 'learning'), 1),
                         '只替换山体主体，保留Mount Kumgang Tourist Region行政区专名。')
        q = questions[qid, 'statement']
        set_visual_eval(qid, 'statement', q['question'].replace('Mount Kumgang', ref(qid, 'evaluation'), 1),
                        [x['text'].replace('Mount Kumgang', ref(qid, 'evaluation'), 1) for x in q['options']],
                        '仅转换每项陈述的山体主体，保留被考行政区名称。')

    if 'Q1774046' in keep:
        qid = 'Q1774046'
        title = 'Does Democracy Need to be More Populist?'
        assert title in evidence[qid]['new_knowledge']['new_evidence']
        before = learning[qid]['text']
        learning[qid]['text'] = before.replace('gave a lecture', f"gave a lecture titled '{title}'", 1)
        change(qid, 'grounded_missing_learning_attribute', 'source/learning:text', before,
               learning[qid]['text'], '原始new_evidence明确给出演讲标题，补回学习遗漏；未从gold反向编造。')
        set_visual_train(qid, learning[qid]['text'].replace('Marc Jongen', ref(qid, 'learning'), 1),
                         '同步文本与视觉学习中的演讲标题。', 'grounded_missing_learning_attribute')

    # "Who?" questions whose unique picture-reference option reveals the
    # answer are rewritten symmetrically to an attribute supported by learning.
    for qid, stem, values, correct, template in [
        ('Q512678', "What is Maroš Šefčovič's all-time rank by length of service among EU Commissioners in the supplied account?",
         ['first', 'second', 'third', 'fourth'], 'second',
         'In the supplied account, Maroš Šefčovič ranks {value} among EU Commissioners ever by length of service.'),
        ('Q61899299', "In which minute did Strahinja Pavlović score in AC Milan's 1–0 win against Roma on 2 November 2025?",
         ['29th', '39th', '49th', '59th'], '39th',
         "Strahinja Pavlović scored in the {value} minute of AC Milan's 1–0 win against Roma on 2 November 2025."),
    ]:
        if qid not in keep:
            continue
        name = entities[qid]['name']
        for probe in ['direct', 'statement']:
            old_q = copy.deepcopy(questions[qid, probe])
            gold = answers[qid]['answers'][probe]
            alternatives = iter(x for x in values if x != correct)
            opts = [correct if letter == gold else next(alternatives) for letter in LETTERS]
            q = copy.deepcopy(old_q)
            q['question'] = stem if probe == 'direct' else 'Which statement is accurate?'
            q['options'] = [dict(id=letter, text=value if probe == 'direct' else template.format(value=value))
                            for letter, value in zip(LETTERS, opts)]
            questions[qid, probe] = q
            reason = '把只有正确项是图片指代的身份题改为来源支持的属性题，四项采用同一结构。'
            change(qid, 'identity_option_shortcut_reconstruction', f'source/questions:{probe}', old_q, q, reason)
            set_visual_eval(qid, probe, replace_subject(q['question'], qid, 'evaluation', [name]),
                            [replace_subject(x['text'], qid, 'evaluation', [name]) for x in q['options']],
                            reason, 'identity_option_shortcut_reconstruction')

    # Conservatively shorten already-transformed non-person descriptions;
    # substring replacement had corrupted surrounding region/official names.
    if 'Q4706020' in keep:
        qid = 'Q4706020'
        body = (f"In July 2026, {ref(qid, 'learning')} and related fortifications in Iran were inscribed "
                'on the UNESCO World Heritage List under criterion (iii), recognising the cultural tradition '
                "of the Nizari Isma'ilis.")
        set_visual_train(qid, body, '去除误替换的地理区域名，保留来源支持的时间、遗产资格及标准。', 'visual_reference_grammar')
    if 'Q7896483' in keep:
        qid = 'Q7896483'
        body = (f"{ref(qid, 'learning')} is the University of Toronto's law faculty. Its motto is "
                'Regnum juris regnum pacis, often translated as "Rule of Law, Rule of Peace." '
                'Its teaching origins date to 1887; it officially opened in 1889 and operates on the St. George campus.')
        set_visual_train(qid, body, '移除由正式名称替换造成的重复冠词，保留被考格言及学习背景。', 'visual_reference_grammar')
    if 'Q87749354' in keep:
        qid = 'Q87749354'
        stems = {'direct': f"What is the proposed mission name for sending the spare twin of {ref(qid, 'evaluation')} to the Moon?",
                 'statement': f"Which statement about the spare twin of {ref(qid, 'evaluation')} is accurate?"}
        for probe, stem in stems.items():
            q = questions[qid, probe]
            set_visual_eval(qid, probe, stem, [x['text'] for x in q['options']],
                            '修复the spacecraft twin的嵌套指代语法。', 'visual_reference_grammar')
    for qid in sorted(keep):
        rows_to_fix = [visual_train[qid]] + [visual_eval[f"{learning[qid]['id']}-{t}"] for t in ['direct', 'statement']]
        for row in rows_to_fix:
            content = row['messages'][-1]['content']
            after = re.sub(r'\b([Tt]he)\s+the\b', r'\1', content)
            change(qid, 'visual_reference_grammar', row['id'], content, after, '删除视觉转换引入的重复冠词。')
            row['messages'][-1]['content'] = after

    print('Writing independent image files and cropped name cards...', flush=True)
    active_catalog = []
    for item in catalog:
        qid = item['qid']
        if qid not in keep:
            continue
        item = copy.deepcopy(item)
        role = item['role']
        asset = roles[qid][role]
        old_relative = asset['file']
        old_hash = asset['sha256']
        old_path = INPUT / old_relative
        crop = 'pantheon.world/' in item['source_url'] or (qid == 'Q100152219' and role == 'learning')
        if crop:
            assert 'annotation' not in asset, 'Review target-box coordinates before cropping annotated images'
            with Image.open(old_path) as im:
                assert im.size == (1200, 630)
                im = ImageOps.exif_transpose(im).convert('RGB')
                if qid == 'Q16150184':
                    # A profile is available, but the full circle also prints
                    # ALIJI on the jersey. Keep the head, not that name cue.
                    box = (150, 195, 325, 335)
                    cropped = im.crop(box)
                    method = 'profile_without_printed_jersey_surname'
                elif 'pantheon.world/' in item['source_url']:
                    # A conservative ellipse inside the photograph circle;
                    # x < 454 excludes the adjacent printed-name box.
                    box = (98, 132, 450, 528)
                    cropped = im.crop(box)
                    mask = Image.new('L', cropped.size, 0)
                    ImageDraw.Draw(mask).ellipse((2, 2, cropped.width - 3, cropped.height - 3), fill=255)
                    background = Image.new('RGB', cropped.size, 'white')
                    background.paste(cropped, (0, 0), mask)
                    cropped = background
                    method = 'inner_ellipse_portrait_only'
                else:
                    box = (795, 35, 1195, 535)
                    cropped = im.crop(box)
                    method = 'right_portrait_only'
                relative = f'images/{qid}/{Path(old_relative).stem}_portrait.png'
                target = output / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                cropped.save(target, format='PNG')
                asset.update(file=relative, original_sha256=asset.get('original_sha256', old_hash),
                             sha256=sha(target), width=cropped.width, height=cropped.height, dhash=dhash(cropped))
                transformation = dict(method=method, input_file=old_relative, input_sha256=old_hash,
                                      crop_box_xyxy=list(box), output_file=relative, identity_inferred_from_face=False)
                asset['repair_transform'] = transformation
                item.update(file=relative, sha256=asset['sha256'], size=list(cropped.size),
                            embedded_metadata=[], repair_transform=transformation)
                change(qid, 'image_text_crop', f'images:{role}', old_relative, relative,
                       '裁除人物卡姓名、职业、年份等文字，仅保留照片区域；裁切不代表身份已认证。',
                       role=role, before_file=old_relative, after_file=relative, before_sha256=old_hash,
                       after_sha256=asset['sha256'], transformation=transformation)
        else:
            target = output / old_relative
            target.parent.mkdir(parents=True, exist_ok=True)
            # Independent copies: subsequent edits cannot change the original.
            shutil.copy2(old_path, target)
        item['semantic_identity_approved'] = False
        item['approved_for_entity_pair'] = False
        active_catalog.append(item)

    # Rebuild all active views and nested source copies, not just shared JSONL.
    train_text, train_visual, eval_text, eval_visual = [], [], [], []
    labels, sample_audit, knowledge_samples, quality_status, active_evidence = [], [], [], [], []
    for qid in entities:
        if qid not in keep:
            continue
        unit_id = learning[qid]['id']
        tt = dict(id=unit_id, qid=qid, messages=[dict(role='assistant', content=learning[qid]['text'])])
        vt = visual_train[qid]
        vt['images'] = [roles[qid]['learning']['file']]
        train_text.append(tt)
        train_visual.append(vt)
        for probe in ['direct', 'statement']:
            q = questions[qid, probe]
            eval_text.append(render(qid, probe, q['question'], [o['text'] for o in q['options']], False))
            ve = visual_eval[f'{unit_id}-{probe}']
            ve['images'] = [roles[qid]['evaluation']['file']]
            eval_visual.append(ve)
            label = dict(id=f'{unit_id}-{probe}', qid=qid, answer=answers[qid]['answers'][probe], probe_id=probe)
            labels.append(label)
            sample_audit.append(dict(sample_id=label['id'], unit_id=unit_id, qid=qid, probe_id=probe,
                                     entity_name=entities[qid]['name'], entity_kind=entities[qid]['kind'],
                                     split='paired', answer=label['answer'],
                                     learning_image_files=[roles[qid]['learning']['file']],
                                     eval_image_files=[roles[qid]['evaluation']['file']],
                                     automatic_passed=False, structural_validation_passed=True,
                                     human_approved=False, image_semantics_reviewed=False,
                                     source_fingerprint=unit_id))
        nk = evidence[qid]['new_knowledge']
        active_evidence.append(dict(id=unit_id, qid=qid, new_knowledge=nk,
                                    dated_support=evidence[qid].get('dated_support'),
                                    original_record_reference=f'provenance/input_package/source/evidence.jsonl#{qid}',
                                    evidence_status='candidate_not_independently_certified', automatic_passed=False,
                                    review=dict(verdict='not_semantically_certified',
                                                reason='本次修复验证结构和已裁决问题；不继承旧accept或verified结论。')))
        sample = dict(new_knowledge=nk, learning=learning[qid]['text'], questions=[
            dict(type=t, question=questions[qid, t]['question'],
                 options=[x['text'] for x in questions[qid, t]['options']],
                 answer_index=LETTERS.index(answers[qid]['answers'][t])) for t in ['direct', 'statement']])
        knowledge_samples.append(dict(id=unit_id, qid=qid, fingerprint=unit_id, automatic_passed=False,
                                      human_approved=False, content=dict(entity=dict(name=entities[qid]['name'],
                                      kind=entities[qid]['kind']), sample=sample),
                                      original_record_reference=f'provenance/input_package/source/knowledge_samples.jsonl#{qid}'))
        entities[qid]['samples'] = [dict(sample_id=unit_id, fingerprint=unit_id,
                                        learning_text=learning[qid]['text'], new_knowledge=nk,
                                        question_types=['direct', 'statement'])]
        entities[qid]['entity_processing_state'] = 'candidate_after_reviewed_repairs'
        entities[qid]['requires_state_review'] = True
        quality_status.append(dict(qid=qid, unit_id=unit_id,
            origin='existing1572' if qid in historical_units else 'new450',
            historical_tier=historical_units.get(qid, {}).get('audit_tier'),
            temporal_candidate=new_temporal.get(qid),
            first_public_date=nk.get('first_public_date'), first_publication_independently_verified=False,
            image_identity_independently_verified=False,
            approved_for_post_cutoff_experiment=False,
            content_status='repaired_candidate' if any(x['qid'] == qid for x in changes) else 'no_specific_issue_found',
            note_zh='保留不等于外部事实、身份或首次公开时间认证；日期缺失不直接判错。'))

    active_sources = dict(entities=[entities[q] for q in entities if q in keep],
                          learning=[learning[q] for q in entities if q in keep],
                          questions=[questions[q, t] for q in entities if q in keep for t in ['direct', 'statement']],
                          answers=[answers[q] for q in entities if q in keep],
                          evidence=active_evidence, knowledge_samples=knowledge_samples)
    for name, rows in active_sources.items():
        save_rows(output / f'source/{name}.jsonl', rows)
    views = dict(train_text=train_text, train_multimodal=train_visual, eval_text=eval_text, eval_multimodal=eval_visual)
    directions = {'T2T': ['train_text', 'eval_text'], 'T2V': ['train_text', 'eval_multimodal'],
                  'V2T': ['train_multimodal', 'eval_text'], 'V2V': ['train_multimodal', 'eval_multimodal']}
    for name, rows in views.items():
        save_rows(output / f'shared/{name}.jsonl', rows)
    for direction, (train, evaluate) in directions.items():
        for split, view in [('train', train), ('eval', evaluate)]:
            target = output / direction / f'{split}.jsonl'
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output / f'shared/{view}.jsonl', target)
    for split, view in [('train', 'train_text'), ('eval', 'eval_text')]:
        save_rows(output / f'text_all/{split}.jsonl', views[view])
    save_rows(output / 'text_all/eval_labels.jsonl', labels)
    save_rows(output / 'audit/eval_labels.jsonl', labels)
    save_rows(output / 'audit/sample_audit.jsonl', sample_audit)
    save_rows(output / 'audit/image_roles.jsonl', [roles[q] for q in entities if q in keep])
    save_rows(output / 'audit/image_catalog.jsonl', active_catalog)
    save_rows(output / 'audit/changes.jsonl', changes)
    save_rows(output / 'audit/unit_quality_status.jsonl', quality_status)
    repaired_qids = {x['qid'] for x in changes}
    dispositions = [dict(qid=qid, unit_id=original_learning[qid]['id'], name=original_entities[qid]['name'],
                         action='delete' if qid not in keep else ('repair' if qid in repaired_qids else 'retain'),
                         deletion_reasons=delete_reasons.get(qid, []),
                         repair_categories=sorted({x['category'] for x in changes if x['qid'] == qid}),
                         directions_affected=list(directions),
                         approved_for_post_cutoff_experiment=False) for qid in entities]
    save_rows(output / 'audit/unit_dispositions.jsonl', dispositions)
    with (output / 'audit/unit_dispositions.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['qid', 'name', 'action', 'reasons_zh', 'repair_categories'])
        writer.writeheader()
        for x in dispositions:
            writer.writerow(dict(qid=x['qid'], name=x['name'], action=x['action'],
                                 reasons_zh='；'.join(r['reason_zh'] for r in x['deletion_reasons']),
                                 repair_categories=';'.join(x['repair_categories'])))
    save(output / 'source/source_provenance.json', dict(
        schema='entity1025-reviewed-repair-provenance-v1', selected_units=len(keep),
        input_package='provenance/input_package', input_inventory_sha256=frozen[INPUT / 'package_inventory.json'],
        original_zip_sha256='4642a17fc2c47b2b513ca4998761c12eac48aef21d7ef0e8129f7fa84beae1c2',
        review='provenance/review/adjudications.jsonl',
        source_counts=dict(Counter(x['origin'] for x in quality_status)),
        first_publication_or_identity_certification=False))
    numbered = sum('annotation' in roles[q][r] for q in keep for r in ['learning', 'evaluation'])
    manifest = dict(schema='xmodal-entity-reviewed-repair-v1', name=RELEASE,
                    release_status='structurally_validated_repaired_candidate_not_semantically_certified',
                    input_knowledge_units=len(entities), knowledge_units=len(keep),
                    deleted_units=len(delete_reasons), repaired_units=len(repaired_qids),
                    unchanged_content_units=len(keep) - len(repaired_qids),
                    questions_per_direction=2 * len(keep), images=2 * len(keep),
                    numbered_image_roles=numbered, human_visual_identity_approved=0,
                    directions=directions, answer_letters_preserved=True,
                    original_inputs_modified=False, parent_inventory_verified_before_and_after=True,
                    source_counts=dict(Counter(x['origin'] for x in quality_status)),
                    deletion_category_unit_counts=dict(Counter(c for reasons in delete_reasons.values()
                                                               for c in {x['category'] for x in reasons})),
                    repair_category_unit_counts={c: len({x['qid'] for x in changes if x['category'] == c})
                                                for c in sorted({x['category'] for x in changes})})
    save(output / 'build_manifest.json', manifest)
    # Existing launch scripts are copied as tools; no experiment is launched.
    shutil.copytree(INPUT / 'scripts', output / 'scripts')
    shutil.copy2(ROOT / 'scripts/verify_entity1025_repaired.py', output / 'verify_package.py')
    (output / 'provenance/code').mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), output / 'provenance/code/repair_entity1025_dataset.py')
    shutil.copy2(ROOT / 'scripts/verify_entity1025_repaired.py', output / 'provenance/code/verify_entity1025_repaired.py')
    # Preserve correct letters for retained units despite all view reconstruction.
    assert all(answers[q] == original_answers[q] for q in keep)
    report = f'''# entity1025 修复记录（2026-11-03）

原始 {len(entities)} 组；删除 {len(delete_reasons)} 组；保留 {len(keep)} 组。其中 {len(repaired_qids)} 组有内容修复，{len(keep)-len(repaired_qids)} 组内容保持原样。
每个方向学习 {len(keep)} 条、评测 {2*len(keep)} 题（direct / statement 各 {len(keep)}）；保留 {2*len(keep)} 张学习/测试图片。

修复：恢复被误替换的完整姓名、绰号、节目名及其他人物姓名；补回证据明确支持而学习漏掉的标题；采用已有受控修题候选；修复图片指代语法；将仅正确项为“图中人”的题改为来源支持的对称属性题；裁掉姓名卡的姓名、职业、出生年份。

删除：错图或占位图、无法可靠确认归属的特定冲突图、现有来源不足以修复的答案/学习冲突、明确提前公开的事实，以及无法安全重构的题目捷径。删除均按知识单元执行，四方向同步，未按模型对错筛题。原图没有备用图；旧包中对应候选图与原图哈希相同，不能用另一侧图片伪造独立测试图。

只缺首次公开日期、只缺图片归属材料、或只有旧事件日期的条目没有一律删除。历史 Robert Dowd 父母选项误判已撤销；没有依据该误判删除。Tom Courtenay 的部分排除风险保留在逐组状态中。

`audit/unit_dispositions.csv` / `.jsonl` 为全部单元的保留、修复及删除原因；`audit/changes.jsonl` 记录逐字段前后值、图片裁切坐标及哈希。删除原因有重叠，不能相加作为删除总数。旧审查和原始活动文件放在 `provenance/`，仅供追溯，不是本版训练输入。

本版是完成已知问题清理的候选版，尚未逐条独立认证外部事实、图片人物身份及知识首次公开时间。已有修题候选的合并不等于新颖性审查通过；旧 `accept`、`verified`、`automatic_passed=true` 没有继承为本版语义批准。具体状态见 `audit/unit_quality_status.jsonl`。

## 使用

四个方向位于 `T2T/`、`T2V/`、`V2T/`、`V2V/`。图片路径相对于本数据包根目录。两种题型和四方向严格配对，四方向不是四份独立知识。标签见 `audit/eval_labels.jsonl`，保留原始正确字母；部分题干/候选属性已变化，旧评测输出不能直接合并计分。

在已有项目虚拟环境中运行 `python verify_package.py` 可全量校验，不需要GPU。训练脚本未启动；`scripts/train_experiment.sh` 仍保留原PT默认值，若要SFT须另行指定合适的训练数据和命令。
'''
    (output / 'README.md').write_text(report, encoding='utf-8')
    (output / 'REPAIR_REPORT_ZH.md').write_text(report + '\n修复类别与数量：\n\n```json\n' +
         json.dumps(manifest, ensure_ascii=False, indent=2) + '\n```\n', encoding='utf-8')
    print('Validating active source, paired views, and all images...', flush=True)
    validation = validate(output, check_inventory=False)
    save(output / 'validation_report.json', validation)
    print('Rechecking original package is unchanged...', flush=True)
    check_parent(inventory)
    for path, expected in frozen.items():
        assert sha(path) == expected
    file_hashes = {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob('*')) if p.is_file()}
    save(output / 'package_inventory.json', dict(schema='xmodal-portable-package-inventory-v1',
         file_count_excluding_inventory=len(file_hashes), sha256=file_hashes))
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'data' / RELEASE)
    args = parser.parse_args()
    main(args.output.resolve())
