"""Read-only dataset audit; observations do not certify semantic correctness."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import difflib
import hashlib
import html
import json
from pathlib import Path
import re
import time
import unicodedata
from urllib.parse import unquote, urlsplit

from PIL import Image, ImageOps


def rows(path):
    with path.open(encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def norm(value):
    return ' '.join(unicodedata.normalize('NFKC', value).casefold().split())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def save_rows(path, values):
    with path.open('w') as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False) + '\n')


def years(text):
    return [int(x) for x in re.findall(r'\b((?:18|19|20)\d{2})\b', text)]


def image_source_key(url):
    parsed = urlsplit(url)
    path = unquote(parsed.path)
    if any(x in parsed.netloc for x in ['wikimedia.org', 'wikipedia.org']):
        match = re.search(r'/commons/(?:thumb/)?[0-9a-f]/[0-9a-f]{2}/([^/]+)', path)
        if match:
            return 'wikimedia:' + match.group(1).replace('_', ' ').casefold()
        if '/wiki/File:' in path:
            return 'wikimedia:' + path.split('/wiki/File:', 1)[1].replace('_', ' ').casefold()
    # Several image APIs select the subject using a query parameter (for example,
    # Pantheon's person?id=...). Keep it to avoid merging different subjects.
    return parsed.netloc.casefold() + path + ('?' + parsed.query if parsed.query else '')


def dhash(image):
    image = ImageOps.exif_transpose(image).convert('L').resize((9, 8))
    pixels = list(image.getdata())
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | int(pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data/entity_merged2022_position_annotated'))
    parser.add_argument('--output', type=Path, default=Path('reports/entity1025_review_20261103/full_audit'))
    args = parser.parse_args()
    data, out = args.data.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    entities = {x['qid']: x for x in rows(data / 'source/entities.jsonl')}
    learning = {x['qid']: x for x in rows(data / 'source/learning.jsonl')}
    questions = rows(data / 'source/questions.jsonl')
    answers = {x['qid']: x['answers'] for x in rows(data / 'source/answers.jsonl')}
    evidence = {x['qid']: x for x in rows(data / 'source/evidence.jsonl')}
    knowledge = {x['qid']: x for x in rows(data / 'source/knowledge_samples.jsonl')}
    train_t = {x['qid']: x for x in rows(data / 'T2T/train.jsonl')}
    train_v = {x['qid']: x for x in rows(data / 'V2T/train.jsonl')}
    test_t = {x['id']: x for x in rows(data / 'T2T/eval.jsonl')}
    test_v = {x['id']: x for x in rows(data / 'T2V/eval.jsonl')}
    labels = {x['id']: x for x in rows(data / 'audit/eval_labels.jsonl')}
    catalog = rows(data / 'audit/image_catalog.jsonl')
    roles = {x['qid']: x for x in rows(data / 'audit/image_roles.jsonl')}
    new_roles = {x['qid']: x for x in rows(data / 'audit/independent_position_annotated_450/image_roles.jsonl')}
    detections = {(x['qid'], x['role']): x for x in rows(data / 'audit/existing_cleaned_1572/multi_person_detection.jsonl')}
    history = out / 'history'
    prior_questions = rows(history / 'knowledge/question_audit.jsonl')
    prior_units = {x['qid']: x for x in rows(history / 'knowledge/unit_audit.jsonl')}
    prior_pending = {x['qid']: x for x in rows(history / 'pending/unit_ledger.jsonl')}
    findings = []

    def flag(category, status, qid, scope, detail, **extra):
        findings.append(dict(category=category, status=status, qid=qid,
                             name=entities[qid]['name'], scope=scope, detail=detail, **extra))

    # Active labels, questions and prompts, rather than ancestor option positions.
    checks = {}
    checks['unique_entities'] = len(entities) == len(train_t) == len(train_v) == 2022
    checks['label_ids'] = set(test_t) == set(test_v) == set(labels)
    checks['unique_question_keys'] = len(questions) == len({(x['qid'], x['type']) for x in questions}) == 4044
    checks['every_entity_has_two_questions'] = set(Counter(x['qid'] for x in questions).values()) == {2}
    checks['shared_views'] = all(
        (data / folder / 'train.jsonl').read_bytes() == (data / 'shared' / (tr + '.jsonl')).read_bytes()
        and (data / folder / 'eval.jsonl').read_bytes() == (data / 'shared' / (te + '.jsonl')).read_bytes()
        for folder, tr, te in [('T2T', 'train_text', 'eval_text'), ('T2V', 'train_text', 'eval_multimodal'),
                               ('V2T', 'train_multimodal', 'eval_text'), ('V2V', 'train_multimodal', 'eval_multimodal')])
    checks['learning_text_agrees_with_source'] = all(train_t[q]['messages'][0]['content'] == learning[q]['text'] for q in entities)
    for qid in entities:
        old = train_t[qid]['messages'][0]['content']
        visual = train_v[qid]['messages'][0]['content'].removeprefix('<image>\n')
        visual = visual.removeprefix('Focus on the person marked 1 with a red box.\n')
        if old == visual:
            flag('unchanged_visual_learning_text', 'confirmed_observation', qid, 'learning',
                 'Learning text is unchanged after removing image/target-focus prefix.')
        if re.search(r'\bthe\s+the\b', visual, re.I):
            flag('repeated_article', 'confirmed_surface_error', qid, 'learning', visual)
        src = knowledge[qid]['content']['sample']
        if norm(src['learning']) != norm(old):
            flag('source_learning_disagreement', 'requires_review', qid, 'learning',
                 'Packaged knowledge sample differs from active learning.', active=old, source=src['learning'])
        if not evidence[qid]['new_knowledge'].get('first_public_date'):
            flag('first_public_date_missing', 'provenance_gap', qid, 'knowledge',
                 'A later Wikipedia edit is not independent evidence of first public disclosure.')

    gold_literal_miss = []
    question_records = []
    prompt_groups = defaultdict(list)
    for question in questions:
        qid, typ, uid = question['qid'], question['type'], question['id']
        sid = uid + '-' + typ
        gold = answers[qid][typ]
        options = {x['id']: x['text'] for x in question['options']}
        assert set(options) == set('ABCD') and gold in options
        assert labels[sid]['answer'] == gold
        text = test_t[sid]['messages'][-1]['content']
        visual = test_v[sid]['messages'][-1]['content']
        plain_visual = visual.removeprefix('<image>\n').removeprefix('Focus on the person marked 1 with a red box.\n')
        if plain_visual == text:
            flag('unchanged_visual_question_text', 'confirmed_observation', qid, typ,
                 'Question and options are identical without the image/target-focus prefix.', sample_id=sid)
        if re.search(r'\bthe\s+the\b', visual, re.I):
            flag('repeated_article', 'confirmed_surface_error', qid, typ, visual, sample_id=sid)
        # Compare options independently of boilerplate or entity substitution in the stem.
        vo = dict(re.findall(r'^([ABCD])\. (.*)$', visual, re.M))
        assert set(vo) == set('ABCD')
        if len({norm(s) for s in options.values()}) < 4:
            flag('duplicate_options_text', 'confirmed_error', qid, typ, str(options), sample_id=sid)
        if len({norm(s) for s in vo.values()}) < 4:
            flag('duplicate_options_visual', 'confirmed_error', qid, typ, str(vo), sample_id=sid)
        changed_options = [letter for letter in options if vo[letter] != options[letter]]
        if changed_options and typ == 'direct':
            flag('direct_option_rewrite', 'requires_semantic_review', qid, typ,
                 'Visual conversion changes direct-answer options; may change proper names.',
                 sample_id=sid, text_options=options, visual_options=vo, gold=gold)
        refs = [letter for letter, value in vo.items() if 'shown in the image' in value]
        if len(refs) == 1:
            flag('one_option_has_image_reference', 'shortcut_candidate', qid, typ,
                 'Only one option contains an image reference; inspect option symmetry.',
                 sample_id=sid, reference_letter=refs[0], is_gold=refs[0] == gold,
                 text_question=question['question'], visual_options=vo)
        # These are candidates, never a semantic entailment verdict.
        if typ == 'direct' and re.search(r'(?<!\w)' + re.escape(norm(options[gold])) + r'(?!\w)', norm(question['question'])):
            flag('answer_phrase_in_stem', 'shortcut_candidate', qid, typ,
                 'Gold answer occurs literally in the question; context must be reviewed.',
                 sample_id=sid, question=question['question'], answer=options[gold])
        for answer_text, count in Counter(norm(x) for x in options.values()).items():
            assert answer_text
        if typ == 'direct' and norm(options[gold]) not in norm(learning[qid]['text']):
            gold_literal_miss.append(dict(qid=qid, sample_id=sid, question=question['question'],
                                          answer=options[gold], options=options, learning=learning[qid]['text']))
        prompt_groups[norm(text)].append((sid, qid, norm(options[gold])))
        original = knowledge[qid]['content']['sample']
        src_q = next(x for x in original['questions'] if x['type'] == typ)
        src_gold = src_q['options'][src_q['answer_index']]
        if norm(src_q['question']) != norm(question['question']) or {norm(x) for x in src_q['options']} != {norm(x) for x in options.values()} or norm(src_gold) != norm(options[gold]):
            flag('source_question_disagreement', 'requires_review', qid, typ,
                 'Knowledge-sample question/answer differs from the active source question.',
                 sample_id=sid, active_question=question, original_question=src_q)
        question_records.append(dict(sample_id=sid, qid=qid, probe_id=typ, gold=gold,
                                     gold_text=options[gold], group='new450' if qid in new_roles else 'existing1572'))
    for text, group in prompt_groups.items():
        if len(group) > 1:
            for sid, qid, gold in group:
                flag('duplicate_test_prompt', 'requires_review' if len({x[2] for x in group}) == 1 else 'conflicting_labels',
                     qid, 'evaluation', 'Identical text prompts shared across records.', sample_id=sid, matching_records=group)

    # Historical conclusions are transferred only after matching all content, including the correct text.
    qmap = {(x['qid'], x['type']): x for x in questions}
    history_matches, history_mismatches = [], []
    for prior in prior_questions:
        qid, typ = prior['qid'], prior['question_type']
        current = qmap[qid, typ]
        gold_current = next(x['text'] for x in current['options'] if x['id'] == answers[qid][typ])
        gold_prior = next(x['text'] for x in prior['options'] if x['id'] == prior['gold'])
        matched = norm(current['question']) == norm(prior['question']) and {norm(x['text']) for x in current['options']} == {norm(x['text']) for x in prior['options']} and norm(gold_current) == norm(gold_prior)
        (history_matches if matched else history_mismatches).append(dict(qid=qid, probe_id=typ))
        if matched and (prior['clear_problem'] or prior['review_needed']):
            flag('historical_content_review', 'historical_previously_marked_issue' if prior['clear_problem'] else 'historical_pending_review',
                 qid, typ, prior['reason_zh'], temporal_verdict=prior['temporal_verdict'], structural_verdict=prior['structural_verdict'],
                 source='history/knowledge/question_audit.jsonl')
    assert len(history_mismatches) == 0, 'Do not reuse historical conclusions for changed content.'
    for qid, prior in prior_pending.items():
        flag('historical_followup', 'historical_followup_' + prior['partition'], qid, 'knowledge', prior['reason_zh'],
             action=prior['recommended_action'], source='history/pending/unit_ledger.jsonl')

    # Full image decoding, exact hashes, source keys, low-resolution and perceptual duplicate screening.
    inventory = json.loads((data / 'package_inventory.json').read_text())['sha256']
    inventory_errors = []
    image_records = []
    hashes = {}
    for i, (name, expected) in enumerate(inventory.items(), 1):
        path = data / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[name] = actual
        if actual != expected:
            inventory_errors.append(name)
        if i % 1000 == 0:
            print('hashes', i, '/', len(inventory), flush=True)
    for cat in catalog:
        qid, role, name = cat['qid'], cat['role'], cat['file']
        asset = roles[qid][role]
        assert hashes[name] == cat['sha256'] == asset['sha256']
        with Image.open(data / name) as image:
            image.load()
            width, height = image.size
            perceptual = dhash(image)
            frames = getattr(image, 'n_frames', 1)
        if min(width, height) < 100:
            flag('low_image_resolution', 'quality_candidate', qid, role,
                 'Shortest image side is below 100 pixels.', file=name, width=width, height=height)
        if frames > 1:
            flag('animated_image', 'confirmed_observation', qid, role,
                 'Image has multiple frames; inference pipeline may use only the first.', file=name, frames=frames)
        source_key = image_source_key(asset['source_url'])
        original_sha = asset.get('original_sha256', cat.get('original_sha256', hashes[name]))
        image_records.append(dict(qid=qid, role=role, file=name, sha256=hashes[name], original_sha256=original_sha,
                                  source_key=source_key, dhash=perceptual, width=width, height=height,
                                  source_evidence=cat.get('source_evidence'), annotated='annotation' in asset))
        if cat.get('source_evidence') == 'no_target_attribution':
            flag('image_no_target_attribution', 'provenance_gap', qid, role,
                 'Packaged image catalogue records no source attribution to the target entity.', file=name)
        if 'annotation' in asset:
            ann = asset['annotation']
            x, y, w, h = ann['target_detection']['box']
            if not (0 <= x < width and 0 <= y < height and w > 0 and h > 0 and x + w <= width + 2 and y + h <= height + 2):
                flag('annotation_outside_image', 'confirmed_geometry_error', qid, role, str(ann), file=name, size=[width, height])
            flag('numbered_target_not_identity_verified', 'requires_source_confirmation', qid, role,
                 ann.get('evidence', ''), file=name,
                 candidate_count=ann['target_detection'].get('candidate_count'))
        if qid in new_roles:
            record = new_roles[qid][role]['record']
            source = record.get('source_record', {})
            ext = source.get('extmetadata', {})
            birth = re.search(r'born[ _,]+(\d{4})', unquote(entities[qid].get('wikipedia_url', '')))
            if birth:
                birth_year = int(birth.group(1))
                description = ' '.join(str(record.get(k, '')) for k in ['source_title', 'source_description'])
                description += ' ' + str(source.get('result', {}).get('categories', ''))
                description += ' ' + str(ext.get('DateTimeOriginal', {}).get('value', ''))
                early = [year for year in years(description) if year < birth_year + 14]
                if early:
                    flag('image_birth_timeline_conflict', 'requires_source_confirmation', qid, role,
                         description, file=name, birth_year=birth_year, early_years=early,
                         entity_url=entities[qid]['wikipedia_url'])
        # Existing multi-person detections are inherited hints, not a fresh detector pass.
        det = detections.get((qid, role))
        if det and det['file'] == name and det.get('person_count', 0) > 1:
            flag('multiple_people_inherited_detector', 'target_ambiguity_candidate', qid, role,
                 'Historical detector found multiple people; check spatial reference and caption.',
                 file=name, person_count=det['person_count'])

    exact_overlap = {}
    for key in ['sha256', 'original_sha256', 'source_key']:
        groups = defaultdict(list)
        for record in image_records:
            groups[record[key]].append(record)
        overlaps = []
        for value, group in groups.items():
            if len(group) > 1:
                overlap = dict(key=value, images=[{k: x[k] for k in ['qid', 'role', 'file']} for x in group],
                               crosses_learning_test=len({x['role'] for x in group}) > 1)
                overlaps.append(overlap)
                for x in group:
                    flag('shared_image_' + key, 'duplicate_candidate' if key == 'source_key' else 'confirmed_duplicate_bytes',
                         x['qid'], x['role'], 'Images share ' + key, group=overlap)
        exact_overlap[key] = overlaps
    near_duplicates = []
    for i, left in enumerate(image_records):
        for right in image_records[i + 1:]:
            distance = (left['dhash'] ^ right['dhash']).bit_count()
            if distance <= 5:
                near_duplicates.append(dict(distance=distance,
                    left={k: left[k] for k in ['qid', 'role', 'file']},
                    right={k: right[k] for k in ['qid', 'role', 'file']},
                    crosses_learning_test=left['role'] != right['role']))
    near_duplicates.sort(key=lambda x: (x['distance'], x['left']['qid'], x['right']['qid']))
    for candidate in near_duplicates:
        for side in ['left', 'right']:
            x = candidate[side]
            flag('perceptually_similar_images', 'visual_review_candidate', x['qid'], x['role'],
                 '64-bit difference hash distance <= 5; similarity is not proof of duplication.', pair=candidate)

    counts = Counter(x['category'] for x in findings)
    by_unit = defaultdict(list)
    for i, finding in enumerate(findings, 1):
        finding['finding_id'] = f'F{i:05d}'
        by_unit[finding['qid']].append(finding)
    unit_records = []
    for qid, entity in entities.items():
        hist = prior_units.get(qid, {})
        pending = prior_pending.get(qid, {})
        unit_records.append(dict(qid=qid, name=entity['name'], group='new450' if qid in new_roles else 'existing1572',
                                 historical_tier=hist.get('audit_tier'), historical_followup_partition=pending.get('partition'),
                                 finding_categories=dict(Counter(x['category'] for x in by_unit[qid])),
                                 finding_ids=[x['finding_id'] for x in by_unit[qid]], semantic_approved=False))
    summary = dict(data_root=str(data), scope='All records: structural, textual and metadata screening; all images: hash/full decode/perceptual screening. Historical semantic decisions content-matched for existing1572. New450 semantic correctness is not certified.',
                   checks=checks, source_input_sha256={str(x.relative_to(data)): hashlib.sha256(x.read_bytes()).hexdigest() for x in [data/'source/questions.jsonl',data/'source/learning.jsonl',data/'source/answers.jsonl',data/'audit/image_roles.jsonl']},
                   coverage=dict(units=len(entities), questions=len(questions), images=len(image_records),
                                 inventory_files=len(inventory), inventory_errors=inventory_errors,
                                 historical_question_matches=len(history_matches), historical_question_mismatches=len(history_mismatches)),
                   findings_by_category=dict(counts), findings_by_status=dict(Counter(x['status'] for x in findings)),
                   unique_units_by_category={k:len({x['qid'] for x in findings if x['category']==k}) for k in counts},
                   historical_unit_tiers=dict(Counter(x['audit_tier'] for x in prior_units.values())),
                   historical_followup_partitions=dict(Counter(x['partition'] for x in prior_pending.values())),
                   exact_overlap_groups={k:len(v) for k,v in exact_overlap.items()}, perceptual_duplicate_candidates=len(near_duplicates),
                   non_literal_direct_answer_count=len(gold_literal_miss),
                   elapsed_seconds=round(time.monotonic()-started,2), original_dataset_modified=False)
    save_rows(out/'findings.jsonl',findings)
    save_rows(out/'unit_audit.jsonl',unit_records)
    save_rows(out/'question_audit.jsonl',question_records)
    save_rows(out/'image_audit.jsonl',image_records)
    save_rows(out/'nonliteral_direct_answers.jsonl',gold_literal_miss)
    save_rows(out/'similar_image_pairs.jsonl',near_duplicates)
    save(out/'shared_image_sources.json',exact_overlap)
    save(out/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
