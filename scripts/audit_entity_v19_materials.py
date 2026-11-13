"""Independent material audit, without modifying data or invoking any model.

Sample selection and static flags never use predictions/correctness. Existing
results are read only after the review sample has been written.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'runs/entity_v19_transfer_ep5_20261014'
DATA = ROOT / 'entity_v19_1478'
FROZEN = WORK / 'data/frozen_sources'
SCORED = WORK / 'rescoring/native-empty-think-v2'
OUT = ROOT / 'diagnostics/entity_v19_material_audit_20261016'


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def write_rows(path, records):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))


def csvfile(name, records):
    with (OUT / (name + '.csv')).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def normalize(value):
    return ' '.join(unicodedata.normalize('NFKC', value).casefold().split())


def has_name(name, text):
    return bool(re.search(r'(?<!\w)' + re.escape(normalize(name)) + r'(?!\w)', normalize(text)))


def content(row):
    return row['messages'][-1]['content']


def options(text):
    answer = re.findall(r'^([ABCD])\. (.*)$', text, re.M)
    assert len(answer) == 4
    return dict(answer)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'contact_sheets').mkdir(exist_ok=True)
    manifest = json.loads((WORK / 'manifest.json').read_text())
    verified = []
    for source in manifest['sources']:
        p = Path(source['copy'])
        assert hashlib.sha256(p.read_bytes()).hexdigest() == source['sha256'], p
        verified.append(str(p))
    audit = {r['sample_id']: r for r in rows(FROZEN / 'audit/sample_audit.jsonl')}
    source = {r['id']: r for r in rows(FROZEN / 'source/knowledge_samples.jsonl')}
    roles = {r['unit_id']: r for r in rows(FROZEN / 'audit/image_roles.jsonl')}
    text = {r['id']: r for r in rows(FROZEN / 'T2T/eval.jsonl')}
    image = {r['id']: r for r in rows(FROZEN / 'T2V/eval.jsonl')}
    visual_train = {r['id']: r for r in rows(FROZEN / 'V2V/train.jsonl')}
    text_train = {r['id']: r for r in rows(FROZEN / 'T2T/train.jsonl')}
    units = {r['unit_id']: r for r in audit.values()}
    assert len(units) == len(roles) == 1225
    # Freeze a stratified screening sample without reading prediction artifacts.
    rng = random.Random(20261016)
    persons = sorted(uid for uid, r in units.items() if r['entity_kind'] == 'person')
    other = sorted(set(units) - set(persons))
    selected = rng.sample(persons, 24) + rng.sample(other, 8)
    sample = []
    for number, uid in enumerate(selected, 1):
        a, role = units[uid], roles[uid]
        sample.append(dict(number=number, unit_id=uid, qid=a['qid'], entity_name=a['entity_name'],
                           entity_kind=a['entity_kind'],
                           learning_image=str(DATA / role['learning']['file']),
                           evaluation_image=str(DATA / role['evaluation']['file']),
                           learning_source_url=role['learning'].get('source_url'),
                           evaluation_source_url=role['evaluation'].get('source_url'),
                           text_learning=content(text_train[uid]),
                           visual_learning=content(visual_train[uid]),
                           direct=content(image[uid + '-direct']),
                           statement=content(image[uid + '-statement'])))
    write_rows(OUT / 'review_sample.jsonl', sample)
    save(OUT / 'sampling_protocol.json', dict(seed=20261016, population=1225,
         strata={'person': {'population': len(persons), 'sample': 24},
                 'other': {'population': len(other), 'sample': 8}},
         sampled_before_reading_predictions=True, correctness_used_for_selection=False,
         purpose='Assistant visual screening; not human identity verification',
         required_identity_evidence='Visual plausibility alone does not certify named identity',
         original_data_modified=False))

    font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 17)
    for page in range(4):
        sheet = Image.new('RGB', (1640, 1240), 'white')
        draw = ImageDraw.Draw(sheet)
        for pos, r in enumerate(sample[page * 8:(page + 1) * 8]):
            x, y = (pos % 2) * 820, (pos // 2) * 310
            title = f"{r['number']:02d} {r['entity_name']} ({r['qid']})"
            draw.text((x + 5, y + 3), title[:75], fill='black', font=font)
            for col, (key, label) in enumerate((('learning_image', '学习图'), ('evaluation_image', '测试图'))):
                box = (x + col * 410, y + 48, x + col * 410 + 405, y + 302)
                with Image.open(r[key]) as im:
                    thumb = ImageOps.contain(ImageOps.exif_transpose(im).convert('RGB'), (400, 250))
                    sheet.paste(thumb, (box[0] + (405 - thumb.width) // 2,
                                       box[1] + (254 - thumb.height) // 2))
                draw.text((x + col * 410 + 5, y + 27), label, fill='black', font=font)
        sheet.save(OUT / 'contact_sheets' / f'page_{page + 1}.jpg', quality=93)

    features = []
    length_records = []
    for sid, a in audit.items():
        uid = a['unit_id']
        t, v = content(text[sid]), content(image[sid])
        t_without_marker = v.replace('<image>', '').strip()
        choices = options(v)
        wc = {key: len(re.findall(r"\b[\w]+(?:['’-][\w]+)*\b", value)) for key, value in choices.items()}
        winners = [key for key, value in wc.items() if value == max(wc.values())]
        q = source[uid]['content']['sample']['questions']
        question = next(r['question'] for r in q if r['type'] == a['probe_id'])
        features.append(dict(source_id=sid, unit_id=uid, qid=a['qid'], entity_name=a['entity_name'],
                             entity_kind=a['entity_kind'], question_type=a['probe_id'],
                             unchanged_text=t.strip() == t_without_marker,
                             full_name_in_visual_eval=has_name(a['entity_name'], v),
                             full_name_in_visual_learning=has_name(a['entity_name'], content(visual_train[uid])),
                             explicit_visual_deictic=bool(re.search(r'\b(?:shown|depicted|pictured)\b|\bin (?:this|the) (?:image|photo)', v, re.I)),
                             question=question, visual_eval=v))
        length_records.append(dict(source_id=sid, unit_id=uid, question_type=a['probe_id'],
                                   longest_option_tie_count=len(winners),
                                   longest_option_expected_correct=(1 / len(winners) if a['answer'] in winners else 0),
                                   **{'words_' + key: value for key, value in wc.items()}))
    write_rows(OUT / 'question_features.jsonl', features)
    csvfile('option_length_features', length_records)
    indexed = {r['source_id']: r for r in features}
    unchanged = {r['source_id'] for r in features if r['unchanged_text']}
    # The flag removes questions only in a derived diagnostic, never in the dataset.
    groups = {'all': set(indexed), 'unchanged_text': unchanged,
              'changed_text': set(indexed) - unchanged}
    unchanged_units = {indexed[sid]['unit_id'] for sid in unchanged}
    groups['exclude_unchanged_units'] = {sid for sid, r in indexed.items() if r['unit_id'] not in unchanged_units}
    save(OUT / 'static_summary.json', dict(
        verified_frozen_sources=len(verified), total_units=len(units), total_questions=len(features),
        entity_kinds=dict(Counter(r['entity_kind'] for r in units.values())),
        unchanged_text_questions=len(unchanged), unchanged_text_units=len({indexed[sid]['unit_id'] for sid in unchanged}),
        full_name_in_visual_eval=sum(r['full_name_in_visual_eval'] for r in features),
        full_name_in_visual_learning_units=len({r['unit_id'] for r in features if r['full_name_in_visual_learning']}),
        no_explicit_visual_deictic=sum(not r['explicit_visual_deictic'] for r in features),
        longest_option_expected_accuracy={probe: sum(r['longest_option_expected_correct'] for r in length_records
            if r['question_type'] == probe) / sum(r['question_type'] == probe for r in length_records)
            for probe in ('direct', 'statement')},
        caveats=['Static string flags are screening indicators, not proof of semantic answerability or leakage',
                 'Length heuristic is exploratory and is not evidence of model mechanism'],
    ))
    csvfile('unchanged_questions', [{k: r[k] for k in ('source_id', 'unit_id', 'qid', 'entity_name', 'question_type', 'question')}
                                   for r in features if r['unchanged_text']])

    # Predictions are read only after sample, rules and static flags are frozen.
    subset_scores, subset_controls = [], []
    for model in ('qwen3', 'glm46'):
        for mode in ('base', 'TEXT', 'VISION'):
            folder = (SCORED / 'base' / model) if mode == 'base' else (
                SCORED / 'evaluation' / model / 'seed_42' / mode / 'epoch_5')
            results = rows(folder / 'results.jsonl')
            by_condition = defaultdict(list)
            for r in results:
                by_condition[r['test']].append(r)
            for group, ids in groups.items():
                for probe in ('overall', 'direct', 'statement'):
                    for test, values in by_condition.items():
                        subset = [r for r in values if r['source_id'] in ids
                                  and (probe == 'overall' or r['probe_id'] == probe)]
                        if not subset:
                            continue
                        subset_scores.append(dict(model=model, state=mode, group=group, question_type=probe,
                                                  test=test, questions=len({r['source_id'] for r in subset}),
                                                  units=len({r['unit_id'] for r in subset}), rows=len(subset),
                                                  accuracy_pct=100 * sum(r['correct'] for r in subset) / len(subset)))
                    control_ids = {r['source_id'] for r in by_condition['no_image']} & ids
                    if probe != 'overall':
                        control_ids = {sid for sid in control_ids if indexed[sid]['question_type'] == probe}
                    if not control_ids:
                        continue
                    values = {}
                    for test in ('image', 'no_image', 'wrong_image'):
                        subset = [r for r in by_condition[test] if r['source_id'] in control_ids]
                        values[test] = 100 * sum(r['correct'] for r in subset) / len(subset)
                    subset_controls.append(dict(model=model, state=mode, group=group, question_type=probe,
                                                questions=len(control_ids), **{k + '_pct': v for k, v in values.items()},
                                                correct_minus_no_image_pp=values['image'] - values['no_image'],
                                                correct_minus_wrong_image_pp=values['image'] - values['wrong_image']))
    csvfile('subset_scores', subset_scores)
    csvfile('subset_controls', subset_controls)
    print(json.dumps(json.loads((OUT / 'static_summary.json').read_text()), ensure_ascii=False, indent=2))
    print('Sample frozen and 4 contact sheets generated:', OUT)


if __name__ == '__main__':
    main()
