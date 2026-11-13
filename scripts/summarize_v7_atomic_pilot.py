#!/usr/bin/env python3
"""Export four-direction results, paired changes and a hashed artifact inventory."""
import csv
import json
import shutil
from pathlib import Path

from pilot_v7_atomic_supervision import OUT, ARMS
from run_v7_full_experiment import read, rows, save, sha, ROOT


def main():
    validation = read(OUT / 'validation.json'); assert validation['status'] == 'pass'
    summary = read(OUT / 'summary.json')
    assert len(summary['arms']) == len(summary['neutral_protocol']) == 5
    cells = []
    for variant in ('original', 'atomic'):
        for modality, directions in (('text', ('T2T', 'T2V')), ('vision', ('V2T', 'V2V'))):
            arm = variant + '_' + modality
            for direction, target in zip(directions, ('T', 'V')):
                for protocol, condition, source in (
                    ('direct', 'direct_' + target, summary['arms'][arm]['metrics']),
                    ('self_heading', 'self_' + target, summary['arms'][arm]['metrics']),
                    ('self_identifier', 'neutral_self_' + target, summary['neutral_protocol'][arm]['metrics']),
                    ('oracle_external_text', 'oracle_' + target, summary['arms'][arm]['metrics'])):
                    m = source[condition]; valid = m['n'] - m['format_errors']
                    cells.append(dict(variant=variant, arm=arm, direction=direction, protocol=protocol,
                                      n=m['n'], correct=m['correct'], accuracy_all=m['correct'] / m['n'],
                                      valid=valid, accuracy_valid=m['correct'] / valid if valid else None,
                                      format_errors=m['format_errors'], truncated=m['truncated']))
    with (OUT / 'four_directions.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(cells[0])); writer.writeheader(); writer.writerows(cells)
    save(OUT / 'four_directions.json', cells)
    paired = {}
    for mod in ('text', 'vision'):
        a = {r['id']: r for r in rows(OUT / ('original_' + mod) / 'predictions.jsonl')}
        b = {r['id']: r for r in rows(OUT / ('atomic_' + mod) / 'predictions.jsonl')}
        paired[mod] = {}
        for condition in ('direct_T', 'direct_V'):
            keys = [key for key in a if a[key]['condition'] == condition]
            paired[mod][condition] = {'n': 16,
                                      'wrong_to_correct': sum(not a[k]['correct'] and b[k]['correct'] for k in keys),
                                      'correct_to_wrong': sum(a[k]['correct'] and not b[k]['correct'] for k in keys),
                                      'unchanged_correct': sum(a[k]['correct'] and b[k]['correct'] for k in keys),
                                      'unchanged_wrong': sum(not a[k]['correct'] and not b[k]['correct'] for k in keys)}
    save(OUT / 'paired_changes.json', paired)
    inventory = []
    for arm in ARMS:
        for folder in (OUT / arm, OUT / 'neutral_recall' / arm):
            path = folder / 'predictions.jsonl'
            inventory.append({'path': str(path), 'sha256': sha(path), 'records': len(rows(path)), 'reused': 0})
    assert sum(r['records'] for r in inventory) == 1240
    save(OUT / 'response_inventory.json', inventory)
    snap = OUT / 'source_snapshots'
    for name in ('diagnose_v7_atomic_neutral_recall.py', 'validate_v7_atomic_pilot.py',
                 'review_v7_atomic_recall.py', 'summarize_v7_atomic_pilot.py'):
        source = ROOT / 'scripts' / name; target = snap / name
        if target.exists():
            assert sha(source) == sha(target)
        else:
            shutil.copy2(source, target)
    save(OUT / 'script_manifest.json', {p.name: sha(p) for p in snap.iterdir() if p.is_file()})
    save(OUT / 'execution_notes.json', {
        'generation_calls': 1240, 'reused_generations': 0, 'training_jobs': 4, 'epochs_per_job': 1,
        'final_step_per_job': 128, 'source_images_modified': 0,
        'startup_retry': 'One parallel startup status-write collision occurred before TEXT training began; restart completed. No training or prediction rows were overwritten/repeated for this retry.',
        'admission_cleanup': 'Completed evaluation reservations from multi-arm worker processes were released after report completion; unrelated GPU processes were untouched.',
        'reproduction': 'Use one arm per Python worker invocation, and serialize preflight verification/status writes when starting parallel workers.',
        'protocol_followup': 'Neutral identifier recall is a post-diagnostic intervention applied uniformly to all five arms. Original heading-based and direct results retained.'})
    print('Exported32 four-direction cells,10 prediction files,1240 new generations.')


if __name__ == '__main__':
    main()
