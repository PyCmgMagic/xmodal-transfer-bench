#!/usr/bin/env python3
"""Derive task-relevant consequences of saved observations; no new inference."""
import json

import explore_v7_visual_replay as trial
import pilot_v7_counterfactual_names as pilot
from run_v7_full_experiment import rows, read, save, sha


def main():
    knowledge = {r['id']: json.loads(r['fact']['quote']) for r in rows(trial.SOURCE / 'target_knowledge.jsonl')}
    blocks = {b['source_id']: b for b in rows(trial.SOURCE / 'blocks.jsonl')}
    result = []
    arms = ('warmup', 'text', 'vision') + trial.ARMS
    for arm in arms:
        source = trial.OUT / 'observation_diagnostic' / arm / 'extraction_reviews.jsonl'
        for r in rows(source):
            block = blocks[r['id'].split('extract:', 1)[1]]
            parsed = r['parsed']
            checks = []
            for qid in block['rule_ids']:
                try:
                    observed = pilot.assignment([pilot.independent(knowledge[qid], parsed[c]) for c in 'ABCD'])
                except (TypeError, KeyError):
                    observed = None
                checks.append({'qid': qid, 'true_result': block['rule_results'][qid],
                               'observed_result': observed, 'equal': observed == block['rule_results'][qid]})
            result.append({'arm': arm, 'id': r['id'], 'group': r['group'],
                'source': str(source), 'source_sha256': sha(source),
                'strict_scene_correct': r['scene_correct'], 'rule_result_checks': checks,
                'note': 'Deterministic rule execution on saved parsed fields, not model reasoning. Dark-blue vs black word correspondence is not corrected.'})
    assert len(result) == 25
    summary = {}
    for arm in arms:
        summary[arm] = {}
        for group in (0, 1):
            ss = [r for r in result if r['arm'] == arm and r['group'] == group]
            checks = [c for r in ss for c in r['rule_result_checks']]
            summary[arm][str(group)] = {'scenes': len(ss), 'rule_scene_comparisons': len(checks),
                'matching_full_results': sum(c['equal'] for c in checks), 'new_generations': 0}
    save(trial.OUT / 'symbolic_observation_effects.json', {'summary': summary, 'details': result})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
