#!/usr/bin/env python3
"""Combine every prespecified method, paired rule-level changes and artifacts."""
import csv
import json
import random
from collections import Counter

import explore_v7_memory_router as mem
import explore_v7_observation_router as obs
import explore_v7_material_memory as pt
from run_v7_full_experiment import rows, read, save, sha


def selected(arm, method):
    if method == 'direct':
        path = mem.OUT / arm / 'memory/predictions.jsonl'; stage = 'direct'
    elif method == 'routed_base':
        path = mem.OUT / arm / 'routed_base/predictions.jsonl'; stage = 'choice'
    else:
        path = obs.OUT / arm / 'predictions.jsonl'; stage = 'choice'
    return [r for r in rows(path) if r['stage'] == stage]


def paired(rr, base):
    reference = {(r['source_id'], r['input_mod']): r for r in base}
    pairs = [(r, reference[(r['source_id'], r['input_mod'])]) for r in rr]
    assert len(pairs) == len(rr)
    byrule = {}
    for r, b in pairs:
        byrule.setdefault(r['qid'], []).append(int(r['correct']) - int(b['correct']))
    gains = [sum(v)/len(v) for v in byrule.values()]
    rng = random.Random(42)
    boot = sorted(sum(rng.choices(gains, k=len(gains)))/len(gains) for _ in range(10000))
    return {'n_rules': len(gains), 'n_questions': len(rr),
            'base_wrong_trained_right': sum(r['correct'] and not b['correct'] for r,b in pairs),
            'base_right_trained_wrong': sum(b['correct'] and not r['correct'] for r,b in pairs),
            'both_right': sum(b['correct'] and r['correct'] for r,b in pairs),
            'accuracy_gain': sum(gains)/len(gains),
            'descriptive_rule_cluster_bootstrap_95ci': [boot[249], boot[9749]],
            'limits': 'Same18 development rules and1seed; descriptive interval, no independent confirmation or correction for searching multiple methods.'}


def main():
    mem.audit(); obs.audit(); pt.audit(); mem.summarize(); obs.summarize()
    assert read(mem.OUT / 'completion_audit.json')['all_requested_complete']
    result, table, changes = {}, [], {}
    for method in ('direct', 'routed_base', 'observation_route'):
        arms = mem.ARMS + (pt.ARM,) if method != 'observation_route' else obs.ARMS
        if method == 'routed_base': arms += ('oracle',)
        base = selected('base', method)
        result[method] = {}
        for arm in arms:
            rr = selected(arm, method)
            result[method][arm] = {}
            for mod in ('T','V'):
                ss = [r for r in rr if r['input_mod'] == mod]
                if not ss: continue
                assert len(ss) == 36
                stats = mem.metric(ss)
                for key in ('task','relation','evaluation_group','qid'):
                    stats['by_'+key] = {v: mem.metric([r for r in ss if r[key] == v]) for v in sorted({r[key] for r in ss})}
                result[method][arm][mod] = stats
                delta = paired(ss, [r for r in base if r['input_mod'] == mod])
                changes[method+':'+arm+':'+mod] = delta
                table.append(dict(method=method, arm=arm, input_mod=mod, correct=stats['correct'], n=stats['n'],
                    accuracy=stats['accuracy'], valid_accuracy=stats['valid_accuracy'], format_errors=stats['format_errors'],
                    direct_correct=stats['by_task']['direct']['correct'], diagnosis_correct=stats['by_task']['diagnosis']['correct'],
                    gain_over_matched_base=delta['accuracy_gain'], gained=delta['base_wrong_trained_right'], lost=delta['base_right_trained_wrong']))
    save(mem.OUT / 'all_methods.json', result)
    save(mem.OUT / 'paired_changes.json', changes)
    with (mem.OUT / 'all_methods.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    target = {r['qid']: r['answer'] for r in rows(mem.OUT / 'targets.jsonl') if r['arm'] == 'text_memory'}
    normalize = lambda s: ' '.join(s.split())
    reconstruction = {}
    for arm in mem.ARMS + (pt.ARM,):
        records = rows(mem.OUT / arm / 'memory/predictions.jsonl')
        reconstruction[arm] = {}
        for stage in ('blind_recall','image_recall'):
            ss = [r for r in records if r['stage'] == stage]
            reconstruction[arm][stage] = {'n': len(ss),
                'exact_normalized_target': sum(normalize(r['response']) == normalize(target[r['qid']]) for r in ss),
                'truncated': sum(r['finish_reason'] == 'length' for r in ss),
                'empty': sum(not r['response'].strip() for r in ss),
                'not_an_exhaustive_semantic_score': True}
    save(mem.OUT / 'target_reconstruction.json', reconstruction)
    # Validate the actual model used for every final inference against frozen identity.
    frozen = read(mem.OUT / 'manifest.json')['models']
    for arm in mem.ARMS + (pt.ARM,):
        cfg = read(mem.OUT / arm / 'memory/eval_manifest.json')['model']
        if arm not in mem.TRAIN_ARMS + (pt.ARM,):
            assert cfg == frozen[{'base':'base','text_ep10':'text','vision_ep10':'vision'}[arm]]
        else:
            cp = read(mem.OUT / arm / 'checkpoint_manifest.json')
            assert cfg == {'path':cp['path'], 'sha256':cp['sha256']}
        assert read(mem.OUT / arm / 'routed_base/eval_manifest.json')['model'] == frozen['base']
    assert read(mem.OUT / 'oracle/routed_base/eval_manifest.json')['model'] == frozen['base']
    for arm in obs.ARMS:
        assert read(obs.OUT / arm / 'eval_manifest.json')['model'] == frozen['base']
    # Plot actual formal training loss; never mix smoke or old experiments.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,4.5))
    train_info = {}
    for arm in mem.TRAIN_ARMS + (pt.ARM,):
        lr = [r for r in rows(mem.OUT / arm / 'checkpoint/logging.jsonl') if 'loss' in r]
        steps = [int(r['global_step/max_steps'].split('/')[0]) for r in lr]
        ax.plot(steps, [r['loss'] for r in lr], label=arm, linewidth=1.5)
        train_info[arm] = read(mem.OUT / arm / 'checkpoint_manifest.json')
    ax.set(xlabel='Optimizer step', ylabel='Assistant token CE loss', yscale='log',
           title='Clean-base memory LoRA: fixed final step 144')
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(mem.OUT/'training_loss.png',dpi=180); plt.close(fig)
    save(mem.OUT / 'training_summary.json', train_info)
    main_count = read(mem.OUT / 'completion_audit.json')['new_generations'] + read(pt.OUT / 'completion_audit.json')['generations']
    obs_count = read(obs.OUT / 'completion_audit.json')['new_generations']
    smoke_count = sum(len(rows(mem.OUT / ('smoke_'+arm) / 'reload/predictions.jsonl')) for arm in mem.TRAIN_ARMS) + len(rows(pt.OUT / 'smoke/reload/predictions.jsonl'))
    save(mem.OUT / 'result_inventory.json', {'completed':True,
        'main_generations':main_count, 'observation_generations':obs_count, 'successful_smoke_generations':smoke_count,
        'total_successful_generations':main_count+obs_count+smoke_count, 'formal_training_jobs':4, 'original_images_unchanged':3000,
        'files':{str(p.relative_to(mem.OUT)):sha(p) for p in sorted(mem.OUT.glob('*.json')) if p.name!='result_inventory.json'}})
    print(json.dumps(table,ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
