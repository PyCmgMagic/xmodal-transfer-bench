#!/usr/bin/env python3
"""Independent audit/reporting for the frozen SFT/DPO/GRPO pilot."""
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/v7_preference_learning_20261020'
ARMS = ('base', 'warmup', 'sft', 'dpo', 'grpo')


def rows(p):
    return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]


def save(p, z):
    p.write_text(json.dumps(z, ensure_ascii=False, indent=2) + '\n')


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def metric(rr):
    n = len(rr)
    valid = sum(not r['format_error'] for r in rr)
    correct = sum(r['scored_correct'] for r in rr)
    return dict(n=n, correct=correct, accuracy=correct/n if n else None,
                format_errors=n-valid, valid_accuracy=correct/valid if valid else None,
                truncated=sum(r['finish_reason'] == 'length' for r in rr))


def key(r):
    return ':'.join(r[k] for k in ('condition', 'protocol', 'input_mod'))


def paired(left, right):
    a = {r['id']: r for r in left}
    b = {r['id']: r for r in right}
    assert a.keys() == b.keys()
    gains = sum(not a[i]['scored_correct'] and b[i]['scored_correct'] for i in a)
    losses = sum(a[i]['scored_correct'] and not b[i]['scored_correct'] for i in a)
    discord = gains + losses
    p = min(1., 2 * sum(math.comb(discord, j) for j in range(min(gains, losses)+1)) / 2**discord) if discord else 1.
    groups = defaultdict(list)
    for i in a:
        groups[a[i]['qid']].append(int(b[i]['scored_correct'])-int(a[i]['scored_correct']))
    ids = sorted(groups)
    rng = random.Random(42)
    boot = []
    for _ in range(5000):
        values = [v for q in rng.choices(ids, k=len(ids)) for v in groups[q]]
        boot.append(sum(values)/len(values))
    boot.sort()
    return dict(n=len(a), gained=gains, lost=losses, delta=(gains-losses)/len(a),
                mcnemar_exact_two_sided=p, rule_cluster_bootstrap_ci95=[boot[125], boot[4874]],
                warning='Exploratory, 8 rules / one seed; no multiplicity correction or population-level claim.')


def main():
    from v7_preference_reward import components
    import run_v7_evaluation as ev
    rr = rows(OUT / 'scored_predictions.jsonl')
    assert len(rr) == 1160
    probes = {p['id']: p for p in rows(OUT / 'probes.jsonl')}
    assert len(probes) == 232
    for arm in ARMS:
        assert {r['id'] for r in rr if r['arm'] == arm} == probes.keys()
    for r in rr:
        p=probes[r['id']]
        assert r['input_sha256'] == ev.digest({'messages':p['messages'],'images':p.get('images',[])})
        assert r['gold'] == p['gold']
    metrics = []
    for arm in ARMS:
        for k in sorted({key(r) for r in rr}):
            subset = [r for r in rr if r['arm'] == arm and key(r) == k]
            metrics.append(dict(arm=arm, test=k, **metric(subset)))
    with (OUT / 'summary.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(metrics[0])); w.writeheader(); w.writerows(metrics)
    rule_metrics = []
    for arm in ARMS:
        for qid in sorted({r['qid'] for r in rr}):
            for k in sorted({key(r) for r in rr}):
                subset = [r for r in rr if r['arm'] == arm and r['qid'] == qid and key(r) == k]
                rule_metrics.append(dict(arm=arm, qid=qid, test=k, **metric(subset)))
    save(OUT / 'per_rule_metrics.json', rule_metrics)
    comparisons = []
    for before in ('base', 'warmup', 'sft'):
        for after in ('warmup', 'sft', 'dpo', 'grpo'):
            if before == after or (before == 'sft' and after == 'warmup'):
                continue
            for k in sorted({key(r) for r in rr if r['condition'] != 'recall'}):
                a = [r for r in rr if r['arm'] == before and key(r) == k]
                b = [r for r in rr if r['arm'] == after and key(r) == k]
                comparisons.append(dict(before=before, after=after, test=k, **paired(a,b)))
    save(OUT / 'paired_comparisons.json', comparisons)
    by_group = defaultdict(list)
    for r in rr:
        if r['condition'] == 'original':
            by_group[(r['arm'], r['protocol'], r['input_mod'], r['task'], r['evaluation_group'])].append(r)
    save(OUT / 'original_task_breakdown.json', [dict(arm=k[0], protocol=k[1], input_mod=k[2],
         task=k[3], evaluation_group=k[4], **metric(v)) for k,v in sorted(by_group.items())])

    # Recompute every logged reward from preserved completions, independent of trainer metrics.
    reward_rows = []; reward_groups = []; seen_batches = set(); duplicate_batches = 0
    for batch in rows(OUT / 'grpo/checkpoint/completions.jsonl'):
        batch_sig = json.dumps(batch, sort_keys=True)
        if batch_sig in seen_batches:
            duplicate_batches += 1
            continue
        seen_batches.add(batch_sig)
        grouped = defaultdict(list)
        for i, (text, gold, logged, prompt) in enumerate(zip(batch['completion'], batch['solution'], batch['V7RuleApplicationReward'], batch['prompt'])):
            score = components(text, gold)
            assert abs(score['reward'] - float(logged)) < 1e-6
            assert gold not in prompt, 'Reward metadata leaked to model prompt'
            z = dict(step=int(batch['step'][i]), **score, advantage=float(batch['advantages'][i]))
            reward_rows.append(z); grouped[(z['step'], prompt)].append(z)
        for (step, _), group in grouped.items():
            assert len(group) == 4
            reward_groups.append(dict(step=step, zero_reward_variance=len({r['reward'] for r in group})==1,
                                      all_zero_advantage=all(abs(r['advantage']) < 1e-7 for r in group)))
    assert len(reward_rows) == 512 and len(reward_groups) == 128
    reward_audit = dict(samples=len(reward_rows), groups=len(reward_groups), reward_recomputation='pass',
                        duplicate_log_batches=duplicate_batches,
                        duplicate_note='Exact repeated trainer flush rows excluded from statistics; raw log preserved.',
                        full_solution_absent_from_prompts=True,
                        zero_reward_variance_groups=sum(r['zero_reward_variance'] for r in reward_groups),
                        zero_advantage_groups=sum(r['all_zero_advantage'] for r in reward_groups))
    for label, sub in [('all', reward_rows), ('first_quarter', reward_rows[:128]), ('last_quarter', reward_rows[-128:])]:
        reward_audit[label] = dict(n=len(sub), mean_reward=sum(r['reward'] for r in sub)/len(sub),
                                  rule_exact=sum(r['rule_exact'] for r in sub)/len(sub),
                                  keep_exact=sum(r['keep_exact'] for r in sub)/len(sub),
                                  valid=sum(r['valid'] for r in sub)/len(sub))
    save(OUT / 'grpo_reward_audit.json', reward_audit)

    # Formal outputs are preserved as raw examples, never used to change rewards/checkpoints.
    knowledge = {k['id']: k for k in rows(OUT / 'knowledge.jsonl')}
    original = []
    for r in rr:
        if r['condition'] == 'original' and r['protocol'] == 'reasoning':
            original.append(dict(arm=r['arm'], id=r['id'], qid=r['qid'], input_mod=r['input_mod'],
                gold=r['gold'], correct=r['scored_correct'], format_error=r['format_error'],
                finish_reason=r['finish_reason'], true_rule=json.loads(knowledge[r['qid']]['fact']['quote']),
                prompt=probes[r['id']]['messages'][-1]['content'], response=r['response']))
    with (OUT / 'original_reasoning_review.jsonl').open('w') as f:
        for r in original: f.write(json.dumps(r, ensure_ascii=False)+'\n')

    # Check same-scene/same-chosen supervision across algorithms.
    sft = {r['id']:r for r in rows(OUT / 'sft.jsonl')}
    dpo = {r['id']:r for r in rows(OUT / 'dpo.jsonl')}
    grpo = {r['id']:r for r in rows(OUT / 'grpo.jsonl')}
    assert sft.keys() == dpo.keys() == grpo.keys()
    for i in sft:
        assert sft[i]['messages'] == dpo[i]['messages']
        assert sft[i]['messages'][:-1] == grpo[i]['messages']
        assert sft[i]['messages'][-1]['content'] == grpo[i]['solution']
        assert 'Options:' not in str(sft[i]['messages'])
    from v7_preference_reward import canonical_conditions
    same_conditions = sum(canonical_conditions(json.loads(dpo[i]['messages'][-1]['content'])) ==
                          canonical_conditions(json.loads(dpo[i]['rejected_response'])) for i in dpo)
    save(OUT / 'dpo_preference_coverage.json', dict(pairs=len(dpo),same_rule_parameter_conditions=same_conditions,
         limitation='All synthetic rejected answers retain correct condition field/operator/value. Half differ in mode, half only in candidate decisions. Does not directly contrast hallucinated numeric/shape/color parameters.'))
    blocks = rows(OUT / 'blocks.jsonl')
    train = {r['observation_signature'] for r in blocks if r['split']=='train'}
    hold = {r['observation_signature'] for r in blocks if r['split']=='holdout'}
    assert len(train) == len(hold) == 128 and not train & hold
    save(OUT / 'independent_audit.json', dict(status='pass', predictions=1160, same_scenes_and_chosen_targets=True,
         all_eval_input_hashes_and_labels_match_frozen_probes=True,
         train_holdout_scene_disjoint=True, final_checkpoints_only=True,
         analysis_source_sha256=digest(Path(__file__)), reward_numeric_equivalence_fixed=True))
    import torch
    from safetensors import safe_open
    torch.set_num_threads(4)
    baseline = OUT / 'warmup/checkpoint/checkpoint-128/adapter_model.safetensors'
    updates = {}
    for arm in ('sft', 'dpo', 'grpo'):
        folder = OUT / arm / 'checkpoint' / ('checkpoint-128' if arm == 'grpo' else 'checkpoint-64')
        args = json.loads((folder / 'args.json').read_text())
        assert args['adapters'] == [str(baseline.parent)]
        if arm != 'sft':
            assert args['ref_adapters'] == [str(baseline.parent)]
        total = delta = 0.; changed = 0
        with safe_open(str(baseline), framework='pt', device='cpu') as a, safe_open(str(folder/'adapter_model.safetensors'), framework='pt', device='cpu') as b:
            assert a.keys() == b.keys()
            for k in a.keys():
                x=a.get_tensor(k).float();y=b.get_tensor(k).float()
                total+=(x*x).sum().item();delta+=((y-x)**2).sum().item();changed+=not x.equal(y)
            updates[arm]=dict(changed_tensors=changed, total_tensors=len(a.keys()), relative_l2_from_warmup=(delta/total)**.5)
        assert changed > 0
    save(OUT / 'parameter_update_audit.json', updates)

    lines = ['# 本轮训练数据示例', '', '仅TEXT来源；8条固定规则；保留原图与正式题。训练没有选择题选项或正确选项字母。', '',
             '共同起点：原学习材料提取规则定义，4种问法各重复8次，共256行，2epoch。以下三组从同一个最终warmup权重开始。', '',
             'SFT：128个新四候选场景，监督正确规则条件与四个KEEP布尔值。DPO：相同场景与相同正确回答，负例交替为错误AND/OR或翻错一个决策。GRPO：相同场景，仅生成提示；gold solution为奖励元数据，不进入提示。', '',
             '```json', json.dumps(next(iter(sft.values())), ensure_ascii=False, indent=2), '```', '',
             'DPO对应负例：', '', '```json', dpo[next(iter(sft))]['rejected_response'], '```', '',
             'GRPO奖励：0.5×精确规则条件 + 0.5×四候选决策正确比例。非法格式为0。整数4与浮点4.0等价。', '',
             '训练场景128、独立新留出128，按完整无序四候选场景隔离；并非每个单候选都不重复。正式原题只用于场景排除、标签核查及最后评测，未作训练。新留出题的选项位置在规则内平衡随机化，与KEEP组合编号独立。', '',
             '仅两类结构、单seed。DPO负例不是on-policy模型错误；GRPO采样/步数更多，不能视为同计算预算。']
    (OUT / 'TRAINING_DATA_ZH.md').write_text('\n'.join(lines)+'\n')
    plot()
    print('Independent analysis complete:', OUT)


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(11,7))
    for arm, ax in zip(('warmup','sft','dpo'), axes.flat):
        logs = [r for r in rows(OUT / arm / 'checkpoint/logging.jsonl') if 'loss' in r]
        steps = [int(r['global_step/max_steps'].split('/')[0]) for r in logs]
        ax.plot(steps, [r['loss'] for r in logs], label='training loss')
        if arm == 'dpo' and 'rewards/accuracies' in logs[-1]:
            ax2=ax.twinx(); ax2.plot(steps,[r['rewards/accuracies'] for r in logs],color='orange',alpha=.6)
            ax2.set_ylabel('Training preference accuracy',color='orange')
        ax.set(title=arm, xlabel='Optimizer step', ylabel='Loss'); ax.grid(alpha=.2)
    logs=[r for r in rows(OUT / 'grpo/checkpoint/logging.jsonl') if 'reward' in r]
    ax=axes[1,1]; steps=[int(r['global_step/max_steps'].split('/')[0]) for r in logs]
    ax.plot(steps,[r['reward'] for r in logs],label='rollout reward',alpha=.8)
    ax.plot(steps,[r['frac_reward_zero_std'] for r in logs],label='zero-variance group',alpha=.4)
    ax.set(title='GRPO (training only)',xlabel='Optimizer step',ylabel='Reward / group indicator');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Different objectives: loss/reward are not comparable across methods')
    fig.tight_layout();fig.savefig(OUT / 'training_curves.png',dpi=180);plt.close(fig)


if __name__ == '__main__': main()
