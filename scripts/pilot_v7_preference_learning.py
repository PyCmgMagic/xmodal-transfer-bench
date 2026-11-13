#!/usr/bin/env python3
"""Source-only SFT/DPO/GRPO learning pilot; immutable v7 formal tests."""
import argparse
import copy
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import random
import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import run_v7_evaluation as ev
import explore_v7_memory_router as mem
import pilot_v7_learning_first as old
from build_v7_experiment_data import name
from run_v7_full_experiment import ROOT, MODELS, environment, rows, read, save, sha, train_command
from run_v7_fullparam_ep10 import dense_identity
from v7_gpu_resources import reserve
from v7_preference_reward import components, parse, canonical_conditions

OUT = ROOT / 'runs/v7_preference_learning_20261020'
DATA = ROOT / 'rules_repaired_single_image_v7'
PLUGIN = ROOT / 'scripts/v7_preference_reward.py'
PY = ROOT / '.venv/bin/python'
ARMS = ('base', 'warmup', 'sft', 'dpo', 'grpo')
HISTORICAL = {'scale1k_' + s for s in ('0955_programmatic', '0270_programmatic', '0085_programmatic',
              '0953_programmatic', '0966_programmatic', '0963_programmatic', '0274_programmatic', '0698_ai_generated')}
SYSTEM = ('Recall and apply the learned named rule. Return a JSON object only, with keys mode, conditions, keep. '
          'mode is "all" or "any". conditions is the two-element list of objects with field, operator, value. '
          'keep is four JSON booleans in candidate A,B,C,D order. Use operators eq, ge, lt. No explanation.')
REASON_SYSTEM = 'Apply the learned named rule. Explain its conditions, check every candidate, then give the requested final option.'
REASON_SUFFIX = '\n\nExplain the rule and the candidate checks. End with Answer: followed by one uppercase option letter.'


def write(p, rr):
    ev.save_rows(p, rr)


def sig(obs):
    return ev.digest(sorted(json.dumps({k: int(v) if isinstance(v, float) and v.is_integer() else v
                                      for k, v in o.items()}, sort_keys=True) for o in obs))


def observed(p):
    text = p['messages'][-1]['content']
    if '\n\nTest input:\n' not in text:
        return None
    text = text.split('\n\nTest input:\n', 1)[1].split('\n\nOptions:\n', 1)[0]
    result = []
    for _, description in re.findall(r'([A-D]) has (.*?)(?=; [A-D] has|$)', text.rstrip('.')):
        d = dict(re.findall(r'(\w+): ([^,;]+)', description))
        if 'value' in d:
            d['value'] = float(d['value'])
        result.append(d)
    return result if len(result) == 4 else None


def target(rule, obs):
    keep = [any(old.flags(rule, o)) if rule['mode'] == 'any' else all(old.flags(rule, o)) for o in obs]
    assert '; '.join(f'{c}={"KEEP" if b else "OMIT"}' for c, b in zip('ABCD', keep)) == old.independent_assignment(rule, obs)
    return dict(mode=rule['mode'], conditions=rule['conditions'], keep=keep)


def encoded(z):
    return json.dumps(z, ensure_ascii=False, separators=(',', ':'))


def text_obs(obs):
    return '; '.join(c + ' has ' + ', '.join(f'{f}: {v}' for f, v in o.items()) for c, o in zip('ABCD', obs)) + '.'


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'manifest.json').exists():
        validate(); return
    allk = [r['content'] for r in rows(DATA / 'audit/rule_knowledge.jsonl')]
    chosen = []
    for topology in ('disjunctive_filter', 'interval_filter'):
        candidates = []
        for k in allk:
            if k['id'] in HISTORICAL:
                continue
            try:
                r = json.loads(k['fact']['quote'])
            except ValueError:
                continue
            if r.get('selection_topology') != topology:
                continue
            if topology == 'disjunctive_filter' and {(c['field'], c['operator']) for c in r['conditions']} != {('shape', 'eq'), ('color', 'eq')}:
                continue
            if topology == 'interval_filter' and {(c['field'], c['operator']) for c in r['conditions']} != {('value', 'ge'), ('value', 'lt')}:
                continue
            candidates.append(k)
        candidates.sort(key=lambda k: hashlib.sha256(('preference-pilot-42:' + k['id']).encode()).hexdigest())
        assert len(candidates) >= 4
        chosen += candidates[:4]
    ids = {k['id'] for k in chosen}; assert len(ids) == 8 and not ids & HISTORICAL
    # Original inputs are accessed ONLY for exclusion, label auditing and final evaluation.
    formal = rows(DATA / 'T2T/eval.jsonl')
    forbidden = {sig(o) for p in formal if (o := observed(p)) is not None}
    occupied = set(forbidden); blocks = []; practice = []; preference = []; rollout = []; hold = []; warm = []
    for k in chosen:
        r = json.loads(k['fact']['quote'])
        for j in range(32):
            warm.append(dict(id=f'warm:{k["id"]}:{j}', qid=k['id'], messages=mem.messages(
                mem.FORMS[j % len(mem.FORMS)].format(name=name(k)), mem.compact(k))))
        for split in ('train', 'holdout'):
            positions = list(range(4)) * 4
            random.Random(f'choice-order-42:{k["id"]}:{split}').shuffle(positions)
            for j in range(16):
                rng = random.Random(f'preference-42:{k["id"]}:{split}:{j}')
                mask = j
                for _ in range(10000):
                    obs = []
                    for pos in range(4):
                        keep = bool(mask & (1 << pos))
                        if r['mode'] == 'any':
                            wanted = {c['field']: c['value'] for c in r['conditions']}
                            flags = rng.choice([(1, 0), (0, 1), (1, 1)]) if keep else (0, 0)
                            o = dict(shape=wanted['shape'] if flags[0] else rng.choice([s for s in ['circle', 'triangle', 'square', 'pentagon', 'hexagon', 'star', 'oval', 'diamond'] if s != wanted['shape']]),
                                     color=wanted['color'] if flags[1] else rng.choice([s for s in ['red', 'blue', 'green', 'gray', 'black', 'white', 'yellow', 'purple', 'brown', 'orange'] if s != wanted['color']]))
                        else:
                            lo = next(c['value'] for c in r['conditions'] if c['operator'] == 'ge')
                            hi = next(c['value'] for c in r['conditions'] if c['operator'] == 'lt')
                            vals = [lo, hi - .25, round(rng.uniform(lo, hi - .001), 3)] if keep else [lo - .25, hi, hi + .25, round(rng.uniform(hi + .01, hi + 5), 3)]
                            o = {'value': rng.choice(vals)}
                        obs.append(o)
                    if sig(obs) not in occupied:
                        break
                else:
                    raise RuntimeError('scene generation exhausted')
                occupied.add(sig(obs)); z = target(r, obs); gold = encoded(z)
                block = dict(id=f'{split}:{k["id"]}:{j}', qid=k['id'], split=split, observations=obs,
                             observation_signature=sig(obs), gold=gold)
                blocks.append(block)
                msgs = [dict(role='system', content=SYSTEM), dict(role='user', content=f'Apply {name(k)}.\nCandidates: ' + text_obs(obs))]
                if split == 'train':
                    base = dict(id=block['id'], qid=k['id'], messages=msgs, observations=obs, solution=gold)
                    rollout.append(copy.deepcopy(base))
                    s = copy.deepcopy(base); s['messages'].append(dict(role='assistant', content=gold)); practice.append(s)
                    # Half knowledge errors, half execution errors; same JSON style as chosen.
                    bad = copy.deepcopy(z)
                    if j % 2:
                        bad['keep'][j % 4] = not bad['keep'][j % 4]
                    else:
                        bad['mode'] = 'all' if z['mode'] == 'any' else 'any'
                        bad['keep'] = [all(old.flags(r, o)) if bad['mode'] == 'all' else any(old.flags(r, o)) for o in obs]
                    assert components(gold, gold)['reward'] == 1 and components(encoded(bad), gold)['reward'] < 1
                    d = copy.deepcopy(s); d['rejected_response'] = encoded(bad); preference.append(d)
                else:
                    # New MCQs diagnose generalization independently of the training JSON format.
                    result = old.assignment(r, obs)
                    opts = ['; '.join(f'{c}={"KEEP" if b else "OMIT"}' for c, b in zip('ABCD', bits)) for bits in itertools.product((False, True), repeat=4)]
                    opts.remove(result); rng.shuffle(opts); opts = opts[:3]; opts.insert(positions[j], result)
                    body = f'Apply {name(k)}. Report KEEP/OMIT for each candidate; KEEP means selected and OMIT means not selected.\n\nTest input:\nThe observed candidate set is: ' + text_obs(obs)
                    body += '\n\nOptions:\n' + '\n'.join(c + '. Result: ' + v + '.' for c, v in zip('ABCD', opts)) + ev.SUFFIX
                    hold.append(dict(id='holdout:' + block['id'], qid=k['id'], condition='new_holdout', input_mod='T', fmt='mcq',
                                     task='direct', evaluation_group='result_application', protocol='short', gold='ABCD'[positions[j]],
                                     messages=[dict(role='system', content=ev.SYSTEM), dict(role='user', content=body)]))
    for rr in (warm, practice, preference, rollout):
        random.Random(42).shuffle(rr)
    assert len(practice) == len(preference) == len(rollout) == len(hold) == 128
    assert len(warm) == 256
    labels = {r['id']: r for r in rows(DATA / 'audit/eval_labels.jsonl')}
    probes = copy.deepcopy(hold)
    # Equal, predefined four scenes per rule also get the explicit reasoning protocol.
    for p in hold:
        if int(p['id'].split(':')[-1]) < 4:
            probes.append(reasoning(p))
    for direction, mod in [('T2T', 'T'), ('T2V', 'V')]:
        for p in rows(DATA / direction / 'eval.jsonl'):
            if p['qid'] not in ids:
                continue
            label = labels[p['id']]; z = ev.prepare_row(p, DATA, direction)
            z.update(id='original:' + mod + ':' + p['id'], source_id=p['id'], condition='original', input_mod=mod,
                     fmt='mcq', protocol='short', gold=label['answer'], task=label['question_type'], evaluation_group=label['evaluation_group'])
            if mod == 'T':
                obs = observed(p); assert obs is not None
                k = next(k for k in chosen if k['id'] == p['qid']); r = json.loads(k['fact']['quote'])
                expected = old.assignment(r, obs)
                options = dict(re.findall(r'(?m)^([A-D])\. (.*)$', p['messages'][-1]['content']))
                assert expected in options[label['answer']], (p['id'], expected)
            probes.extend([z, reasoning(z)])
    for k in chosen:
        probes.append(dict(id='recall:' + k['id'], qid=k['id'], input_mod='T', condition='recall', fmt='fact',
                           protocol='recall', gold=mem.compact(k), messages=mem.messages(mem.RECALL_Q.format(name=name(k)))))
    for n, rr in [('knowledge', chosen), ('blocks', blocks), ('warmup', warm), ('sft', practice), ('dpo', preference), ('grpo', rollout), ('probes', probes)]:
        write(OUT / (n + '.jsonl'), rr)
    sources = [Path(__file__), PLUGIN, Path(old.__file__), Path(mem.__file__), ROOT / 'CROSS_MODAL_RESEARCH_PROTOCOL.md',
               ROOT / 'scripts/run_v7_full_experiment.py', ROOT / 'scripts/run_v7_evaluation.py',
               DATA / 'audit/rule_knowledge.jsonl', DATA / 'audit/eval_labels.jsonl', DATA / 'image_hashes.jsonl',
               DATA / 'T2T/eval.jsonl', DATA / 'T2V/eval.jsonl']
    inventory = dense_identity(MODELS['qwen3'])
    for image in rows(DATA / 'image_hashes.jsonl'):
        assert sha(DATA / image['path']) == image['source_sha256'] == image['repaired_sha256']
    manifest = dict(created_at=ev.now(), research='Learning-method pilot, source-only TEXT; formal T2T/T2V use the same checkpoint, no test material or external solver.',
                    sources={str(p): sha(p) for p in sources}, files={p.name: sha(p) for p in OUT.glob('*.jsonl')},
                    base=inventory, ids=sorted(ids), arms=ARMS, selection='Four rules per supported topology, SHA256 ordering, excludes previous eight; no selection by scores. Not a claim that these rules have never appeared in historical runs.',
                    training=dict(warmup='256 original-definition rows, 2 epochs, LoRA32/64, LR1e-4, fixed128steps',
                                  continuations='Same warmup checkpoint and 128 new TEXT scenes. SFT and DPO: 2epochs/64steps; GRPO: 1epoch/128steps, four samples per prompt.',
                                  sft_lr=1e-5, dpo_lr=1e-5, grpo_lr=1e-5, dpo_beta=.1, grpo_beta=.04,
                                  reward='0.5 exact rule(mode+both conditions) + 0.5 fraction of four correct decisions; malformed output zero. No reward model or MCQ letters.',
                                  preference='Synthetic oracle chosen; same-style rejected alternates wrong AND/OR and one wrong decision; not on-policy preference data.',
                                  budget_caveat='Exploratory algorithm comparison: unique practice scenes shared; token, rollout, gradient and compute budgets are NOT matched. No pure algorithm-causality claim.'),
                    generation=dict(direct=32, reasoning=512, recall=256, temperature=0, batch=4, thinking=False),
                    counts=dict(Counter(p['condition'] + ':' + p['protocol'] + ':' + p['input_mod'] for p in probes)),
                    success='Primary: source-modal closed-book original and independent scene gains vs base and common warmup, not training reward. All rules/checkpoints/failures retained; no gate based on T2T>T2V.',
                    images_verified=3000, versions={n: importlib.metadata.version(n) for n in ('ms-swift', 'trl', 'transformers', 'torch')})
    save(OUT / 'manifest.json', manifest)
    print('PREPARED', manifest['ids'], manifest['counts'], flush=True)


def reasoning(p):
    z = copy.deepcopy(p); z['id'] = 'reasoning:' + z['id']; z['protocol'] = 'reasoning'
    assert z['messages'][-1]['content'].endswith(ev.SUFFIX)
    z['messages'][-1]['content'] = z['messages'][-1]['content'][:-len(ev.SUFFIX)] + REASON_SUFFIX
    z['messages'][0]['content'] = REASON_SYSTEM
    return z


def validate():
    m = read(OUT / 'manifest.json'); assert m
    for p, h in m['sources'].items():
        assert sha(Path(p)) == h, p
    for p, h in m['files'].items():
        assert sha(OUT / p) == h, p
    return m


def cp(arm, smoke=False):
    return OUT / ('smoke_' + arm if smoke else arm) / 'checkpoint' / ('checkpoint-1' if smoke else 'checkpoint-' + ('128' if arm in ('warmup', 'grpo') else '64'))


def train(arm, gpu, smoke=False):
    validate(); folder = OUT / ('smoke_' + arm if smoke else arm); folder.mkdir(exist_ok=True)
    final = cp(arm, smoke)
    if (final / 'adapter_model.safetensors').exists():
        return
    dataset = OUT / (arm + '.jsonl')
    if smoke:
        dataset = folder / 'input.jsonl'; write(dataset, rows(OUT / (arm + '.jsonl'))[:4])
    cmd = train_command('qwen3', 'sft', dataset, final.parent, epochs=1 if arm == 'grpo' else 2)
    for flag, value in [('--per_device_train_batch_size', 4), ('--gradient_accumulation_steps', 1), ('--learning_rate', 1e-4 if arm == 'warmup' else 1e-5),
                        ('--lora_rank', 32), ('--lora_alpha', 64), ('--save_strategy', 'steps'), ('--save_total_limit', 1),
                        ('--logging_steps', 1), ('--max_length', 2048)]:
        cmd[cmd.index(flag)+1] = str(value)
    cmd += ['--save_steps', '1' if smoke else ('32' if arm != 'grpo' else '32'), '--attn_impl', 'sdpa',
            '--packing', 'false', '--padding_free', 'false', '--use_logits_to_keep', 'false', '--report_to', 'none',
            '--dataloader_num_workers', '0', '--dataloader_persistent_workers', 'false']
    if arm != 'warmup':
        assert (cp('warmup') / 'adapter_model.safetensors').is_file()
        cmd += ['--adapters', str(cp('warmup'))]
    if arm in ('dpo', 'grpo'):
        cmd[1] = 'rlhf'; cmd[cmd.index('--loss_scale')+1] = 'last_round'
        cmd += ['--rlhf_type', arm, '--ref_adapters', str(cp('warmup')), '--beta', '.1' if arm == 'dpo' else '.04']
    if arm == 'grpo':
        cmd += ['--external_plugins', str(PLUGIN), '--reward_funcs', 'v7_rule_application', '--num_generations', '4',
                '--use_vllm', 'false', '--max_completion_length', '256', '--temperature', '.9',
                '--log_completions', 'true', '--steps_per_generation', '1']
    if smoke:
        cmd += ['--max_steps', '1']
    else:
        recover = sorted([p for p in final.parent.glob('checkpoint-*') if (p / 'optimizer.pt').exists()], key=lambda p: int(p.name.split('-')[-1]))
        if recover:
            cmd += ['--resume_from_checkpoint', str(recover[-1]), '--load_args', 'false']
    ledger = ROOT / 'runs/v7_full_experiment'; ledger.mkdir(exist_ok=True)
    env = environment(gpu, DATA); env.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    with reserve(ledger, gpu, str(folder) + ':train', 26000 if arm == 'grpo' else 22000) as reservation:
        save(folder / 'state.json', dict(status='running', command=cmd, started_at=ev.now()))
        with (folder / 'train.log').open('a') as log:
            p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            reservation.bind(p.pid); code = p.wait()
        if code:
            save(folder / 'state.json', dict(status='failed', returncode=code)); raise RuntimeError(f'{arm} failed: {folder}/train.log')
    assert (final / 'adapter_model.safetensors').exists(), str(final)
    state = read(final / 'trainer_state.json')
    losses = [r for r in rows(final.parent / 'logging.jsonl') if 'loss' in r]
    assert losses and all(math.isfinite(r['loss']) for r in losses)
    save(folder / 'checkpoint_manifest.json', dict(path=str(final), sha256=sha(final / 'adapter_model.safetensors'),
         step=state['global_step'], epoch=state['epoch'], first_loss=losses[0]['loss'], last_loss=losses[-1]['loss']))
    save(folder / 'state.json', dict(status='complete', ended_at=ev.now()))


def evaluate(arm, gpu):
    validate(); mem.OUT = OUT; folder = OUT / arm / 'evaluation'
    probes = rows(OUT / 'probes.jsonl'); runner = mem.Runner(folder, gpu, arm)
    with reserve(ROOT / 'runs/v7_full_experiment', gpu, str(folder), 22000) as reservation:
        reservation.bind(os.getpid()); runner.load(MODELS['qwen3'] if arm == 'base' else cp(arm), adapter=arm != 'base')
        for protocol, stage in [('short', 'direct'), ('reasoning', 'reasoning'), ('recall', 'recall')]:
            runner.infer([p for p in probes if p['protocol'] == protocol], stage)
    assert len(runner.rr) == len(probes)
    save(folder / 'state.json', dict(status='complete', n=len(probes), ended_at=ev.now()))


def run_child(stage, arm, gpu, smoke=False):
    cmd = [str(PY), '-u', str(Path(__file__)), '--stage', stage, '--arm', arm, '--gpu', str(gpu)]
    if smoke:
        cmd += ['--smoke']
    subprocess.run(cmd, cwd=ROOT, check=True)


def workflow():
    validate()
    # Model allocations are released between every subprocess.
    run_child('train', 'warmup', '0', True)
    run_child('train', 'warmup', '0')
    def queue_one():
        run_child('evaluate', 'base', '0'); run_child('evaluate', 'warmup', '0')
        run_child('train', 'sft', '0', True); run_child('train', 'sft', '0'); run_child('evaluate', 'sft', '0')
        run_child('train', 'dpo', '0', True); run_child('train', 'dpo', '0'); run_child('evaluate', 'dpo', '0')
    def queue_two():
        run_child('train', 'grpo', '1', True); run_child('train', 'grpo', '1'); run_child('evaluate', 'grpo', '1')
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(queue_one), pool.submit(queue_two)]
        for f in futures:
            f.result()
    run_child('score', 'base', '0')


def score():
    validate(); scored = []; summary = {}
    probes = {p['id']: p for p in rows(OUT / 'probes.jsonl')}
    for arm in ARMS:
        rr = rows(OUT / arm / 'evaluation/predictions.jsonl'); assert len(rr) == len(probes)
        for r in rr:
            p = probes[r['id']]; text = r['response'].strip()
            if p['fmt'] == 'fact':
                correct = re.sub(r'\s+', ' ', text).lower() == re.sub(r'\s+', ' ', p['gold']).lower(); pred = None
            elif p['protocol'] == 'short':
                pred = ev.letter(text); correct = pred == p['gold']
            else:
                matches = re.findall(r'(?im)^\s*Answer:\s*([A-D])\s*[.!]?\s*$', text)
                pred = matches[0] if len(matches) == 1 else None; correct = pred == p['gold']
            z = {**r, 'arm': arm, 'scored_correct': correct, 'scored_answer': pred,
                 'format_error': p['fmt'] == 'mcq' and pred is None}; scored.append(z)
        summary[arm] = {}
        for key in sorted({p['condition'] + ':' + p['protocol'] + ':' + p['input_mod'] for p in probes.values()}):
            subset = [r for r in scored if r['arm'] == arm and r['condition'] + ':' + r['protocol'] + ':' + r['input_mod'] == key]
            good = sum(r['scored_correct'] for r in subset); valid = sum(not r['format_error'] for r in subset)
            summary[arm][key] = dict(n=len(subset), correct=good, accuracy=good / len(subset), format_errors=len(subset)-valid,
                                    valid_accuracy=good / valid if valid else None,
                                    by_task={t: dict(n=sum(r.get('task') == t for r in subset), correct=sum(r['scored_correct'] for r in subset if r.get('task') == t)) for t in ('direct', 'diagnosis')})
    write(OUT / 'scored_predictions.jsonl', scored); save(OUT / 'summary.json', summary)
    lines = ['# SFT / DPO / GRPO 来源模态学习试验', '', '同一个干净Qwen3-VL-4B先学习8条固定规则，再从相同warmup检查点分支；新训练场景、正式测试题、原图相互隔离。训练为TEXT，所有组的T2T/T2V由同一检查点闭卷回答。', '',
             '| 测试 | 基座 | 规则SFT起点 | 继续SFT | DPO | GRPO |', '|---|---:|---:|---:|---:|---:|']
    for key in next(iter(summary.values())):
        lines.append('| ' + key + ' | ' + ' | '.join(f'{summary[a][key]["correct"]}/{summary[a][key]["n"]} ({summary[a][key]["accuracy"]:.1%})' for a in ARMS) + ' |')
    lines += ['', '限制：仅8条规则、单seed、两种可程序验证的规则结构。DPO负例为程序生成，不是模型实际错误；GRPO采样/优化预算较大，不是同计算预算对照。回忆逐字匹配只是保守辅助诊断，不代表语义应用能力。以闭卷留出和原题结果判断学习效果，不能用偏好准确率或训练奖励代替。', '',
              '全部分项及格式错误见summary.json，原始回答保留于各组evaluation/predictions.jsonl。配置、数据和源文件哈希见manifest.json。']
    (OUT / 'report_zh.md').write_text('\n'.join(lines) + '\n')
    for image in rows(DATA / 'image_hashes.jsonl'):
        assert sha(DATA / image['path']) == image['source_sha256'] == image['repaired_sha256']
    save(OUT / 'completion_audit.json', dict(status='pass', predictions=len(scored), images_verified=3000, completed_at=ev.now()))
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--stage', choices=['prepare', 'train', 'evaluate', 'workflow', 'score'], required=True)
    p.add_argument('--arm', choices=ARMS, default='warmup'); p.add_argument('--gpu', default='0'); p.add_argument('--smoke', action='store_true'); a = p.parse_args()
    if a.stage == 'prepare': prepare()
    elif a.stage == 'train': train(a.arm, a.gpu, a.smoke)
    elif a.stage == 'evaluate': evaluate(a.arm, a.gpu)
    elif a.stage == 'workflow': workflow()
    else: score()
