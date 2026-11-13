#!/usr/bin/env python3
"""Learning-side repair: fixed TEXT data, suffix/content loss ablation, original tests."""
import argparse
import copy
import gc
import json
import math
import os
import random
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import pilot_v7_learning_first as old
import explore_v7_memory_router as mem
import run_v7_evaluation as ev
from build_v7_experiment_data import name
from run_v7_full_experiment import ROOT, MODELS, environment, rows, read, save, sha, train_command
from v7_gpu_resources import reserve

OUT = ROOT / 'runs/v7_learning_repaired_20261019'
DATA = old.DATA
PLUGIN = ROOT / 'scripts/v7_content_loss.py'
ARMS = ('suffix_control', 'answer_content')
EPOCHS = 3
STEPS = 480
FACT_SYSTEM = 'Answer the factual question about the learned named rule accurately. Return only the requested fact.'
LOCAL_SYSTEM = 'Apply the learned named rule to this candidate. Return exactly KEEP or OMIT, without explanation.'


def norm(s):
    return ' '.join(s.strip().split()).casefold()


def row(q, k, answer, channel, ident, system=FACT_SYSTEM):
    return {'id': ident, 'qid': k['id'], 'channel': channel,
            'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': q},
                         {'role': 'assistant', 'content': answer}]}


def atomic(k, heldout=False):
    r = json.loads(k['fact']['quote']); n = name(k)
    if r['selection_topology'] == 'disjunctive_filter':
        values = {c['field']: str(c['value']) for c in r['conditions']}
        stems = [('shape', 'shape', 'alternative shape condition'), ('color', 'color', 'alternative color condition')]
    else:
        values = {'lower': str(r['conditions'][0]['value']), 'upper': str(r['conditions'][1]['value'])}
        stems = [('lower', 'included lower bound', 'inclusive minimum'), ('upper', 'excluded upper bound', 'exclusive maximum')]
    result = []
    for field, label, alias in stems:
        forms = ([f'For {n}, give the {alias} value only.', f'What value defines the {label} in the stored definition of {n}?']
                 if heldout else [f'What is the {label} for {n}?', f'State the {label} used by {n}.',
                    f'Which {label} does {n} specify?', f'Recall only the {label} in {n}.',
                    f'Under {n}, what is the required {label}?', f'Give {n}\'s {label}, without the rest of the rule.'])
        for j, prompt in enumerate(forms):
            result.append(row(prompt, k, values[field], 'atomic_fact', f'atomic:{heldout}:{k["id"]}:{field}:{j}'))
    return result


def locals_for(k):
    r = json.loads(k['fact']['quote']); rng = random.Random('local-repaired:42:' + k['id'])
    if r['selection_topology'] == 'disjunctive_filter':
        shapes = ['triangle', 'square', 'pentagon', 'diamond', 'circle', 'hexagon', 'star', 'oval']
        colors = ['blue', 'red', 'green', 'black', 'gray', 'white', 'yellow', 'orange', 'purple', 'brown']
        candidates = [{'shape': s, 'color': c} for s in shapes for c in colors]
    else:
        candidates = [{'value': i / 8} for i in range(-32, 249)]
    buckets = defaultdict(list)
    for obs in candidates:
        fs = tuple(old.flags(r, obs)); buckets[any(fs) if r['mode'] == 'any' else all(fs)].append(obs)
    for pool in buckets.values(): rng.shuffle(pool)
    chosen = []
    for keep in (False, True):
        pool = buckets[keep]
        if r['selection_topology'] == 'disjunctive_filter' and keep:
            both = next(o for o in pool if all(old.flags(r, o)))
            pool.remove(both); pool.insert(0, both)
        if r['selection_topology'] == 'interval_filter':
            boundaries = [o for o in pool if o['value'] in {r['conditions'][0]['value'], r['conditions'][1]['value']}]
            for o in reversed(boundaries): pool.remove(o); pool.insert(0, o)
        chosen += [(o, keep, 'train' if i < 8 else 'holdout') for i, o in enumerate(pool[:10])]
    train, test = [], []
    for j, (obs, keep, split) in enumerate(chosen):
        description = ', '.join(f'{field}: {value}' for field, value in obs.items())
        z = row(f'Apply {name(k)} to one candidate with {description}. Should this candidate be kept or omitted?',
                k, 'KEEP' if keep else 'OMIT', 'local_result', f'local:{k["id"]}:{j}', LOCAL_SYSTEM)
        z['observations'] = [obs]
        assert old.independent_assignment(r, [obs]).split('=')[1] == z['messages'][-1]['content']
        (train if split == 'train' else test).append(z)
    return train, test


def as_probe(z, condition):
    p = copy.deepcopy(z); p['gold'] = p['messages'].pop()['content']
    p.update(condition=condition, input_mod='T', fmt='fact')
    return p


def validate(images=False):
    m = read(OUT / 'manifest.json')
    assert m['script_sha256'] == sha(Path(__file__))
    for p, h in m['sources'].items(): assert sha(Path(p)) == h, p
    for p, h in m['files'].items(): assert sha(OUT / p) == h, p
    if images:
        for r in rows(DATA / 'image_hashes.jsonl'):
            assert sha(DATA / r['path']) == r['source_sha256'] == r['repaired_sha256']
    return m


def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT / 'manifest.json').exists(): validate(); return
    old.validate(images=True)
    targets = rows(old.OUT / 'target_knowledge.jsonl'); replay = rows(old.OUT / 'replay_knowledge.jsonl')
    lookup = {k['id']: k for k in targets + replay}
    occupied = {r['signature'] for r in read(old.OUT / 'label_and_split_audit.json')['original_text_labels_recalculated']}
    blocks = old.construct(targets, 'repaired-target', 32, 12, occupied)
    replay_blocks = old.construct(replay, 'repaired-replay', 16, 4, occupied)
    training, probes, label_checks = [], [], []
    for k in targets:
        for j, form in enumerate(mem.FORMS):
            training.append(row(form.format(name=name(k)), k, mem.compact(k), 'full_fact', f'full:{k["id"]}:{j}', mem.MEM_SYSTEM))
        training += atomic(k)
        probes += [as_probe(z, 'atomic_holdout') for z in atomic(k, True)]
        local_train, local_test = locals_for(k)
        training += local_train
        probes += [as_probe(z, 'local_seen') for z in local_train[:2] + local_train[8:10]]
        probes += [as_probe(z, 'local_holdout') for z in local_test]
        probes.append({'id': 'recall:' + k['id'], 'qid': k['id'], 'condition': 'recall_control', 'input_mod': 'T',
                       'fmt': 'fact', 'gold': mem.compact(k), 'messages': mem.messages(mem.RECALL_Q.format(name=name(k)))})
    for b in blocks:
        i = int(b['id'].split(':')[-1])
        for q in b['rule_ids']:
            tasks = ('direct', 'diagnosis') if b['split'] == 'holdout' else (('direct',) if i % 2 == 0 else ('diagnosis',))
            for task in tasks:
                bb = copy.deepcopy(b)
                if b['split'] == 'train':
                    wanted = (i // 2) % 4
                    other = [s for s in bb['options'] if s != bb['results'][q]]
                    random.Random(f'option:42:{q}:{i}').shuffle(other)
                    other.insert(wanted, bb['results'][q]); bb['options'] = other
                    # One quarter of diagnosis examples accept the submitted report.
                    if task == 'diagnosis':
                        bb['submitted_report'] = bb['results'][q] if (i // 2) in (0, 5, 10, 15) else next(s for s in other if s != bb['results'][q])
                p = old.practice(bb, lookup[q], task)
                p['id'] = 'repaired:' + p['id']
                if b['split'] == 'train':
                    training.append(old.as_training(p, 'target_mcq'))
                    p['condition'] = 'target_seen'
                else: p['condition'] = 'target_holdout'
                probes.append(p)
                rule = json.loads(lookup[q]['fact']['quote'])
                expected = old.independent_assignment(rule, bb['observations'])
                assert expected == bb['results'][q]
                opts = dict(re.findall(r'(?m)^([A-D])\. (.*)$', p['messages'][-1]['content']))
                action = 'Accept report.' if bb['submitted_report'] == expected else 'Replace report.'
                matches = [c for c, s in opts.items() if s.split('Result: ')[1].rstrip('.') == expected
                           and (task == 'direct' or s.startswith(action))]
                assert matches == [p['gold']]
                label_checks.append({'id': p['id'], 'gold': p['gold'], 'expected': expected})
    for b in replay_blocks:
        for q in b['rule_ids']:
            task = 'direct' if int(b['id'].split(':')[-1]) % 2 == 0 else 'diagnosis'
            p = old.practice(b, lookup[q], task, reference=True)
            p['id'] = 'repaired:' + p['id']
            if b['split'] == 'train': training.append(old.as_training(p, 'independent_application_replay'))
            else:
                p['condition'] = 'retention_control'; probes.append(p)
    # Reuse exact original inputs and separate oracle controls; no answer rewriting.
    previous = rows(old.OUT / 'probes.jsonl')
    probes += [copy.deepcopy(p) for p in previous if p['condition'] in ('original', 'oracle_control')]
    # 4 replay holdout blocks/family x 4 rules x 1 task =32; add the other question type for 64 controls.
    for b in replay_blocks:
        if b['split'] != 'holdout': continue
        for q in b['rule_ids']:
            task = 'diagnosis' if int(b['id'].split(':')[-1]) % 2 == 0 else 'direct'
            p = old.practice(b, lookup[q], task, reference=True); p['id'] = 'repaired:' + p['id']
            p['condition'] = 'retention_control'; probes.append(p)
    random.Random(42).shuffle(training)
    assert len(training) == 640 and len(probes) == 680
    assert len({r['id'] for r in training}) == 640 and len({p['id'] for p in probes}) == 680
    inputs = [ev.digest(r['messages'][:-1]) for r in training]
    assert len(set(inputs)) == len(inputs), 'Unnecessary duplicate training inputs'
    assert all(not r.get('images') and all('<image>' not in m['content'] for m in r['messages']) for r in training)
    positions = defaultdict(Counter)
    for r in training:
        if r['channel'] == 'target_mcq': positions[(r['qid'], r['task'])][r['gold']] += 1
    assert len(positions) == 16 and all(set(c.values()) == {4} and set(c) == set('ABCD') for c in positions.values())
    train_sigs = {b['observation_signature'] for b in blocks + replay_blocks if b['split'] == 'train'}
    test_sigs = {b['observation_signature'] for b in blocks + replay_blocks if b['split'] == 'holdout'}
    assert not train_sigs & test_sigs
    train_apps = {ev.digest(r['messages'][:-1]) for r in training}
    assert all(ev.digest(p['messages']) not in train_apps for p in probes if p['condition'] in ('target_holdout', 'atomic_holdout', 'local_holdout', 'original'))
    ev.save_rows(OUT / 'training.jsonl', training); ev.save_rows(OUT / 'probes.jsonl', probes)
    ev.save_rows(OUT / 'knowledge.jsonl', targets); ev.save_rows(OUT / 'blocks.jsonl', blocks + replay_blocks)
    ev.save_rows(OUT / 'label_checks.jsonl', label_checks)
    # Every training row is actually encoded before training; suffix layout is checked explicitly.
    os.environ.update(environment('0', DATA))
    from swift.model import get_model_processor
    from swift.template import get_template
    _, processor = get_model_processor(str(MODELS['qwen3']), model_type='qwen3_vl', load_model=False)
    template = get_template(processor, template_type='qwen3_vl', max_length=8192, enable_thinking=False)
    template.set_mode('train'); tok = processor.tokenizer
    suffix = tok.encode('<|im_end|>\n', add_special_tokens=False)
    encoded = []
    for r in training:
        z = template.encode({'messages': copy.deepcopy(r['messages'])})
        labels = [i for i in z['labels'] if i != -100]
        assert labels == tok.encode(r['messages'][-1]['content'], add_special_tokens=False) + suffix
        assert len(z['input_ids']) < 8192
        encoded.append({'id': r['id'], 'channel': r['channel'], 'input_tokens': len(z['input_ids']),
                        'content_tokens': len(labels) - len(suffix), 'suffix_tokens': len(suffix)})
    ev.save_rows(OUT / 'encoding_audit.jsonl', encoded)
    sources = [old.OUT / 'manifest.json', old.OUT / 'target_knowledge.jsonl', old.OUT / 'replay_knowledge.jsonl',
               old.OUT / 'probes.jsonl', DATA / 'image_hashes.jsonl', DATA / 'audit/rule_knowledge.jsonl',
               DATA / 'audit/eval_labels.jsonl', PLUGIN, Path(old.__file__), Path(mem.__file__),
               ROOT / 'scripts/run_v7_full_experiment.py', ROOT / 'scripts/run_v7_evaluation.py', ROOT / 'scripts/v7_gpu_resources.py']
    manifest = {'created_at': ev.now(), 'script_sha256': sha(Path(__file__)), 'base': read(old.OUT / 'manifest.json')['base'],
                'sources': {str(p): sha(p) for p in sources}, 'files': {p.name: sha(p) for p in OUT.glob('*.jsonl')},
                'arms': ARMS, 'suffix_ids': suffix,
                'training': {'rules': 8, 'modality': 'TEXT', 'rows': 640, 'unique_inputs': 640, 'epochs': EPOCHS,
                    'steps': STEPS, 'batch': 2, 'accumulation': 2, 'lr': 3e-5, 'lora_rank': 32, 'lora_alpha': 64,
                    'seed': 42, 'freeze_vision_aligner': True, 'channels': dict(Counter(r['channel'] for r in training)),
                    'ablation': 'Same data, steps, initialization and per-row weighting; only inclusion of template suffix in loss differs.'},
                'generation': {'direct': 32, 'blind_recall': 256, 'short_fact': 32, 'temperature': 0, 'thinking': False, 'batch': 4},
                'evaluation_counts': dict(Counter(p['condition'] for p in probes)),
                'decision': {'source_screen': 'Original T2T >=60%, gain >=15pp, unseen text gain >=15pp; main format errors <=2%. Descriptive developer screen only.',
                    'training_fit': 'Report all 256 target training MCQ; >=90% is diagnostic fit target, not checkpoint selection.',
                    'cross_modal': 'Same final checkpoint; original PNG/options/questions unchanged; no materials/recall/routing in main test; never select by T2T>T2V.'},
                'limits': ['Historical eight development rules, two subfamilies, one seed; not independent confirmation.',
                    'Prior pilot used different data/steps; only the two current loss arms are a matched ablation.',
                    'Training includes source-modality practice beyond original explanations; not original-material-only SFT.',
                    'All rules retained; no full1000 or VISION-training conclusion.'],
                'images_verified': 3000}
    save(OUT / 'manifest.json', manifest)
    save(OUT / 'data_audit.json', {'status': 'pass', 'unique_training_inputs': 640, 'all_rows_encoded': True,
        'independent_application_labels': True, 'per_rule_per_task_letters': {str(k): dict(c) for k,c in positions.items()},
        'train_holdout_observations_disjoint': True, 'training_formal_tests_disjoint': True,
        'facts_source': 'Original text material and matching formal rules; existing pilot prefix audit retained.',
        'local_conditions_covered': {k['id']: sorted({str(old.flags(json.loads(k['fact']['quote']), r['observations'][0]))
             for r in training if r['qid'] == k['id'] and r['channel'] == 'local_result'}) for k in targets}})
    snap = OUT / 'source_snapshots'; snap.mkdir(exist_ok=True)
    for p in [Path(__file__), PLUGIN, Path(old.__file__), Path(mem.__file__)]: shutil.copy2(p, snap / p.name)
    examples = ['# 修复版实际训练数据', '', '两组使用同一份640行训练文件；只改变模板结束符是否参与loss。', '',
        '每规则：4条完整事实＋12条参数问答＋16条独立局部判断＋32道选择题。另有128条独立规则带材料解题回放。',
        '局部判断答案直接给KEEP/OMIT，不在答案前先给正确规则。训练仅文字；原图片只用于评测。', '']
    for channel in ('full_fact', 'atomic_fact', 'local_result', 'target_mcq', 'independent_application_replay'):
        examples += ['## ' + channel, '', '```json', json.dumps(next(r for r in training if r['channel'] == channel), ensure_ascii=False, indent=2), '```', '']
    (OUT / 'TRAINING_DATA_ZH.md').write_text('\n'.join(examples))
    validate(); print('PREPARED', manifest['training'], manifest['evaluation_counts'], flush=True)


def cp(arm, smoke=False):
    return OUT / ('smoke_' + arm if smoke else arm) / 'checkpoint' / f'checkpoint-{1 if smoke else STEPS}'


def train(arm, gpu, smoke=False):
    m = validate(); folder = cp(arm, smoke).parent.parent; folder.mkdir(exist_ok=True)
    with ev.run_lock(folder / 'train.lock'):
        checkpoint = cp(arm, smoke)
        if not (checkpoint / 'adapter_model.safetensors').exists():
            dataset = OUT / 'training.jsonl'
            if smoke:
                rr = rows(dataset)
                dataset = folder / 'input.jsonl'
                ev.save_rows(dataset, [next(r for r in rr if r['channel'] == c) for c in sorted({r['channel'] for r in rr})])
            recover = [p for p in checkpoint.parent.glob('checkpoint-*') if all((p/n).exists() for n in ('optimizer.pt','scheduler.pt','rng_state.pth','adapter_model.safetensors','trainer_state.json'))]
            resume = max(recover, key=lambda p: read(p/'trainer_state.json')['global_step']) if recover and not smoke else None
            cmd = train_command('qwen3', 'sft', dataset, checkpoint.parent, EPOCHS, resume=resume)
            for flag, value in [('--per_device_train_batch_size','2'), ('--gradient_accumulation_steps','2' if not smoke else '1'),
                                ('--learning_rate','3e-5'), ('--lora_rank','32'), ('--lora_alpha','64'),
                                ('--save_strategy','steps'), ('--logging_steps','1')]: cmd[cmd.index(flag)+1] = value
            cmd += ['--save_steps', '1' if smoke else '160', '--attn_impl','sdpa', '--external_plugins',str(PLUGIN),
                    '--loss_type','v7_' + arm, '--packing','false', '--padding_free','false','--use_logits_to_keep','false']
            if smoke:
                cmd += ['--max_steps','1']; cmd[cmd.index('--warmup_ratio')+1] = '0'
            env = environment(gpu, DATA)
            env.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', V7_CONTENT_SUFFIX_IDS=json.dumps(m['suffix_ids']),
                       V7_CONTENT_LOSS_AUDIT=str(folder / 'loss_audit.jsonl'))
            save(folder/'state.json', {'status':'waiting_for_gpu','command':cmd,'gpu':gpu})
            with reserve(old.LEDGER, gpu, str(folder)+':train', 21000) as reservation:
                with (folder/'train.log').open('a') as log:
                    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                    reservation.bind(proc.pid)
                    save(folder/'state.json', {'status':'running','command':cmd,'pid':proc.pid,'gpu':gpu})
                    code = proc.wait()
                if code: raise RuntimeError((arm, code, str(folder/'train.log')))
        state = read(checkpoint/'trainer_state.json'); args = read(checkpoint/'args.json')
        assert state['global_step'] == (1 if smoke else STEPS)
        assert args['model'] == str(MODELS['qwen3']) and args['adapters'] in (None, [])
        assert args['freeze_vit'] and args['freeze_aligner'] and args['lora_rank'] == 32
        assert state['epoch'] >= EPOCHS - 1e-6 or smoke
        from safetensors import safe_open
        with safe_open(str(checkpoint/'adapter_model.safetensors'), framework='pt', device='cpu') as f:
            keys = list(f.keys()); bs = [k for k in keys if 'lora_B' in k]
            assert bs and all(bool(f.get_tensor(k).count_nonzero()) for k in bs)
            assert not any('.visual.' in k or '.merger.' in k for k in keys)
        losses = [r['loss'] for r in rows(checkpoint.parent/'logging.jsonl') if 'loss' in r]
        assert losses and all(math.isfinite(v) for v in losses)
        save(folder/'checkpoint_manifest.json', {**ev.checkpoint(checkpoint), 'global_step':state['global_step'],
             'epoch':state['epoch'], 'nonzero_lora_B':len(bs), 'final_loss':losses[-1], 'selection':'fixed final; no test selection'})
        save(folder/'state.json', {'status':'complete','checkpoint':str(checkpoint)})


def evaluate(arm, gpu, smoke=False):
    validate(); mem.OUT = OUT
    folder = OUT / ('smoke_'+arm if smoke else arm) / ('reload' if smoke else 'evaluation')
    folder.mkdir(parents=True, exist_ok=True)
    probes = rows(OUT/'probes.jsonl')
    if smoke:
        probes = [next(p for p in probes if p['condition']=='original' and p['input_mod']==mod) for mod in ('T','V')]
    with ev.run_lock(folder/'eval.lock'):
        run = mem.Runner(folder, gpu, arm)
        if run.seen != {p['id'] for p in probes}:
            with reserve(old.LEDGER, gpu, str(folder)+':evaluation', 22000):
                run.load(cp=MODELS['qwen3'] if arm=='base' else cp(arm,smoke), adapter=arm!='base')
                run.infer([p for p in probes if p['fmt']=='mcq'], 'direct')
                run.infer([p for p in probes if p['fmt']=='fact' and p['condition']!='recall_control'], 'short_fact')
                run.infer([p for p in probes if p['condition']=='recall_control'], 'blind_recall')
                del run.engine
                import torch
                gc.collect(); torch.cuda.empty_cache()
        assert run.seen == {p['id'] for p in probes}
        if smoke:
            assert all(r['response'].strip() for r in run.rr)
            save(folder.parent/'validated.json', {'status':'pass','checkpoint_reload':True,'text_image_generation':True})
        save(folder/'state.json', {'status':'complete','saved':len(run.rr)})


def summarize():
    validate(images=True)
    result, changes, csvrows = {}, {}, []
    pp = {p['id']: p for p in rows(OUT/'probes.jsonl')}
    indexed = {}
    for arm in ('base',)+ARMS:
        rr = rows(OUT/arm/'evaluation/predictions.jsonl')
        assert len(rr)==len(pp) and {r['id'] for r in rr}==set(pp)
        groups = {}; indexed[arm] = {r['id']:r for r in rr}
        for r in rr:
            p = pp[r['id']]
            assert r['input_sha256']==ev.digest({'messages':p['messages'],'images':p.get('images',[])})
            if p['fmt']=='mcq': assert r['predicted']==ev.letter(r['response']) and r['correct']==(r['predicted']==p['gold'])
        for condition in sorted({p['condition'] for p in pp.values()}):
            for mod in ('T','V'):
                ss = [r for r in rr if r['condition']==condition and r['input_mod']==mod]
                if not ss: continue
                if ss[0]['fmt']=='mcq': g = old.metric(ss)
                else:
                    g = {'n':len(ss),'correct':sum(norm(r['response'])==norm(r['gold']) for r in ss),
                         'empty':sum(not r['response'].strip() for r in ss),
                         'truncated':sum(r['finish_reason']=='length' for r in ss),
                         'scoring':'literal whitespace/case-normalized; paraphrases not semantically graded'}
                    g['accuracy'] = g['correct']/g['n']
                groups[condition+'_'+mod] = g
                csvrows.append({'arm':arm,'condition':condition,'input_mod':mod,'n':g['n'],'correct':g['correct'],'accuracy':g['accuracy']})
        result[arm] = groups
    from report_v7_learning_first import paired
    for arm in ARMS:
        changes[arm] = {condition+'_'+mod:paired(list(indexed['base'].values()),list(indexed[arm].values()),condition,mod)
            for condition,mod in [('original','T'),('original','V'),('target_holdout','T'),('target_seen','T')]}
    save(OUT/'summary.json', result); save(OUT/'paired_changes.json', changes)
    import csv
    with (OUT/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
    decisions={}
    for arm in ARMS:
        g=result[arm]
        checks={'source_absolute_ge60':g['original_T']['accuracy']>=.6,
                'source_gain_ge15pp':changes[arm]['original_T']['gain']>=.15,
                'unseen_text_gain_ge15pp':changes[arm]['target_holdout_T']['gain']>=.15,
                'main_format_errors_le2percent':sum(g[k]['format_errors'] for k in ('original_T','original_V'))/32<=.02}
        decisions[arm]={'checks':checks,'source_screen_pass':all(checks.values()),
                        'training_fit_ge90':g['target_seen_T']['accuracy']>=.9,
                        'gain_gap':changes[arm]['original_T']['gain']-changes[arm]['original_V']['gain']}
    save(OUT/'decisions.json',decisions)
    comparisons=[]
    for p in pp.values():
        if p['condition']=='original':
            comparisons.append({**p,'answers':{a:{k:indexed[a][p['id']][k] for k in ('response','correct','predicted','finish_reason')} for a in indexed}})
    ev.save_rows(OUT/'original_comparisons.jsonl', comparisons)
    losses={}
    for arm in ARMS:
        audits=rows(OUT/arm/'loss_audit.jsonl'); bycount=defaultdict(list)
        for r in audits:
            if r['step_before_update']<320:continue
            for i,n in enumerate(r['supervised_tokens']):
                bycount[str(n)].append({k:r[k][i] for k in ('all_token_ce','answer_content_ce','suffix_ce','first_answer_token_ce')})
        losses[arm]={n:{'rows':len(v),**{k:sum(z[k] for z in v)/len(v) for k in v[0]}} for n,v in bycount.items()}
    save(OUT/'loss_breakdown.json',{'last_epoch_online_grouped_by_supervised_length':losses,
         'limit':'Single-token content pools MCQ, local and atomic rows. This is not channel-isolated final-checkpoint loss.'})
    lines=['# 学习端修复对照结果','','同一640行TEXT训练数据，8条历史开发规则，3 epoch、480步；两组只改变结束符是否参与loss。',
           '干净Qwen3-VL-4B基座、LoRA32、LR3e-5；原题、原选项、原PNG、直接字母协议保持不变。',
           '新增练习只用文字来源；不是原始解释SFT，不能与历史不同预算试验作因果比较。','',
           '| 组 | 原T2T | 原T2V | 全部已见文字选择题 | 未见文字选择题 |','|---|---:|---:|---:|---:|']
    def cell(g,k):v=g[k];return f'{v["correct"]}/{v["n"]} ({v["accuracy"]:.1%})'
    for a,g in result.items():lines.append('| '+' | '.join([a]+[cell(g,k) for k in ('original_T','original_V','target_seen_T','target_holdout_T')])+' |')
    lines+=['','| 组 | 独立参数问答 | 已见局部判断 | 未见局部判断 | 完整规则逐字回忆 | 独立规则带材料控制 |','|---|---:|---:|---:|---:|---:|']
    for a,g in result.items():lines.append('| '+' | '.join([a]+[cell(g,k) for k in ('atomic_holdout_T','local_seen_T','local_holdout_T','recall_control_T','retention_control_T')])+' |')
    lines+=['','| 组 | T2T增益 | T2V增益 | 来源学习开发筛查 |','|---|---:|---:|---|']
    for a in ARMS:lines.append(f'| {a} | {changes[a]["original_T"]["gain"]*100:+.2f}pp | {changes[a]["original_V"]["gain"]*100:+.2f}pp | {"通过，仍需独立规则确认" if decisions[a]["source_screen_pass"] else "未通过，不扩大"} |')
    lines+=['','原题每方向16道、每规则2道，不能从小样本分差声称稳定迁移差距。训练题256道均评测，没有只挑容易样本。',
            '参数和局部短答采用严格文字匹配，完整回忆采用空白及大小写归一化；同义答案可能被低估，需查看原答。',
            '逐题原答见original_comparisons.jsonl及各组evaluation/predictions.jsonl；题型、分组、格式错误见summary.json。',
            'source_screen_pass为预先固定开发门槛，非显著性检验；T2T高于T2V不是筛选条件。',
            '未更改原1000条规则、3000张图或历史结果。本轮未训练VISION模型；不能据此确认视觉学习已修复。']
    (OUT/'report_zh.md').write_text('\n'.join(lines)+'\n')
    save(OUT/'completion_audit.json',{'status':'pass','predictions_per_checkpoint':680,'models':list(indexed),
        'images_verified':3000,'all_original_inputs_unchanged':True,
        'prediction_hashes':{a:sha(OUT/a/'evaluation/predictions.jsonl') for a in indexed},
        'final_checkpoints':{a:read(OUT/a/'checkpoint_manifest.json') for a in ARMS}})
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True,choices=('prepare','run','train','evaluate','summarize'))
    ap.add_argument('--arm',choices=('base',)+ARMS,default='answer_content');ap.add_argument('--gpu',default='0')
    a=ap.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='summarize':summarize()
    elif a.stage=='train':train(a.arm,a.gpu)
    elif a.stage=='evaluate':evaluate(a.arm,a.gpu)
    else:
        if a.arm!='base':
            train(a.arm,a.gpu,True);evaluate(a.arm,a.gpu,True);train(a.arm,a.gpu)
        evaluate(a.arm,a.gpu)


if __name__=='__main__':main()
