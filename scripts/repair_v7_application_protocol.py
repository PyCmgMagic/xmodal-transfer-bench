#!/usr/bin/env python3
"""Fixed development trial: explicit result-to-option readout, no new training."""
import argparse
import copy
import dataclasses
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import run_v7_evaluation as ev
from run_v7_full_experiment import ROOT, environment, read, rows, save, sha
from run_v7_fullparam_ep10 import dense_identity
from v7_gpu_resources import reserve

OUT = ROOT / 'runs/v7_application_protocol_repair_20261019'
DATA = ROOT / 'rules_repaired_single_image_v7'
SOURCE = ROOT / 'runs/v7_fullparam_sft_ep10_20261019'
AUDIT = ROOT / 'runs/v7_ep10_learning_audit_20261019'
ARMS = ('base', 'text', 'vision')
PROTOCOLS = {'strict32': 32, 'strict768': 768, 'explicit768': 768}
SYSTEM = '''Solve the multiple-choice question using the supplied reference and observations.
Candidate IDs, state names, record codes and option letters are different namespaces. A computed candidate C does not imply option C. Compare the computed result with the CONTENT of every option.
For report-audit questions, check the rule parameters, every reported result, and whether the report should be accepted or replaced. A correct result with a wrong rule parameter does not make the complete report correct.
For image inputs, read only the test image as observations. Reference examples are not test observations. Keep candidate groups separate and count only objects/cells required by the rule. Do not invent unreadable details.
Return one JSON object with these string fields, and no other text:
{"observation":"short task-relevant observation", "rule_used":"short applicable conditions/values", "computed_result":"explicit result before choosing an option", "report_action":"accept, replace, or not_applicable", "option_text":"copy the COMPLETE chosen option text, excluding its A./B./C./D. prefix", "answer":"A or B or C or D"}
Use concise factual records rather than a long explanation. The answer field is always the OPTION letter. Do not merely repeat the reference.'''


def options(text):
    return dict(re.findall(r'(?m)^([A-D])\. (.*)$', text))


def parse(text, protocol, opts):
    if protocol != 'explicit768':
        letter = ev.letter(text)
        return {'predicted': letter, 'format_error': letter is None,
                'option_consistent': None, 'fields': None}
    try:
        value = json.loads(text.strip())
        keys = ('observation', 'rule_used', 'computed_result', 'report_action', 'option_text', 'answer')
        valid = (isinstance(value, dict) and set(value) == set(keys)
                 and all(isinstance(value[k], str) and value[k].strip() for k in keys)
                 and value['answer'] in opts
                 and value['report_action'] in ('accept', 'replace', 'not_applicable'))
        pred = value['answer'] if valid else None
        consistent = value['option_text'].strip() == opts[pred].strip() if valid else False
        # A mismatched copy is retained as a consistency error. Never silently
        # recover a favorable letter from candidate IDs, results or option text.
        return {'predicted': pred, 'format_error': not valid,
                'option_consistent': consistent, 'fields': value}
    except (ValueError, TypeError, KeyError):
        return {'predicted': None, 'format_error': True,
                'option_consistent': False, 'fields': None}


def parser_checks():
    opts = dict(A='Result: C.', B='Result: D.', C='Result: B.', D='Result: A.')
    good = dict(observation='candidate C is triangle', rule_used='triangle first',
                computed_result='C', report_action='not_applicable', option_text=opts['A'], answer='A')
    assert parse(json.dumps(good), 'explicit768', opts)['predicted'] == 'A'
    wrong = dict(good, answer='C')
    z = parse(json.dumps(wrong), 'explicit768', opts)
    assert z['predicted'] == 'C' and not z['option_consistent']
    assert parse('Candidate C', 'strict32', opts)['format_error']
    assert parse('A', 'strict32', opts)['predicted'] == 'A'
    for bad in ('', '{}', json.dumps(dict(good, answer='option A')), json.dumps(dict(good, answer=['A']))):
        assert parse(bad, 'explicit768', opts)['format_error']


def prepare():
    parser_checks()
    OUT.mkdir(exist_ok=True)
    if (OUT / 'manifest.json').exists():
        validate(); return
    # Freeze exclusion lists before loading predictions or scoring labels.
    excluded = {r['id'] for r in rows(AUDIT / 'selected_knowledge.jsonl')}
    exclusion_files = {str(AUDIT / 'selected_knowledge.jsonl'): sha(AUDIT / 'selected_knowledge.jsonl')}
    for p in sorted((ROOT / 'runs').glob('v7*/**/*knowledge*.jsonl')):
        if 'source_snapshots' in p.parts or 'frozen_sources' in p.parts: continue
        try:
            rr = rows(p)
        except (ValueError, OSError): continue
        if not 1 <= len(rr) <= 200: continue
        ids = {r.get('content', r).get('id') for r in rr}
        ids = {x for x in ids if isinstance(x, str) and x.startswith('scale1k_')}
        if ids:
            excluded.update(ids); exclusion_files[str(p)] = sha(p)
    knowledge = {r['content']['id']: r['content'] for r in rows(DATA / 'audit/rule_knowledge.jsonl')}
    groups = defaultdict(list)
    for k in knowledge.values():
        if k['id'] not in excluded: groups[k['fact']['relation']].append(k)
    selected = []
    for rel, group in sorted(groups.items()):
        ordered = sorted(group, key=lambda k: hashlib.sha256(('application-protocol-repair:42:' + k['id']).encode()).hexdigest())
        first = ordered[0]; family = json.loads(first['fact']['quote'])['structural_subfamily_id']
        second = next(k for k in ordered[1:] if json.loads(k['fact']['quote'])['structural_subfamily_id'] != family)
        selected.extend((first, second))
    assert len(selected) == 18 and not ({k['id'] for k in selected} & excluded)
    ev.save_rows(OUT / 'knowledge.jsonl', selected)
    save(OUT / 'selection.json', {'method': 'SHA256 seed42; 2 different subfamilies per relation; no performance selection',
        'ids': [k['id'] for k in selected], 'excluded_ids': sorted(excluded), 'exclusion_sources': exclusion_files,
        'limits': 'Excluded indexed small knowledge files and previous audit. Historical full1000 models saw these rules. Not a pristine unseen benchmark.'})
    save(OUT / 'decision_criteria.json', {'fixed_before_generation': True, 'scope': 'oracle application only, not closed-book transfer',
        'application_gate': {'accuracy_each_test_modality': .8, 'models_required': list(ARMS),
                             'format_error_rate_max': .02},
        'next_stage': 'Only after the application gate is met, validate fact acquisition and closed-book retrieval. No automatic full1000 training.',
        'statistics': '18 development rules, one seed. Report by rule/relation/task; no request-level significance claims.'})
    ids = {k['id'] for k in selected}
    labels = {r['id']: r for r in rows(DATA / 'audit/eval_labels.jsonl')}
    train = {r['id']: r for r in rows(SOURCE / 'data/sft_text.jsonl')}
    probes = []
    for modality, direction in (('T', 'T2T'), ('V', 'T2V')):
        for original in rows(SOURCE / 'data' / f'eval_{direction}.jsonl'):
            if original['qid'] not in ids: continue
            gold = labels[original['id']]
            for protocol in PROTOCOLS:
                p = copy.deepcopy(original)
                text = p['messages'][-1]['content']; opts = options(text); assert set(opts) == set('ABCD')
                if protocol == 'explicit768':
                    assert text.rstrip().endswith(ev.SUFFIX)
                    text = text.rstrip()[:-len(ev.SUFFIX)]
                    text += '\n\nCompute the requested result, then match the entire chosen option. Return the JSON record specified by the system.'
                    p['messages'][0]['content'] = SYSTEM
                p['messages'][-1]['content'] = 'Correct learning material for this question:\n' + train[p['qid']]['messages'][-1]['content'] + '\n\n' + text
                p.update(id=f'{protocol}:{modality}:{original["id"]}', source_id=original['id'],
                    input_mod=modality, protocol=protocol, options=opts, gold=gold['answer'],
                    task=gold['question_type'], evaluation_group=gold['evaluation_group'],
                    relation=knowledge[p['qid']]['fact']['relation'], condition='oracle')
                probes.append(p)
    assert len(probes) == 216
    ev.save_rows(OUT / 'probes.jsonl', probes)
    images = rows(DATA / 'image_hashes.jsonl'); assert len(images) == 3000
    for r in images: assert sha(DATA / r['path']) == r['source_sha256'] == r['repaired_sha256']
    sources = [DATA / 'audit/rule_knowledge.jsonl', DATA / 'audit/eval_labels.jsonl', DATA / 'image_hashes.jsonl',
               SOURCE / 'data/sft_text.jsonl', SOURCE / 'data/eval_T2T.jsonl', SOURCE / 'data/eval_T2V.jsonl', AUDIT / 'manifest.json']
    models = read(AUDIT / 'manifest.json')['models']
    save(OUT / 'manifest.json', {'created_at': ev.now(), 'script_sha256': sha(Path(__file__)), 'models': models,
        'sources': {str(p): sha(p) for p in sources}, 'files': {p.name: sha(p) for p in OUT.glob('*') if p.is_file()},
        'rules': 18, 'relations': 9, 'requests_per_arm':216, 'arms': ARMS, 'protocols_max_tokens':PROTOCOLS,
        'batch': 4, 'temperature': 0, 'thinking': False, 'seed': 42, 'all3000images_unchanged': True,
        'limits': ['Correct text material is supplied in ALL conditions; no closed-book claims.',
                   'Explicit JSON and strict outputs have different format requirements, which are reported separately.',
                   'VISION denotes training source of checkpoint; test references here are text.',
                   'New development rule selection is not independent of historical full1000 training.']})
    validate()


def validate():
    m = read(OUT / 'manifest.json'); assert m['script_sha256'] == sha(Path(__file__))
    for p, h in m['sources'].items(): assert sha(Path(p)) == h, p
    for p, h in m['files'].items(): assert sha(OUT / p) == h, p


def run(arm, gpu):
    validate(); m = read(OUT / 'manifest.json'); folder = OUT / arm; folder.mkdir(exist_ok=True)
    with ev.run_lock(OUT / (arm + '.lock')):
        raw = folder / 'predictions.jsonl'
        rr, notes = ev.read_rows(raw, allow_partial_tail=True) if raw.exists() else ([], [])
        if notes: ev.save_rows(raw, rr)
        seen = {r['id'] for r in rr}; assert len(seen) == len(rr)
        pending = [p for p in rows(OUT / 'probes.jsonl') if p['id'] not in seen]
        if pending:
            save(folder / 'state.json', {'status': 'waiting_for_gpu', 'gpu':gpu, 'saved':len(rr), 'pid':os.getpid()})
            with reserve(ROOT / 'runs/v7_full_experiment', gpu, str(folder) + ':protocol-repair', 24000) as res:
                res.bind(os.getpid()); os.environ.update(environment(gpu, DATA)); os.environ.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
                import torch
                from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig
                from swift.utils import seed_everything
                torch.set_num_threads(4); seed_everything(42)
                assert dense_identity(Path(m['models'][arm]['path'])) == m['models'][arm]
                engine = TransformersEngine(m['models'][arm]['path'], model_type='qwen3_vl', torch_dtype=torch.bfloat16,
                    device_map='cuda:0', max_batch_size=4, attn_impl='sdpa')
                engine.template.enable_thinking = False
                for protocol, budget in PROTOCOLS.items():
                    for mod in ('T', 'V'):
                        todo = [p for p in pending if p['protocol']==protocol and p['input_mod']==mod]
                        for start in range(0, len(todo), 4):
                            batch = todo[start:start+4]
                            outputs = engine.infer([InferRequest(messages=copy.deepcopy(p['messages']), images=p.get('images', [])) for p in batch],
                                RequestConfig(max_tokens=budget, temperature=0, seed=42, return_details=True), use_tqdm=False)
                            assert len(outputs)==len(batch)
                            for p, output in zip(batch, outputs):
                                if isinstance(output, Exception): raise output
                                choice = output.choices[0]; response = choice.message.content or ''
                                parsed = parse(response, protocol, p['options'])
                                record = {k:v for k,v in p.items() if k not in ('messages','images')}
                                full = dataclasses.asdict(output); full.pop('prompt_token_ids', None); full.pop('prompt_logprobs', None)
                                record.update(response=response, **parsed, correct=parsed['predicted']==p['gold'],
                                    consistent_correct=parsed['predicted']==p['gold'] and (protocol!='explicit768' or parsed['option_consistent']),
                                    finish_reason=choice.finish_reason, raw_response=full,
                                    input_sha256=ev.digest({'messages':p['messages'], 'images':p.get('images', [])}))
                                with raw.open('a') as f:
                                    f.write(json.dumps(record, ensure_ascii=False, default=str)+'\n'); f.flush(); os.fsync(f.fileno())
                                rr.append(record)
                            save(folder / 'state.json', {'status':'running', 'gpu':gpu, 'saved':len(rr), 'expected':216, 'pid':os.getpid(), 'updated_at':ev.now()})
                            print(arm, protocol, mod, len(rr), '/',216, flush=True)
        assert len(rr)==216
        save(folder / 'state.json', {'status':'complete', 'saved':216, 'finished_at':ev.now()})


def summarize():
    m=read(OUT/'manifest.json'); summary={}; lines=['# 规则应用流程修复：固定18规则开发对照', '',
        'Qwen3-VL-4B基座及10epoch TEXT/VISION模型；全部条件提供正确文字材料，仅测试规则应用，不是闭卷知识迁移。原图/题目/标签未改。', '',
        'strict32是原协议；strict768只增加长度；explicit768要求明确结果、完整选项文本和最终选项字母。', '',
        '| 模型 | 协议 | 测试模态 | 总准确率 | 有效输出准确率 | direct | diagnosis | 格式错 | 选项复制不一致 | 截断 |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        path=OUT/arm/'predictions.jsonl'; rr=ev.read_rows(path, allow_partial_tail=True)[0] if path.exists() else []
        def metric(ss):
            n=len(ss); c=sum(r['correct'] for r in ss); valid=sum(not r['format_error'] for r in ss)
            return dict(n=n, correct=c, accuracy=c/n if n else None, valid_accuracy=c/valid if valid else None,
                format_errors=n-valid, empty=sum(not r['response'].strip() for r in ss),
                truncated=sum(r['finish_reason']=='length' for r in ss),
                inconsistent_options=sum(r['protocol']=='explicit768' and not r['format_error'] and not r['option_consistent'] for r in ss),
                consistent_correct=sum(r['consistent_correct'] for r in ss),
                predictions=dict(Counter(str(r['predicted']) for r in ss)))
        info={'state':read(OUT/arm/'state.json'), 'metrics':{}}
        for protocol in PROTOCOLS:
            info['metrics'][protocol]={}
            for mod in ('T','V'):
                ss=[r for r in rr if r['protocol']==protocol and r['input_mod']==mod]; s=metric(ss)
                for key in ('task','relation','evaluation_group','qid'):
                    s['by_'+key]={v:metric([r for r in ss if r[key]==v]) for v in sorted({r[key] for r in ss})}
                info['metrics'][protocol][mod]=s
                def fmt(v): return f'{v:.1%}' if v is not None else '-'
                by=s['by_task']; desc=lambda t: f'{by[t]["correct"]}/{by[t]["n"]}' if t in by else '-'
                lines.append(f'| {arm} | {protocol} | {mod} | {s["correct"]}/{s["n"]} ({fmt(s["accuracy"])}) | {fmt(s["valid_accuracy"])} | {desc("direct")} | {desc("diagnosis")} | {s["format_errors"]} | {s["inconsistent_options"]} | {s["truncated"]} |')
        completed=len(rr)==216
        info['complete']=completed
        info['gate_pass']=completed and all(info['metrics']['explicit768'][mod]['accuracy']>=.8
            and info['metrics']['explicit768'][mod]['format_errors']/36<=.02 for mod in ('T','V'))
        summary[arm]=info
    save(OUT/'summary.json',summary)
    lines += ['', '门槛：三个模型在explicit768下文字与图片测试各至少80%，格式错误率不超过2%。这是预先固定的开发筛查条件，不是显著性检验。', '',
        '当前全部完成：'+str(all(v['complete'] for v in summary.values()))+'；应用门槛全部通过：'+str(all(v['gate_pass'] for v in summary.values()))+'。', '',
        '18条规则、单seed；direct/diagnosis共享知识，不能把请求数当独立规则数。选项不一致单列，另保存consistent_correct；不从computed_result或选项文本事后恢复有利答案。完整回答在各模型predictions.jsonl。']
    (OUT/'report_zh.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--stage', choices=('prepare','run','summarize','parser-check'), required=True)
    ap.add_argument('--arm', choices=ARMS); ap.add_argument('--gpu', default='0'); a=ap.parse_args()
    if a.stage=='prepare': prepare()
    elif a.stage=='summarize': summarize()
    elif a.stage=='parser-check': parser_checks(); print('parser checks passed')
    else:
        assert a.arm
        run(a.arm,a.gpu)
