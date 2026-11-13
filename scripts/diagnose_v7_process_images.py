#!/usr/bin/env python3
"""Separate posthoc image controls for the completed eight-rule process model."""
import argparse
import copy
import json
import os
import re
from pathlib import Path

import confirm_v7_process_learning as previous
import explore_v7_memory_router as mem
import run_v7_evaluation as ev
from build_v7_experiment_data import name
from run_v7_full_experiment import ROOT, MODELS, read, rows, save, sha
from v7_gpu_resources import reserve

SOURCE = ROOT / 'runs/v7_process_confirmation_20261020'
OUT = ROOT / 'runs/v7_process_image_diagnostic_20261021'
CP = SOURCE / 'process/checkpoint/checkpoint-480'


def validate():
    m = read(OUT / 'manifest.json')
    assert m
    for p, h in m['sources'].items():
        assert sha(Path(p)) == h, p
    for p, h in m['files'].items():
        assert sha(OUT / p) == h, p
    assert ev.checkpoint(CP) == m['checkpoint']
    return m


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'manifest.json').exists():
        return validate()
    assert read(SOURCE / 'final_completion_audit.json')['status'] == 'pass'
    knowledge = {k['id']: k for k in rows(SOURCE / 'knowledge.jsonl')}
    originals = [p for p in rows(SOURCE / 'probes.jsonl')
                 if p['condition'] == 'original' and p['protocol'] == 'reasoning']
    text_by_scene = {(p['qid'], p['task']): p for p in originals if p['input_mod'] == 'T'}
    probes = []
    for p in originals:
        k = knowledge[p['qid']]
        text = text_by_scene[p['qid'], p['task']]
        base = p['messages'][-1]['content']
        assert base.endswith(previous.SUFFIX)
        z = copy.deepcopy(p)
        z.update(id='context_recall:' + p['id'], condition='context_recall', protocol='recall',
                 fmt='fact', gold=mem.compact(k), original_id=p['id'])
        z['messages'][0]['content'] = mem.MEM_SYSTEM
        z['messages'][-1]['content'] = base[:-len(previous.SUFFIX)] + (
            '\n\nDo not solve the question. State only the exact stored definition of ' + name(k) +
            ', including its conditions, parameters, and output convention.')
        probes.append(z)
        for condition in ('oracle_rule', 'oracle_observations', 'oracle_both'):
            if p['input_mod'] == 'T' and condition != 'oracle_rule':
                continue
            z = copy.deepcopy(p)
            z.update(id=condition + ':' + p['id'], condition=condition, original_id=p['id'])
            prefix = ''
            if condition in ('oracle_rule', 'oracle_both'):
                prefix += 'Diagnostic supplied correct rule:\n' + mem.compact(k) + '\n\n'
            if condition in ('oracle_observations', 'oracle_both'):
                description = '; '.join(c + ': ' + ', '.join(f'{f}={v}' for f, v in o.items())
                                        for c, o in zip('ABCD', text['observations']))
                prefix += 'Diagnostic supplied observation annotations:\n' + description + '\n\n'
            z['messages'][-1]['content'] = prefix + base
            probes.append(z)
        if p['input_mod'] == 'V':
            probes.append(dict(id='extract:' + p['id'], original_id=p['id'], qid=p['qid'], task=p['task'],
                condition='extract', protocol='extract', input_mod='V', fmt='json',
                gold=json.dumps(dict(zip('ABCD', text['observations'])), sort_keys=True), images=p['images'],
                messages=[dict(role='system', content='Read the image literally. Do not infer a selection rule or solve a question.'),
                    dict(role='user', content='<image>\nTranscribe the four candidates A, B, C, D. Return only one JSON object keyed by A, B, C, D. For each candidate, record the displayed numeric value under "value", or the visible object type under "category" and material under "material". Omit fields not present in this kind of image. Do not output KEEP/OMIT or a rule.')]))
    assert len(probes) == 112 and len({p['id'] for p in probes}) == 112
    ev.save_rows(OUT / 'static_probes.jsonl', probes)
    sources = [Path(__file__), SOURCE / 'manifest.json', SOURCE / 'probes.jsonl', SOURCE / 'knowledge.jsonl',
               SOURCE / 'process/evaluation/predictions.jsonl', ROOT / 'scripts/confirm_v7_process_learning.py',
               ROOT / 'scripts/explore_v7_memory_router.py', ROOT / 'scripts/run_v7_evaluation.py',
               ROOT / 'scripts/run_v7_full_experiment.py', ROOT / 'scripts/v7_gpu_resources.py',
               previous.DATA / 'image_hashes.jsonl']
    manifest = dict(created_at=ev.now(), sources={str(p): sha(p) for p in sources},
        files={'static_probes.jsonl': sha(OUT / 'static_probes.jsonl')}, checkpoint=ev.checkpoint(CP),
        generation=dict(direct=32, reasoning=768, recall=256, extract=384, temperature=0, thinking=False, batch=4),
        static_requests=112, dynamic_self_recall_requests=16, total_requests=128,
        scope='Posthoc diagnosis on the previous fixed eight rules. Not a training intervention or official transfer score.',
        caveats=['All oracle contexts are capability controls and excluded from closed-book scores.',
                 'Self-context uses the same checkpoint, verbatim recalled text, with no correction or model switch.',
                 'Exact definition and literal observation matching are conservative coverage checks; paraphrases need review.',
                 'Positive/negative oracle differences locate sensitivities, not unique causal attribution.'])
    save(OUT / 'manifest.json', manifest)
    return manifest


def dynamic_probes(predictions):
    recalled = {r['original_id']: r for r in predictions if r['condition'] == 'context_recall'}
    pp = []
    for p in rows(SOURCE / 'probes.jsonl'):
        if (p['condition'], p['protocol'], p['input_mod']) != ('original', 'reasoning', 'V'):
            continue
        r = recalled[p['id']]
        z = copy.deepcopy(p)
        z.update(id='self_context:' + p['id'], original_id=p['id'], condition='self_context',
                 recalled_response_sha256=ev.digest(r['response']))
        z['messages'][-1]['content'] = ('Your earlier unverified recollection of this rule:\n' + r['response'] +
                                       '\n\n' + z['messages'][-1]['content'])
        pp.append(z)
    assert len(pp) == 16
    return pp


def run(gpu):
    m = validate()
    mem.OUT = OUT
    folder = OUT / 'evaluation'
    folder.mkdir(exist_ok=True)
    with ev.run_lock(folder / 'eval.lock'):
        runner = mem.Runner(folder, gpu, 'process_image_diagnostic')
        if len(runner.rr) != m['total_requests']:
            save(folder / 'state.json', dict(status='waiting_for_gpu', expected=128, gpu=gpu))
            with reserve(previous.LEDGER, gpu, str(folder), 22000) as res:
                res.bind(os.getpid())
                runner.load(CP, adapter=True)
                static = rows(OUT / 'static_probes.jsonl')
                for protocol in ('recall', 'extract', 'reasoning'):
                    runner.infer([p for p in static if p['protocol'] == protocol], protocol)
                dynamic = dynamic_probes(runner.rr)
                ev.save_rows(OUT / 'self_context_probes.jsonl', dynamic)
                runner.infer(dynamic, 'reasoning')
        assert len(runner.rr) == len(runner.seen) == 128
        save(folder / 'state.json', dict(status='complete', n=128, ended_at=ev.now()))
    score()


def score():
    validate()
    pp = rows(OUT / 'static_probes.jsonl')
    rr = rows(OUT / 'evaluation/predictions.jsonl')
    dynamic = dynamic_probes(rr)
    assert rows(OUT / 'self_context_probes.jsonl') == dynamic
    probes = {p['id']: p for p in pp + dynamic}
    assert len(rr) == len({r['id'] for r in rr}) == len(probes) == 128
    summaries, details = {}, []
    for r in rr:
        p = probes[r['id']]
        assert r['input_sha256'] == ev.digest(dict(messages=p['messages'], images=p.get('images', [])))
        response = r['response'].strip()
        if p['protocol'] == 'recall':
            valid = bool(response)
            good = ' '.join(response.lower().split()) == ' '.join(p['gold'].lower().split())
        elif p['protocol'] == 'extract':
            try:
                parsed = json.loads(re.sub(r'^```(?:json)?\s*|\s*```$', '', response))
                valid = isinstance(parsed, dict) and set(parsed) == set('ABCD')
            except ValueError:
                parsed, valid = None, False
            checks = []
            for c, fields in json.loads(p['gold']).items():
                for field, truth in fields.items():
                    value = parsed.get(c, {}).get(field) if valid and isinstance(parsed[c], dict) else None
                    try:
                        match = float(value) == float(truth) if field == 'value' else str(value).lower().strip() == str(truth).lower().strip()
                    except (ValueError, TypeError):
                        match = False
                    checks.append(dict(candidate=c, field=field, expected=truth, observed=value, correct=match))
            good = all(c['correct'] for c in checks)
            details.append(dict(id=r['id'], raw_response=response, parsed=parsed, field_checks=checks,
                                literal_scene_match=good, caveat='Literal annotations only; aliases or image ambiguity need review.'))
        else:
            answer = re.findall(r'(?im)^\s*Answer:\s*([A-D])\s*[.!]?\s*$', response)
            valid = len(answer) == 1
            good = valid and answer[0] == p['gold']
        key = p['condition'] + ':' + p['input_mod']
        z = summaries.setdefault(key, dict(n=0, correct=0, invalid=0, truncated=0))
        z['n'] += 1; z['correct'] += int(good); z['invalid'] += int(not valid)
        z['truncated'] += r['finish_reason'] == 'length'
    for z in summaries.values():
        z['accuracy'] = z['correct'] / z['n']
    baseline = read(SOURCE / 'summary.json')['process']
    result = dict(status='complete', requests=128, metrics=summaries,
        original_closed_book={key: baseline['original:reasoning:' + key] for key in ('T', 'V')},
        caveat='Diagnostic controls remain separate. Recall/extraction are exact-match diagnostics, not full semantic grading.')
    save(OUT / 'summary.json', result)
    ev.save_rows(OUT / 'extraction_review.jsonl', details)
    lines = ['# 图片错误拆解（事后控制）', '', '固定上一轮过程模型和原图，共128次新生成；不计入正式迁移分数。', '',
             '| 控制/模态 | 正确或精确匹配 | 无效 | 截断 |', '|---|---:|---:|---:|']
    for key, z in summaries.items():
        lines.append(f'| {key} | {z["correct"]}/{z["n"]}（{z["accuracy"]:.1%}） | {z["invalid"]} | {z["truncated"]} |')
    lines += ['', 'context_recall是在完整题目上下文中只回忆定义；extract只读取原图；oracle_rule提供正确规则；oracle_observations保留图片并提供文字观测；oracle_both同时提供两者；self_context提供同一模型未经修正的图片上下文回忆。', '',
              '闭卷基线原题推理：文字16/16、图片5/16。回忆采用完整定义精确匹配，近义改写不自动视为知识错误。图像字段采用字面标注匹配，类别别名和图片歧义须人工复核。oracle差异不是唯一因果解释，不能当成已修复的迁移成绩。']
    (OUT / 'report_zh.md').write_text('\n'.join(lines) + '\n')
    save(OUT / 'completion_audit.json', dict(status='pass', requests=128,
        prompt_hashes_verified=True, raw_predictions_sha256=sha(OUT / 'evaluation/predictions.jsonl'),
        summary_sha256=sha(OUT / 'summary.json'), ended_at=ev.now()))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=('prepare', 'run', 'score'), required=True)
    ap.add_argument('--gpu', default='1')
    args = ap.parse_args()
    if args.stage == 'prepare':
        prepare()
    elif args.stage == 'run':
        run(args.gpu)
    else:
        score()
