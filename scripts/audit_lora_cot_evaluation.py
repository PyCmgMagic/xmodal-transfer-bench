#!/usr/bin/env python3
"""Independently re-score frozen CoT predictions and write the final matrices."""
import argparse
import json
from pathlib import Path
import re
import time
import traceback

import run_v7_evaluation as ev

ROOT = Path(__file__).resolve().parents[1]


def independent_answer(text):
    if not isinstance(text, str) or text.count('<think>') != text.count('</think>'):
        return None
    final = text.rsplit('</think>', 1)[-1].strip()
    occurrences = re.findall(r'\bFINAL\s*[:：]\s*([A-D])\b', final)
    if len(occurrences) != 1:
        return None
    ending = re.fullmatch(r'(?s).*\bFINAL\s*[:：]\s*([A-D])\s*', final)
    return ending[1] if ending else None


def audit(work):
    manifest = json.loads((work/'manifest.json').read_text())
    state = json.loads((work/'status.json').read_text())
    assert state['status'] == 'complete' and state['completed_jobs'] == 10
    for path, sha in manifest['sources'].items():
        assert ev.file_hash(path) == sha, path
    for path, sha in manifest['frozen_files'].items():
        assert ev.file_hash(work/path) == sha, path
    images = json.loads((work/'eval_image_hashes.json').read_text())
    for path, sha in images.items():
        assert ev.file_hash(path) == sha, path
    reports = []
    for job in manifest['jobs']:
        out = work/'jobs'/job['name']
        assert json.loads((out/'state.json').read_text())['status'] == 'complete'
        runtime = json.loads((out/'runtime.json').read_text())
        assert runtime['use_chat_template'] and runtime['explicit_cot'] and not runtime['native_thinking']
        assert not runtime['template_is_thinking']
        assert runtime['adapter'] == job['adapter']
        labels = ev.id_map(ev.read_rows(work/'data'/job['dataset']/'eval_labels.jsonl')[0], 'labels')
        for direction in job['directions']:
            folder = out/direction
            inputs = ev.read_rows(work/'data'/job['dataset']/f'{direction}.jsonl')[0]
            raw = ev.id_map(ev.read_rows(folder/'raw.jsonl')[0], 'raw')
            scored = ev.id_map(ev.read_rows(folder/'results.jsonl')[0], 'scored')
            assert set(raw) == set(scored) == {r['id'] for r in inputs} == set(labels)
            report = json.loads((folder/'report.json').read_text())
            assert report['raw_sha256'] == ev.file_hash(folder/'raw.jsonl')
            correct = valid = truncated = explained = 0
            for r in inputs:
                sid = r['id']; pred = raw[sid]; result = scored[sid]
                payload = {k:r[k] for k in ('messages','images') if k in r}
                assert [m['role'] for m in payload['messages']] == ['system','user']
                assert pred['prompt_sha256'] == ev.digest(payload)
                assert pred['adapter'] == job['adapter']
                selected = independent_answer(pred['response'])
                assert selected == result['predicted']
                assert labels[sid]['answer'] == result['gold']
                assert (selected == result['gold']) == result['correct']
                correct += selected == result['gold']
                valid += selected is not None
                truncated += pred['finish_reason'] == 'length'
                explained += len(pred['response'].rsplit('FINAL',1)[0].strip()) >= 20
            n = len(inputs)
            assert report['total'] == n and report['correct'] == correct and report['valid'] == valid
            assert report['accuracy'] == correct/n and report['invalid'] == n-valid
            assert report['truncated'] == truncated and report['with_explanation'] == explained
            reports.append(report)
    assert len(reports) == 20 and sum(r['total'] for r in reports) == 42180
    audit_record = {'status':'pass','completed_at':ev.now(),'cells':20,'predictions':42180,
                    'trained_predictions':33744,'base_predictions':8436,'eval_images_verified':len(images),
                    'sources_unchanged':True,'prompt_and_adapter_alignment':True,
                    'independent_rescoring_matches':True,'invalid':sum(r['invalid'] for r in reports),
                    'truncated':sum(r['truncated'] for r in reports),
                    'auditor_sha256':ev.file_hash(__file__)}
    ev.save_json(work/'final_completion_audit.json',audit_record)
    lines = ['# LoRA 全量 CoT 四方向评测结果','',
             '两套数据、SFT/PT、TEXT/VISION共8个最终第5 epoch LoRA；另有同协议基座对照。'
             '完整42,180条预测已独立重评分，输入、权重及原图哈希检查通过。','',
             'CoT为Instruct模型的显式逐步解释提示，1536 tokens上限、200词内解释要求、temperature=0、seed=42。'
             '仅解析末尾唯一FINAL标记；无效输出计入错误分母。','']
    for dataset, title in [('rules','规则类'),('entities','实体类')]:
        rr = [r for r in reports if r['dataset']==dataset]
        base = {r['direction']:r for r in rr if r['objective']=='base'}
        lines += [f'## {title}','', '| 训练目标 | T2T | T2V | V2V | V2T |',
                  '|---|---:|---:|---:|---:|']
        for objective in ('sft','pt'):
            lookup = {r['direction']:r for r in rr if r['objective']==objective}
            cells = [f"{lookup[d]['accuracy']:.2%} ({lookup[d]['correct']}/{lookup[d]['total']})"
                     for d in ('T2T','T2V','V2V','V2T')]
            lines.append('| '+objective.upper()+' | '+' | '.join(cells)+' |')
        lines += ['',f"同协议基座：文字测试 {base['T2T']['accuracy']:.2%}，图片测试 {base['T2V']['accuracy']:.2%}。",
                  '', '| 目标 | T2T较基座 | T2V较基座 | V2V较基座 | V2T较基座 |',
                  '|---|---:|---:|---:|---:|']
        for objective in ('sft','pt'):
            lookup = {r['direction']:r for r in rr if r['objective']==objective}
            cells = [f"{100*(lookup[d]['accuracy']-base['T2V' if d.endswith('V') else 'T2T']['accuracy']):+.2f} pp"
                     for d in ('T2T','T2V','V2V','V2T')]
            lines.append('| '+objective.upper()+' | '+' | '.join(cells)+' |')
        lines += ['', '| 任务 | 方向 | 无效答案 | 生成截断 | 有解释 |', '|---|---|---:|---:|---:|']
        for r in rr:
            lines.append(f"| {r['objective']}/{r['learning_modality']} | {r['direction']} | {r['invalid']} | {r['truncated']} | {r['with_explanation']}/{r['total']} |")
        lines.append('')
    lines += ['## 解读边界','',
              '每种训练目标的四方向由两个学习模型组成：TEXT测T2T/T2V，VISION测V2V/V2T。'
              '基座没有学习模态，只报告两种测试输入条件。','',
              '该评测不认证数据的语义或时间新颖性。规则104道复合记录任务及实体数据既有审计限制继续适用。'
              '可见CoT是生成解释，不等同于内部计算的忠实记录。规则分组见各方向report.json；'
              '四方向差值也可能包含图片识别、提示与原训练材料表达差异的影响。','',
              '逐题原始回答、解释、生成token、截断标记及prompt/adapter哈希在jobs/*/*/raw.jsonl；'
              '逐题评分在results.jsonl，汇总在summary.csv。']
    (work/'report_zh.md').write_text('\n'.join(lines)+'\n')
    ev.save_json(work/'audit_status.json',{'status':'complete','completed_at':ev.now(),
                 'report':str(work/'report_zh.md'),'audit':str(work/'final_completion_audit.json')})
    print(json.dumps(audit_record,ensure_ascii=False),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,default=ROOT/'runs/qwen3_lora_cot_eval_20261025')
    parser.add_argument('--wait',action='store_true')
    args = parser.parse_args();work=args.work.resolve()
    try:
        if args.wait:
            ev.save_json(work/'audit_status.json',{'status':'waiting_for_evaluation','started_at':ev.now()})
            while True:
                status=json.loads((work/'status.json').read_text())
                if status['status']=='complete':
                    break
                if status['status']=='failed':
                    raise RuntimeError('Evaluation failed; inspect status.json and coordinator_error.json')
                time.sleep(60)
        audit(work)
    except BaseException as exc:
        ev.save_json(work/'audit_status.json',{'status':'failed','error':str(exc),
                     'traceback':traceback.format_exc(),'updated_at':ev.now()})
        raise


if __name__=='__main__':
    main()
