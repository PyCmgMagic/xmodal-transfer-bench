#!/usr/bin/env python3
"""Validate all raw requests/results and compare readout protocols."""
import argparse
import json
from pathlib import Path

import repair_v7_application_protocol as first
import repair_v7_application_two_stage as second
import repair_v7_isolated_option_readout as isolated
from run_v7_full_experiment import read, rows, save, sha
import run_v7_evaluation as ev


def validate(complete=False):
    first.validate();second.validate();isolated.validate();first.parser_checks()
    report={'checked_at':ev.now(),'all_original_sources_unchanged':True,'arms':{},'complete':True}
    comparison=[]
    all_requests=0
    for arm in first.ARMS:
        one=first.OUT/arm/'predictions.jsonl';two=second.OUT/arm/'predictions.jsonl'
        rr=rows(one) if one.exists() else [];ss=rows(two) if two.exists() else []
        assert len({r['id'] for r in rr})==len(rr)
        assert len({r['id'] for r in ss})==len(ss)
        one_probes={p['id']:p for p in rows(first.OUT/'probes.jsonl')}
        two_probes={p['id']:p for p in rows(second.OUT/'worksheet_probes.jsonl')}
        cp=second.OUT/arm/'choice_probes.jsonl'
        if cp.exists():two_probes.update({p['id']:p for p in rows(cp)})
        drafts={r['input_mod']+':'+r['source_id']:r for r in ss if r['stage']=='worksheet'}
        for data,probes,kind in ((rr,one_probes,'first'),(ss,two_probes,'second')):
            for r in data:
                p=probes[r['id']]
                assert r['gold']==p['gold'] and r['qid']==p['qid']
                assert r['input_sha256']==ev.digest({'messages':p['messages'],'images':p.get('images',[])})
                if kind=='first':
                    parsed=first.parse(r['response'],r['protocol'],p['options'])
                    for key in ('predicted','format_error','option_consistent','fields'):assert r[key]==parsed[key]
                    assert r['correct']==(r['predicted']==r['gold'])
                    assert r['consistent_correct']==(r['correct'] and (r['protocol']!='explicit768' or r['option_consistent']))
                elif r['stage']=='choice':
                    assert r['predicted']==ev.letter(r['response'])
                    assert r['correct']==(r['predicted']==r['gold'])
                    draft=drafts[r['input_mod']+':'+r['source_id']]
                    assert r['worksheet_response_sha256']==ev.digest(draft['response'])
                    assert draft['response'] in p['messages'][-1]['content']
                else:
                    assert '\n\nOptions:\n' not in p['messages'][-1]['content']
                    assert r['correct'] is None
        arm_complete=len(rr)==216 and len(ss)==144
        if arm_complete:assert read(one.parent/'state.json')['status']==read(two.parent/'state.json')['status']=='complete'
        report['complete'] &= arm_complete
        report['arms'][arm]={'first_records':len(rr),'second_records':len(ss),'complete':arm_complete,
                            'first_sha256':sha(one) if one.exists() else None,'second_sha256':sha(two) if two.exists() else None}
        all_requests+=len(rr)+len(ss)
        iso_path=isolated.OUT/arm/'predictions.jsonl'
        iso=rows(iso_path) if iso_path.exists() else []
        iso_probes={p['id']:p for p in rows(isolated.OUT/f'{arm}_probes.jsonl')}
        assert len({r['id'] for r in iso})==len(iso)
        for r in iso:
            p=iso_probes[r['id']]
            assert r['gold']==p['gold'] and r['predicted']==ev.letter(r['response'])
            assert r['correct']==(r['predicted']==r['gold'])
            assert r['input_sha256']==ev.digest({'messages':p['messages'],'images':[]})
            assert drafts[r['input_mod']+':'+r['source_id']]['response'] in p['messages'][-1]['content']
            assert r['worksheet_response_sha256']==ev.digest(drafts[r['input_mod']+':'+r['source_id']]['response'])
        if len(iso)==72:assert read(iso_path.parent/'state.json')['status']=='complete'
        report['arms'][arm].update(isolated_records=len(iso),isolated_sha256=sha(iso_path) if iso_path.exists() else None)
        report['arms'][arm]['complete']=arm_complete and len(iso)==72
        report['complete'] &= len(iso)==72
        all_requests+=len(iso)
        for mod in ('T','V'):
            record={'model':arm,'test_modality':mod,'oracle':True}
            selections={protocol:[r for r in rr if r['protocol']==protocol and r['input_mod']==mod] for protocol in first.PROTOCOLS}
            selections['two_stage']=[r for r in ss if r['stage']=='choice' and r['input_mod']==mod]
            selections['isolated_readout']=[r for r in iso if r['input_mod']==mod]
            for protocol,data in selections.items():
                n=len(data);c=sum(r['correct'] for r in data);valid=sum(not r['format_error'] for r in data)
                record[protocol]={'n':n,'correct':c,'accuracy':c/n if n else None,'valid_accuracy':c/valid if valid else None,'format_errors':n-valid}
                for key in ('task','relation','evaluation_group','qid'):
                    parts={}
                    for value in sorted({r[key] for r in data}):
                        subset=[r for r in data if r[key]==value]
                        valid_n=sum(not r['format_error'] for r in subset)
                        correct_n=sum(r['correct'] for r in subset)
                        parts[value]={'n':len(subset),'correct':correct_n,'accuracy':correct_n/len(subset),
                            'valid_accuracy':correct_n/valid_n if valid_n else None,'format_errors':len(subset)-valid_n}
                    record[protocol]['by_'+key]=parts
                base={r['source_id']:r for r in selections['strict32']};common=[r for r in data if r['source_id'] in base]
                record[protocol]['paired_vs_strict32']={'common':len(common),
                    'wrong_to_correct':sum(not base[r['source_id']]['correct'] and r['correct'] for r in common),
                    'correct_to_wrong':sum(base[r['source_id']]['correct'] and not r['correct'] for r in common),
                    'response_changed':sum(base[r['source_id']]['response']!=r['response'] for r in common)}
                if protocol=='explicit768':
                    record[protocol]['consistent_correct']=sum(r['consistent_correct'] for r in data)
            comparison.append(record)
    assert read(first.OUT/'independent_text_label_audit.json')['status']=='pass'
    images=rows(first.DATA/'image_hashes.jsonl');assert len(images)==3000
    for im in images:assert sha(first.DATA/im['path'])==im['source_sha256']==im['repaired_sha256']
    cap=first.OUT/'caption_control';cap_manifest=read(cap/'manifest.json')
    assert cap_manifest['script_sha256']==sha(first.ROOT/'scripts/test_v7_boundary_clarification.py')
    assert cap_manifest['files']['probes.jsonl']==sha(cap/'probes.jsonl')
    cap_probes={r['id']:r for r in rows(cap/'probes.jsonl')}
    cap_path=cap/'predictions.jsonl';cap_rr=rows(cap_path) if cap_path.exists() else []
    for r in cap_rr:
        p=cap_probes[r['id']]
        assert r['gold']==p['gold'] and r['predicted']==ev.letter(r['response'])
        assert r['correct']==(r['predicted']==r['gold'])
        assert r['input_sha256']==ev.digest({'messages':p['messages'],'images':p['images']})
    report['complete'] &= len(cap_rr)==24
    all_requests+=len(cap_rr)
    candidate=read(first.OUT/'data_candidate/manifest.json')
    assert candidate['script_sha256']==sha(first.ROOT/'scripts/build_v7_visual_boundary_clarification.py')
    assert sha(Path(candidate['image']))==candidate['original_image_sha256']
    assert sha(Path(candidate['source']))==candidate['source_sha256']
    for name,h in candidate['files'].items():assert sha(first.OUT/'data_candidate'/name)==h
    report.update(total_generated=all_requests,original_images_verified_unchanged=3000,caption_control_records=len(cap_rr),
        independent_text_labels_checked=36,visual_preview_notes=36,status='pass' if report['complete'] else 'in_progress')
    if complete:assert report['complete'],report
    save(first.OUT/'completion_audit.json',report)
    save(first.OUT/'comparison.json',comparison)
    lines=['# 同题应用流程对照','','全部条件提供正确文字材料。不是训练效果或闭卷四方向迁移成绩。','',
        '| 模型 | 测试模态 | 原短答案 | 只增长度 | 明确结果JSON | 分步执行再选择 | 隔离选项映射 |','|---|---|---:|---:|---:|---:|---:|']
    for c in comparison:
        def fmt(p):
            s=c[p];return f'{s["correct"]}/{s["n"]} ({s["accuracy"]:.1%})' if s['n'] else '-'
        lines.append('| '+c['model']+' | '+c['test_modality']+' | '+' | '.join(fmt(p) for p in ('strict32','strict768','explicit768','two_stage','isolated_readout'))+' |')
    lines+=['','全部完成：'+str(report['complete'])+'。全部3000张原PNG及冻结来源哈希已核对不变。',
        '','18条规则、单seed；分步方案是在首轮失败后追加的开发修复，未独立确认。格式错误照常计错，有效输出与完整选项一致性单列。']
    (first.OUT/'comparison_zh.md').write_text('\n'.join(lines)+'\n')
    files=[p for p in first.OUT.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.md','.py') and p.name!='result_inventory.json']
    save(first.OUT/'result_inventory.json',{'created_at':ev.now(),'sha256':{str(p.relative_to(first.OUT)):sha(p) for p in files}})
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--complete',action='store_true');a=ap.parse_args();validate(a.complete)
