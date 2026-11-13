#!/usr/bin/env python3
"""Audit whether supervised answers distinguish differing stored rule parameters."""
import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from build_v7_experiment_data import name
from run_v7_full_experiment import ROOT, rows, save, sha

IDENTITY_KEYS={'predicate_name','priority_name','index_name','selector_name','procedure_name'}


def normalize_string(text,names):
    if not names:return text
    pattern=r'\b(?:'+'|'.join(re.escape(n) for n in sorted(names,key=len,reverse=True))+r')\b'
    return re.sub(pattern,lambda m:names[m[0]],text)


def normalized_rule(value,names):
    if isinstance(value,dict):
        return {normalize_string(k,names):normalized_rule(v,names) for k,v in value.items()
                if k not in IDENTITY_KEYS and k not in {'structural_subfamily_id','difficulty','field_semantics','observation_unit'} and not k.endswith('_topology')}
    if isinstance(value,list):return [normalized_rule(v,names) for v in value]
    return normalize_string(value,names) if isinstance(value,str) else value


def run(out):
    out.mkdir(parents=True,exist_ok=True)
    dataset=ROOT/'rules_repaired_single_image_v7';training=ROOT/'runs/v7_full_experiment/data'
    knowledge=[r['content'] for r in rows(dataset/'audit/rule_knowledge.jsonl')]
    inputs={a:{r['qid']:r for r in rows(training/f'{a}.jsonl')} for a in ('sft_text','sft_vision','pt_text','pt_vision')}
    for a,rr in inputs.items():assert len(rr)==1000
    sources=[dataset/'audit/rule_knowledge.jsonl']+[training/f'{a}.jsonl' for a in inputs]
    details=[];groups=defaultdict(list)
    for k in knowledge:
        rule=json.loads(k['fact']['quote']);names={n.strip():f'LABEL{i}' for i,n in enumerate(name(k).split(','))}
        signature=normalized_rule(rule,names)
        signature_text=json.dumps(signature,sort_keys=True,ensure_ascii=False)
        for arm,rr in inputs.items():
            row=rr[k['id']];target=row['messages'][-1]['content']
            if arm.startswith('pt_'):
                assert target.removeprefix('<image>')==inputs[arm.replace('pt_','sft_')][k['id']]['messages'][-1]['content']
            normalized=normalize_string(target.removeprefix('<image>'),names)
            item={'qid':k['id'],'relation':k['fact']['relation'],'arm':arm,'normalized_target_sha256':hashlib.sha256(normalized.encode()).hexdigest(),'normalized_rule_signature':signature,'normalized_target':normalized,'original_target':target}
            if k['fact']['relation']=='three_way_category':
                numbers={Fraction(s) for s in re.findall(r'(?<![A-Za-z_\d.])-?\d+(?:\.\d+)?',target)}
                item['threshold_literal_coverage']={b:Fraction(str(rule['thresholds'][b])) in numbers for b in ('lower','upper')}
                item['expected_thresholds']=rule['thresholds']
            details.append(item);groups[(arm,k['fact']['relation'],normalized)].append((k['id'],signature_text))
    collisions=[]
    for (arm,relation,target),group in groups.items():
        if len({sig for qid,sig in group})<2:continue
        collisions.append({'arm':arm,'relation':relation,'n':len(group),'distinct_rules':len({sig for _,sig in group}),'normalized_target':target,'rules':[{'qid':qid,'normalized_rule':json.loads(sig)} for qid,sig in group]})
    summaries=[]
    for arm in inputs:
        for relation in sorted({k['fact']['relation'] for k in knowledge}):
            relrows=[r for r in details if r['arm']==arm and r['relation']==relation]
            gg=[g for g in collisions if g['arm']==arm and g['relation']==relation]
            summaries.append({'arm':arm,'relation':relation,'rules':len(relrows),'collision_groups':len(gg),'rules_in_collisions':sum(g['n'] for g in gg),'max_group':max([g['n'] for g in gg],default=0)})
    numeric={}
    for arm in inputs:
        group=[r for r in details if r['arm']==arm and r['relation']=='three_way_category']
        numeric[arm]={'n':len(group),'both_threshold_literals_present':sum(all(r['threshold_literal_coverage'].values()) for r in group),'neither_threshold_literal_present':sum(not any(r['threshold_literal_coverage'].values()) for r in group),'note':'Literal numeric coverage is a necessary-only text audit, not semantic OCR/graph decoding. A threshold might be implicit in image; positive token presence alone does not prove correct binding.'}
    from run_v7_evaluation import save_rows
    save_rows(out/'supervision_rows.jsonl',details);save_rows(out/'target_collisions.jsonl',collisions)
    save(out/'supervision_audit.json',{'status':'complete','sources':{str(p):sha(p) for p in sources},'script_sha256':sha(Path(__file__)),'rules':1000,'rows_checked':len(details),'same_supervised_material_between_sft_and_pt':True,'by_relation':summaries,'three_way_numeric_coverage':numeric,'normalization':'Replace fictional rule/category/state names only; remove rule topology/difficulty/identity metadata from semantic signature. Preserve actual conditions, thresholds, mappings, edges, operations, attributes and numerical coefficients. Output text otherwise unchanged.','meaning':'An identical target template can be predicted using names in the user prompt and generic instructions without recovering omitted parameters from the image. This proves a lack of explicit discriminating token supervision for these fields, not that image information is absent or gradients cannot ever encode it.','limits':['Collisions concern supervised target strings, not equality of image inputs.','Text/vision objectives are not semantically matched in the historical full run.','5historical epochs mean5 material exposures/rule; augmented8rule pilots have64 exposures/rule and different targets; they are not fair capacity comparisons.']})
    lines=['# 全量训练监督信息审计','',
           '检查实际送入历史完整实验的四份1000行训练文件；不是仅检查准备中的候选文件。SFT与PT对应模态的材料正文逐条相同（PT视觉多一个图像标签）。','',
           '| 规则族 | 规则数 | 文字监督中目标重复且规则参数不同 | 图片监督中目标重复且规则参数不同 |',
           '|---|---:|---:|---:|']
    for rel in sorted({k['fact']['relation'] for k in knowledge}):
        tt=next(r for r in summaries if r['arm']=='sft_text' and r['relation']==rel);vv=next(r for r in summaries if r['arm']=='sft_vision' and r['relation']==rel)
        lines.append(f"| {rel} | {tt['rules']} | {tt['rules_in_collisions']} | {vv['rules_in_collisions']} |")
    lines+=['','只将规则/类别等虚构名称替换成占位符，不删数字、不删条件。把同一规则的不同名称视作同一内容。图片监督中的重复组可直接查看 `target_collisions.jsonl`。','',
            f"117条三分类中，文字监督同时包含两个阈值数字的有{numeric['sft_text']['both_threshold_literals_present']}条，图片监督有{numeric['sft_vision']['both_threshold_literals_present']}条。这里只统计文本中是否明写数值，不能否认图中还编码了这些数值。",'',
            '例如0090的真实阈值是0.1/0.55，视觉监督要求“数出左图和中图的边界”；0089使用0.45/0.9，替换类别名字后，整段监督文本仍逐字相同。模型忽略具体边界也可能预测好这种通用答案，故低loss不证明记住边界。','',
            '这影响历史所有复用同一训练文件的模型，也同时影响SFT与PT的视觉目标。修复时应让正确解码出的具体参数进入监督目标，并验证移除学习图后能否回忆；不能只增加epoch或只看loss。', '',
            '这些统计不意味着所有视觉知识都无法学习，也不意味着图像没有信息；它们证明原目标没有迫使模型在输出中区分一些对评测至关重要的参数。', '']
    (out/'supervision_report_zh.md').write_text('\n'.join(lines));print('\n'.join(lines[:20]))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'runs/v7_access_diagnostics_20261015/supervision_audit');a=ap.parse_args();run(a.out.resolve())
