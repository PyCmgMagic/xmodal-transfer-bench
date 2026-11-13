#!/usr/bin/env python3
"""Report shared-task-baseline gains without conflating reference controls."""
import argparse
import csv
import json
from pathlib import Path

from pilot_v7_task_then_transfer import ARMS, DATA, validate_sources
from run_v7_full_experiment import read, rows, save, sha


def display(m):
    return f"{m['correct']}/{m['n']} ({m['accuracy']:.1%})"


def summarize(out):
    reports={a:read(out/a/'report.json') for a in ARMS}
    assert all(r.get('status')=='complete' for r in reports.values())
    manifest=read(out/'manifest.json');preds={a:rows(out/a/'predictions.jsonl') for a in ARMS}
    table=[]
    for a in ARMS:
        for c,m in reports[a]['metrics'].items():
            table.append({'arm':a,'condition':c,**{k:m[k] for k in ('n','correct','accuracy','valid_accuracy','format_errors','empty','truncated')}})
    with (out/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    directions=[]
    warm={r['id']:r for r in preds['warmup']}
    for a in ('text','vision'):
        for d in ('T2T','T2V','V2T','V2V'):
            c='direct_'+d[-1];group=[r for r in preds[a] if r['condition']==c]
            assert len(group)==16
            old=[warm[r['id']] for r in group];m=reports[a]['metrics'][c];baseline=reports['warmup']['metrics'][c]
            directions.append({'arm':a,'direction':d,'source_modality_matches_direction':d[0]==a[0].upper(),
                'primary_direction':d in (('T2T','T2V') if a=='text' else ('V2T','V2V')),
                'metrics':m,'shared_task_baseline':baseline,
                'gain_percentage_points':100*(m['accuracy']-baseline['accuracy']),
                'paired_wrong_to_right':sum(not b['correct'] and r['correct'] for r,b in zip(group,old)),
                'paired_right_to_wrong':sum(b['correct'] and not r['correct'] for r,b in zip(group,old)),
                'source_file':str(out/a/'predictions.jsonl'),'input_condition':c})
    # Category-label order queries copy the names already present in their prompt.
    fact_inventory={p['id']:p for p in rows(out/'probes.jsonl') if p['fmt']=='fact'}
    noncopy={}
    for a in ARMS:
        noncopy[a]={}
        for c in ('fact_no_image','fact_with_image'):
            rr=[r for r in preds[a] if r['condition']==c and 'category labels in below/between/above order' not in fact_inventory[r['id']]['messages'][-1]['content']]
            noncopy[a][c]={'n':len(rr),'correct':sum(r['correct'] for r in rr),'accuracy':sum(r['correct'] for r in rr)/len(rr),'truncated':sum(r['finish_reason']=='length' for r in rr)}
    diagnostics={}
    for a in ('text','vision'):
        same='direct_T' if a=='text' else 'direct_V';cross='direct_V' if a=='text' else 'direct_T'
        source_gain=reports[a]['metrics'][same]['accuracy']-reports['warmup']['metrics'][same]['accuracy']
        transfer_gain=reports[a]['metrics'][cross]['accuracy']-reports['warmup']['metrics'][cross]['accuracy']
        diagnostics[a]={'source_gain_pp':100*source_gain,'transfer_gain_pp':100*transfer_gain,
            'transfer_ratio':transfer_gain/source_gain if source_gain>=.25 else None,
            'ratio_gate':'Exploratory only: source gain must be >=25pp (4/16 questions). Small n still prevents a robust transfer claim.'}
    content_audit={}
    canonical={p['qid']:p['gold'] for p in rows(out/'probes.jsonl') if p['condition']=='card_training_prompt'}
    for a in ARMS:
        content_audit[a]={}
        for c in ('fact_no_image','fact_with_image'):
            rr=[r for r in preds[a] if r['condition']==c and 'category labels in below/between/above order' not in fact_inventory[r['id']]['messages'][-1]['content']]
            content_audit[a][c]={'n':len(rr),'requested_fact_exact':sum(r['correct'] for r in rr),'whole_correct_card_instead_of_short_fact':sum(r['response'].strip()==canonical[r['qid']] for r in rr),'note':'Full exact card is sufficient evidence of correct requested fact despite failing short-output format. Other nonexact responses ungraded for semantics.'}
    auxiliary={a:read(out/a/'task_wording_recall/report.json') for a in ('warmup','text','vision')}
    aliases={a:read(out/a/'rule_name_alias/report.json') for a in ('warmup','text','vision')}
    revision=read(out/'text_alias/report.json')
    save(out/'summary.json',{'manifest_sha256':sha(out/'manifest.json'),'metrics':table,'directions':directions,'facts_excluding_name_copy':noncopy,'fact_content_audit':content_audit,'transfer_diagnostics':diagnostics,'auxiliary_two_stage':auxiliary,'name_alias_diagnosis':aliases,'diagnosis_informed_text_alias_intervention':revision,'directions_are_input_aliases':True,'unique_primary_questions_per_arm':32,'target_rules':8,'status':'complete'})
    lines=['# 独立规则练习后，再测试知识跨模态迁移','',
        '本轮只是一轮 Qwen3-VL-4B 的 8 条目标规则开发实验。16 条其他规则用于共同的题型训练；随后文字、原图两组从同一个题型适配器出发，各做 1 epoch 的知识 SFT。目标规则的测试观察和答案未进入训练。','',
        '| 模型阶段 | 不给材料·文字测试 | 不给材料·图片测试 | 给正确文字材料·文字测试 | 给正确文字材料·图片测试 |',
        '|---|---:|---:|---:|---:|']
    for a,label in [('base','基础模型'),('warmup','只练习题型'),('text','练习题型后学文字'),('vision','练习题型后学原图')]:
        mm=reports[a]['metrics'];lines.append('| '+label+' | '+' | '.join(display(mm[c]) for c in ('direct_T','direct_V','oracle_text_T','oracle_text_V'))+' |')
    lines+=['','主结果不给模型学习材料，只输出 A/B/C/D。给材料的结果只能说明当场读取材料后的解题能力，不能当作已经学会、也不能计入迁移准确率。','',
        '| 目标知识训练 | 方向 | 正确数 | 相对共同题型基线的变化 | 原错→对 | 原对→错 |','|---|---|---:|---:|---:|---:|']
    for r in directions:
        if r['primary_direction']:
            lines.append(f"| {r['arm']} | {r['direction']} | {display(r['metrics'])} | {r['gain_percentage_points']:+.1f} 个百分点 | {r['paired_wrong_to_right']} | {r['paired_right_to_wrong']} |")
    lines+=['','每个方向只有 16 题，每题相当于 6.25 个百分点；同一规则有两题，样本还存在组内关联。T2T/V2T 的测试输入相同，T2V/V2V 也相同，因此每个模型只计算 32 个独立的主评测输入；不能把四方向标签当作额外样本。','',
        '## 辅助诊断','',
        '| 阶段 | 完整复述·训练问法 | 完整复述·新问法 | 给原学习图的完整复述 | 已见开发题给材料 |',
        '|---|---:|---:|---:|---:|']
    for a in ARMS:
        mm=reports[a]['metrics'];lines.append('| '+a+' | '+display(mm['card_training_prompt'])+' | '+display(mm['card_new_prompt'])+' | '+display(mm['card_image_prompt'])+' | '+display(mm['seen_dev_text_reference'])+' |')
    lines+=['','完整复述只按字面相同评分，措辞不同不代表知识一定错误。所有原始输出、结束原因和格式错误均保留。已见开发题是容量检查，并非独立泛化结果。', '',
        '短事实回答不能直接用严格匹配率当作知识正确率：文字组无图的26个非名称抄写探针中，有22个返回了完全正确的整条规则，尽管没有遵守“只给短事实”的格式；原图组有图时也有21/26返回正确整条规则。此问题已单独记在 `summary.json/fact_content_audit`，不把这些回答误称为知识错误。', '',
        '## 进一步定位：调用与名称绑定','',
        '让模型先按照实际题目中的规则名称回忆，再将原始回忆交回同一模型答题。没有提供或纠正外部规则材料；这多了一次生成，只能作为两步诊断。','',
        '| 阶段 | 按实际题目名称复述·字面相同 | 两步文字题 | 两步图片题 |','|---|---:|---:|---:|']
    for a in ('warmup','text','vision'):
        z=auxiliary[a]
        if z.get('status')=='complete':lines.append(f"| {a} | {z['task_wording_recall_exact']}/8 | {z['metrics']['direct_T']['correct']}/16 | {z['metrics']['direct_V']['correct']}/16 |")
    lines+=['',
        '文字组按训练/新复述问法都能正确复述8/8规则，但按真实题目名称问时，三分类的4条都不能字面正确复述。集合筛选的4条内容正确；程序化占比规则0972内容也正确，仅名称分隔符不同。其余三分类可见标签顺序错置、借用另一条规则的阈值或整个规则族。', '',
        '一次名称对照只改变三分类名称顺序和分隔符、保留其他文字，文字组的严格完整复述从0/4变为4/4，warmup与vision仍是0/4。**名称顺序也提供类别顺序线索，所以不能据此把评测名称改成低/中/高顺序来提高成绩。**正确的修复方向是让训练识别名称别名和排列，同时保持原评测。','']
    if revision.get('status')=='complete':
        lines+=['### 诊断后的单变量文字组对照','',
            '新增 `text_alias`：仍从同一warmup出发，512行/128步/1epoch，助手规则答案、集合题训练行与原题均不变；只给三分类训练提示覆盖全部6种名称排列及逗号/斜线分隔。此对照是在看到诊断后提出的开发修复，不是预注册的独立验证。','',
            '| 文字知识训练 | 原始文字题 | 原始图片题 | 新问法完整复述 |','|---|---:|---:|---:|']
        for label,z in [('原文字组',reports['text']),('名称不变性训练',revision)]:
            lines.append('| '+label+' | '+' | '.join(display(z['metrics'][c]) for c in ('direct_T','direct_V','card_new_prompt'))+' |')
        repaired_recall=read(out/'text_alias/task_wording_recall/report.json')
        if repaired_recall.get('status')=='complete':
            lines+=['',f"名称训练后，按实际题目名称完整回忆达到 {repaired_recall['task_wording_recall_exact']}/8；用模型自己的回忆再答题达到文字 {repaired_recall['metrics']['direct_T']['correct']}/16、图片 {repaired_recall['metrics']['direct_V']['correct']}/16。原文字组对应为4/8（字面相同）、11/16、9/16。直接选择题仍未提升，说明修复了名称调用，但还没有修复直接答题时的自动调用。"]
            save(out/'alias_intervention_results.json',{'direct':revision['metrics'],'two_stage':repaired_recall,'primary_original_reference':str(out/'text/report.json'),'intervention_manifest':str(out/'text_alias/training_manifest.json')})
        lines+=['']
    lines+=['## 判断与后续约束','',
        '1. 共同题型训练提高了给正确材料的文字解题，且开发题容量检查通过；但它没有自动带来无材料主评测提升。',
        '2. 文字组已经记住小规模规则，直接做题仍低分；至少在这组实验中，瓶颈包含知识调用与应用衔接，而不是完全没有学习。',
        '3. 原图组给学习图可复述8/8，不给图不能正确复述任何完整规则。只能说明它学会了图像条件下的回答，尚不能证明形成稳定的无图可检索知识。SFT监督答案也含文字知识，不能把它解释为纯像素知识写入。',
        '4. 主结果的同模态控制未稳定改善，因此不计算有意义的迁移保持率，也不能把跨模态分数解释为迁移能力差。',
        '5. 原图保留。名称别名已在文字小样本修复，但直接答题调用仍需桥接；图片到无图的回忆能力另设门槛。未通过前不扩到1000条/三模型长训练。',
        '6. 若研究允许把“先自行回忆规则、再解题”定义为统一评测协议，它可在不添加外部学习资料的情况下将记忆与迁移拆开。必须所有基线/方向使用相同两步协议，并单独保留直接答题结果；当前8条开发结果不足以替代正式实验。若坚持单步选择题，则还需要独立规则上的自动调用训练与题型重放，不能用两步分数顶替主结果。','',
        '## 数据与训练约束','',
        '- 原 v7 与全部 3000 张 PNG 保持不变，训练前后检查 SHA256。',
        '- 筛选在训练前按固定哈希顺序完成；排除原学习图不明确提供阈值的 target_object_share 结构，未按目标分数挑规则。24 张入选学习图均先人工检查。',
        '- 题型训练：512 行，320 个不同输入；开发规则的已有两题允许用于训练，不能再当作测试集。包括 4 种选项轮换与完整结果/选择字母两种输出。',
        '- 目标知识训练：每组 512 行，8 条规则各 16 种提示重复 4 次；不是 512 条独立知识。文字组没有图片，图片组使用原学习图，输入文字不含该规则的具体条件或阈值。',
        '- 图片组通过图片到文字的 SFT 学习，监督答案明确包含规则知识。这不是只对图像做无监督训练；仅凭这两组不能证明模型参数更新只利用了图片像素。',
        '- 两组最终 checkpoint 都固定为第 1 epoch，LoRA r8/alpha32/dropout0.05，学习率 1e-4，batch4，128 步，冻结视觉编码器和对齐层。两组均接续相同 warmup LoRA，但重置优化器。',
        '- 这轮只验证 SFT 思路，没有把原始材料 PT 的定义改掉，也没有做全量重新训练。','',
        '## 文件','',
        '- `warmup.jsonl`、`text.jsonl`、`vision.jsonl`：实际训练输入。',
        '- `manifest.json`、`image_eligibility.jsonl`、`model_hashes.json`：设计、来源与筛选记录。',
        '- 各阶段的 `checkpoint_manifest.json`、`train.log`、`predictions.jsonl`、`report.json`：权重、日志和逐题输出。',
        '- `summary.csv` / `summary.json`：总表、四方向别名、题型/规则族/分项、相对基线变化。',
        '- `task_wording_recall/`、`rule_name_alias/`：两步诊断及名称敏感性对照；`text_alias/`为诊断后的小规模训练修复。','']
    (out/'report_zh.md').write_text('\n'.join(lines))
    validate_sources(out)
    print('\n'.join(lines[:22]))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=Path('runs/v7_task_then_transfer_20261015/pilot_v2'));a=ap.parse_args();summarize(a.out.resolve())
