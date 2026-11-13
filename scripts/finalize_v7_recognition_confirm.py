#!/usr/bin/env python3
"""Finalize the completed fixed-rule experiment without launching new inference."""
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path

import confirm_v7_rule_recognition as pilot
import confirm_v7_paired_application as paired
import summarize_v7_recognition_confirm as summary
import run_v7_evaluation as ev
from build_v7_experiment_data import name
from run_v7_full_experiment import ROOT, read, rows, save, sha
from v7_gpu_resources import ledger

OUT=pilot.OUT


def finalize():
    summary.summarize()
    ss=read(OUT/'summary.json')
    titles={'warmup':'共同起点','text_explain':'TEXT解释','text_match':'TEXT辨认','text_mixed':'TEXT混合',
            'vision_explain':'VISION解释','vision_match':'VISION辨认','vision_mixed':'VISION混合'}
    def score(a,c):
        r=ss[a][c]
        return f"{r['correct']}/{r['n']}（{r['accuracy']:.1%}）"
    def delta(a,c):
        return f"{100*(ss[a][c]['accuracy']-ss['warmup'][c]['accuracy']):+.1f}个百分点"
    lines=['# 结果解读与下一步','',
        '这轮完成6个1epoch SFT适配器和共同起点对照，4872次回答。其中4704次选择题回答格式错误为0，168份规则回忆逐条人工审核。新8条规则、单seed；没有做全量重训，也没有做PT。','',
        '## 文字辨认的局部收益复现，但原题迁移没有整体解决','',
        f"同观察同选项、只换规则名的配对对照：TEXT辨认文字应用{score('text_match','paired_T')}，相对起点{delta('text_match','paired_T')}；图片应用{score('text_match','paired_V')}，相对起点{delta('text_match','paired_V')}。无图改写定义辨认{score('text_match','recognition_para_noimage')}。因此收益不局限于逐字重复训练定义，也出现于未训练的应用观察，但仍有约一半应用回答错误。",
        '',
        f"原v7题四位置合计：TEXT辨认文字{score('text_match','named_T')}，起点{score('warmup','named_T')}；图片{score('text_match','named_V')}，起点{score('warmup','named_V')}。图片原题没有改善，不能把配对题的收益写成原v7全部题已修复。原顺序16题另见主报告，轮换不增加独立知识数量。",
        '',
        '## 混合练习没有自动结合两种能力','',
        f"TEXT混合能在训练system下完整回忆{ss['text_mixed']['recall']['recall']['correct']}/8，但无图改写定义辨认{score('text_mixed','recognition_para_noimage')}；配对文字{score('text_mixed','paired_T')}，图片{score('text_mixed','paired_V')}。直接各放32行并不等于两项目标均学好。应停止把这份50/50配置视为已验证修复。",
        '',
        'token审计：混合组解释目标共9856个assistant内容token，辨认目标共256个；均不含模板或EOS。两类行数相同，目标长度不同。这提示后续需检查实际loss归一化和任务权重，但本实验没有隔离权重、监督次数、提示形式等因素，不能断言某一个就是原因。',
        '',
        '## 视觉学习仍依赖学习图，名称检索没有打通','',
        f"VISION解释带原学习图完整回忆{ss['vision_explain']['recall']['recall_image']['correct']}/8，去图仅{ss['vision_explain']['recall']['recall']['correct']}/8。VISION辨认带图改写定义{score('vision_match','recognition_para_image')}，去图{score('vision_match','recognition_para_noimage')}。前者是在学习图当场可见时完成任务，不能据此认定知识已能凭名称提取。",
        '',
        f"VISION混合配对文字{score('vision_mixed','paired_T')}、图片{score('vision_mixed','paired_V')}；完整回忆无图{ss['vision_mixed']['recall']['recall']['correct']}/8、带学习图{ss['vision_mixed']['recall']['recall_image']['correct']}/8。本轮视觉辨认候选仍含文字定义，视觉端没有收益也不能归为完全无文字监督的纯视觉学习结论。",
        '',
        '## direct与diagnosis分开检查','',
        '| 训练 | direct文字 | diagnosis文字 | direct图片 | diagnosis图片 |',
        '|---|---:|---:|---:|---:|']
    for a in summary.ARMS:
        cells=[]
        for c,t in [('paired_T','direct'),('paired_T','diagnosis'),('paired_V','direct'),('paired_V','diagnosis')]:
            r=ss[a][c]['by_task'][t];cells.append(f"{r['correct']}/{r['n']}（{r['accuracy']:.1%}）")
        lines.append('| '+titles[a]+' | '+' | '.join(cells)+' |')
    lines+=['','## 同图换名是否影响回答','',
        '每个配对block和选项位置包含四个名称，其正确答案恰好覆盖A/B/C/D。始终答同一字母只能答对1/4。下面统计32组中，四名称都输出相同答案的组数；这是名称敏感性描述，不是独立显著性检验。','',
        '| 训练 | 文字始终同答/32 | 图片始终同答/32 |', '|---|---:|---:|']
    for a in summary.ARMS:
        lines.append('| '+titles[a]+' | '+' | '.join(str(ss[a][c]['paired_checks']['all4_same_answer']) for c in ('paired_T','paired_V'))+' |')
    lines+=['','## 下一步建议','',
        '1. 保留已经有局部收益的TEXT辨认作为参照，先修复从名称检索到应用的连接，再决定是否扩大规模。所有方法都同时报告原v7和名称配对对照，避免只挑有提升的一张表。',
        '2. 在同组小实验中单独控制提示协议：当前训练和辨认用规则说明system，应用用短答案system，plain_recall还去掉system。先验证提示变化是否影响知识提取；主测试输入不给正确事实。',
        '3. 对混合目标核对实际loss归一化，比较明确任务权重或分阶段训练。保持规则、步数及最后checkpoint固定，不按测试结果挑checkpoint。',
        '4. 视觉端优先解决“看图会答、去图不会”的关联问题。若采用图像生成文字再做无图回放，应单列为图像衍生文字桥接训练，不能混称纯VISION迁移。当前没有证据支持继续盲目增加epoch即可解决。',
        '',
        '本轮提供了复现证据和失败边界，尚不能将这8条上的结果外推1000条，也不能宣称数据集和完整四方向实验已修复。']
    (OUT/'interpretation_zh.md').write_text('\n'.join(lines)+'\n')
    with (OUT/'report_zh.md').open('a') as f:
        f.write('\n## 实质结论\n\n'+ '\n\n'.join((lines[6],lines[8])) + '\n\n完整分题型、名称敏感性和下一步分析见 interpretation_zh.md；具体训练输入见 TRAINING_EXAMPLE_ZH.md。\n')
    # Illustrative block selected only after all scoring, never used in training.
    probes=rows(paired.OUT/'probes.jsonl')
    pr={a:{r['id']:r for r in rows(paired.OUT/a/'predictions.jsonl')} for a in summary.ARMS}
    blocks=rows(paired.OUT/'blocks.jsonl')
    def gain(b):
        pp=[p for p in probes if p['block']==b['id'] and p['modality']=='T' and p['shift']==0]
        return sum(pr['text_match'][p['id']]['correct']-pr['warmup'][p['id']]['correct'] for p in pp)
    b=max(blocks,key=gain)
    pp=[p for p in probes if p['block']==b['id'] and p['modality']=='T' and p['shift']==0]
    kk={k['id']:k for k in rows(OUT/'target_knowledge.jsonl')}
    ex=['# 一个实际应用对照','',
        '此例在全部评测完成后按TEXT辨认相对共同起点的正确数差值选择，只用于解释结果；不是预指定代表样本，不能用于估计总体提升。训练和整体评测不受选例影响。','',
        '下列四题使用同一观察和选项，仅规则名变化；完整第一题：','', '```text',pp[0]['messages'][-1]['content'],'```','',
        '| 名称 | 正确 | 共同起点 | TEXT解释 | TEXT辨认 | TEXT混合 |','|---|---|---|---|---|---|']
    for p in pp:
        ex.append('| '+name(kk[p['qid']])+' | '+p['gold']+' | '+' | '.join(pr[a][p['id']]['response'] for a in ('warmup','text_explain','text_match','text_mixed'))+' |')
    (OUT/'APPLICATION_EXAMPLE_ZH.md').write_text('\n'.join(ex)+'\n')
    scripts=[ROOT/'scripts'/s for s in ('confirm_v7_rule_recognition.py','confirm_v7_paired_application.py',
        'summarize_v7_recognition_confirm.py','finalize_v7_recognition_confirm.py')]
    for s in scripts:shutil.copy2(s,OUT/'source_snapshots'/s.name)
    versions={p:importlib.metadata.version(p) for p in ('torch','transformers','ms-swift','peft','qwen-vl-utils','safetensors')}
    save(OUT/'environment_manifest.json',{'python':sys.version,'versions':versions,'timestamp':ev.now(),
        'scripts':{str(p):sha(p) for p in scripts}})
    with ledger(pilot.LEDGER) as entries:
        active={k:v for k,v in entries.items() if str(OUT) in k}
    assert not active,active
    files={}
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.name!='final_artifact_manifest.json' and p.suffix in ('.json','.jsonl','.md','.csv','.py') and 'checkpoint' not in p.relative_to(OUT).parts:
            files[str(p.relative_to(OUT))]=sha(p)
    save(OUT/'final_artifact_manifest.json',{'status':'complete','completed_at':ev.now(),
        'new_training_jobs':6,'new_responses_total':4872,'recall_manual_reviews':168,
        'active_gpu_reservations_for_this_run':active,'files':files})
    print('Final report and artifact hashes written.',flush=True)


if __name__=='__main__':finalize()
