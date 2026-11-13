#!/usr/bin/env python3
"""Interpret completed confirmation without changing training or inference."""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/v7_process_confirmation_20261020'
ARMS = ('base', 'definitions', 'definition_replay', 'process')
NAMES = dict(base='基座', definitions='定义SFT', definition_replay='定义续训', process='具体过程SFT')


def read(p):
    return json.loads(p.read_text())


def rows(p):
    return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]


def main():
    audit = read(OUT/'independent_final_audit.json')
    assert audit['status']=='pass'
    m, summary = read(OUT/'manifest.json'), read(OUT/'summary.json')
    scored = rows(OUT/'scored_predictions.jsonl')
    assert len(scored)==2656
    passing=read(OUT/'completion_audit.json')['source_application_gate_passed']
    keys=('new_holdout:reasoning:T','original:reasoning:T','original:reasoning:V',
          'new_holdout:short:T','original:short:T','original:short:V','new_train:reasoning:T','recall:recall:T','field_recall:field:T')
    table=['| 测试 | 基座 | 定义SFT | 定义续训 | 具体过程SFT |','|---|---:|---:|---:|---:|']
    for key in keys:
        table.append('| '+key+' | '+' | '.join(f'{summary[a][key]["correct"]}/{summary[a][key]["n"]}（{summary[a][key]["accuracy"]:.1%}）' for a in ARMS)+' |')
    lines=['# 新规则具体过程学习：最终解释','',
        '**本轮来源应用可行性门槛'+('通过' if passing else '未通过')+'。** 这是8条固定规则、一个seed的TEXT学习确认，不是完整1000条或四方向最终结果。','']+table
    lines+=['','## 实际干预及证据范围','',
        '干净Qwen3-VL-4B先做640行定义/原子事实SFT、3epoch；两组从本轮最终checkpoint-480分支，继续定义复习或具体过程学习，各640行、3epoch/480步、LR1e-5。全部使用LoRA32/alpha64、冻结视觉/对齐层。旧适配器不参与初始化。',
        '过程组训练256个新文字场景，每个场景同时有直接字母和具体比较过程监督。256个独立留出场景覆盖direct/diagnosis与16种KEEP组合。正式原题、标签与3000张原PNG保留，测试不给材料、不检索、不切换到其他模型。',
        '小规模历史数据扫描后，按SHA256选择4条类别/材质OR和4条区间规则，全部参与学习与评测。它们可能出现在旧全量实验中，但未继承旧权重；不能宣称历史从未涉及。类别/材质域及更广的开闭区间与旧形状/颜色开发集不同，因此这是扩展域检验。',
        f'两组续训监督token分别{m["training"]["token_totals"]["definition_replay"]}与{m["training"]["token_totals"]["process"]}；更新步数一致，内容与token不一致，不能声称排除了额外训练token的作用。']
    lines+=['','## 同协议配对比较','',
        '以下比较具体过程与等更新步数的定义续训；区间按规则簇bootstrap，仅8簇，p值为描述性逐题McNemar，不能代替总体确认。', '']
    comparisons=read(OUT/'paired_comparisons.json')
    for key in keys[:6]:
        p=next(r for r in comparisons if r['before']=='definition_replay' and r['after']=='process' and r['test']==key)
        ci=p['rule_cluster_bootstrap_ci95']
        lines.append(f'- {key}：改善{p["gained"]}题、退步{p["lost"]}题，差值{p["delta"]:+.1%}；规则簇95%区间[{ci[0]:+.1%}, {ci[1]:+.1%}]，描述性p={p["descriptive_mcnemar_p"]:.4g}。')
    lines+=['','## 来源模态与图片测试','']
    if passing:
        lines.append('按预先固定的操作门槛，来源模态应用取得了值得继续确认的增益。需结合规则回忆、实际条件比较和最终选项检查，确认收益是否来自正确应用。不能只据T2T与T2V差值宣布纯知识迁移障碍。')
    else:
        lines.append('按预先固定门槛，本轮尚未完成来源模态应用修复。即使某些留出或图片分项较高，也应如实保留原题、基线及格式结果，不扩大为完整数据集成功。需依据具体错误区分规则调用、条件执行及选项对应，不能直接将图片低分归为迁移失败。')
    lines+=['','| 协议 | 过程组T2T | 过程组T2V | T2T−T2V |','|---|---:|---:|---:|']
    for proto in ('short','reasoning'):
        t=summary['process'][f'original:{proto}:T']['accuracy'];v=summary['process'][f'original:{proto}:V']['accuracy']
        lines.append(f'| {proto} | {t:.1%} | {v:.1%} | {t-v:+.1%} |')
    lines+=['','每方向仅16题，单题6.25个百分点；差值为描述性结果。训练及测试协议不按差值选择，不要求T2T必须高于T2V。',
        '显式推理为模型自身普通解释生成，native thinking关闭，所有组温度0、最多768token；仍为原四选一题。短答案上限32token，单独报告，不能用推理高分替换短答案。']
    lines+=['','## 格式、截断与题型','', '| 模型 | 新留出推理格式错/截断 | 原题T2T推理格式错/截断 | 原题T2V推理格式错/截断 |','|---|---:|---:|---:|']
    for a in ARMS:
        vals=[]
        for key in ('new_holdout:reasoning:T','original:reasoning:T','original:reasoning:V'):
            z=summary[a][key];vals.append(f'{z["format_errors"]}/{z["truncated"]}（共{z["n"]}题）')
        lines.append('| '+NAMES[a]+' | '+' | '.join(vals)+' |')
    lines+=['','无效/缺失最终答案按错误计入总准确率，有效输出准确率见summary.json。截断与格式错误有重叠，不能相加；低有效输出率不代表底层规则能力为零。',
        '题型分项：','', '| 测试/题型 | 基座 | 定义SFT | 定义续训 | 具体过程SFT |','|---|---:|---:|---:|---:|']
    for key in ('new_holdout:reasoning:T','original:reasoning:T','original:reasoning:V'):
        for task in ('direct','diagnosis'):
            vals=[summary[a][key]['by_task'][task] for a in ARMS]
            lines.append('| '+key+'/'+task+' | '+' | '.join(f'{z["correct"]}/{z["n"]}（{z["accuracy"]:.1%}）' for z in vals)+' |')
    traces=rows(OUT/'trace_diagnostics.jsonl')
    lines+=['','## 轨迹解析的实际覆盖','',
        '以下只统计固定Result行及精确观测值/规则参数比较模板；没有匹配模板的自由解释不可直接当作语义错误。完整定义行精确匹配同样只是保守辅助诊断。','',
        '| 模型 | 新留出Result可解析/256 | Result正确/256 | 四行比较完全匹配/256 |','|---|---:|---:|---:|']
    for a in ARMS:
        rr=[r for r in traces if r['arm']==a and r['condition']=='new_holdout'];assert len(rr)==256
        lines.append(f'| {NAMES[a]} | {sum(r["result_parseable"] for r in rr)}/256 | {sum(r["result_correct"] for r in rr)}/256 | {sum(r["exact_operand_lines_all"] for r in rr)}/256 |')
    grouped=defaultdict(list)
    for r in scored:
        if r['condition']=='original' and r['protocol']=='reasoning':grouped[r['arm']].append(r)
    examples={a:sorted(rr,key=lambda r:(r['input_mod'],r['qid'],r['id'])) for a,rr in grouped.items()}
    (OUT/'original_reasoning_review.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2)+'\n')
    lines+=['','## 可复核材料与限制','',
        '主评测4×664=2656次生成，全部原始回答保留在各组evaluation/predictions.jsonl；原题推理逐条对照另存original_reasoning_review.json。数据独立审计、原题负载、3000图像哈希、模型权重身份、最终epoch3检查点均通过。',
        '训练数据样例见TRAINING_DATA_ZH.md，loss见training_loss.png，按规则结果见per_rule_metrics.json。训练loss下降、定义复述或给定偏好对准确率均不作为应用学会的替代证据。',
        '本轮没有VISION学习组，也没有给材料控制；不能报告V2T/V2V或把图片错误全部解释成纯迁移。8规则/1seed、属性域变化、监督token不同均须保留。若门槛未通过，继续定位失败；若通过，在更多固定规则与训练seed上确认后才扩大，不挑选本轮高分规则。']
    manual_path = OUT/'manual_error_review.json'
    if manual_path.exists():
        manual = read(manual_path)
        assert len(manual['items']) == 32
        lines += ['', '## 原题理由逐条人工复核', '',
            '完整复核过程组原题T2T/T2V各16条推理回答，并查看16张原图拼图；其中3张另看完整图片。没有声称人工复核全部2656条输出。此项为事后定性诊断，不改变预先固定的最终字母评分或可行性门槛。',
            f'文字题最终字母{summary["process"]["original:reasoning:T"]["correct"]}/16正确，但理由与条件、结果、报告操作和选项均一致的为{manual["counts"]["T"]["coherent"]}/16。Lilren直接题把wood换成metal却碰巧选对C；Lilren和Somvad核查题的候选结果正确，但错误写成Accept report，而最终选项为Replace。',
            f'图片题最终字母{summary["process"]["original:reasoning:V"]["correct"]}/16正确，按同一人工准则仅{manual["counts"]["V"]["coherent"]}/16理由完整一致。若干正确字母来自错误规则/错误结果，不能当作应用学会。图片中既有正确识别物品却改写规则，也有区间规则被物品类别替代、读数错误及选项不一致；纸板样卡含布样的图文语义差异另作保留。',
            '例如Sanvat文字题正确调用2<x<4，图片题能识别手电，却把规则改为flashlight OR metal。Kinsem手机图被解释为toolbox/metal选择。此类输出提示图像上下文下的知识调用不稳定；仅凭这些样本不能确定是权重遗忘、读取问题还是模态触发的条件调用失败，更不能证明纯知识迁移失败。',
            '每条真实条件、文字观测、原始理由和复核说明见manual_error_review.json；原图拼图仅为审阅副本，不参与训练或正式评测。']
        if 'new_holdout_answer_error_review' in manual:
            counts = manual['new_holdout_answer_error_review']['counts']
            lines += ['', '过程组新文字留出推理的全部10道最终答案错误也已复核：'+
                f'{counts["wrong_rule_parameter"]}道错误参数、{counts["option_mapping"]}道Result正确但选项映射错误、'+
                f'{counts["false_equality"]}道把不同类别判断相等、{counts["category_alias_ambiguity"]}道roll/material roll近义边界。'+
                '最后一项源自程序按字符串相等定义类别真值，语义上有歧义；不能全部当作模型逻辑缺陷。本轮不更改冻结标签或据此筛题。此项没有声称已人工确认其余246道字母正确回答的全部理由。']
    operand_path = OUT/'operand_diagnostics_summary.json'
    if operand_path.exists():
        operands = read(operand_path)['summary']['process']
        lines += ['', '## 归一化比较操作诊断', '',
            '事后解析过程组的固定候选比较模板，数值4与4.0视为等价；自由解释无法匹配时只记未覆盖。主评分不变。',
            '| 测试 | 四候选模板可解析 | 全部规则参数正确 | 全部观测值正确 | 自身比较/逻辑一致 | Result真值正确 |',
            '|---|---:|---:|---:|---:|---:|']
        for key in ('new_holdout:T', 'original:T', 'original:V'):
            z = operands[key]
            lines.append('| '+key+' | '+' | '.join(f'{z[k]}/{z["n"]}' for k in
                ('all_candidate_comparisons_parseable','all_rule_parameters_match','all_observations_match',
                 'all_comparisons_self_consistent','result_matches_truth'))+' |')
        lines += ['', '规则参数与观测诊断采用四候选完全可解析的保守覆盖；不等于语义总正确率，也不能用Result正确替换正式四选一分数。明细见operand_diagnostics.jsonl。']
    (OUT/'analysis_zh.md').write_text('\n'.join(lines)+'\n')
    print('Wrote',OUT/'analysis_zh.md','source gate:',passing)


if __name__=='__main__':
    main()
