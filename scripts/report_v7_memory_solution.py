#!/usr/bin/env python3
"""Write the human report only after all fixed development/confirmation runs pass audit."""
from pathlib import Path
import json

from run_v7_full_experiment import ROOT, rows, read, save, sha

DEV = ROOT / 'runs/v7_memory_router_20261019'
CONF = ROOT / 'runs/v7_memory_router_confirm_20261019'


def main():
    inventory = read(DEV / 'result_inventory.json')
    confirm_inventory = read(CONF / 'final_inventory.json')
    assert inventory['completed'] and confirm_inventory['complete']
    development = read(DEV / 'all_methods.json')
    confirmation = read(CONF / 'confirmation_results.json')
    reconstruction = read(CONF / 'target_reconstruction.json')
    def cell(s): return f'{s["correct"]}/{s["n"]}（{s["accuracy"]:.1%}）'
    def row(summary, arm):
        return f'| {arm} | {cell(summary[arm]["T"])} | {cell(summary[arm]["V"])} |'
    lines = ['# 本轮实际找到的提升路线及其边界', '',
        '**“具体事实记忆适配器 → 未纠正的名称回忆 → 冻结原基座执行 → 原选项映射”在另一批18条规则上复现了闭卷提升；单步答题和纯图像来源学习仍未修复。**', '',
        '本轮做完4组开发训练、2组新规则确认训练，以及原10epoch模型的调用对照、材料式PT和图片观察转录对照。原v7规则、题目、标签、3000张PNG和历史训练数据没有修改。所有预先选中的规则和失败回答都计分，没有提供正确事实补丁。', '',
        '## 1. 优先看另一批规则的确认结果', '',
        '确认按原18个结构子类另取18条不同规则，排除已索引小实验knowledge文件中的规则；固定SHA256选择，不看成绩选题。从干净原基座重新训练，沿用开发配方，不按确认评测选checkpoint。每个测试模态36题，含direct/diagnosis各18题。它是新规则ID确认，仍只有一个seed、相同题目模板，不能外推为完整1000条成绩。', '',
        '**下表闭卷条件没有学习图、学习材料、检索、真实观察转录或人工纠正。** 第一阶段只给规则名称，原样传递模型的回忆。基座也使用完全相同的回忆和解题流程，避免把多步骤推理收益当作训练收益。', '',
        '| 知识来源 | 文字测试 | 图片测试 |', '|---|---:|---:|']
    for arm in ('base','text_memory','vision_bridge'):
        lines.append(row(confirmation['routed_base'], arm))
    lines.extend(['', '`text_memory`是文字监督；`vision_bridge`含原学习图和额外无图文字事实回放，因此不能把后者命名为纯V2T/V2V。图片测试仍使用原测试PNG。', '',
        '| 组 | 文字direct | 文字diagnosis | 图片direct | 图片diagnosis |', '|---|---:|---:|---:|---:|'])
    for arm in ('base','text_memory','vision_bridge'):
        s=confirmation['routed_base'][arm]
        parts=[f'{s[m]["by_task"][t]["correct"]}/18' for m in ('T','V') for t in ('direct','diagnosis')]
        lines.append('| '+arm+' | '+' | '.join(parts)+' |')
    lines.extend(['', '同一批模型保持原来的单步短答案时：', '',
        '| 模型 | 文字测试 | 图片测试 |', '|---|---:|---:|'])
    for arm in ('base','text_memory','vision_bridge'):
        lines.append(row(confirmation['direct'],arm))
    lines.extend(['', '**单步没有相应增益。新高分属于改变调用流程后的方法结果，不能替换旧单步主表，也不能宣称原模型在任何提问方式下都已掌握应用。**', '',
        '## 2. 怎样改的', '',
        '训练端从原基座重新开始，使用轻量LoRA，并把目标改成包含具体条件、数值、映射与输出约定的事实材料，减少依赖示例配文。TEXT用名称回忆请求；VISION事实组始终带原学习PNG；桥接组一半带原图、一半只给名称并监督同一事实。桥接中的无图文字目标是额外监督，不能隐藏。', '',
        '每组18规则×32次呈现=576行，在这个扩充文件上1epoch/144步；rank32、alpha64、dropout0.05、LR1e-4、batch4、无梯度累积、seed42、max8192，视觉与对齐层冻结。32次呈现不能描述成每条材料只学习一遍。所有新组使用最终144步，未根据评测选最好checkpoint。原正式选择题观察、测试图、选项和标签不进训练。', '',
        '测试端先仅凭规则名称让记忆模型输出规则；再把完整、未修正的回忆交给原基座，隐藏选项完成工作记录；最后只用自己的记录匹配四个原选项。所有执行模型都是同一原Qwen3-VL-4B基座，不是换更大模型。没有原材料检索、人工补正确规则或外部确定性执行器。它增加了生成步骤，推理成本高于直接输出字母。', '',
        '这项训练修改是事实目标、提示、更新范围、容量与重复次数的组合，不能单凭这轮把收益唯一归因于某一个超参数。跨checkpoint比较也不是等预算因果实验。', '',
        '## 3. 开发实验的全部路线', '', '| 知识来源 | 文字测试 | 图片测试 |', '|---|---:|---:|'])
    for arm in ('base','text_ep10','vision_ep10','text_memory','vision_memory','vision_bridge','text_pt_memory'):
        lines.append(row(development['routed_base'],arm))
    lines.extend(['',
        '旧10epoch TEXT模型也从显式调用受益，说明并非全部知识都没存进去；它仍有名称串规则和事实混淆。新TEXT记忆组在开发和确认均18/18逐字复现目标；新桥接组均17/18，仅匹配表的一条串成另一套规则。确认桥接错误是Vastor把sealed container/surgical mask/forceps兼容表换成另一条barrel/sack/crate/toolbox表，错误回忆原样保留。', '',
        '纯图像事实组：原图条件18/18重构目标，去图0/18逐字重构，原答还有明显规则类型混淆；它没有建立可靠的名称调用。材料式PT对照同576行、同规则顺序、rank和步数，但没有问答上下文；名称回忆也没有稳定恢复。0/18是目标重构分，不是对所有潜在内部知识的证明。', '',
        '图片先转录观察的完整对照：', '', '| 知识来源 | 原图直接工作记录 | 先转录图片 |', '|---|---:|---:|'])
    for arm in ('base','text_memory','vision_memory','vision_bridge','oracle'):
        before=development['routed_base'][arm]['V']
        after=development['observation_route'][arm]['V']
        lines.append(f'| {arm} | {cell(before)} | {cell(after)} |')
    lines.extend(['',
        '观察转录由基座只看原测试图生成，不给正确文字观察；五种知识来源都复用同一份未经纠正的转录。它会把文件夹称为笔记本、把两行格盘描述成三行、遗漏或重复计数，没有成为稳定改进。原图直接输入保留为候选流程。', '',
        '## 4. 给正确材料的应用控制与停止条件', '', '| 样本 | 文字 | 图片 |', '|---|---:|---:|'])
    for label, summary in (('开发18规则',development),('确认另18规则',confirmation)):
        s=summary['routed_base']['oracle']
        lines.append(f'| {label} | {cell(s["T"])} | {cell(s["V"])} |')
    gates={}
    for arm, source_mod in [('text_memory','T'),('vision_bridge','V')]:
        s=confirmation['routed_base'][arm][source_mod]
        b=confirmation['routed_base']['base'][source_mod]
        o=confirmation['routed_base']['oracle']
        gates[arm]={'source_accuracy':s['accuracy'],'source_gain':s['accuracy']-b['accuracy'],
                    'source_gain_pass':s['accuracy']>=.6 and s['accuracy']-b['accuracy']>=.15,
                    'oracle_each_modality_80pct':all(o[m]['accuracy']>=.8 for m in ('T','V')),
                    'format_pass':all(confirmation['routed_base'][arm][m]['format_errors']/36<=.02 for m in ('T','V'))}
        gates[arm]['all_predeclared_checks_pass']=all(gates[arm][k] for k in ('source_gain_pass','oracle_each_modality_80pct','format_pass'))
    save(CONF/'decision_results.json',gates)
    lines.extend(['', '预设开发筛查要求来源方向闭卷至少60%、比方法匹配基座高至少15个百分点；另要求oracle两个模态至少80%、格式错率≤2%。确认结果的逐项判定保存于decision_results.json。即使学习增益通过，也不以它掩盖未通过的应用控制或纯视觉失败。', '',
        '保留该方法作为可重复的改进候选；本轮不自动扩到1000条，不发布“全部四方向已修好”的数据集，不把桥接收益当纯视觉迁移。后续扩大规模需固定方法后检验容量、更多规则及多个seed；纯视觉来源仍需另找能形成无图知识调用的学习方案。', '',
        '## 5. 错误与可复核证据', '',
        'Ravvat的同一道算术诊断：旧TEXT模型即便拿到正确公式，把3×0.5+4×1.5+5×0.25−2算成7.25；新记忆组正确回忆事实后，冻结基座算出6.75并选对原选项。记录在concrete_example.json。两次上下文/执行流程不同，只作具体行为证据。', '',
        '剩余错误包含把pouch视为protective cover、序列旋转执行错、状态机没有逐次正确查表、报告轨迹多带初始状态、读图计数和物品类型错。解题记录会出错，不能以“能复述规则”替代最终准确率。', '',
        '原1000规则完整名称审计没有完全重复名称（1000个不同完整名称），因此不能用同名不同规则解释本次串规则；这不等于所有名称token或单个状态词都没有共享。name_collision_audit.json保留完整结论。', '',
        '## 6. 验证、统计与文件', '',
        f'开发正式训练4组，确认2组；本轮已保存 {inventory["total_successful_generations"]+confirm_inventory["generations"]+confirm_inventory["smoke_generations"]} 次成功生成，含 {inventory["successful_smoke_generations"]+confirm_inventory["smoke_generations"]} 次checkpoint重载回答。失败的预检记录另存preflight_revision目录，没有作为成功实验分数。请求数不是独立样本数。', '',
        '全部预定预测ID、实际输入哈希、回忆/转录原样传递、原始回答解析、模型权重身份、LoRA更新与冻结范围、最终epoch/步数、原训练数据和3000张PNG均已核验。所有评分都保留格式失败；逐字重构与语义判断分开。18规则作为相关簇计算描述性bootstrap区间，未做多方法选择校正，单seed不能称广泛统计确认。', '',
        '- 开发完整表：all_methods.csv、all_methods.json、paired_changes.json。',
        '- 确认完整表：../v7_memory_router_confirm_20261019/confirmation_results.json、decision_results.json。',
        '- 实际训练行与解释：TRAINING_DATA_ZH.md、text_memory.jsonl、vision_memory.jsonl、vision_bridge.jsonl、text_pt_memory/train.jsonl。',
        '- 原始回答：各组memory/predictions.jsonl、routed_base/predictions.jsonl、observation_route/*/predictions.jsonl。',
        '- 曲线与checkpoint：training_loss.png、training_summary.json、各组checkpoint_manifest.json。',
        '- 核验：completion_audit.json、result_inventory.json、original_data_validation.json、确认final_inventory.json。',
        '- 复现命令：根目录instruct.md。', '',
        '确认目录的通用manifest继承了开发脚本的一些样本说明；该目录的confirmation_manifest.json、confirmation_results.json与本报告明确界定其实际新规则样本及三组活动模型。'])
    text='\n'.join(lines)+'\n'
    (DEV/'interpretation_zh.md').write_text(text)
    (CONF/'interpretation_zh.md').write_text('# 新规则确认结果\n\n详细方法、开发负结果和边界见 ../v7_memory_router_20261019/interpretation_zh.md。\n\n'+
        '\n'.join(['| 知识来源 | 文字测试 | 图片测试 |','|---|---:|---:|']+[row(confirmation['routed_base'],a) for a in ('base','text_memory','vision_bridge')])+ '\n\n无学习材料、仅名称回忆后由冻结原基座解题；18条新规则、每模态36题、单seed。桥接含额外文字监督，不能写成纯视觉迁移。\n')
    save(DEV/'final_report_manifest.json',{'report_sha256':sha(DEV/'interpretation_zh.md'),'confirmation_report_sha256':sha(CONF/'interpretation_zh.md'),
        'writer_sha256':sha(Path(__file__)),'total_successful_generations':inventory['total_successful_generations']+confirm_inventory['generations']+confirm_inventory['smoke_generations'],
        'formal_training_jobs':6,'original_pngs_unchanged':3000})
    print(text[:5000])


if __name__=='__main__': main()
