#!/usr/bin/env python3
"""Render the completed pilot, with controls and limitations kept separate."""
import json
import shutil
from pathlib import Path

import pilot_v7_counterfactual_names as pilot
import run_v7_evaluation as ev
from run_v7_full_experiment import ROOT, read, rows, save, sha


def ratio(m):
    return f"{m['correct']}/{m['n']}（{m['accuracy']:.2%}）"


def main():
    out = pilot.OUT
    audit = read(out / 'completion_audit.json')
    assert audit['status'] == 'complete' and audit['new_generations'] == 2448
    for filename in ('model_hash_validation.json', 'original_data_hash_validation.json'):
        assert read(out / filename)['status'] == 'pass'
    summary = read(out / 'summary.json')
    arms = summary['arms']
    recalls = read(out / 'semantic_review_summary.json')
    assert summary['construction']['status'] == 'pass'
    probes = {p['id']: p for p in rows(out / 'probes.jsonl')}
    reviews = {(r['arm'], r['qid']): r for r in rows(out / 'semantic_reviews.jsonl')
               if r['condition'] == 'recall'}
    # Fixed first examples in saved probe order; no subset accuracy is computed.
    examples = []
    for arm in ('text', 'vision'):
        rr = rows(out / arm / 'predictions.jsonl')
        lookup = {r['id']: r for r in rr}
        own = {p['id']: p for p in rows(out / arm / 'self_probes.jsonl')}
        selected = set()
        for r in rr:
            if r.get('shift') != 0 or not r['condition'].startswith('self_'):
                continue
            named = lookup[r['id'][5:]]
            if r['condition'] == 'self_T':
                wanted = (r['correct'] and not named['correct']) if arm == 'text' else not r['correct']
                kind = 'recall_rescued_text_answer' if arm == 'text' else 'wrong_recall_and_wrong_answer'
                if not wanted:
                    continue
            elif not r['correct']:
                kind = 'wrong_image_answer_group_' + str(r['group'])
            else:
                continue
            if kind in selected:
                continue
            selected.add(kind)
            review = reviews[arm, r['qid']]
            examples.append({
                'arm': arm, 'kind': kind, 'id': r['id'], 'qid': r['qid'],
                'gold': r['gold'], 'named_response': named['response'], 'self_response': r['response'],
                'own_recall': review['response'], 'recall_core_correct': review['core_facts_correct'],
                'named_input': probes[named['id']], 'self_input': own[r['id']],
                'raw_prediction_file': str(out / arm / 'predictions.jsonl'),
                'raw_prediction_file_sha256': sha(out / arm / 'predictions.jsonl'),
                'interpretation_limit': 'Observed recall/answer only. No native CoT collected; perceptual and application errors are not separable here.'})
    assert len(examples) == 6
    ev.save_rows(out / 'diagnostic_examples.jsonl', examples)

    lines = [
        '# 原图不变、只换规则名的配对实验（完成）', '',
        '这次把题面改成必须区分规则身份的成组选择题。共同基线直接答题约25%；文字训练后能完整回忆8/8条规则，先自行回忆再做文字题达到72.50%，直接答题仍为26.88%。视觉训练组尚未解决脱离学习图后的规则调用，不能据此宣布跨模态迁移已修复。', '',
        '## 实际改动', '',
        '沿用v7的8条真实规则、8张原学习图和5张原测试图。对同一观察固定四个选项，仅替换四个规则名称，四题的正确字母恰好覆盖A/B/C/D。每个选项都是其中一条真实规则产生的完整结果。报告核验题还固定同一份提交报告，其中一个规则接受，另三个分别替换为不同的正确结果。应用题与报告核验题分别保留，不合并题型。', '',
        f'具体中文例子及原图见 [EXAMPLE_ZH.md]({out / "EXAMPLE_ZH.md"})。原v7题目没有被覆盖，本轮是单独的候选构造。', '',
        '完全忽略名称、对组内四题始终给同一有效答案时，只能命中一个答案键，即25%。匿名控制每个固定题面只生成一次，再与四个规则的答案键比较，匹配率均为25%；这是构造性质，不能作为独立的“匿名题准确率正常”实验证据，也不保证所有未训练模型都恰好25%。本轮共同基线的具名题23.75%/24.38%是实际生成结果。', '',
        '## 模型、训练及计数', '',
        '模型为Qwen3-VL-4B-Instruct。共同对照不是裸基座，而是此前独立开发规则上的任务warmup适配器；本轮8条规则与该warmup的开发规则不重叠。TEXT和VISION均从该同一parent继续SFT，重新初始化optimizer。', '',
        '每组512条：8条规则各重复64次，目标是完整条件、组合关系和应用方式；文字组与视觉组的事实监督相同。视觉组输入始终带各自原学习图；没有混入无图事实训练。训练目标由学习侧结构规则生成，不含本轮测试观察、选项或答案。它不同于历史1000条材料各一次曝光的1 epoch，也不同于旧版不完整的视觉配文监督。', '',
        '两组各1 epoch、128步；batch=4、梯度累积=1、lr=1e-4、LoRA r=8/alpha=32/dropout=0.05、seed=42、max_length=8192、冻结vision/aligner。使用固定最终checkpoint-128，不按评测成绩选模型。两组各252个LoRA_B均非零。末次记录loss：TEXT=3.164e-05，VISION=6.39e-06；这些loss不能替代选择题成绩。', '',
        '本轮仅SFT，没有训练PT/FT，没有开启原生CoT。每组816次新生成：16次规则回忆、320次直接答题、320次自身回忆后答题、80次正确材料对照、80次匿名控制；三组共2448次，原始输出均保留。', '',
        '每个测试模态只有40个语义问题：5图×2题型×4规则名；四种选项轮换后是160次请求。总计10个配对题块、8条规则、5个场景，不能把轮换和换名后的请求当成独立样本。', '',
        '## 四方向主结果：直接答题，不给材料', '',
        '| 方向 | 对应共同基线 | 训练后 |',
        '|---|---:|---:|',
    ]
    directions = [('text', 'T', 'T2T'), ('text', 'V', 'T2V'),
                  ('vision', 'T', 'V2T'), ('vision', 'V', 'V2V')]
    for arm, mod, direction in directions:
        lines.append(f'| {direction} | {ratio(arms["warmup"]["named_" + mod])} | {ratio(arms[arm]["named_" + mod])} |')
    lines += ['', '单步答题全部仍接近25%，这部分没有出现足以支持修复成功的提升。', '',
              '## 两步诊断：先凭规则名自行回忆，再答题', '',
              '第一步只有规则名和回忆指令，没有测试观察或学习材料；第二步原样加入该模型自己的回忆，不纠正错误，不提供外部正确规则。VISION两步也使用无图回忆，不能把给学习图后读出的规则混入这一栏。', '',
              '| 方向 | 对应共同基线 | 训练后 |', '|---|---:|---:|']
    for arm, mod, direction in directions:
        lines.append(f'| {direction} | {ratio(arms["warmup"]["self_" + mod])} | {ratio(arms[arm]["self_" + mod])} |')
    lines += ['', '文字学习后的T2T、T2V出现描述性提升，说明在这批小样本上可以用独立回忆步骤显现学习效果；它不能替换单步主成绩。视觉训练组两步仍接近25%。', '',
              '## 学习侧核验：全部48份规则回忆', '',
              '| 适配器 | 只给名称，无学习图 | 名称＋原学习图 |', '|---|---:|---:|']
    for arm, label in [('warmup', '共同基线'), ('text', 'TEXT'), ('vision', 'VISION')]:
        lines.append(f'| {label} | {recalls[arm]["recall"]["core_correct"]}/8 | {recalls[arm]["recall_image"]["core_correct"]}/8 |')
    lines += ['',
        '48份完整回答已逐条检查条件、ANY/ALL及区间端点。TEXT无图8条全部完整正确；给学习图时Lamdal误报上界12（应14），Vevlev误报[12,16)（应[9,14)）。VISION给学习图时8条均正确，无图时却把形状颜色规则和不同区间大多统一回忆成[2,6]，Tetkil回忆成[1,4]。', '',
        '因此，VISION的极低训练loss与“带学习图时能生成正确规则”可以同时成立；它们并没有证明规则已经能按名称脱图调用。也不能据此断言模型一般能力被训坏，或断言知识完全没有写入权重。', '',
        '## 外部正确材料对照（单独报告）', '',
        '给当前题对应的正确文字规则，仅评测选项位置shift0，每格40题。这是查观察读取/规则应用能力的控制，不能算无材料迁移成绩。', '',
        '| 适配器 | 文字题 | 图片题 |', '|---|---:|---:|']
    for arm, label in [('warmup', '共同基线'), ('text', 'TEXT'), ('vision', 'VISION')]:
        lines.append(f'| {label} | {ratio(arms[arm]["oracle_T"])} | {ratio(arms[arm]["oracle_V"])} |')
    lines += ['',
        '即使给正确规则，图片题也只有45%—50%。知识调用之外，图中观察读取与后续应用仍有瓶颈；本轮短答案没有给出推理，无法进一步断定具体错在计数、属性识别、边界判断还是选项对应。该控制不是严格性能上界，提示词和选项轮换也会影响结果。', '',
        '同一shift0比较：TEXT自身回忆后文字31/40、图片20/40；TEXT正确材料对照分别28/40、18/40；VISION自身回忆后分别10/40、10/40，正确材料分别31/40、20/40。', '',
        '## 两种题型及规则结构分项', '',
        '每种题型每输入模态80次请求；形状颜色OR组96次、区间组64次。样本高度相关，下表只作描述。', '',
        '| 适配器／协议／测试输入 | 应用direct | 报告核验diagnosis | 形状颜色OR | 数值区间 |', '|---|---:|---:|---:|---:|']
    for arm in pilot.ARMS:
        for protocol in ('named', 'self'):
            for mod in ('T', 'V'):
                m = arms[arm][protocol + '_' + mod]
                vals = [ratio(m['by_task'][t]) for t in ('direct', 'diagnosis')]
                vals += [ratio(m['by_group'][str(g)]) for g in (0, 1)]
                lines.append('| ' + f'{arm}/{protocol}/{mod}' + ' | ' + ' | '.join(vals) + ' |')
    lines += ['',
        'TEXT自身回忆后，图片区间题只有17/64，而文字区间题52/64；图片形状颜色题61/96、文字64/96。数值图题值得优先排查，但这里只有2张区间测试图，不能推广到全部数值规则。', '',
        '## 配对行为与具体回答', '',
        '每模态有40个“题块×选项位置”单元，每单元含四个规则名。共同基线直接答题四名同答：文字33/40、图片36/40；TEXT直接答题31/40、32/40；VISION直接答题37/40、35/40。三个适配器直接答题均没有四名全部答对的单元。TEXT自行回忆后，文字四名同答降为0/40，四名全对达到9/40；图片仍没有四名全对单元。', '',
        '具体例子使用EXAMPLE_ZH中的同一原图与选项：Donsov应保留菱形候选C，正确选项A。TEXT直接答C；自行正确回忆“菱形或黄色”后答A。VISION无图却回忆成“数值>=2且<=6”，答C。这提供了可核对的规则调用证据。', '',
        '同图Deldev应保留白色A和五边形D，正确选项D；TEXT回忆的规则完整正确，但图片题选了只保留D的选项B。这表明回忆正确后仍可能答错；只凭一个最终字母不能确定其内部错误原因。', '',
        f'6个按固定顺序抽取的诊断例子、完整输入及输出见 [diagnostic_examples.jsonl]({out / "diagnostic_examples.jsonl"})。未生成CoT，未把分析者的猜测冒充模型给出的理由。', '',
        '## 审核及复现资料', '',
        '所有2448份输出均无缺失、空输出、格式错误或截断，因此本轮总准确率与有效输出准确率相同。校验通过：四方向ID集合及标签一致；320个具名构造由两份独立求解逻辑核对；全部3000张原PNG哈希不变；12份原数据来源文件及13份Qwen3基座/模板文件与历史哈希一致；输入、原始输出、LoRA、两步回忆引用哈希均核对。', '',
        '本轮训练和评测已结束，自己的GPU占用登记已释放，未终止无关进程。失败启动及续跑原因保留于execution_notes.json；未修改配置、安装包或重生成已保存回答。', '',
        f'- [四方向CSV]({out / "four_directions.csv"})、[全指标JSON]({out / "summary.json"})。',
        f'- [全部规则回忆审核]({out / "semantic_reviews.jsonl"})、[完成审核]({out / "completion_audit.json"})。',
        f'- [构造manifest]({out / "manifest.json"})、[文字训练数据]({out / "text.jsonl"})、[视觉训练数据]({out / "vision.jsonl"})。',
        f'- 原始回答：[共同基线]({out / "warmup/predictions.jsonl"})、[TEXT]({out / "text/predictions.jsonl"})、[VISION]({out / "vision/predictions.jsonl"})。', '',
        '复现顺序：select_v7_counterfactual_images.py验证选择；pilot_v7_counterfactual_names.py执行构造/训练/评测；review_v7_counterfactual_names.py保存逐条语义审核；summarize_v7_counterfactual_names.py --complete核验并汇总；report_v7_counterfactual_names.py生成本报告。最终哈希校验结果和代码快照保存在本目录。语义审核是对固定48份已保存回答的明确判断，脚本不是通用自动语义评分器。', '',
        '## 结论与下一步', '',
        '可以保留这种必须区分规则身份的成组构造，作为减少题面/选项捷径的候选；目前不应直接替换完整1000条最终集。它只覆盖筛选规则的OR和区间两种子结构，不包含三分类等知识族。规则选择依据原图能否区分四个结果，没有读取模型分数，但它仍是可区分性筛选出的开发样本；其中两条规则曾出现在旧开发试验，并非独立确认集。', '',
        '下一步优先修复视觉学习后的“去学习图、按名称调用规则”，先在固定开发组检验无图规则回忆，再用未参与调参的独立规则确认。若训练仍只在原学习图存在时有效，就不宜把低迁移分数解释成纯粹的跨模态迁移失败。任何加入无图知识答案训练的方案都应单列为桥接监督，保留纯视觉学习对照。', '',
        '同时单独检查数值原图的观察读取：先只问图中候选值，再给正确值做相同规则应用，以区分视觉读取和规则运算。完成这些门槛后，再考虑扩大规则数与补充PT对照。以上是后续建议，本轮没有执行额外训练或这些新诊断。', '',
        '当前证据支持“小样本文字规则可学，显式自行回忆可显现学习效果”；不支持“单步答题或视觉学习已经修好”。', '',
    ]
    (out / 'report_zh.md').write_text('\n'.join(lines))
    paths = [Path(pilot.__file__), Path(__file__)]
    paths += [ROOT / 'scripts' / f for f in (
        'select_v7_counterfactual_images.py', 'summarize_v7_counterfactual_names.py',
        'review_v7_counterfactual_names.py', 'run_v7_evaluation.py', 'run_v7_full_experiment.py',
        'pilot_v7_task_then_transfer.py', 'pilot_v7_atomic_supervision.py',
        'build_v7_experiment_data.py', 'v7_gpu_resources.py')]
    snapshot = out / 'source_snapshots'
    snapshot.mkdir(exist_ok=True)
    for source in paths:
        dest = snapshot / source.name
        if dest.exists():
            assert sha(dest) == sha(source), source
        else:
            shutil.copy2(source, dest)
    save(out / 'script_manifest.json', {
        'scripts': {str(p.resolve()): sha(p) for p in paths},
        'snapshots': {p.name: sha(snapshot / p.name) for p in paths}})
    save(out / 'final_artifact_manifest.json', {
        'status': 'complete', 'created_at': ev.now(),
        'files': {str(p.relative_to(out)): sha(p) for p in [
            out / name for name in ('report_zh.md', 'EXAMPLE_ZH.md', 'summary.json',
                'four_directions.csv', 'four_directions.json', 'completion_audit.json',
                'semantic_reviews.jsonl', 'semantic_review_summary.json',
                'diagnostic_examples.jsonl', 'model_hash_validation.json',
                'original_data_hash_validation.json', 'script_manifest.json')]}})
    print(out / 'report_zh.md')


if __name__ == '__main__':
    main()
