"""Report only complete matched vLLM arms; keep historical short results distinct."""
import csv
import json
import random

import eval_v7_cot100 as ex
import eval_v7_cot100_vllm as run
from run_v7_full_experiment import rows, save


def main():
    run.setup()
    reports = {arm: ex.summarize(arm) for arm in ('base', 'trained')}
    complete = all(r['status'] == 'complete' for r in reports.values())
    save(run.OUT / 'summary.json', dict(status='complete' if complete else 'partial', arms=reports))
    if not complete:
        print(json.dumps({a:r['saved'] for a,r in reports.items()})); return
    all_rows = {a: rows(run.OUT / a / 'predictions.jsonl') for a in reports}
    differences = {}
    # Paired bootstrap resamples rules, retaining the two tasks within each rule.
    for protocol in ('direct_cot', 'self_recall_cot'):
        for mod in ('T', 'V'):
            paired = [{r['id']:r for r in all_rows[a] if r['protocol'] == protocol and r.get('modality') == mod}
                      for a in ('base', 'trained')]
            assert set(paired[0]) == set(paired[1]) and len(paired[0]) == 100
            clusters = {}
            for key in paired[0]:
                b, t = paired[0][key], paired[1][key]
                clusters.setdefault(b['qid'], []).append(int(t['correct']) - int(b['correct']))
            values = [sum(v)/len(v) for _,v in sorted(clusters.items())]
            assert len(values) == 50
            rng = random.Random(42)
            boot = sorted(sum(rng.choice(values) for _ in values)/len(values) for _ in range(10000))
            differences[protocol+'_'+mod] = dict(trained_minus_base=sum(values)/len(values),
                cluster_bootstrap_95_interval=[boot[249],boot[9749]], rule_clusters=50)
    save(run.OUT / 'paired_rule_differences.json', differences)
    lines = ['# 扩大到每模态100题的CoT测试', '',
        '完成范围：50条已学规则，每条direct和diagnosis各1题；100道文字题及其100道配对原图题。基座和训练后模型各完成450次生成，共900次。', '',
        '训练后模型：Qwen3-VL-4B-Instruct，64规则的TEXT定义/原子事实SFT，seed42、3epoch最终checkpoint-3840。训练不含正式应用选择题。它不是此前只学习8条规则的atomic检查点，因此不能把两轮差异只解释为扩样。', '',
        '| 模型 | 作答条件 | 文字题 | 图片题 |',
        '|---|---|---:|---:|']
    labels = {'base':'未训练基座','trained':'训练后模型'}
    names = {'direct_cot':'直接解释后答题','self_recall_cot':'自己回忆后解释答题'}
    for a in ('base','trained'):
        for p in ('direct_cot','self_recall_cot'):
            t,v = [reports[a]['metrics'][p+'_'+m] for m in ('T','V')]
            assert t['n'] == v['n'] == 100
            lines.append(f"| {labels[a]} | {names[p]} | {t['correct']}/100（{t['accuracy']:.0%}） | {v['correct']}/100（{v['accuracy']:.0%}） |")
    lines += ['', '准确率分母固定为100，包含空输出、格式错误及截断。自我回忆仅输入规则名称，不提供正确规则；回忆内容未经修正，交回同一个模型。', '',
        '| 模型 | 条件/题目模态 | direct | diagnosis | 有效输出准确率 | 无效终答 | 截断 |',
        '|---|---|---:|---:|---:|---:|---:|']
    flat = []
    for a in ('base','trained'):
        for p in ('direct_cot','self_recall_cot'):
            for m in ('T','V'):
                r = reports[a]['metrics'][p+'_'+m]
                d, g = [r['by_task'][task] for task in ('direct','diagnosis')]
                lines.append(f"| {labels[a]} | {names[p]}/{'文字' if m=='T' else '图片'} | {d['correct']}/50（{d['correct']/50:.0%}） | {g['correct']}/50（{g['correct']/50:.0%}） | {r['valid_accuracy']:.1%} | {r['invalid_final']} | {r['truncated']} |")
                flat.append(dict(model=a,protocol=p,modality=m,n=r['n'],correct=r['correct'],
                    accuracy_percent=r['accuracy']*100,valid=r['valid'],
                    valid_accuracy_percent=r['valid_accuracy']*100,invalid=r['invalid_final'],
                    truncated=r['truncated'],direct_percent=d['correct']/50*100,
                    diagnosis_percent=g['correct']/50*100))
    with (run.OUT/'summary.csv').open('w') as h:
        writer=csv.DictWriter(h,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    lines += ['', '训练后相对基座的变化（按50条规则配对bootstrap，95%区间；单位：百分点）：']
    for key,r in differences.items():
        low,high=r['cluster_bootstrap_95_interval']
        lines.append(f"- {key}：{r['trained_minus_base']*100:+.1f}，区间[{low*100:+.1f}, {high*100:+.1f}]。")
    recall=run.read(run.OUT/'recall_manual_audit.json')
    errors=run.read(run.OUT/'text_error_manual_audit.json')
    if recall and errors:
        assert recall['correct']==48 and recall['n']==50
        lines += ['', '## 为什么能回忆，却仍答错', '',
            '逐条人工核查50次独立回忆：48/50（96%）的核心谓词、条件值和逻辑连接正确。两条混淆分别是Namtiv（把protective cover OR fabric记成mail sack OR paper）与Mosvat（把bucket OR plastic记成bottle OR metal）。这不是完整输出格式的正确率，也不意味着带题目时同样能稳定检索。', '',
            '例1：Cedvim独立回忆正确说出“bottle OR metal”，直接看题却说成“bottle made of plastic”，文字题选错B；先放回它自己的正确回忆后，逐项得到A、B、C保留，选对D。这里的差别发生在场景中的规则检索与使用，并非模型完全没记住知识。', '',
            '文字自我回忆后仍错12题，按最早可见的主要偏差归类：回忆错4题、规则应用错2题、报告核对错4题、选项匹配错2题。原始回答与逐条注释见text_error_manual_audit.json。类别可能同时出现，此表不声称解释模型内部计算。', '',
            '例2：Merves文字诊断题正确得出“只有B保留”，但原报告是“只有C保留”，模型却称二者相符，选择接受报告。属于报告核对失误。', '',
            '例3：Cedlis图片题在自我回忆后正确逐项判断“仅候选C保留”，却输出FINAL: C。正确选择是D，因为D选项才表示“仅C保留”。这是候选编号A/B/C/D与选项编号A/B/C/D的混淆；同一题文字版能选对D。Cedvim图片题也正确得到保留A、B、C，却把对应选项D误认成C。', '',
            '因此，扩大样本后可以确认：单独回忆的知识记忆已经较好，直接带题检索仍不稳定；回忆放回上下文显著改善文字应用，但跨模态图片题还受观察、应用与选项匹配影响。88%对55%的差距发生在“先自我回忆”的辅助条件，不能当成原始闭卷训练方案已修复，也不能归因于纯粹的跨模态迁移能力。']
    lines += ['', '## 可复查的原始回答', '',
        '完整原始生成及输出token保存在base/predictions.jsonl、trained/predictions.jsonl。自动抽样只按ID排序，包含答对、选错及格式错误，不能代替人工核查。']
    examples=[]
    for a in ('base','trained'):
        for p in ('direct_cot','self_recall_cot'):
            for m in ('T','V'):
                for category in ('correct','wrong_final','invalid_final'):
                    pool=[r for r in all_rows[a] if r['protocol']==p and r.get('modality')==m and
                          ('correct' if r['correct'] else 'invalid_final' if r['predicted'] is None else 'wrong_final')==category]
                    examples += [dict(category=category,**r) for r in sorted(pool,key=lambda r:r['id'])[:3]]
    ex.ev.save_rows(run.OUT/'explanation_samples.jsonl',examples)
    short=run.read(run.OUT/'matched_historical_short_report.json')
    if short:
        lines += ['', '## 历史短答案对照（不是本轮CoT）', '',
            '| 模型 | 同100题文字 | 同100题图片 |', '|---|---:|---:|']
        for a in ('base','trained'):
            t,v=[short['results'][a]['metrics'][m] for m in ('T','V')]
            lines.append(f"| {labels[a]} | {t['accuracy']:.0%} | {v['accuracy']:.0%} |")
        lines += ['', '这些旧短答案使用Swift/Transformers，本轮使用vLLM及训练适配器的独立BF16合并副本；因此相差多少不能全部归因于CoT。合并会有BF16舍入差异，没有重新训练。']
    lines += ['', '## 复现与边界', '',
        '本轮两组统一vLLM 0.11.0、temperature=0、seed42、连续批处理并发上限32；CoT最多3072tokens、要求200词内；回忆最多768tokens。采用Instruct普通模板、显式要求解释，不强制<think>。严格解析唯一末尾FINAL: A/B/C/D，或单独字母。', '',
        '原图、正式题、选项、标签保持不变；100道文字题的标签用独立规则执行器重新计算通过。模型/数据/图像哈希在manifest.json；合并审计逐tensor核对252处语言层更新，315个视觉tensor完全不变。engine_audit.json另行核查生成token解码、实际提示词及同模型自我回忆来源。', '',
        '100题来自50条规则，覆盖四种集合筛选结构，并非整个1000规则数据集。只评估TEXT学习后的T2T/T2V；回忆后作答额外携带同模型生成的规则文字，是辅助条件。解释是模型生成的可见文字，不能直接等同内部真实计算过程。', '',
        '此前Transformers慢速部分输出及失败兼容测试单独保留，不进入本报告。']
    (run.OUT/'report_zh.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='complete',metrics=flat),ensure_ascii=False))


if __name__ == '__main__':
    main()
