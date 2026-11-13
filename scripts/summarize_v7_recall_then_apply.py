#!/usr/bin/env python3
"""Attach explicit factual review and independently verify the saved diagnostic."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import run_v7_evaluation as ev
from run_v7_full_experiment import ROOT, read, rows, save, sha
from diagnose_v7_recall_then_apply import application_messages


# Explicit review judgments of the saved 2026-10-15 outputs, not an automatic
# semantic grader. The response hash binds each judgment to the reviewed text.
RECALLED_FAMILIES = {
    '0310': 'scalar_index_rule', '0202': 'scalar_index_rule', '0209': 'scalar_index_rule',
    '0355': 'scalar_index_rule', '0116': 'scalar_index_rule', '0011': 'scalar_index_rule',
    '0432': 'ordering_priority', '0124': 'scalar_index_rule',
    '0223': 'mapping_codebook', '0224': 'three_way_category', '0897': 'mapping_codebook',
    '0891': 'mapping_codebook', '0480': 'mapping_codebook', '0133': 'mapping_codebook',
    '0014': 'mapping_codebook', '0454': 'mapping_codebook',
    '0139': 'scalar_index_rule', '0528': 'scalar_index_rule', '0074': 'scalar_index_rule',
    '0249': 'scalar_index_rule', '0597': 'ordering_priority', '0601': 'scalar_index_rule',
    '0947': 'scalar_index_rule', '0160': 'ordering_priority',
    '0958': 'ordering_priority', '0086': 'scalar_index_rule', '0654': 'ordering_priority',
    '0084': 'ordering_priority', '0974': 'three_way_category', '0730': 'mapping_codebook',
    '0283': 'three_way_category', '0740': 'three_way_category',
    '0198': 'scalar_index_rule', '0097': 'scalar_index_rule', '0995': 'scalar_index_rule',
    '0194': 'scalar_index_rule',
}
NOTES = {
    '0310': '应为同一批全是protective glove且数量>=6，并有四种记录代码；却回答圆/三角/正方形加权计分。',
    '0202': '应为同一批全是circle且数量>=4，并有四种记录代码；却回答storage basket/protective wrap/sample jar加权计分。',
    '0209': '应为square形状或yellow颜色，并有四种记录代码；却回答形状计分。',
    '0355': '应为toolbox或wood材料，并有四种记录代码；却回答物品计分。',
    '0480': '应为tray/utility bag和case/sample jar的四个有序对映射；回忆改成protective cover等四个单物体映射。',
    '0133': '保留有序对形式，但bottle crate/insulated box和cooler bag/sealed container全部被换成其他物体。',
    '0014': '应将fabric roll/sample card/storage bin/tool case映射到四标签；却编成protective cover/sample jar等有序对。',
    '0454': '应将harvest bag/watering can/spray bottle/basket映射到四标签；却编成storage crate等有序对。',
    '0601': '应为4*helmet_count-2*cable_reel_count+3，整数；却回答0.5/1.5/0.25形状权重、偏置2和两位小数。',
    '0947': '应为5*square_count+2*pentagon_count+2，整数；却回答0.5/1.5/0.25形状权重和两位小数。',
    '0974': '应为有蓝圆的方框占比，阈值0.4/0.65；却改成蓝色检验封条占比，阈值0.15/0.6。',
    '0283': '应为有蓝圆的方框占比，阈值0.25/0.7；却改成有完整物体的托盘区域，阈值0.45/0.6。',
    '0740': '占比含义大致保留，但正确阈值0.2/0.8被改成0.15/0.6。',
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    out = a.out.resolve()
    manifest = read(out / 'manifest.json')['config']
    raw = rows(out / 'predictions.jsonl')
    recalls = rows(out / 'recall_probes.jsonl')
    probes = rows(out / 'application_probes.jsonl')
    applications = rows(out / 'application_inputs.jsonl')
    refs = {p['qid']: p for p in recalls}
    recall_index = {(r['mode'], r['qid']): r for r in raw if r['stage'] == 'recall'}
    raw_index = {(r['mode'], r['stage'], r['protocol'], r['id']): r for r in raw}
    input_index = {(p['mode'], p['protocol'], p['id']): p for p in applications}
    assert len(raw) == len(raw_index) == 504
    assert len(input_index) == len(applications) == 432
    assert len(recalls) == 36 and len(probes) == 72
    for p in applications:
        original = next(q for q in probes if q['id'] == p['id'])
        material = recall_index[p['mode'], p['qid']]['response']
        expected = application_messages(original, p['protocol'], material, refs[p['qid']]['reference'])
        assert p['messages'] == expected
        r = raw_index[p['mode'], 'application', p['protocol'], p['id']]
        assert r['prompt_sha256'] == ev.digest(expected)
        assert r['predicted'] == ev.letter(r['response'])
        assert r['correct'] == (r['predicted'] == p['gold'])
    for r in [r for r in raw if r['stage'] == 'recall']:
        assert r['prompt_sha256'] == ev.digest(refs[r['qid']]['messages'])
    for path, expected in manifest['sources'].items():
        assert sha(Path(path)) == expected
    for filename, expected in manifest['adapter']['sha256'].items():
        assert sha(Path(manifest['adapter']['path']) / filename) == expected

    reviewed = []
    for p in recalls:
        r = recall_index['adapter', p['qid']]
        short = p['qid'].split('_')[1]
        family = RECALLED_FAMILIES[short]
        category = 'wrong_rule_family' if family != p['relation'] else 'wrong_core_facts'
        reviewed.append({'qid': p['qid'], 'mode': 'adapter', 'review_method': 'agent_manual_semantic_review',
            'expected_family': p['relation'], 'recalled_family': family,
            'fully_correct_rule': False, 'failure_category': category,
            'notes_zh': NOTES.get(short, f"应为{p['relation']}，却解释成{family}，原规则关键条件或操作未复述。"),
            'response_sha256': ev.digest(r['response']), 'reference_sha256': ev.digest(p['reference']),
            'reference': p['reference'], 'response': r['response']})
    assert len(reviewed) == len(RECALLED_FAMILIES) == 36
    ev.save_rows(out / 'factual_review.jsonl', reviewed)
    counts = Counter(r['failure_category'] for r in reviewed)
    assert counts == {'wrong_rule_family': 27, 'wrong_core_facts': 9}
    single = rows(out / 'single_request_verification/predictions.jsonl')
    assert len(single) == 4 and all(r['identical_prompt_to_batch8'] for r in single)
    single_review = []
    for r in single:
        # All four raw responses separately inspected. Kimcam changes priority
        # details but still invents an ordering task in place of selection.
        single_review.append({'qid': r['qid'], 'fully_correct_rule': False,
            'response_sha256': ev.digest(r['response']), 'review_method': 'agent_manual_semantic_review',
            'notes_zh': '同样存在规则家族或关键数值错误；Kimcam单条变成另一套排序规则，仍不是筛选规则。'
                         if r['qid'] == 'scale1k_0084_programmatic' else '与批量回忆逐字相同，原有事实错误复现。'})
    ev.save_rows(out / 'single_request_verification/factual_review.jsonl', single_review)

    accept_counts = {}
    for mode in ('base', 'adapter'):
        for protocol in ('direct', 'self_recalled_material', 'correct_material'):
            g = [r for r in raw if r['stage'] == 'application' and r['mode'] == mode
                 and r['protocol'] == protocol and r['question_type'] == 'diagnosis']
            predicted, gold = 0, 0
            for r in g:
                text = input_index[mode, protocol, r['id']]['messages'][-1]['content']
                options = dict(re.findall(r'^([A-D])\. (.*)$', text, re.M))
                assert len(options) == 4
                predicted += options[r['predicted']].startswith('Accept report.')
                gold += options[r['gold']].startswith('Accept report.')
            accept_counts[mode + ':' + protocol] = {'n': len(g), 'predicted_accept': predicted, 'gold_accept': gold}

    # Independently calculate two illustrative material-given failures.
    assert [2 + 3*3 + 1, 0 + 4*3 + 1, 2 + 2*3 + 1, 0 + 3*3 + 1] == [12, 13, 9, 10]
    assert 3 + 4*3 + 1 == 16
    silrov_direct = raw_index['adapter', 'application', 'correct_material',
                              'scale1k_0160_ai_generated-scale1k_0160_ai_generated_direct']
    silrov_audit = raw_index['adapter', 'application', 'correct_material',
                             'scale1k_0160_ai_generated-scale1k_0160_ai_generated_diagnosis']
    assert (silrov_direct['gold'], silrov_direct['predicted']) == ('C', 'B')
    assert (silrov_audit['gold'], silrov_audit['predicted']) == ('C', 'D')
    validation = {'status': 'passed', 'unique_main_requests': 504, 'additional_single_requests': 4,
        'total_saved_responses': 508, 'rule_counts_by_family': dict(Counter(p['relation'] for p in recalls)),
        'question_counts_by_type': dict(Counter(p['question_type'] for p in probes)),
        'all_432_application_prompts_verified': True, 'all_432_scores_independently_recalculated': True,
        'recall_uses_only_original_training_user_prompt': True, 'self_material_from_same_model_only': True,
        'tracked_sources_and_adapter_hashes_unchanged': True, 'new_training': False,
        'reviewed_adapter_recalls': 36, 'full_rule_correct': 0, 'failure_categories': dict(counts),
        'diagnosis_accept_counts': accept_counts, 'finished_at': ev.now()}
    save(out / 'validation.json', validation)
    report = read(out / 'report.json')
    report['recall']['adapter']['semantic_factual_review'] = 'complete:36/36 have key factual errors,27 wrong family,9 wrong core facts'
    report['recall']['adapter']['fully_correct_rules'] = 0
    report['recall']['adapter']['failure_categories'] = dict(counts)
    report['recall']['base']['semantic_factual_review'] = 'not exhaustively graded; not reported as factual accuracy'
    report['factual_review_file'] = str(out / 'factual_review.jsonl')
    report['single_request_verification'] = read(out / 'single_request_verification/report.json')
    report['diagnosis_accept_counts'] = accept_counts
    save(out / 'report.json', report)
    prefix = (out / 'report_zh.md').read_text().split('同一规则的两题有关联')[0]
    extra = '''
## 回忆内容的逐条事实核验

对训练后模型的36段回忆逐条比对结构化规则和真实训练答案，0/36完整正确：27条连规则家族都错了，另外9条家族相近但映射、对象、权重或阈值错误。这是语义核验，不是用字符串不相同判错，也不是说每一个事实片段都错。详见factual_review.jsonl。

例如：Narmir应判断“同一批全是防护手套且数量至少6个”，并依两个条件记录代码。模型回忆成“圆、三角、方形分别按0.5、1.5、0.25计分，再加2”。Somdim/Devtol/Vadrev应按有蓝圆的方框占比、阈值0.4和0.65分档，模型改成蓝色检验封条占比、阈值0.15和0.6。36段中18段回忆为加权计分内容。

这些例子说明本批完整5epoch模型未能可靠把规则名绑定到对应事实。不能用“只是没有把记住的规则迁移到选择题”解释全部失败。也不能据此认定所有1000条都没学会，或把此前8规则强化试验能复述7/8的结果覆盖掉；两者模型和训练数据不同。

## 排除批量推理和适配器切换的局部复测

另起新模型实例，只加载适配器，不做base/adapter切换，batch=1原提示复测4个已发现错误。4/4依然有关键事实错误，其中3/4回答逐字相同，Kimcam改成另一套错误排序规则。由于这是按已发现错误挑选的案例，只用于核查现象是否复现，不能当独立准确率样本。无截断。

## 提供正确材料之后仍然存在的障碍

训练后模型给正确材料38/72，较直接18/72增加20题净正确数。逐题为修复23题、破坏3题；因此材料可用性有明显影响，但不是唯一障碍。

同样给正确材料时，集合筛选为8/8，而计分为2/8、变换程序为1/8。每类只有8题，不做总体推断。Silrov应为storage_box_count + 3*supply_tray_count + 1；四个集合实际分数12、13、9、10，标签C与独立计算一致，模型选B。另一题3个storage box和4个supply tray应得16，模型却选“接受报告29”。这两个例子确认给材料后仍有真实的应用/报告核验错误，不是这两个标签算错。

36道diagnosis中正确报告有10条，训练后模型直接接受35条，给正确材料仍接受27条；基础模型给材料接受15条。这里有明显的接受报告倾向，但不证明全部错误都由该倾向造成。

## 对后续修复的约束

1. 优先补规则名到具体事实的提取训练：规则类型、每项条件/运算符、阈值和边界、完整映射。对同一事实用多种自然问法，开发验证使用未见措辞。
2. 将长篇完整规则复述与短事实提取分开监控，按任务记录loss和准确率。控制任务采样或答案权重，验证是否真正改善事实绑定；低总体loss不能替代这一检查。
3. 再增加由结构化规则计算的新观察应用题和报告核验题，平衡正确/错误报告。原始评测题和答案不得加入训练，图片保持原样。
4. 下一轮只做8到12条规则的1epoch对照，并预先固定开发验证；必须同时看到事实回忆和原始题的可重复收益再扩到100/1000。PT保留原始材料式基线，添加问答后的方案另列为增强SFT。

本次仅做诊断，没有启动新训练，也未改数据或图片。该流程尝试了“先回忆再应用”，当前没有带来提升；不能把它报告成有效修复。

## 范围和复现

主诊断504次请求，单条复测4次，合计508份原始响应。432次最终选择题均无空输出、格式错误或截断；基础模型规则回忆31/36截断，训练后回忆0/36截断。基础模型回忆截断不影响已经逐条确认的训练后事实错误，但限制对基础模型长篇回忆的解释。

同一规则两题有关联；这只是Qwen3 SFT TEXT最终5epoch模型的开发诊断，不能外推全量、其他模型、PT或VISION。使用自己的回忆会增加模型自行生成的上下文和一次生成步骤，与原始直接盲测协议分开。提供正确材料的结果属于外部材料控制，不能作为最终无材料成绩。未开启原生Thinking或全量COT。

完整输入：application_inputs.jsonl；完整响应：predictions.jsonl；回忆逐条对照：recall_review_zh.md；语义核验：factual_review.jsonl；校验：validation.json；单条复测：single_request_verification。
'''
    (out / 'report_zh.md').write_text(prefix + extra)
    save(out / 'status.json', {'status': 'complete', 'saved': 504, 'additional_requests': 4,
                              'factual_review': 'complete_for_adapter', 'updated_at': ev.now()})
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
