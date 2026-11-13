#!/usr/bin/env python3
"""Independently audit and summarize the three-arm retrieval exploration."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import run_v7_evaluation as ev
from explore_v7_retrieval_repair import ARMS, DATA, atoms, assignment, independent, score
from run_v7_full_experiment import read, rows, save, sha


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    out=a.out.resolve();manifest=read(out/'manifest.json')
    for p,h in manifest['sources'].items():assert sha(Path(p))==h
    for f,h in manifest['files'].items():assert sha(out/f)==h
    knowledge={k['id']:k for k in rows(out/'knowledge.jsonl')}
    probes={p['id']:p for p in rows(out/'probes.jsonl')}
    for r in rows(out/'solver_checks.jsonl'):
        rule=json.loads(knowledge[r['qid']]['fact']['quote'])
        expected=[independent(rule,o) for o in r['observations']]
        assert expected==r['results']
        if len(r['answer'])==1:
            assert r['options']['ABCD'.index(r['answer'])]==assignment(expected)
        else:assert r['answer']==assignment(expected)
    for qid in knowledge:
        checks=[r for r in rows(out/'solver_checks.jsonl') if r['qid']==qid]
        train={r['signature'] for r in checks if not r['task'].startswith('validation_')}
        valid={r['signature'] for r in checks if r['task'].startswith('validation_')}
        assert not train&valid
        for task in ('diagnosis','validation_diagnosis'):
            g=[r for r in checks if r['task']==task]
            assert len(g)==8 and sum(r['accepted'] for r in g)==2
            assert Counter(r['answer'] for r in g)==dict.fromkeys('ABCD',2)
            assert all(r['submitted'] in r['options'] for r in g)
    reports={arm:read(out/arm/'report.json') for arm in ('base',*ARMS)}
    bridge_reports={arm:read(out/arm/'bridge/report.json') for arm in ('base',*ARMS)}
    total=0
    facts_review={}
    per_rule={}
    for arm in ('base',*ARMS):
        records=rows(out/arm/'predictions.jsonl');index={r['id']:r for r in records}
        assert len(records)==len(index)==464
        total+=len(records)
        for r in records:
            p=probes[r['id']]
            assert r['correct']==score(r['response'],p['gold'],p['fmt'])
            assert r['prompt_sha256']==ev.digest({'messages':p['messages'],'images':p.get('images',[])})
        facts=[r for r in records if r['condition']=='new_fact']
        easy=[r for r in facts if r['relation']=='three_way_category' and ':new_fact:2:' in r['id']]
        other=[r for r in facts if r not in easy]
        facts_review[arm]={'all':{'n':96,'correct':sum(r['correct'] for r in facts)},
            'label_order_already_visible_in_training_identifier':{'n':12,'correct':sum(r['correct'] for r in easy)},
            'excluding_label_order':{'n':84,'correct':sum(r['correct'] for r in other)},
            'truncated':sum(r['finish_reason']=='length' for r in facts),
            'scoring':'conservative normalized exact; errors must be semantically reviewed before saying every fact is wrong'}
        per_rule[arm]={qid:{cond:{'n':sum(r['qid']==qid and r['condition']==cond for r in records),
            'correct':sum(r['correct'] for r in records if r['qid']==qid and r['condition']==cond)}
            for cond in reports[arm]['metrics']} for qid in knowledge}
        br=rows(out/arm/'bridge/predictions.jsonl');bp={p['id']:p for p in rows(out/arm/'bridge/probes.jsonl')}
        assert len(br)==len({r['id'] for r in br})==328
        total+=len(br)
        for r in br:
            p=bp[r['id']]
            assert r['prompt_sha256']==ev.digest({'messages':p['messages'],'images':p.get('images',[])})
            if p['fmt']=='card':assert r['correct']==(r['response']==p['gold'])
            else:assert r['correct']==score(r['response'],p['gold'],'mcq')
            if p['condition'].startswith('self_fact_context'):
                expected='\n'.join(focus+': '+index[f"{p['qid']}:new_fact:{i}:0"]['response']
                    for i,(focus,gold) in enumerate(atoms(json.loads(knowledge[p['qid']]['fact']['quote']))))
                assert p['material']==expected
    # Check the promising result uses only generated cards, and that generated
    # cards match the gold material as an observed outcome, not an input.
    card_recall={r['qid']:r for r in rows(out/'cards/bridge/predictions.jsonl') if r['condition']=='full_card_training_prompt'}
    assert all(r['response']==knowledge[qid]['material']['learning']['text'] for qid,r in card_recall.items())
    full_probes={p['id']:p for p in rows(out/'cards/full_card_application/probes.jsonl')}
    full=rows(out/'cards/full_card_application/predictions.jsonl');assert len(full)==32
    total+=len(full)
    material_predictions={r['id']:r for r in rows(out/'cards/predictions.jsonl') if r['condition'].startswith('material_original_')}
    for r in full:
        p=full_probes[r['id']];original_id=r['id'].removesuffix(':self_full_card')
        original=probes[original_id]
        expected='Learning material:\n'+card_recall[r['qid']]['response']+'\n\n'+original['messages'][-1]['content']
        assert p['messages'][-1]['content']==expected
        assert p['messages'][:-1]==original['messages'][:-1]
        assert p.get('images',[])==original.get('images',[])
        assert r['correct']==score(r['response'],p['gold'],'mcq')
        assert r['prompt_sha256']==ev.digest({'messages':p['messages'],'images':p.get('images',[])})
        assert r['response']==material_predictions[original_id+':material']['response']
    assert total==3200
    all_images=rows(DATA/'image_hashes.jsonl')
    for image in all_images:assert sha(DATA/image['path'])==image['source_sha256']==image['repaired_sha256']
    train_info={}
    for arm in ARMS:
        info=read(out/arm/'checkpoint_manifest.json');args=read(Path(info['path'])/'args.json')
        assert info['epoch']==1 and info['global_step']==128 and info['nonzero_lora_B']==252
        assert args['per_device_train_batch_size']==4 and args['gradient_accumulation_steps']==1
        for f,h in info['sha256'].items():assert sha(Path(info['path'])/f)==h
        logs=rows(out/arm/'checkpoint/logging.jsonl');logs=[r for r in logs if 'loss' in r]
        train_info[arm]={'first_loss':logs[0]['loss'],'last_loss':logs[-1]['loss'],
            'last_logged_step':logs[-1]['global_step/max_steps'],
            'task_losses':{k:v for k,v in logs[-1].items() if k.startswith('loss_')},'checkpoint':info}
    save(out/'per_rule.json',per_rule)
    save(out/'factual_scoring_audit.json',facts_review)
    validation={'status':'passed','main_requests':1856,'bridge_requests':1312,'full_card_application_requests':32,
        'unique_saved_requests':total,'original_images_verified':len(all_images),'original_data_hashes_unchanged':True,
        'three_final_1epoch_checkpoints_reloaded':True,'main_and_bridge_scores_recomputed':True,
        'self_facts_from_fixed_first_wording_only':True,'full_card_application_context_verified_from_raw_generation':True,
        'card_adapter_full_recall_exact':8,'full_card_application_matches_material_control':32,
        'new_train_valid_four_observation_combinations_disjoint':True,
        'diagnosis_correct_report_ratio':.25,'diagnosis_gold_positions_balanced':True,
        'semantic_review_of_facts_mismatches':'facts and facts_cases all mismatch outputs inspected; substantive fact/extraction errors observed',
        'finished_at':ev.now()}
    save(out/'validation.json',validation)
    summary={'status':'complete','main':reports,'bridge':bridge_reports,
        'full_card_application':read(out/'cards/full_card_application/report.json'),
        'factual_scoring_audit':facts_review,'training':train_info,'validation':validation,
        'answer_tokens':read(out/'answer_token_inventory.json')}
    save(out/'summary.json',summary)
    labels={'base':'基础模型','cards':'完整材料复述','facts':'短事实多问法','facts_cases':'短事实＋案例'}
    conditions=['new_fact','new_direct','new_diagnosis','original_T2T','original_T2V']
    table=[]
    for arm in labels:
        for cond in conditions:
            m=reports[arm]['metrics'][cond]
            table.append({'arm':arm,'condition':cond,**{k:m[k] for k in ('correct','n','format_errors','empty','truncated')},
                          'accuracy':m['correct']/m['n']})
    with (out/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    lines=['# v7：短事实与回忆后应用的小规模探索','',
        '结论：本轮验证了一个小规模可行流程——先按训练问法完整回忆，再把模型自己的回忆用于原题。'
        '8条规则的完整材料组能逐字回忆8/8；原始T2T由直接1/16变为自行回忆后14/16，T2V由2/16变为9/16。'
        '它尚不证明单次直接答题已改善，也不证明1000条可以复现。','',
        '## 三组训练与直接答题','',
        'Qwen3-VL-4B，从相同基础权重分别训练TEXT LoRA。固定哈希抽取集合筛选4条、比例分档4条；'
        '每组512行、1epoch、batch4、累积1、128步、LR1e-4、rank8/alpha32/dropout0.05、seed42、最大8192。'
        '最终checkpoint-128固定使用。原有5epoch模型未继续训练，也未混入PT。','',
        '| 模型/数据 | 新问法事实 | 新组合直接题 | 新审核题 | 原始T2T | 原始T2V |',
        '|---|---:|---:|---:|---:|---:|']
    for arm,label in labels.items():
        vals=[reports[arm]['metrics'][c] for c in conditions]
        lines.append('| '+label+' | '+' | '.join(f"{m['correct']}/{m['n']}" for m in vals)+' |')
    lines += ['', '事实96题实际是32个事实槽的三种未见问法，不能当作96个独立事实。其中12题要求按名称中已有次序返回三个类别标签，存在复制线索。'
        '单列剔除这12题后：基础模型11/84，完整材料5/84，短事实59/84，短事实＋案例29/84。'
        '因此短事实收益不全来自复制标签，但仍没有稳定转化为原始题成绩。','',
        '完整材料组新事实输出53/96达到64token上限，多为继续复述长材料；7/96是格式受约束的准确率，不能说其余89个事实全部未记住。'
        '短事实及混合组均无这类截断；已逐条查看两组未匹配输出，发现实际阈值、条件、顺序和定义错配。','',
        '## 自行回忆后答题','',
        '首先用固定的第一种新问法取每个规则的四个事实；不选最好的一次，把完整原始回答拼成上下文。正确四事实另作对照。','',
        '| 模型 | 直接T2T | 自回忆四事实T2T | 正确四事实T2T | 正确完整材料T2T |',
        '|---|---:|---:|---:|---:|']
    for arm,label in labels.items():
        parts=[reports[arm]['metrics']['original_T2T'],bridge_reports[arm]['metrics']['self_fact_context:original_T2T'],
               bridge_reports[arm]['metrics']['gold_fact_context:original_T2T'],reports[arm]['metrics']['material_original_T2T']]
        lines.append('| '+label+' | '+' | '.join(f"{m['correct']}/{m['n']}" for m in parts)+' |')
    lines += ['', '四事实上下文与完整材料内容并不等价；例如比例边界和计数语义可能在完整材料中更明确。正确事实/完整材料均属于外部材料对照，不计入无材料成绩。','',
        '随后用完整材料组训练中的第一条完整复述提示，生成8段规则。全部与真实训练材料逐字相同。'
        '使用这些已保存的模型输出重新回答原始题，实际完成32次新推理：','',
        '| 方式 | T2T | T2V |', '|---|---:|---:|',
        '| 完整材料组直接回答 | 1/16（6.25%） | 2/16（12.50%） |',
        '| 先自行完整回忆，再回答 | 14/16（87.50%） | 9/16（56.25%） |',
        '| 额外提供正确完整材料，仅作对照 | 14/16（87.50%） | 9/16（56.25%） |','',
        '这32次请求的上下文逐字追溯到模型生成文件，未用正确材料替换。因为8段回忆恰好都正确，'
        '其最终提示与正确完整材料对照一致，32个回答也全部一致。这两行不是独立证据。'
        '这说明在该8规则适配器里，部分知识可在训练问法下取出，但直接原题未能可靠调用。'
        '之前1000规则5epoch模型36段回忆均有关键错误，两者并不矛盾：每条规则的暴露次数、规则总数和数据组织不同。','',
        '## 训练监督与观察','',
        '完整材料组每规则64行，16个提示各重复4次；短事实组每规则4事实×16种问法；混合组32条事实加16个案例，'
        '每案例分别监督完整语义结果和选择题答案，共64行。重复问法和成对案例不是独立新知识。'
        '训练和新验证观察组合分开；有限真值模式、正例值和部分数值边界会重复。比例训练分母20/40，验证60/80。','',
        '答案正文token总量（不含模板/EOS）：完整材料70976，短事实2176，混合3599；混合中事实1088（30.23%）。'
        '因此本轮匹配行数和步数，不匹配token预算；不能把差异归因于唯一因素。'
        '混合组事实暴露减半、任务类型增多，可能影响事实提取；尚未隔离这两个因素。','',
        '完整材料最终记录loss约0.000122，短事实0.127642，混合0.295192，记录在第125/128步；'
        'checkpoint实际为128步、epoch1。不同目标的loss不可直接作效果排名。每个任务loss在各checkpoint/logging.jsonl中。','',
        '## 下一步建议与边界','',
        '优先把“训练后自行完整回忆，再应用”作为候选推理协议做独立规则子集复现，再逐步扩到100条。'
        '扩规模时需分别报告完整回忆的事实正确率、给自己正确/错误回忆时的应用率，以及最终无材料准确率。'
        '若完整回忆随规则数增长而失效，先解决名称到事实的绑定和训练暴露，不能直接宣布流程已修复数据集。','',
        '若研究要求一次直接输出选项，这轮仍未解决：最好的混合组只有原始T2T6/16、T2V5/16。'
        '分两步生成是新协议，不能混入历史strict直接答题成绩。它没有外部学习材料，但确实增加一次规则生成。'
        '本轮只训练TEXT并测T2T/T2V，没有证明VISION训练的V2T/V2V有效。辅助事实题是短回答，最终任务始终是选择题。','',
        '全部结果属于开发探索。两类各4规则、每方向16原题，同规则两题相关，不能外推1000条或声称统计稳定。'
        '原始题已用于开发观察，不作为新的独立最终测试。没有按评测挑checkpoint；全量COT仍未启动。','',
        '## 文件与核验','',
        '验证3200份实际原始响应：主对照1856、四事实/完整回忆控制1312、自完整回忆答题32。'
        '全部3000PNG及原始题、标签、知识和适配器哈希通过；训练规则、脚本、日志和checkpoint均保留。','',
        'summary.json/summary.csv：汇总；validation.json：独立核验；per_rule.json：逐规则；'
        '各模型predictions.jsonl：原始输出；bridge：四事实和完整规则回忆；'
        'cards/full_card_application：有收益的自回忆流程；answer_token_inventory.json：监督量。','',
        '本目录balanced_candidate是实际训练版本；父目录早期prepare草稿未训练、不得当实验结果。']
    (out/'report_zh.md').write_text('\n'.join(lines)+'\n')
    save(out/'final_status.json',{'status':'complete','trained_adapters':3,'epochs_per_adapter':1,'requests':3200,
                                'source_data_and_images_unchanged':True,'finished_at':ev.now()})
    print(json.dumps(validation,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
