#!/usr/bin/env python3
"""Verify new-rule replication, original tasks and balanced paired diagnostics."""
import copy
import csv
import json
from collections import Counter
from pathlib import Path

import confirm_v7_rule_recognition as pilot
import confirm_v7_paired_application as paired
import run_v7_evaluation as ev
from run_v7_full_experiment import ROOT,MODELS,rows,read,save,sha

OUT=pilot.OUT
ARMS=('warmup',*pilot.ARMS)


def audit():
    pilot.validate(OUT);paired.validate()
    kk={k['id']:k for k in rows(OUT/'target_knowledge.jsonl')}
    sel=read(OUT/'selection.json');assert not set(kk)&set(sel['excluded_ids'])
    for path,r in sel['prior_file_inventory'].items():assert sha(Path(path))==r['sha256']
    for arm in pilot.ARMS:
        tr=rows(OUT/(arm+'.jsonl'));assert len(tr)==512
        modality,style=arm.split('_')
        for qid in kk:
            rr=[r for r in tr if r['qid']==qid];assert len(rr)==64
            expected={'explain':64} if style=='explain' else {'match':64} if style=='match' else {'explain':32,'match':32}
            assert Counter(r['channel'] for r in rr)==Counter(expected)
        for r in tr:
            assert len(r['messages'])==3 and r['messages'][0]['role']=='system'
            assert len(r.get('images',[]))==(modality=='vision')
            if modality=='vision':assert r['images']==[str(pilot.DATA/'images'/r['qid']/'learning_visual.png')]
            assert 'Test input:' not in str(r) and 'test_visual' not in str(r)
            if r['channel']=='explain':assert r['messages'][-1]['content']==pilot.card(kk[r['qid']])
            else:
                assert r['messages'][-1]['content']==r['gold']
                assert r['option_rule_ids']['ABCD'.index(r['gold'])]==r['qid']
                for l,q in zip('ABCD',r['option_rule_ids']):
                    assert f'{l}. '+pilot.prior.definition(kk[q]) in r['messages'][-2]['content']
        st=read(OUT/arm/'checkpoint_manifest.json');assert st['global_step']==128
        assert ev.checkpoint(Path(st['path']))['sha256']==st['sha256']
    pp=rows(OUT/'probes.jsonl');pairpp=rows(paired.OUT/'probes.jsonl');responses={}
    for arm in ARMS:
        folder=OUT/arm;rr=rows(folder/'predictions.jsonl');assert len(rr)==440
        extra=rows(folder/'self_probes.jsonl');assert len(extra)==128
        lookup={p['id']:p for p in pp+extra};assert len(lookup)==440 and set(lookup)=={r['id'] for r in rr}
        recall={r['qid']:r['response'] for r in rr if r['condition']=='recall'};base={p['id']:p for p in pp}
        for p in extra:
            old=base[p['id'].removeprefix('self:')];text=recall[p['qid']]
            assert p['messages'][-1]['content']=='Your own prior rule recall (uncorrected):\n'+text+'\n\n'+old['messages'][-1]['content']
            assert p['recall_response_sha256']==ev.digest(text)
        pr=rows(paired.OUT/arm/'predictions.jsonl');assert len(pr)==256
        plook={p['id']:p for p in pairpp};assert set(plook)=={r['id'] for r in pr}
        for r in rr+pr:
            p=(lookup if r['id'] in lookup else plook)[r['id']]
            assert r['input_sha256']==ev.digest({'messages':p['messages'],'images':p.get('images',[])})
            assert r['response']==(r['raw_response']['choices'][0]['message']['content'] or '')
            if r['fmt']=='mcq':
                pred=ev.letter(r['response']);assert pred==r['predicted']
                assert r['correct']==(pred==p['gold']) and r['format_error']==(pred is None)
        assert read(folder/'eval_state.json')['status']=='complete'
        assert read(paired.OUT/arm/'eval_state.json')['status']=='complete'
        responses[arm]=rr+pr
    # Independently recompute all original gold results and all counterfactual keys.
    for case in rows(OUT/'original_case_audit.jsonl'):
        rule=json.loads(kk[case['qid']]['fact']['quote'])
        assert pilot.assignment([pilot.independent(rule,o) for o in case['observations']])==case['expected']
    for block in rows(paired.OUT/'blocks.jsonl'):
        result={q:pilot.assignment([pilot.independent(json.loads(kk[q]['fact']['quote']),o) for o in block['observation']])
                for q in block['rule_ids']}
        assert result==block['rule_results'] and len(set(result.values()))==4
    for reference in ('model_hash_validation.json','original_data_hash_validation.json'):
        old=read(ROOT/'runs/v7_rule_recognition_20261016'/reference)
        for r in old['checked_files']:assert sha(Path(r['path']))==r['sha256']
        save(OUT/reference,{'status':'pass','reference':str(ROOT/'runs/v7_rule_recognition_20261016'/reference),
                             'checked_files':old['checked_files']})
    reviews=rows(OUT/'semantic_reviews.jsonl')
    assert len(reviews)==168 and len({(r['arm'],r['id']) for r in reviews})==168
    assert {(r['arm'],r['id']) for r in reviews}=={(a,r['id']) for a,rr in responses.items() for r in rr if r['fmt']=='card'}
    for r in reviews:
        actual=next(x for x in responses[r['arm']] if x['id']==r['id'])
        assert r['response_sha256']==ev.digest(actual['response'])
        assert isinstance(r['correct'],bool) and r['response']==actual['response']
        assert r['qid']==actual['qid'] and r['condition']==actual['condition']
    save(OUT/'completion_audit.json',{'status':'pass','new_training_jobs':6,'new_responses':4872,
        'original_protocol_responses':3080,'paired_protocol_responses':1792,'manual_recall_reviews':168,
        'all_original3000PNG_unchanged':True,'original_dataset_and_base_model_unchanged':True,
        'training_labels_checked':3072,'new_rule_ids_disjoint_from_indexed_development':True,
        'raw_replies_and_inference_inputs_verified':True,'final_epoch1_only':True,
        'evaluation_gold_never_in_model_messages':True})
    return responses,reviews


def summarize():
    responses,reviews=audit();summary={};table=[];dirs=[]
    for arm,rr in responses.items():
        metrics={}
        for c in sorted({r['condition'] for r in rr if r['fmt']=='mcq'}):
            group=[r for r in rr if r['condition']==c];metrics[c]=pilot.metric(group)
            if c.startswith(('named_','self_','oracle_','paired_')):
                metrics[c]['by_rule']={q:pilot.metric([r for r in group if r['qid']==q]) for q in pilot.EXPECTED}
                metrics[c]['by_task']={t:pilot.metric([r for r in group if r['task']==t]) for t in ('direct','diagnosis')}
                def evaluation_group(r):
                    return r.get('evaluation_group',{'direct':'result_application','diagnosis':'rule_report_audit'}[r['task']])
                metrics[c]['by_evaluation_group']={t:pilot.metric([r for r in group if evaluation_group(r)==t])
                    for t in sorted({evaluation_group(r) for r in group})}
                metrics[c]['by_group']={str(g):pilot.metric([r for r in group if r['group']==g]) for g in (0,1)}
                metrics[c]['by_shift']={str(s):pilot.metric([r for r in group if r['shift']==s]) for s in range(4)}
                if not c.startswith('paired_'):metrics[c]['original_order']=pilot.metric([r for r in group if r['shift']==0])
                else:
                    blocks=[[r for r in group if (r['block'],r['shift'])==key] for key in sorted({(r['block'],r['shift']) for r in group})]
                    assert len(blocks)==32 and all(len(g)==4 and {r['gold'] for r in g}==set('ABCD') for g in blocks)
                    metrics[c]['paired_checks']={'block_times_position':32,'all4_same_answer':sum(len({r['predicted'] for r in g})==1 for g in blocks),
                        'all4_correct':sum(all(r['correct'] for r in g) for g in blocks)}
            table.append({'arm':arm,'condition':c,**{k:metrics[c][k] for k in ('n','correct','accuracy','format_errors','empty','truncated')}})
        metrics['recall']={c:{'correct':sum(r['correct'] for r in reviews if r['arm']==arm and r['condition']==c),'n':8}
            for c in ('recall','recall_image','plain_recall')}
        summary[arm]=metrics
        if arm!='warmup':
            for d in ('T2T','T2V','V2T','V2V'):
                dirs.append({'arm':arm,'direction':d,'matches_training_source':d[0]==('V' if arm.startswith('vision') else 'T'),
                    'original_v7':metrics['named_'+d[-1]],'paired_rebuilt':metrics['paired_'+d[-1]]})
    save(OUT/'summary.json',summary);save(OUT/'four_directions.json',dirs)
    with (OUT/'comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    def score(a,c,original=False):
        r=summary[a][c]['original_order'] if original else summary[a][c]
        return f"{r['correct']}/{r['n']}（{r['accuracy']:.1%}）"
    titles={'warmup':'共同起点','text_explain':'TEXT解释','text_match':'TEXT辨认','text_mixed':'TEXT混合',
        'vision_explain':'VISION解释','vision_match':'VISION辨认','vision_mixed':'VISION混合'}
    lines=['# 新规则复现：解释、辨认、混合监督','',
        '本轮固定另一组8条规则：Dostan、Morkil、Ratlod、Lavrit、Litkal、Saldal、Levsim、Sovned。与已索引的小实验训练/知识/探针规则及共同warmup训练规则分开。不是新采集数据，历史1000条全量模型曾使用这套材料；本轮未从那些模型初始化。','',
        '## 同图同选项、只换名称的配对应用','',
        '这部分沿用上一轮构造方式，固定新规则后从它们原16张测试图中保留4张具有四种不同结果的图；每图构造direct和diagnosis。每输入模态32道语义题，轮换4个选项位置得到128次回答。主输入不给学习材料。','',
        '| 训练 | 文字应用 | 图片应用 | 无图辨认改写定义 | 无图完整回忆 |','|---|---:|---:|---:|---:|']
    for a in ARMS:
        lines.append('| '+titles[a]+' | '+' | '.join(score(a,c) for c in ('paired_T','paired_V','recognition_para_noimage'))+f" | {summary[a]['recall']['recall']['correct']}/8 |")
    lines+=['','## 原v7题不重构的结果','',
        '以下保留原题面、原图及原选项内容。原顺序为每输入模态16题；四种位置合计64次回答。与上表属于不同题集，不能直接混合，也不能拿不同题集的百分比当训练提升。','',
        '| 训练 | 原顺序文字16题 | 原顺序图片16题 | 四位置文字64次 | 四位置图片64次 |','|---|---:|---:|---:|---:|']
    for a in ARMS:
        lines.append('| '+titles[a]+' | '+' | '.join([score(a,'named_T',True),score(a,'named_V',True),score(a,'named_T'),score(a,'named_V')])+' |')
    lines+=['','## 监督与预算','',
        '六组均Qwen3-VL-4B，同一warmup LoRA起点；1epoch、512行、128步、batch4/accum1、LR1e-4、r8/alpha32/dropout0.05、seed42、max8192、冻结vision/aligner。解释组64次/规则，辨认组64次/规则，混合组32次解释＋32次辨认/规则。使用固定最后checkpoint，不按成绩选checkpoint。',
        '',
        '训练先于读取该组测试内容固定，应用观察与正式应用答案均不进训练。每份学习图、16张测试图已人工查看；关键属性、区间及候选数量与文本一致。原3000PNG、数据集、基座权重哈希均通过。',
        '',
        '匹配行数和优化步数，不匹配文字token量或事实曝光量：辨认输入显示四份候选定义，目标只有字母；混合将两种练习各减半。VISION辨认/混合也显示文字候选事实，不能叫无文字事实监督的视觉学习。',
        '',
        'recognition_*是给四份候选定义的辨认诊断；oracle_*额外提供正确规则；self_*使用模型自己的无图回忆，保留错误原文。以上均与无材料单步应用分开，未开原生CoT。',
        '',
        '四方向：TEXT适配器来源对应T2T/T2V，VISION来源对应V2T/V2V；每个适配器另两个方向复用相同测试输入，不当作新的独立样本。four_directions.json保存四方向映射与两套题集成绩。',
        '',
        '独立性边界：8条不同规则的预固定复现、单seed、两种选择器结构。32/128/64等回答数包含相关重复；并非128条独立知识，不外推1000条。该规则库已用于历史开发，因此称新规则复现，而非独立新基准。',
        '',
        'summary.json包含逐规则、题型、结构、选项位置及配对检查；各适配器原始回答在predictions.jsonl，配对回答在paired_application/<arm>/predictions.jsonl；semantic_reviews.jsonl含168份人工关键事实审核。']
    (OUT/'report_zh.md').write_text('\n'.join(lines)+'\n')
    print('Summary and independent raw/gold/hash audits complete.',flush=True)


if __name__=='__main__':summarize()
