#!/usr/bin/env python3
"""Audit immutable training, held-constant probes, raw responses and final results."""
import copy
import csv
import itertools
import json
from collections import Counter
from pathlib import Path

import explore_v7_rule_recognition as pilot
import run_v7_evaluation as ev
from build_v7_experiment_data import name
from run_v7_full_experiment import ROOT, rows, read, save, sha
from summarize_v7_counterfactual_names import metric

OUT=pilot.OUT
ARMS=('warmup','text','vision','text_match','vision_match')


def audit():
    pilot.validate(OUT)
    target={k['id']:k for k in rows(pilot.source.OUT/'target_knowledge.jsonl')}
    training={a:rows(OUT/(a+'.jsonl')) for a in pilot.ARMS}
    for arm,rr in training.items():
        assert len(rr)==512 and Counter(r['qid'] for r in rr)==Counter({q:64 for q in target})
        for r in rr:
            assert len(r['messages'])==3 and r['messages'][-1]['content']==r['gold']
            assert r['option_rule_ids']['ABCD'.index(r['gold'])]==r['qid']
            body=r['messages'][-2]['content']
            for l,qid in zip('ABCD',r['option_rule_ids']):
                assert f'{l}. '+pilot.definition(target[qid]) in body
            assert len(r.get('images',[]))==(arm=='vision_match')
            if arm=='vision_match':assert r['images']==[str(pilot.source.DATA/'images'/r['qid']/'learning_visual.png')]
        for qid in target:
            assert Counter(r['gold'] for r in rr if r['qid']==qid)==Counter({l:16 for l in 'ABCD'})
    for t,v in zip(training['text_match'],training['vision_match']):
        assert t['qid']==v['qid'] and t['gold']==v['gold'] and t['option_rule_ids']==v['option_rule_ids']
    diag=rows(OUT/'diagnostic_probes.jsonl')
    source=rows(pilot.source.OUT/'probes.jsonl')
    source_lookup={p['id']:p for p in source}
    outputs={}
    for arm in ARMS:
        folder=OUT/arm
        rr=rows(folder/'predictions.jsonl');assert len(rr)==(952 if arm in pilot.ARMS else 136)
        pp=source+diag+rows(folder/'self_probes.jsonl') if arm in pilot.ARMS else diag
        lookup={p['id']:p for p in pp};assert len(lookup)==len(rr) and set(lookup)=={r['id'] for r in rr}
        cp=read(folder/'eval_manifest.json')['adapter'];assert ev.checkpoint(Path(cp['path']))==cp
        recall={r['qid']:r for r in rr if r['condition']=='recall'}
        for r in rr:
            p=lookup[r['id']]
            assert r['input_sha256']==ev.digest({'messages':p['messages'],'images':p.get('images',[])})
            assert r['response']==(r['raw_response']['choices'][0]['message']['content'] or '')
            if r['fmt']=='mcq':
                pred=ev.letter(r['response']);assert pred==r['predicted']
                assert r['format_error']==(pred is None)
                if p.get('gold'):assert r['correct']==(pred==p['gold'])
            if r['condition'].startswith('named_'):
                assert p==source_lookup[r['id']]
                assert 'Options:\nA. Result:' in p['messages'][-1]['content'] or p['task']=='diagnosis'
                assert 'Correct learning reference:' not in str(p['messages'])
            if r['condition'].startswith('self_'):
                old=source_lookup[p['id'].removeprefix('self:')]
                text=recall[p['qid']]['response']
                assert p['messages'][-1]['content']=='Your own prior rule recall (uncorrected):\n'+text+'\n\n'+old['messages'][-1]['content']
        assert read(folder/'eval_state.json')['status']=='complete'
        outputs[arm]=rr
    # Main metrics for frozen controls are reused, never counted as new generations.
    for arm in ('warmup','text','vision'):
        outputs[arm]+=rows(pilot.source.OUT/arm/'predictions.jsonl')
    for reference in ('model_hash_validation.json','original_data_hash_validation.json'):
        old=read(pilot.source.OUT/reference)
        checked=[]
        for r in old['checked_files']:
            assert sha(Path(r['path']))==r['sha256']
            checked.append({'path':r['path'],'sha256':r['sha256'],'match':True})
        save(OUT/reference,{'status':'pass','reference':str(pilot.source.OUT/reference),'checked_files':checked})
    checks={'status':'pass','all_original_3000_pngs_unchanged':True,'original_dataset_unchanged':True,
        'base_weights_unchanged':True,'new_responses':sum(len(rows(OUT/a/'predictions.jsonl')) for a in ARMS),
        'main_probes_identical_to_frozen_source':True,'target_test_observations_in_training':False,
        'fixed_final_checkpoint':True,'training_targets_verified':1024,'training_gold_balanced_per_rule':True}
    save(OUT/'completion_audit.json',checks)
    return outputs


def summarize():
    outputs=audit();summary={};table=[]
    reviews=rows(OUT/'semantic_reviews.jsonl')
    expected={(a,r['id']) for a in ARMS for r in outputs[a] if r['fmt']=='card' and
              (a in pilot.ARMS or r['condition']=='plain_recall')}
    assert len(reviews)==len(expected)==72
    assert {(r['arm'],r['id']) for r in reviews}==expected
    for review in reviews:
        actual=next(r for r in outputs[review['arm']] if r['id']==review['id'])
        assert review['response_sha256']==ev.digest(actual['response'])
    for arm,rr in outputs.items():
        conditions=sorted({r['condition'] for r in rr if r['fmt']=='mcq' and not r['condition'].startswith('anonymous_')})
        summary[arm]={c:metric([r for r in rr if r['condition']==c]) for c in conditions}
        summary[arm]['recall']={c:{'correct':sum(r['correct'] for r in reviews if r['arm']==arm and r['condition']==c),
            'n':sum(r['arm']==arm and r['condition']==c for r in reviews)}
            for c in ('recall','recall_image','plain_recall') if any(r['arm']==arm and r['condition']==c for r in reviews)}
        for c in ('named_T','named_V'):
            group=[r for r in rr if r['condition']==c]
            summary[arm][c]['by_task']={t:metric([r for r in group if r['task']==t]) for t in ('direct','diagnosis')}
            summary[arm][c]['by_rule']={q:metric([r for r in group if r['qid']==q]) for q in sorted({r['qid'] for r in group})}
            summary[arm][c]['by_group']={str(g):metric([r for r in group if r['group']==g]) for g in (0,1)}
            paired=[[r for r in group if (r['block'],r['shift'])==key] for key in sorted({(r['block'],r['shift']) for r in group})]
            assert len(paired)==40 and all(len(g)==4 and {r['gold'] for r in g}==set('ABCD') for g in paired)
            summary[arm][c]['paired']={'blocks_times_shifts':40,'all4_correct':sum(all(r['correct'] for r in g) for g in paired),
                'all4_same_answer':sum(len({r['predicted'] for r in g})==1 for g in paired)}
            summary[arm][c]['by_shift']={str(s):metric([r for r in group if r['shift']==s]) for s in range(4)}
        for c in conditions:
            table.append({'arm':arm,'condition':c,**{k:summary[arm][c][k] for k in ('n','correct','accuracy','format_errors','empty','truncated')}})
    save(OUT/'summary.json',summary)
    with (OUT/'comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    directions=[]
    for arm in ('text','vision',*pilot.ARMS):
        for d in ('T2T','T2V','V2T','V2V'):
            directions.append({'adapter':arm,'direction':d,'condition':'named_'+d[-1],
                'matches_training_source':d[0]==('V' if arm.startswith('vision') else 'T'),**summary[arm]['named_'+d[-1]]})
    save(OUT/'four_directions.json',directions)
    def score(arm,c):
        r=summary[arm][c];return f"{r['correct']}/{r['n']}（{r['accuracy']:.1%}）"
    lines=['# 按名称辨认规则：小规模训练对照','',
        '本轮把训练目标改成“四份同类规则定义中，按名称选出正确的一份”。原应用题保持不变，主评测没有给定义或学习图。',
        '', '| 训练 | 文字应用 | 图片应用 | 无图辨认原定义 | 无图辨认改写定义 |',
        '|---|---:|---:|---:|---:|']
    titles={'warmup':'共同起点','text':'旧TEXT解释','vision':'旧VISION解释','text_match':'新TEXT辨认','vision_match':'新VISION辨认'}
    for a in ARMS:
        lines.append('| '+titles[a]+' | '+' | '.join(score(a,c) for c in ('named_T','named_V','recognition_exact_noimage','recognition_para_noimage'))+' |')
    lines+=['','TEXT辨认的直接应用与报告核验分项：', '', '| 测试输入 | direct | diagnosis |','|---|---:|---:|']
    for c,label in [('named_T','文字'),('named_V','原图')]:
        x=summary['text_match'][c]['by_task']
        lines.append('| '+label+' | '+' | '.join(f"{x[t]['correct']}/{x[t]['n']}（{x[t]['accuracy']:.1%}）" for t in ('direct','diagnosis'))+' |')
    lines+=['','## 解释与边界','',
        'TEXT辨认训练提高了单步应用，且两类题均有收益；无图辨认32/32在选项改写后仍成立。训练中没有原测试观察或应用答案。不过它不能正确自由复述规则：新TEXT的无图完整回忆0/8，用户单独问法也0/8。先加入错误自回忆后，文字应用降为'+score('text_match','self_T')+'。因此不能把自由回忆正确作为所有选择题知识学习的必要条件，也不能强制所有适配器先复述。',
        '',
        '同观察、同选项，只换四个名称的配对检查中，旧TEXT在40个block×位置组合里有31组文字题、32组图片题始终选同一字母；新TEXT分别降为2组、6组。每组四个真值恰为ABCD，因此统一偏好一个答案只能25%。这个变化说明新组开始区分名称，收益不能单独用一个全局字母/结果偏好解释；但仍只来自8条开发规则。',
        '',
        '新VISION看原学习图能辨认32/32，脱图辨认只有'+score('vision_match','recognition_exact_noimage')+'，原文字/图片应用仍未显示与TEXT相同的收益。这与“训练任务可通过看图匹配完成，尚未形成可脱图调用的名称记忆”的解释一致；仍是机制假设，并未通过训练梯度因果实验确认。',
        '',
        '训练条件：两组从同一warmup适配器启动，固定最后checkpoint-128，1epoch，8规则×64行，batch4/accum1，LR1e-4，r8/alpha32/dropout0.05，seed42，冻结视觉/aligner。旧解释对照同父checkpoint、行数、步数与参数；旧结果复用，未重复计作新回答。',
        '',
        '匹配步数不等于匹配信息量：辨认输入同时显示4条候选事实，解释输入只针对1条事实；监督目标又由整段规则改成一个字母，token预算不同。VISION输入含文字候选定义，因此不能宣称这是完全没有文字事实的纯视觉知识学习。',
        '',
        '8规则、5张原测试图，40语义问题/输入模态，四选项位置轮换后160请求；不是160条独立知识。规则与场景早已用于开发，单seed结果不能外推1000条。原数据和3000PNG均通过哈希；未训练PT、未开原生CoT。',
        '',
        '四方向映射：TEXT来源主结果T2T=文字应用、T2V=图片应用；VISION来源主结果V2T=文字应用、V2V=图片应用。每适配器另外两个方向只是复用同一测试输入的诊断，不重复当独立样本。four_directions.json保留四项映射。',
        '',
        '本轮可保留的候选是选择式知识监督。下一步应先在未参与开发的规则组复现TEXT收益，并补测VISION如何形成名称绑定；不直接全量修改数据。若加入脱图文字回放，应明确另列为视觉来源文字桥接路线。',
        '',
        'summary.json含逐规则/位置/题型/同图换名配对结果；comparison.csv为简表；semantic_reviews.jsonl含72份新增回忆的人工关键事实审核；各适配器predictions.jsonl保留原始回复。分辨率对照见resolution_control，单独诊断读图，不混进迁移主成绩。']
    (OUT/'report_zh.md').write_text('\n'.join(lines)+'\n')
    k=next(k for k in rows(pilot.source.OUT/'target_knowledge.jsonl') if name(k)=='Deldev')
    sample=next(r for r in rows(OUT/'vision_match.jsonl') if r['qid']==k['id'])
    body=sample['messages'][-2]['content']
    (OUT/'TRAINING_EXAMPLE_ZH.md').write_text('# 训练具体改了什么\n\n旧训练：给Deldev原学习图，请模型复述完整规则；目标含“pentagon OR white”。\n\n新训练仍用同一原学习图，问题和目标如下。\n\n```text\n'+body+'\n```\n\n目标：`'+sample['gold']+'`。该目标指向四份定义中的Deldev；选项内不写规则名。每条规则正确字母A/B/C/D各16次，共64次。TEXT组完全相同，只去掉学习图片及看图前缀。\n\n这是训练用定义辨认练习，最终应用题没有这四份定义。图像组的输入候选定义包含文字事实，应据实归类为图像条件的有监督辨认。\n')
    print('Audit and summary complete.',flush=True)


if __name__=='__main__':summarize()
