#!/usr/bin/env python3
"""Independently verify generated labels, supervised updates, raw predictions and reports."""
import copy
import csv
import importlib.metadata
import json
import math
import operator
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import explore_v7_application_binding as pilot
import summarize_v7_prompt_loss_bridge as previous_summary
import run_v7_evaluation as ev
from run_v7_full_experiment import ROOT, rows, read, save, sha
from v7_gpu_resources import ledger

OUT=pilot.OUT
LABELS={'warmup':'共同起点','text_equal':'旧TEXT解释＋定义辨认','vision_equal':'旧VISION解释＋定义辨认',
    'text_apply':'TEXT解释＋应用','vision_apply':'原学习图＋应用','vision_apply_replay':'原学习图＋应用＋闭卷练习'}


def independent_data_audit():
    knowledge={r['id']:r for r in rows(OUT/'target_knowledge.jsonl')}
    blocks=rows(OUT/'blocks.jsonl');assert len(blocks)==20
    ops={'eq':operator.eq,'ge':operator.ge,'lt':operator.lt};decisions=0;questions=0
    for b in blocks:
        outputs={}
        for q in b['rule_ids']:
            rule=json.loads(knowledge[q]['fact']['quote']);flags=[]
            for obj in b['observations']:
                bits=[ops[c['operator']](obj[c['field']],c['value']) for c in rule['conditions']]
                flags.append(any(bits) if rule['mode']=='any' else all(bits));decisions+=1
            outputs[q]='; '.join(f'{l}={"KEEP" if flag else "OMIT"}' for l,flag in zip('ABCD',flags))
        assert outputs==b['results'] and len(set(outputs.values()))==4
        for task in ('direct','diagnosis'):
            for shift in range(4):
                for q in b['rule_ids']:
                    p=pilot.application(b,q,knowledge,task,shift)
                    choices=dict(re.findall(r'^([A-D])\. (.+)$',p['messages'][1]['content'],re.M))
                    action=('Accept report. ' if outputs[q]==b['submitted_report'] else 'Replace report. ') if task=='diagnosis' else ''
                    assert choices[p['gold']]==action+'Result: '+outputs[q]+'.';questions+=1
    train=[b for b in blocks if b['split']=='train'];holdout=[b for b in blocks if b['split']=='holdout']
    trainset={pilot.signature(b['observations']) for b in train};heldset={pilot.signature(b['observations']) for b in holdout}
    original={pilot.signature(b['observations']) for b in rows(pilot.previous.source.OUT/'original_case_audit.jsonl')}
    assert len(trainset)==12 and len(heldset)==8 and not trainset&heldset and not (trainset|heldset)&original
    base=rows(OUT/'text_apply.jsonl');byblock={b['id']:b for b in blocks}
    for arm in pilot.ARMS:
        training=rows(OUT/(arm+'.jsonl'));assert len(training)==512
        assert Counter(r['channel'] for r in training)=={'explain':128,'application':384}
        assert Counter(r['gold'] for r in training if r['channel']=='application')=={l:96 for l in 'ABCD'}
        for r,text in zip(training,base):
            assert r['id']==text['id'] and r['qid']==text['qid'] and r['messages'][-1]==text['messages'][-1]
            plain=copy.deepcopy(r)
            if r.get('images'):
                assert r['images']==[str(pilot.previous.source.DATA/'images'/r['qid']/'learning_visual.png')]
                assert r['messages'][1]['content'].startswith(pilot.REFERENCE)
                plain['messages'][1]['content']=plain['messages'][1]['content'][len(pilot.REFERENCE):]
            assert plain['messages']==text['messages']
            if r['channel']=='explain':assert r['messages'][-1]['content']==pilot.previous.source.card(knowledge[r['qid']])
            else:
                bid=':'.join(r['id'].split(':')[:2]);task=r['task'];p=pilot.application(byblock[bid],r['qid'],knowledge,task,r['shift'])
                assert byblock[bid]['split']=='train' and plain['messages'][:-1]==p['messages'] and r['gold']==p['gold']
            assert sum(m['content'].count('<image>') for m in r['messages'])==len(r.get('images',[]))
        images=sum(bool(r.get('images')) for r in training)
        assert images=={'text_apply':0,'vision_apply':512,'vision_apply_replay':320}[arm]
        if arm.endswith('replay'):
            assert sum(r['replayed_without_learning_image'] for r in training)==192
            assert all(r.get('images') for r in training if r['channel']=='explain')
    save(OUT/'independent_data_audit.json',{'status':'pass','object_decisions_checked':decisions,'rotated_block_questions_checked':questions,
        'training_rebuilt_from_knowledge':True,'replay_changes_only_image_and_reference_prefix':True,
        'train_holdout_original_multisets_disjoint':True,'all_training_inputs_and_targets_validated':True})


def metric(rr):
    m=pilot.previous.source.metric(rr)
    for key in ('task','qid','group','shift'):
        values=sorted({r[key] for r in rr if key in r},key=str)
        if values:m['by_'+('rule' if key=='qid' else key)]={str(v):pilot.previous.source.metric([r for r in rr if r.get(key)==v]) for v in values}
    if rr and all('block' in r for r in rr):
        groups=[[r for r in rr if (r['block'],r['shift'])==k] for k in sorted({(r['block'],r['shift']) for r in rr})]
        assert all(len(g)==4 and {r['gold'] for r in g}==set('ABCD') for g in groups)
        m['name_checks']={'n':len(groups),'four_names_same_valid_answer':sum(len({r['predicted'] for r in g})==1 and g[0]['predicted'] in ('A','B','C','D') for g in groups),
            'all_four_correct':sum(all(r['correct'] for r in g) for g in groups),
            'note':'Original observations/options are shared. *_image diagnostics also change the supplied learning image per rule; those groups cannot isolate name-only sensitivity.'}
    if rr and all('shift' in r for r in rr):m['original_order']=pilot.previous.source.metric([r for r in rr if r['shift']==0])
    if 'by_task' in m:m['by_evaluation_group']={label:m['by_task'][task] for task,label in [('direct','result_application'),('diagnosis','rule_report_audit')] if task in m['by_task']}
    return m


def audit():
    import torch
    torch.set_num_threads(2)
    pilot.validate();pilot.previous.source.validate(pilot.previous.source.OUT);independent_data_audit()
    for root in (pilot.previous.OUT,pilot.previous.source.OUT):
        old=read(root/'final_artifact_manifest.json')
        for p,h in old['files'].items():assert sha(root/p)==h,p
    for filename in ('model_hash_validation.json','original_data_hash_validation.json'):
        old=read(pilot.previous.OUT/filename)
        for r in old['checked_files']:assert sha(Path(r['path']))==r['sha256']
        save(OUT/filename,{'status':'pass','reference':str(pilot.previous.OUT/filename),'checked_files':old['checked_files']})
    main=rows(OUT/'main_probes.jsonl');extra=rows(OUT/'diagnostic_probes.jsonl')
    update={};responses={}
    for arm in pilot.CONTROLS+pilot.ARMS:
        own=main+extra if arm in pilot.ARMS else extra
        rr=previous_summary.audited_responses(OUT/arm/'evaluation',own,pilot.checkpoint(arm))
        if arm in pilot.CONTROLS:
            if arm=='warmup':
                before=rows(pilot.previous.source.OUT/arm/'predictions.jsonl')+rows(pilot.previous.paired.OUT/arm/'predictions.jsonl')
                lookup={p['id']:p for p in main};before=[r for r in before if r['id'] in lookup]
                assert len(before)==568
                for r in before:assert r['input_sha256']==ev.digest({'messages':lookup[r['id']]['messages'],'images':lookup[r['id']].get('images',[])})
            else:before=previous_summary.audited_responses(pilot.previous.OUT/'main'/arm,main,pilot.checkpoint(arm))
            rr=before+rr
        responses[arm]=rr
        if arm not in pilot.ARMS:continue
        folder=OUT/arm;cp=pilot.checkpoint(arm);args=read(cp/'args.json');state=read(cp/'trainer_state.json')
        assert state['global_step']==128 and state['epoch']==1
        expected={'learning_rate':1e-4,'num_train_epochs':1,'per_device_train_batch_size':4,'gradient_accumulation_steps':1,
            'loss_type':'v7_equal_example','lora_rank':8,'lora_alpha':32,'lora_dropout':.05,'seed':42,'max_length':8192,
            'freeze_vit':True,'freeze_aligner':True,'packing':False,'padding_free':False}
        for k,v in expected.items():assert args[k]==v,(arm,k,args[k])
        assert args['adapters']==[str(pilot.checkpoint('warmup'))] and args['dataset']==[str(OUT/(arm+'.jsonl'))]
        assert ev.checkpoint(cp)['sha256']==read(folder/'checkpoint_manifest.json')['sha256']
        actual=rows(folder/'loss_batches.jsonl');assert len(actual)==128
        assert {r['step_before_update'] for r in actual}==set(range(128))
        counts=[]
        for r in actual:
            assert math.isfinite(r['loss']) and r['equal_row_weights']==[.25]*4
            assert abs(r['loss']-sum(r['per_example_mean_loss'])/4)<1e-5
            counts+=r['supervised_tokens_including_template_suffix']
        assert Counter(n==3 for n in counts)=={True:384,False:128}
        from safetensors import safe_open
        with safe_open(str(pilot.checkpoint('warmup')/'adapter_model.safetensors'),framework='pt') as parent,safe_open(str(cp/'adapter_model.safetensors'),framework='pt') as final:
            assert set(parent.keys())==set(final.keys());deltas={k:float((final.get_tensor(k).float()-parent.get_tensor(k).float()).abs().max()) for k in final.keys()}
        assert len(deltas)==504 and all(math.isfinite(v) and v>0 for v in deltas.values())
        assert not any('.visual.' in k or '.merger.' in k for k in deltas)
        update[arm]={'changed_tensors':504,'total_tensors':504,'max_abs_delta':max(deltas.values()),'last_loss':actual[-1]['loss']}
        smoke=OUT/('smoke_'+arm);assert read(smoke/'validated.json')['status']=='pass'
        assert len(rows(smoke/'reload/predictions.jsonl'))==2
        assert ev.checkpoint(Path(read(smoke/'checkpoint_manifest.json')['path']))['sha256']==read(smoke/'checkpoint_manifest.json')['sha256']
    reviews=rows(OUT/'semantic_reviews.jsonl')
    expected={(a,r['id']) for a in pilot.ARMS for r in responses[a] if r['fmt']=='card'}
    assert len(reviews)==72 and {(r['arm'],r['id']) for r in reviews}==expected
    knowledge={r['id']:r for r in rows(OUT/'target_knowledge.jsonl')}
    for r in reviews:
        actual=next(x for x in responses[r['arm']] if x['id']==r['id'])
        assert r['response']==actual['response'] and r['response_sha256']==ev.digest(actual['response'])
        assert r['expected']==pilot.previous.source.card(knowledge[r['qid']]) and isinstance(r['correct'],bool) and r['reason']
    save(OUT/'adapter_update_validation.json',update)
    save(OUT/'completion_audit.json',{'status':'pass','new_full_training_jobs':3,'smoke_jobs':3,'new_main_and_diagnostic_answers':4392,
        'smoke_reload_answers':6,'new_answers_total':4398,'reused_main_answers':1704,'manual_recall_reviews':72,
        'all_3000_original_PNG_unchanged':True,'original_data_and_models_unchanged':True,'old_runs_unchanged':True,
        'final_checkpoint_128_only':True,'raw_inputs_replies_parser_verified':True,'all_final_adapters_changed_from_parent':True})
    return responses,reviews


def summarize():
    rr,reviews=audit();summary={};table=[]
    for a,responses in rr.items():
        summary[a]={c:metric([r for r in responses if r['condition']==c]) for c in sorted({r['condition'] for r in responses if r['fmt']=='mcq'})}
        if a in pilot.ARMS:
            own=[r for r in reviews if r['arm']==a]
            summary[a]['recall']={c:{'n':sum(r['condition']==c for r in own),'correct':sum(r['correct'] for r in own if r['condition']==c),
                'truncated':sum(r['finish_reason']=='length' for r in own if r['condition']==c)} for c in sorted({r['condition'] for r in own})}
        for c,m in summary[a].items():
            if c!='recall':table.append({'arm':a,'condition':c,**{k:m[k] for k in ('n','correct','accuracy','valid_accuracy','missing','format_errors','empty','truncated')}})
    save(OUT/'summary.json',summary)
    with (OUT/'comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    save(OUT/'format_errors.json',table)
    save(OUT/'four_directions.json',{'note':'TEXT/VISION refer to knowledge-stage inputs. VISION practice includes textual candidate observations; replay adds closed-book derived-label supervision. Same adapter/input evaluated once, direction aliases are not independent answers.',
        'rows':[{'arm':a,'direction':d,'source_matches_direction':d[0]==('T' if a=='text_apply' else 'V'),
            'source_class':'image-derived closed-book practice bridge' if a.endswith('replay') else 'text knowledge and textual practice' if a=='text_apply' else 'original learning image plus textual practice',
            'original_v7':summary[a]['named_'+d[-1]],'paired_application':summary[a]['paired_'+d[-1]]} for a in pilot.ARMS for d in ('T2T','T2V','V2T','V2V')]})
    gates={}
    for a in pilot.ARMS:
        m=summary[a];base=summary['warmup'];source='seen_closed' if a=='text_apply' else 'seen_image'
        gates[a]={'source_capacity_at_least_80pct':m[source]['accuracy']>=.8,
            'unseen_closed_text_at_least_60pct':m['unseen_closed']['accuracy']>=.6,
            'paired_T_gain_over_parent':m['paired_T']['accuracy']-base['paired_T']['accuracy'],
            'paired_V_gain_over_parent':m['paired_V']['accuracy']-base['paired_V']['accuracy']}
        gates[a]['both_paired_gains_at_least_10pp']=gates[a]['paired_T_gain_over_parent']>=.1 and gates[a]['paired_V_gain_over_parent']>=.1
        gates[a]['screening_gates_all_pass']=all(gates[a][k] for k in ('source_capacity_at_least_80pct','unseen_closed_text_at_least_60pct','both_paired_gains_at_least_10pp'))
    save(OUT/'screening_gates.json',gates)
    behavior={}
    for a,responses in rr.items():
        behavior[a]={'answer_letters':{c:dict(Counter(r['response'] for r in responses if r['condition']==c))
            for c in summary[a] if c!='recall'}}
        if a in pilot.ARMS:
            batches=rows(OUT/a/'loss_batches.jsonl')[-32:]
            behavior[a]['last_32_training_steps']={}
            for label,is_mcq in [('MCQ',True),('explanation',False)]:
                losses=[loss for b in batches for n,loss in zip(b['supervised_tokens_including_template_suffix'],b['per_example_mean_loss']) if (n==3)==is_mcq]
                behavior[a]['last_32_training_steps'][label]={'n':len(losses),'mean_row_loss':sum(losses)/len(losses)}
    behavior['interpretation']={'random_letter_plus_perfect_two_suffix_tokens_reference_loss':math.log(4)/3,
        'assumption':'Only an explanatory reference: the letter is uniformly random among four tokens and both suffix tokens have zero loss. Actual first-token probabilities were not logged, so this is not a measured answer-token loss.',
        'logits_shape':'Swift prepare_logits_to_keep retains only output positions needed for loss. A short output-logit sequence does not imply the prompt was truncated.'}
    save(OUT/'output_behavior.json',behavior)
    save(OUT/'matched_reference_comparison.json',{
        'note':'Same64heldout questions and shift0. Image arm supplies the correct original learning image. This is an information-availability diagnostic, not closed-book transfer.',
        'rows':[{'arm':a,'closed':summary[a]['unseen_closed']['original_order'],'with_learning_image':summary[a]['unseen_image']} for a in pilot.CONTROLS+pilot.ARMS]})
    write_report(summary,gates);write_examples();write_interpretation(summary,behavior,rr)
    scripts=[Path(__file__),ROOT/'scripts/explore_v7_application_binding.py']
    for p in scripts:shutil.copy2(p,OUT/'source_snapshots'/p.name)
    save(OUT/'environment_manifest.json',{'python':sys.version,'timestamp':ev.now(),
        'versions':{p:importlib.metadata.version(p) for p in ('torch','transformers','ms-swift','peft','safetensors')},'scripts':{str(p):sha(p) for p in scripts}})
    with ledger(pilot.previous.source.LEDGER) as entries:active={k:v for k,v in entries.items() if str(OUT) in k}
    assert not active,active
    save(OUT/'final_artifact_manifest.json',{'status':'complete','completed_at':ev.now(),'new_answers':4398,'manual_reviews':72,'active_gpu_reservations':active,
        'files':{str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='final_artifact_manifest.json' and 'checkpoint' not in p.relative_to(OUT).parts and p.suffix in ('.json','.jsonl','.md','.csv','.py')}})
    print('Complete:3training jobs,4398new answers,72semantic reviews; all labels/images/data/models/raw replies audited.',flush=True)


def write_report(ss,gates):
    def score(a,c):
        r=ss[a][c];return f"{r['correct']}/{r['n']}（{r['accuracy']:.1%}）"
    lines=['# 知识应用与名称绑定：小规模探索','',
        '目标是先证明学过知识可被调用，再讨论跨模态迁移。保留原v7、3000张PNG、原短答案评测提示及direct/diagnosis两种选择题。训练新增3组Qwen3-VL-4B LoRA，每组1epoch，固定最终第128步，不按成绩选checkpoint。不是全量1000条修复。','',
        '## 新对象上的学习检查','',
        '每模态分组由同4条规则组成，给同一组对象和相同选项，换规则名会对应四个不同正确答案。训练12组对象；未见验证8组对象；两者及原测试对象集合无重合（包含调换对象顺序）。原子属性和真值模式可以重复。','',
        '闭卷列不提供正确学习材料。带学习图列额外给对应原学习图，只检查图像条件解题。训练内列并非泛化成绩。','',
        '| 模型 | 训练内闭卷 | 训练内带学习图 | 未见组合闭卷 | 未见组合带学习图 |',
        '|---|---:|---:|---:|---:|']
    for a in pilot.CONTROLS+pilot.ARMS:lines.append('| '+LABELS[a]+' | '+' | '.join(score(a,c) for c in ('seen_closed','seen_image','unseen_closed','unseen_image'))+' |')
    lines+=['','seen列各64次回答；unseen闭卷256次、带图64次。后两列选项轮换范围不同，不能直接解释为相同题目上的图像效应。四位置重复并非独立问题。','',
        '同样64道未见题、同样shift0的配对信息条件比较如下。给学习图只算带材料诊断：','',
        '| 模型 | 去掉学习图 | 给对应原学习图 |','|---|---:|---:|']
    for a in pilot.CONTROLS+pilot.ARMS:
        closed=ss[a]['unseen_closed']['original_order']
        lines.append('| '+LABELS[a]+f" | {closed['correct']}/{closed['n']}（{closed['accuracy']:.1%}） | "+score(a,'unseen_image')+' |')
    lines+=['',
        '## 原题与配对应用：不给额外学习材料','',
        '| 模型 | 原v7文字 | 原v7图片 | 配对文字 | 配对图片 |','|---|---:|---:|---:|---:|']
    for a in pilot.CONTROLS+pilot.ARMS:lines.append('| '+LABELS[a]+' | '+' | '.join(score(a,c) for c in ('named_T','named_V','paired_T','paired_V'))+' |')
    lines+=['','原v7每输入模态16题×4位置＝64次；配对每模态32个规则/题目组合×4位置＝128次。warmup/text_equal/vision_equal的主评测复用已有1704份回答，额外诊断均为本轮新生成。主表不把oracle混入闭卷分数。','',
        '## 给正确文字规则的能力控制','',
        '| 模型 | 文字测试＋正确规则 | 图片测试＋正确规则 |','|---|---:|---:|']
    for a in pilot.CONTROLS+pilot.ARMS:lines.append('| '+LABELS[a]+' | '+score(a,'oracle_T')+' | '+score(a,'oracle_V')+' |')
    lines+=['','这些是外部材料oracle，不是闭卷迁移成绩或严格上界。TEXT应用组oracle文字退化到3/16，须同时考虑任务能力退化。','',
        '## 两种题型分开','',
        '| 模型 | 配对direct文字 | 配对diagnosis文字 | 配对direct图片 | 配对diagnosis图片 |','|---|---:|---:|---:|---:|']
    for a in pilot.CONTROLS+pilot.ARMS:
        cells=[]
        for c,t in [('paired_T','direct'),('paired_T','diagnosis'),('paired_V','direct'),('paired_V','diagnosis')]:
            r=ss[a][c]['by_task'][t];cells.append(f"{r['correct']}/{r['n']}（{r['accuracy']:.1%}）")
        lines.append('| '+LABELS[a]+' | '+' | '.join(cells)+' |')
    lines+=['','## 是否真正区分规则名','',
        '| 模型 | 配对文字四名同答 | 配对图片四名同答 | 配对文字四名全对 | 配对图片四名全对 |','|---|---:|---:|---:|---:|']
    for a in pilot.CONTROLS+pilot.ARMS:
        t=ss[a]['paired_T']['name_checks'];v=ss[a]['paired_V']['name_checks']
        lines.append('| '+LABELS[a]+f" | {t['four_names_same_valid_answer']}/{t['n']} | {v['four_names_same_valid_answer']}/{v['n']} | {t['all_four_correct']}/{t['n']} | {v['all_four_correct']}/{v['n']} |")
    lines+=['','## 规则回忆诊断','', '| 新训练组 | 训练system无图完整回忆 | 带学习图完整回忆 | 普通提示无图回忆 |','|---|---:|---:|---:|']
    for a in pilot.ARMS:
        rec=ss[a]['recall'];lines.append('| '+LABELS[a]+' | '+' | '.join(f"{rec[c]['correct']}/{rec[c]['n']}" for c in ('recall','recall_image','plain_recall'))+' |')
    lines+=['','## 预先固定的实用筛查门槛','',
        '训练内来源条件至少80%；未见组合闭卷文字至少60%；原配对文字和图片均比共同起点提升至少10个百分点。这是开发筛查门槛，不是统计显著性检验，不能据此确认全量有效。门槛在完整训练前保存于decision_criteria.json。','',
        '| 新训练组 | 来源条件容量 | 未见闭卷组合 | 两输入应用增幅 | 全部门槛 |','|---|---|---|---|---|']
    for a,g in gates.items():lines.append('| '+LABELS[a]+' | '+' | '.join('通过' if g[k] else '未通过' for k in ('source_capacity_at_least_80pct','unseen_closed_text_at_least_60pct','both_paired_gains_at_least_10pp','screening_gates_all_pass'))+' |')
    lines+=['','## 实施、监督来源和解释边界','',
        '各组512行＝128规则解释＋384应用选择题，解释每规则16行；应用为每拓扑6个新对象集合×2题型×4规则名×4选项位置。direct目标是逐对象结果，diagnosis要核对报告并接受或替换，两者未合并。三组助手目标、行ID、顺序完全相同。',
        '',
        'TEXT没有学习图；VISION每行使用原学习图，练习对象用文字描述。VISION＋闭卷练习只从192条应用练习移除学习图及reference前缀，128条解释均保留图。没有新增或修改PNG，没有在图像组用户输入中补入正确规则定义；但解释答案含明确文字事实，选择标签由人工标注规则求解，闭卷行属于额外的图像衍生答案监督。不能把最后一组当作纯视觉知识迁移，也不能仅凭三组比较证明像素的因果贡献。',
        '',
        '各组从相同warmup适配器起步；该warmup训练规则与本轮目标规则分开。r8/alpha32/dropout0.05、batch4/accum1、LR1e-4、cosine/5%warmup、seed42、冻结vision/aligner、max8192、样本等权CE、1epoch/128步。本轮没有PT或原生CoT。三组先单batch训练、重载文字/图片推理，之后跑固定完整预算。',
        '',
        '旧解释＋定义辨认控制和新解释＋应用组同时改变任务内容及行数分配，只作为方案对比，不能把所有差值归因于单个因素。VISION＋闭卷练习相对VISION应用组只改指定192行图像及reference前缀，但同时改变了学习时信息可用性。',
        '',
        '仍为同8条开发规则、单seed；不是未见规则的独立确认，原1000条模型也见过这些规则。每方向原题仅16道，轮换强相关。应先看来源条件是否学会，不在学习增益不足时报告有意义的迁移保持率。',
        '',
        '训练标签用独立operator求解器、渲染后的选项内容逐条核验。实际checkpoint、loss、LoRA变化、输入/输出SHA、严格解析和原图/原数据/历史结果均已审核；全部72份新回忆逐条读全文核对。原始回答在各组evaluation/predictions.jsonl。所有错误/截断均保留；自由回忆与MCQ格式统计分开。',
        '',
        '训练例子见TRAINING_EXAMPLE_ZH.md，原始数据与预定门槛见manifest.json、decision_criteria.json。summary.json含逐规则、题型、结构、位置分项；four_directions.json标明方向与训练来源归属。',
        '', '实际结论和研究路线见interpretation_zh.md；output_behavior.json保留答案字母分布及训练末段分目标loss，controlled_examples.json保留同题四名的完整输入和原始输出。']
    (OUT/'report_zh.md').write_text('\n'.join(lines)+'\n')


def write_examples():
    rr=rows(OUT/'text_apply.jsonl')
    explanation=next(r for r in rr if r['channel']=='explain' and '0955' in r['qid'])
    case=next(r for r in rr if r['channel']=='application' and r['task']=='direct' and '0955' in r['qid'])
    matching=[r for r in rr if r.get('block')==case['block'] and r.get('shift')==case['shift']]
    lines=['# 本轮如何学习知识、如何测试迁移','',
        '我们要测的是同一份新知识从学习模态迁移到测试模态的能力。前提是模型先在无学习材料的新输入上表现出知识学习。本轮检验应用练习能否建立这个前提，原1000条v7及3000PNG保持不变。','',
        '## 先保留规则解释','', 'user：','```text',explanation['messages'][1]['content'],'```','', 'assistant：','```text',explanation['messages'][2]['content'],'```','',
        '## 再练习由规则推导的新候选集合','', 'user：','```text',case['messages'][1]['content'],'```','', 'assistant：`'+case['messages'][-1]['content']+'`。','',
        '同一对象集合、同一选项排列，只改变规则名称，四份训练例的正确答案如下：','', '| 规则ID | 正确选项 |','|---|---|']
    for r in matching:lines.append('| '+r['qid']+' | '+r['messages'][-1]['content']+' |')
    lines+=['','这要求根据指定名称调用不同规则。diagnosis独立生成提交报告，选择接受还是替换，并输出同样完整的逐对象结果。没有把它并入direct。','',
        'TEXT的输入如上。VISION在问题前增加“Original learning reference: <image>”，只连接该规则现有learning_visual.png；候选对象仍为文字。助手答案与TEXT逐字相同。第三组只在192条应用练习去掉图和前缀，让模型闭卷练习；规则解释仍有图。','',
        '最后一组的额外闭卷答案监督必须单列，它不能被写成纯视觉学习。新练习的规则结果由学习材料对应的结构化规则计算，不使用正式测试答案。训练输入先写完，才读取测试观察检查重合；一旦发现重合脚本会停止，不自动换掉样本。','',
        '训练有12个对象集合；另8个未见文字对象集合检查泛化。正式图片测试继续使用原测试PNG，无新增图。训练内练习只用于容量检查，不能当泛化分数。原题、配对题、给图诊断分别报告，选项轮换不算新独立问题。']
    (OUT/'TRAINING_EXAMPLE_ZH.md').write_text('\n'.join(lines)+'\n')


def write_interpretation(ss,behavior,responses):
    lines=['# 回到研究目标：这一轮解决了什么、还没有解决什么','',
        '我们要测的是：模型通过训练获得一份新知识，之后不再给学习材料，面对文字或图片的新问题，是否能使用这份知识。先验证同模态学习增益，再比较跨模态增益。基座必须恰好25%、loss必须很低、规则能背诵，都不是成功条件。','',
        '本轮已经做完3组1epoch训练、4398次新增回答（含6次smoke重载）、72份规则回忆人工审核。原v7、3000张PNG、原始标签和历史运行保持不变。应用练习方案没有通过预先固定的闭卷门槛；不能把本轮写成数据集已经修复。','',
        '## 1. 视觉端能根据学习图应用规则，但没有证明形成闭卷知识','',
        '在同样64道未见组合、同样shift0题目上，原图应用组去图为18/64（28.1%），带对应原学习图为57/64（89.1%）；共同起点对应15/64（23.4%）和36/64（56.2%）。两种题型均受益：新组带图direct28/32、diagnosis29/32。说明这轮确实提高了给图条件下的应用能力；原学习图仍然是关键条件。','',
        '原图应用组训练内带图64/64，但训练内去图16/64；带图完整回忆8/8，去图0/8。闭卷练习组也没有改变这点：回忆仍0/8；未见闭卷64/256，64组同场景四名对照全部同答。这里只能描述输出没有可靠地按规则名变化，不能断言模型内部完全没有存储知识。','',
        '## 2. 新增短答案应用练习，反而损伤了部分文字解题能力','',
        'TEXT应用组训练内闭卷17/64，未见闭卷67/256；配对图片128题全部回答C，配对文字124/128回答C。给正确文字规则的oracle也只有3/16，共同起点是13/16。该配置有明显的答案塌缩和任务能力退化，不能把它只解释为“知识没记住”或“跨模态差”。原图应用组oracle文字仍16/16，表明各组失败并不完全相同。','',
        '不能靠补充格式解析修复这组：三组各992道选择题均无缺失、空输出、格式错误或截断。72份自由回忆另行人工核对；其中的长回答截断没有混入选择题正确率。','',
        '## 3. 为什么loss下降仍然不代表会选答案','',
        '本轮每个字母答案实际有3个监督token：字母、结束符、换行。如果模型只学会后两个token，字母仍四选一不确定，平均loss就能约为ln(4)/3＝0.462。TEXT应用组最后32步的97条MCQ平均loss为'+f"{behavior['text_apply']['last_32_training_steps']['MCQ']['mean_row_loss']:.3f}"+'，与这个解释性参照接近。我们没有保存逐token概率，因此不能把它当作对每个答案token的实际测量；但原始预测的大量同答已经直接证明应用未学会。','',
        '样本等权CE只平衡长短样本，仍然包括答案中的公共后缀。此前单规则实验已经实测过字母加权，未带来稳定应用收益，因此下一轮不应把改loss权重当成未经验证的万能修复。','',
        '训练日志中较短的logits长度来自Swift的logits_to_keep优化，只保留计算loss所需的输出位置。已核对库实现，不能据此认定训练问题被截断。','',
        '## 4. 一条可直接核查的例子','',
        '新对象A=黑色圆形、B=黑色正方形、C=红色五边形、D=红色三角形。同一题面和相同四个选项下，Dostan/Morkil/Ratlod/Lavrit的正确字母分别为B/A/D/C。原图应用组闭卷四次都答A；给各自原学习图后，四次分别答B/A/D/C。direct与diagnosis仍是分别计分的两种任务。完整输入、原答和输入SHA见controlled_examples.json。','',
        '## 5. 如何推进研究，而不是继续追着分数改题','',
        '先保留原图、原题和全部负结果。当前这套新增应用训练不扩到1000条或三模型。继续反复调整同8条开发规则，无法提供独立的修复证据。','',
        '把两条研究线路明确分开：','',
        '| 线路 | 如何获得和使用知识 | 现在能说什么 |',
        '|---|---|---|',
        '| 原始参数学习：TEXT/VISION，直接选择 | 按原SFT/PT学材料；测试只给新题 | 正式主线继续保留。当前小样本的闭卷学习或调用门槛未通过，负结果是真实结果。 |',
        '| 统一先自行回忆、再选择 | 第一步只有规则名，第二步传入该模型自己未改写的原始回忆和测试题；基线与四方向同协议 | 既往开发实验TEXT达到15/16文字、10/16图片；这是两步协议，不能混入直接strict。仍需不同规则上的确认。 |',
        '| 图像来源文字回放桥接＋自行回忆 | 从学习图得到模型原始解释，再训练无图按名称恢复，之后统一两步 | 既往8规则开发实验达到141/160文字、80/160图片。它有额外的无图文字事实监督，必须标为桥接；不能证明纯视觉参数迁移已解决。 |','',
        '两步方案的第一步只取事实，不要求或审查模型隐藏思维；第二步的最终任务仍是选择题。测试时不查学习材料、不用正确规则替换失败回忆、不丢失败规则。正确材料只进入单列oracle控制。','',
        '若研究要求纯视觉训练和单步直接选择，就必须接受目前尚未修复，不能拿桥接或带材料高分替代。若允许研究统一的知识调用流程，已有证据支持优先验证两步方案，避免再把事实记忆、自动调用、图片感知和规则执行混成单一分数。','',
        '## 6. 下一轮确认应怎样固定','',
        '在生成结果前，按规则ID哈希一次性固定未参与上述小实验开发的规则，按结构分层抽取；不能按模型正确率选规则或事后更换。先逐条审核原学习图确实包含必要事实，若不包含则保留清单并说明范围，不能用答案反推改图。最终评测观察及标签不进训练。','',
        '优先只跑一个模型的1epoch TEXT及原VISION，并把桥接设为单列第三组。为每组同时保留直接strict、自回忆后选择、正确材料oracle；所有组使用相同固定提示和最终checkpoint。主评测保留direct与diagnosis以及T2T/T2V/V2T/V2V归属。新的源模态基线不通过，就报告该配置的学习/调用失败，不计算迁移保持率，也不事后调到通过。','',
        '报告各方向训练前后差值、规则级正确回忆率、全量最终选择率和oracle差距。仅在来源方向有明确增益时讨论跨方向保留了多少增益。选项轮换只作位置稳健性检查，统计单位以规则为主；成功回忆子集只能做条件诊断，主分母保留所有预定规则。多个seed及扩大样本应安排在固定方案后，不能把更多相关轮换当独立证据。','',
        '本轮没有执行这项新规则确认，也没有改变原任务定义。具体建议是先确认已有两步证据是否能泛化；纯视觉缺乏无图知识调用仍作为单独待解决问题，不能宣称已有完整四方向的成功方案。']
    (OUT/'interpretation_zh.md').write_text('\n'.join(lines)+'\n')
    extra={r['id']:r for r in rows(OUT/'diagnostic_probes.jsonl')}
    rr=responses['vision_apply']
    block=next(r['block'] for r in rr if r['condition']=='unseen_closed' and r['task']=='direct')
    examples=[{'input':extra[r['id']],'prediction':r} for r in rr if r.get('block')==block and r.get('shift')==0 and r['condition'] in ('unseen_closed','unseen_image')]
    assert len(examples)==8
    save(OUT/'controlled_examples.json',{'arm':'vision_apply','note':'Four name changes on the same observations/options, with and without the correct original learning image. Example chosen by first generated block, not score.','rows':examples})


if __name__=='__main__':summarize()
