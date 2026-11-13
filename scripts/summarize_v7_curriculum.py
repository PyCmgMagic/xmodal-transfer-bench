#!/usr/bin/env python3
"""Audit continuous curricula and compare fixed final results with midpoint diagnostics."""
from collections import Counter
import csv
import importlib.metadata
import math
from pathlib import Path
import shutil
import sys

import explore_v7_curriculum as pilot
import summarize_v7_prompt_loss_bridge as prior_summary
import run_v7_evaluation as ev
from run_v7_full_experiment import ROOT,rows,read,save,sha
from v7_gpu_resources import ledger

OUT=pilot.OUT
NAMES={'alternate':'逐batch交替','explain_match':'解释→辨认','match_explain':'辨认→解释'}


def metric(rr):
    m=pilot.previous.source.metric(rr)
    if rr and 'task' in rr[0]:
        for key,values in [('task',('direct','diagnosis')),('group',(0,1)),('shift',range(4)),('qid',pilot.previous.source.EXPECTED)]:
            m['by_'+('rule' if key=='qid' else key)]={str(v):pilot.previous.source.metric([r for r in rr if r[key]==v]) for v in values}
        m['by_evaluation_group']={label:m['by_task'][t] for t,label in [('direct','result_application'),('diagnosis','rule_report_audit')]}
        m['original_order']=pilot.previous.source.metric([r for r in rr if r['shift']==0])
        if all('block' in r for r in rr):
            groups=[[r for r in rr if (r['block'],r['shift'])==key] for key in sorted({(r['block'],r['shift']) for r in rr})]
            assert len(groups) in (8,32) and all(len(g)==4 and {r['gold'] for r in g}==set('ABCD') for g in groups)
            m['paired_checks']={'n':len(groups),'all4_same_answer':sum(len({r['predicted'] for r in g})==1 and all(r['predicted'] in ('A','B','C','D') for r in g) for g in groups),
                'all4_correct':sum(all(r['correct'] for r in g) for g in groups)}
            option_matrix={q:Counter() for q in pilot.previous.source.EXPECTED}
            for g in groups:
                options={r['gold']:r['qid'] for r in g}
                for r in g:option_matrix[r['qid']][options.get(r['predicted'],'invalid')]+=1
            m['selected_option_rule_result']={q:dict(counts) for q,counts in option_matrix.items()}
            m['selected_option_rule_result_note']='Maps the emitted option to the rule whose result it represents in this controlled block; this does not reveal internal retrieval or reasoning.'
    return m


def audit():
    import torch
    torch.set_num_threads(2)
    pilot.validate();pilot.previous.validate();pilot.previous.source.validate(pilot.previous.source.OUT)
    for root in (pilot.previous.OUT,pilot.previous.source.OUT):
        ref=read(root/'final_artifact_manifest.json')
        for path,h in ref['files'].items():assert sha(root/path)==h,path
    for filename in ('model_hash_validation.json','original_data_hash_validation.json'):
        old=read(pilot.previous.OUT/filename)
        for r in old['checked_files']:assert sha(Path(r['path']))==r['sha256']
        save(OUT/filename,{'status':'pass','reference':str(pilot.previous.OUT/filename),'checked_files':old['checked_files']})
    masks={r['id']:r['supervised_ids'] for r in rows(pilot.previous.OUT/'template_mask_audit.jsonl') if r['arm']=='text_equal'}
    replies={};lr_sums={};adapter_changes={}
    for arm in pilot.ARMS:
        modality,order=arm.split('_',1)
        tr=rows(OUT/(arm+'.jsonl'));old=rows(pilot.previous.OUT/(modality+'_equal.jsonl'))
        assert sorted(tr,key=lambda r:r['id'])==sorted(old,key=lambda r:r['id']) and len(tr)==512
        assert Counter(r['channel'] for r in tr)=={'explain':256,'match':256}
        assert all('test_visual' not in str(r) and 'Test input:' not in str(r) for r in tr)
        assert all(len(r.get('images',[]))==(modality=='vision') for r in tr)
        plan=read(OUT/(arm+'_plan.json'));assert len(plan)==128
        for i,p in enumerate(plan):
            batch=tr[i*4:i*4+4]
            assert p['row_ids']==[r['id'] for r in batch] and p['supervised_ids_sha256']==[ev.digest(masks[r['id']]) for r in batch]
            assert len({r['channel'] for r in batch})==1
        expected=(['explain','match']*64 if order=='alternate' else [order.split('_')[0]]*64+[order.split('_')[1]]*64)
        assert [p['channel'] for p in plan]==expected
        folder=OUT/arm;actual=rows(folder/'actual_order.jsonl');losses=rows(folder/'loss_batches.jsonl')
        assert len(actual)==len(losses)==128
        for i,(p,r,l) in enumerate(zip(plan,actual,losses)):
            assert r['step_before_update']==l['step_before_update']==i
            assert r['planned_row_ids']==p['row_ids'] and r['observed_supervised_ids_sha256']==p['supervised_ids_sha256']
            assert r['matches_planned_labels'] and r['planned_channel']==p['channel']
            assert math.isfinite(l['loss']) and l['equal_row_weights']==[.25]*4
            counts=l['supervised_tokens_including_template_suffix']
            assert (all(n==3 for n in counts) if p['channel']=='match' else all(n>3 for n in counts))
        args=read(pilot.checkpoint(arm,128)/'args.json')
        expected_args={'dataset_shuffle':False,'train_dataloader_shuffle':False,'train_sampling_strategy':'sequential','group_by_length':False,
            'packing':False,'padding_free':False,'loss_type':'v7_curriculum_equal','per_device_train_batch_size':4,'gradient_accumulation_steps':1,
            'learning_rate':1e-4,'num_train_epochs':1,'seed':42,'data_seed':42,'lora_rank':8,'lora_alpha':32,'lora_dropout':.05,
            'freeze_vit':True,'freeze_aligner':True,'max_length':8192}
        for k,v in expected_args.items():assert args[k]==v,(arm,k,args[k])
        assert args['adapters']==[str(pilot.previous.checkpoint('warmup'))]
        assert args['dataset']==[str(OUT/(arm+'.jsonl'))]
        from safetensors import safe_open
        with safe_open(str(pilot.previous.checkpoint('warmup')/'adapter_model.safetensors'),framework='pt',device='cpu') as parent, \
             safe_open(str(pilot.checkpoint(arm,128)/'adapter_model.safetensors'),framework='pt',device='cpu') as final:
            assert set(parent.keys())==set(final.keys())
            changed={k:float((final.get_tensor(k).float()-parent.get_tensor(k).float()).abs().max()) for k in final.keys()}
        assert all(math.isfinite(v) for v in changed.values()) and any(v>0 for v in changed.values())
        assert all(v>0 for k,v in changed.items() if 'lora_B' in k)
        adapter_changes[arm]={'total_tensors':len(changed),'changed_tensors':sum(v>0 for v in changed.values()),
            'all_lora_B_changed_from_warmup':True,'maximum_abs_delta':max(changed.values()),'per_tensor_max_abs_delta':changed}
        states={}
        for step in (64,128):
            cp=pilot.checkpoint(arm,step);st=read(cp/'trainer_state.json');states[step]=st
            assert st['global_step']==step and st['epoch']==step/128
            manifest=read(folder/('midpoint_manifest.json' if step==64 else 'checkpoint_manifest.json'))
            assert ev.checkpoint(cp)['sha256']==manifest['sha256']
            assert all((cp/f).exists() for f in ('optimizer.pt','scheduler.pt','rng_state.pth'))
        assert states[128]['log_history'][:len(states[64]['log_history'])]==states[64]['log_history']
        # Scheduler state distinguishes continuation from a silently restarted second fit.
        sched={n:torch.load(pilot.checkpoint(arm,n)/'scheduler.pt',map_location='cpu',weights_only=True) for n in (64,128)}
        assert sched[64]['last_epoch']==64 and sched[128]['last_epoch']==128
        # Match the total scheduled LR to the actual recorded objective schedule.
        from transformers import get_cosine_schedule_with_warmup
        param=torch.nn.Parameter(torch.zeros(1));optimizer=torch.optim.AdamW([param],lr=1e-4)
        scheduler=get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=7,num_training_steps=128)
        sums={'explain':0.,'match':0.}
        for p in plan:
            sums[p['channel']]+=optimizer.param_groups[0]['lr'];optimizer.step();scheduler.step()
        # Save as a reference; optimizer/scheduler snapshots and trainer logs
        # remain the authority for the actual installed schedule.
        lr_sums[arm]={'reference_sum_learning_rates_by_task':sums,'midpoint_scheduler':sched[64],
            'final_scheduler':sched[128],'note':'Cosine reference uses7warmup steps (ceil(128*.05)); order and LR exposure are not isolated.'}
        replies[arm]={}
        for scope,n in [('final',128),('midpoint',64)]:
            pp=rows(OUT/('final_probes.jsonl' if n==128 else 'midpoint_probes.jsonl'))
            replies[arm][scope]=prior_summary.audited_responses(folder/scope,pp,pilot.checkpoint(arm,n))
    # Task-specific batch content is identical across schedules for each modality.
    for modality in ('text','vision'):
        for channel in ('explain','match'):
            batches=[]
            for order in pilot.ORDERS:
                plan=read(OUT/(modality+'_'+order+'_plan.json'))
                batches.append([r['row_ids'] for r in plan if r['channel']==channel])
            assert batches[0]==batches[1]==batches[2]
    reviews=rows(OUT/'semantic_reviews.jsonl')
    knowledge={r['id']:r for r in rows(pilot.previous.source.OUT/'target_knowledge.jsonl')}
    expected={(a,s,r['id']) for a,scopes in replies.items() for s,rr in scopes.items() for r in rr if r['fmt']=='card'}
    assert len(reviews)==240 and {(r['arm'],r['scope'],r['id']) for r in reviews}==expected
    for r in reviews:
        actual=next(x for x in replies[r['arm']][r['scope']] if x['id']==r['id'])
        assert r['response_sha256']==ev.digest(actual['response']) and r['response']==actual['response']
        assert r['qid']==actual['qid'] and r['condition']==actual['condition'] and isinstance(r['correct'],bool)
        assert r['expected']==pilot.previous.source.card(knowledge[r['qid']]) and r['reason'] and r['method']
    for a in ('text_match_explain','vision_explain_match'):
        smoke=OUT/('smoke_'+a);assert read(smoke/'validated.json')['status']=='pass'
        st=read(smoke/'checkpoint_manifest.json');assert ev.checkpoint(Path(st['path']))['sha256']==st['sha256']
        assert len(rows(smoke/'reload/predictions.jsonl'))==2
    save(OUT/'learning_rate_exposure.json',lr_sums)
    save(OUT/'adapter_update_validation.json',adapter_changes)
    save(OUT/'completion_audit.json',{'status':'pass','new_training_jobs':6,'smoke_training_jobs':2,
        'new_final_answers':3408,'new_midpoint_answers':864,'new_primary_answers':4272,'smoke_reload_answers':4,
        'manual_recall_reviews':240,'original3000PNG_unchanged':True,'original_data_and_models_unchanged':True,
        'old_runs_unchanged':True,'same_rows_and_within_task_batches':True,'actual_order_checked_all768steps':True,
        'order_check_scope':'All actual supervised-label sequences match plan; shuffle disabled and sequential sampler configured. Input identities are not separately instrumented; identical targets cannot distinguish all row permutations.',
        'optimizer_and_scheduler_continuous_128steps':True,'fixed_final128primary':True,'all_final_adapters_changed_from_warmup':True,
        'raw_replies_and_inputs_verified':True,'application_test_data_not_in_training':True})
    return replies,reviews


def summarize():
    replies,reviews=audit();ss={};table=[];midids={p['id'] for p in rows(OUT/'midpoint_probes.jsonl')}
    for arm,scopes in replies.items():
        ss[arm]={}
        for scope,rr in scopes.items():
            metrics={c:metric([r for r in rr if r['condition']==c]) for c in sorted({r['condition'] for r in rr if r['fmt']=='mcq'})}
            own=[r for r in reviews if r['arm']==arm and r['scope']==scope]
            metrics['recall']={c:{'correct':sum(r['correct'] for r in own if r['condition']==c),
                'n':sum(r['condition']==c for r in own),
                'empty':sum(not r['response'].strip() for r in own if r['condition']==c),
                'truncated':sum(r['finish_reason']=='length' for r in own if r['condition']==c)} for c in sorted({r['condition'] for r in own})}
            ss[arm][scope]=metrics
            for c,m in metrics.items():
                if c!='recall':table.append({'arm':arm,'scope':scope,'condition':c,**{k:m[k] for k in ('n','correct','accuracy','valid_accuracy','missing','format_errors','empty','truncated')}})
        subset=[r for r in scopes['final'] if r['id'] in midids]
        assert len(subset)==144
        ss[arm]['final_same_midpoint_probes']={c:metric([r for r in subset if r['condition']==c]) for c in sorted({r['condition'] for r in subset if r['fmt']=='mcq'})}
    save(OUT/'summary.json',ss)
    with (OUT/'comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    save(OUT/'four_directions.json',[{'arm':a,'direction':d,'matches_training_source':d[0]==('T' if a.startswith('text') else 'V'),
        'original_v7':ss[a]['final']['named_'+d[-1]],'paired_application':ss[a]['final']['paired_'+d[-1]]} for a in pilot.ARMS for d in ('T2T','T2V','V2T','V2V')])
    save(OUT/'overall_application.json',{'note':'For each fixed adapter T2T/V2T share text inputs, T2V/V2V share image inputs. Report each answer once in combined metrics; mappings do not create new independent observations.',
        'arms':{a:{label:pilot.previous.source.metric([r for r in replies[a]['final'] if r['condition'] in conditions])
            for label,conditions in [('original_v7',('named_T','named_V')),('paired_application',('paired_T','paired_V'))]} for a in pilot.ARMS}})
    write_report(ss)
    write_interpretation(ss)
    save(OUT/'format_errors.json',[{'arm':a,'scope':s,'condition':c,**{k:m[k] for k in ('n','missing','format_errors','empty','truncated')}} for a,scopes in ss.items() for s,mm in scopes.items() if s!='final_same_midpoint_probes' for c,m in mm.items() if c!='recall'])
    scripts=[Path(__file__),ROOT/'scripts/explore_v7_curriculum.py',ROOT/'scripts/v7_curriculum_loss.py']
    for p in scripts:shutil.copy2(p,OUT/'source_snapshots'/p.name)
    save(OUT/'environment_manifest.json',{'python':sys.version,'timestamp':ev.now(),
        'versions':{p:importlib.metadata.version(p) for p in ('torch','transformers','ms-swift','peft','qwen-vl-utils','safetensors')},
        'scripts':{str(p):sha(p) for p in scripts}})
    with ledger(pilot.previous.source.LEDGER) as entries:active={k:v for k,v in entries.items() if str(OUT) in k}
    assert not active,active
    save(OUT/'final_artifact_manifest.json',{'status':'complete','completed_at':ev.now(),'new_answers':4276,
        'new_training_jobs':6,'manual_recall_reviews':240,'active_gpu_reservations_for_this_run':active,
        'files':{str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='final_artifact_manifest.json' and 'checkpoint' not in p.relative_to(OUT).parts and p.suffix in ('.json','.jsonl','.csv','.md','.py')}})
    print('All curricula, midpoint diagnostics, raw replies and original hashes audited.',flush=True)


def write_interpretation(ss):
    def score(a,c,scope='final'):
        r=ss[a][scope][c]
        return f"{r['correct']}/{r['n']}（{r['accuracy']:.1%}）"
    def recall(a,c='recall',scope='final'):
        r=ss[a][scope]['recall'][c]
        return f"{r['correct']}/{r['n']}"
    old=read(pilot.previous.OUT/'summary.json')['main']
    lines=['# 训练顺序探索：实际结论','',
        '本轮6组训练和全部评测已完成：Qwen3-VL-4B、同8条开发规则、每组512行、1epoch、128步、同warmup起点。TEXT/VISION分别测试交替、解释→辨认、辨认→解释。新增4272份主/诊断回答及4份smoke重载回答，240份规则回忆逐条核对。原v7与3000张PNG均未修改。','',
        '## 1. 分阶段没有同时修好记忆与应用','',
        f"文字交替组无图完整回忆{recall('text_alternate')}，配对文字{score('text_alternate','paired_T')}、图片{score('text_alternate','paired_V')}；无图改写定义辨认{score('text_alternate','recognition_para_noimage')}。这说明有的训练配置确实能学会在回忆提示下复述规则，不能笼统说所有知识都没学到。",'',
        f"解释→辨认：完整回忆{recall('text_explain_match')}，配对文字{score('text_explain_match','paired_T')}、图片{score('text_explain_match','paired_V')}，定义辨认{score('text_explain_match','recognition_para_noimage')}。",'',
        f"辨认→解释：完整回忆{recall('text_match_explain')}，配对文字{score('text_match_explain','paired_T')}、图片{score('text_match_explain','paired_V')}，定义辨认{score('text_match_explain','recognition_para_noimage')}。",'',
        '两种阶段顺序都没有保住交替组的8/8完整回忆。部分应用或定义辨认分数有所变化，不能据此宣布记忆和迁移均已解决。完整原v7成绩、direct/diagnosis及四方向映射见report_zh.md和summary.json。','',
        f"历史纯TEXT辨认参照：配对文字{old['text_match']['paired_T']['correct']}/128、图片{old['text_match']['paired_V']['correct']}/128，无图完整回忆0/8。其512行全部用于辨认，本轮辨认仅256行，因此它用于展示已有能力取舍，不能单独作为目标干扰的因果证据。",'',
        '## 2. 切换目标后存在能力取舍，但学习率仍是混杂因素','',
        f"TEXT解释→辨认：中途完整回忆{recall('text_explain_match',scope='midpoint')}，最终{recall('text_explain_match')}；无图改写辨认由{score('text_explain_match','recognition_para_noimage','midpoint')}变为{score('text_explain_match','recognition_para_noimage')}。其中Litkal中途能给出正确12≤value<21，最终下界错成13。",'',
        f"TEXT辨认→解释：中途完整回忆{recall('text_match_explain',scope='midpoint')}，最终{recall('text_match_explain')}；无图改写辨认由{score('text_match_explain','recognition_para_noimage','midpoint')}变为{score('text_match_explain','recognition_para_noimage')}。最后4条形状/颜色规则正确，4条数值区间仍错误。",'',
        '这与能力取舍相符，但不是纯遗忘的隔离实验：连续cosine调度下，阶段前半累计参考学习率约占79.5%，后半约占20.5%。该比例不是实际梯度贡献。第64步各组接触两类任务的次数也不同。只能判断这套固定预算课程，没有得到可靠修复；不能推论所有阶段训练都会失败。','',
        '## 3. 视觉端仍依赖学习图，名称尚未稳定用于新题','',
        '| VISION训练顺序 | 带原学习图回忆 | 去图完整回忆 | 配对文字 | 配对图片 |',
        '|---|---:|---:|---:|---:|']
    for order in pilot.ORDERS:
        a='vision_'+order
        lines.append('| '+NAMES[order]+' | '+recall(a,'recall_image')+' | '+recall(a)+' | '+score(a,'paired_T')+' | '+score(a,'paired_V')+' |')
    checks=ss['vision_explain_match']['final']['paired_V']['paired_checks']
    lines+=['',
        f"VISION解释→辨认在同一观察、同一图片、同一选项顺序下，只替换4个规则名称，{checks['all4_same_answer']}/{checks['n']}组仍给出同一个字母。4个名称分别对应4个不同正确答案，所以这种同答只能命中其中一个。32组包含4种位置轮换，实际只有8个语义题面，不能当作32次独立重复。",'',
        '具体direct例子：Dostan、Morkil、Ratlod、Lavrit在同一图上的正确选项分别是A、D、B、C，模型四次都答C。diagnosis版本也出现同样结果。这是输出对名称缺乏区分的证据，不能据此窥知模型内部究竟检索了什么。完整输入及逐条回答见controlled_examples.json。','',
        'VISION训练每行都带原学习图，而事实答案和候选定义含明确文字事实。结果应称图像条件SFT诊断，不能作为无文字事实监督的纯视觉学习证据。本轮未加入去图回放。','',
        '## 4. 正确复述也没有保证正确应用','',
        'TEXT解释→辨认最终checkpoint对Dostan能完整回答“triangle OR blue”。但在原v7文字direct题中，A=蓝色圆形、B=蓝色三角形、C=红色圆形、D=红色三角形，正确应保留A、B、D；模型选择只保留A、B，漏掉仅满足形状条件的D。',
        '',
        '因此至少有两条待解决链路：视觉学习后凭名称恢复具体事实，以及在应用提示下正确调用和执行已学规则。回忆、定义辨认和应用是不同提示任务；这些观察不把全部失败强行归为一种原因。给正确规则的oracle另列，不能混入无材料迁移主成绩。','',
        '## 5. 这次结论如何核验、接下来优先改什么','',
        '已经核对原始回答、输入SHA、实际答案解析、最终checkpoint哈希、相对warmup的LoRA张量变化、768步监督目标顺序、连续optimizer/scheduler状态及原图/原数据/历史结果完整性。选择题的缺失、空输出、格式错误和截断见format_errors.json；自由回忆截断单列于summary.json，不与选择题解析混算。','',
        '当前不宜直接把这个课程扩成1000条。若下一轮要检验纯顺序/遗忘，应先在同512行、同起点下用固定学习率控制两类任务的学习率暴露；这项尚未运行。',
        '',
        '若优先争取实际应用收益，更值得新增的是独立的规则应用训练桥接：从学习规则生成新的文字候选对象与结果，训练模型在应用提示下按指定规则作答；原测试观察、测试图和测试答案均不用于生成训练例。direct和diagnosis保持两种独立任务。VISION继续使用现有学习图，并明确桥接题中额外文字信息的范围。它会改变训练目标，应单列实验，不能冒充原材料式学习的提升。这项也尚未运行。',
        '',
        '后续确认须另外固定未用于反复调试的规则和多个seed。当前同8条规则、单seed，只给机制排查证据，不代表全量1000条已修复。训练具体问答与顺序见TRAINING_EXAMPLE_ZH.md。']
    (OUT/'interpretation_zh.md').write_text('\n'.join(lines)+'\n')


def write_report(ss):
    def score(arm,c,scope='final',original=False):
        m=ss[arm][scope][c];m=m['original_order'] if original else m
        return f"{m['correct']}/{m['n']}（{m['accuracy']:.1%}）"
    def label(arm):
        modality,order=arm.split('_',1)
        return modality.upper()+' '+NAMES[order]
    prior=read(pilot.previous.OUT/'summary.json')['main']
    lines=['# 训练顺序：解释、辨认与交替对照','',
        '六组Qwen3-VL-4B LoRA，固定同8条开发规则、同一warmup起点、每组512行、1epoch、128步。原3000张PNG及v7数据保持不变。新生成4272次回答，另4次smoke重载回答；240份完整规则回忆逐条审核。','',
        '## 固定第128步主评测','',
        '配对题：每输入模态32个语义问题，4种选项位置＝128次回答。原v7：16题×4位置＝64次。两套题分开，不额外给正确学习材料。辨认诊断的选项包含规则定义，单列；规则回忆用训练system，去掉system的回忆也保存。','',
        '| 模型 | 配对文字 | 配对图片 | 原v7文字 | 原v7图片 | 无图改写定义辨认 | 无图完整回忆 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for a in pilot.ARMS:
        r=ss[a]['final']['recall']['recall']
        lines.append('| '+label(a)+' | '+' | '.join(score(a,c) for c in ('paired_T','paired_V','named_T','named_V','recognition_para_noimage'))+f" | {r['correct']}/{r['n']} |")
    lines+=['','历史参照（同题输入、原短答案协议，复用原回答，不计新请求）：','',
        '| 参照 | 配对文字 | 配对图片 | 原v7文字 | 原v7图片 |','|---|---:|---:|---:|---:|']
    for a in ('warmup','text_match','text_mixed','text_equal','vision_mixed','vision_equal'):
        cells=[]
        for c in ('paired_T','paired_V','named_T','named_V'):
            r=prior[a][c];cells.append(f"{r['correct']}/{r['n']}（{r['accuracy']:.1%}）")
        lines.append('| '+a+' | '+' | '.join(cells)+' |')
    lines+=['','历史单独辨认组的目标配比与本轮不同，不能用其分数差单独证明目标干扰；本轮训练顺序的主要对照为新建逐batch交替组。历史等权混合的batch组成及打乱设置也不同，不能把新交替组当作其逐字复现。']
    lines+=['','## 原v7原顺序16题','', '| 模型 | 文字 | 图片 |','|---|---:|---:|']
    for a in pilot.ARMS:lines.append('| '+label(a)+' | '+score(a,'named_T',original=True)+' | '+score(a,'named_V',original=True)+' |')
    lines+=['','## 第64步→第128步：完全相同的中途诊断问题','',
        '下面配对题仅取选项shift0，每输入模态32道；原v7为16道。第128步按同一ID子集取分，不能用第128步完整128次回答与中途32题直接相比。完整回忆各8条，定义辨认含4个选项位置共32次。','',
        '| 模型 | 无图完整回忆 | 无图辨认 | 配对文字 | 配对图片 |','|---|---:|---:|---:|---:|']
    for a in pilot.ARMS:
        before=ss[a]['midpoint']['recall']['recall'];after=ss[a]['final']['recall']['recall']
        cells=[f"{before['correct']}/8 → {after['correct']}/8"]
        for c in ('recognition_para_noimage','paired_T','paired_V'):
            cells.append(score(a,c,'midpoint')+' → '+score(a,c,'final_same_midpoint_probes'))
        lines.append('| '+label(a)+' | '+' | '.join(cells)+' |')
    lines+=['','## direct与diagnosis','', '| 模型 | direct文字 | diagnosis文字 | direct图片 | diagnosis图片 |','|---|---:|---:|---:|---:|']
    for a in pilot.ARMS:
        cells=[]
        for c,t in [('paired_T','direct'),('paired_T','diagnosis'),('paired_V','direct'),('paired_V','diagnosis')]:
            r=ss[a]['final'][c]['by_task'][t];cells.append(f"{r['correct']}/{r['n']}（{r['accuracy']:.1%}）")
        lines.append('| '+label(a)+' | '+' | '.join(cells)+' |')
    lines+=['','## 同一题面更换四个规则名称','',
        '每组四个名称对应四个不同正确选项；四次全部回答相同选项，只能答对其中一次。下表为这种同答现象的组数。每输入模态8个语义题面×4种选项位置，共32组；位置轮换并非独立重复。','',
        '| 模型 | 文字四名同答 | 图片四名同答 | 文字四名全对 | 图片四名全对 |',
        '|---|---:|---:|---:|---:|']
    for a in pilot.ARMS:
        t=ss[a]['final']['paired_T']['paired_checks'];v=ss[a]['final']['paired_V']['paired_checks']
        lines.append('| '+label(a)+f" | {t['all4_same_answer']}/{t['n']} | {v['all4_same_answer']}/{v['n']} | {t['all4_correct']}/{t['n']} | {v['all4_correct']}/{v['n']} |")
    lines+=['','## 实施与边界','',
        '各模态三组使用相同的256行解释与256行辨认，每条规则各32行；同一目标内部的64个batch内容及顺序完全相同。逐batch交替组E,M各一批交替；分阶段组前64步一种目标，后64步另一种。batch4/accum1、样本等权CE、LR1e-4、连续cosine调度、LoRA r8/alpha32/dropout0.05、冻结视觉与对齐模块、seed42、max8192。',
        '',
        '关闭dataset_shuffle与train_dataloader_shuffle；每一步按标签token哈希核对计划，全部768步验证。第64步checkpoint仅诊断；主模型固定128步，optimizer与scheduler未重置。顺序改变两种任务遇到早期/晚期学习率的机会，结果不能把纯遗忘与学习率因素完全分开。',
        '顺序核验记录的是实际监督标签序列，结合关闭shuffle及sequential采样设置；没有另外记录每一步的完整输入身份。相同目标文本或相同选项字母的样本互换，单靠标签哈希无法区分。',
        '',
        '交替组中途看过解释和辨认各32批；分阶段组中途看过单一目标64批。第64步是暴露量不同的描述性诊断。训练中没有正式测试观察或应用答案，未按测试结果选checkpoint，也没有针对错误重跑。',
        '',
        'VISION所有训练行均带原学习图；解释目标和辨认候选包含文字事实，不能称无文字事实监督的纯视觉训练。本轮没有去图回放。未开原生CoT、未做材料PT。',
        '',
        '这是同8条开发规则、单seed、两种选择器结构的排查，选项轮换相互相关，不能据此外推全量1000条或宣称统计显著。四方向映射见four_directions.json；同输入复用不算新独立题。',
        '',
        'summary.json含逐规则、结构、题型、选项位置和同图换名敏感性；format_errors.json区分无效/空白/截断，正式准确率将格式错误计错，有效输出准确率另存。semantic_reviews.jsonl保存240份人工事实审核；所有原始回答位于<arm>/final或midpoint/predictions.jsonl。',
        '',
        '结论与具体失败例子见interpretation_zh.md，训练输入和顺序示例见TRAINING_EXAMPLE_ZH.md。']
    (OUT/'report_zh.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':summarize()
