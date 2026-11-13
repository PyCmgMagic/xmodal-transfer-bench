#!/usr/bin/env python3
"""Audit full-parameter SFT against identical-init LoRA controls."""
import csv
import json
import math
import shutil
from collections import Counter
from pathlib import Path

import explore_v7_full_parameter as p
import summarize_v7_application_binding as previous
from run_v7_full_experiment import rows,read,save,sha
import run_v7_evaluation as ev
from v7_gpu_resources import ledger

OUT=p.OUT
LABELS={'merged_parent':'共同起点（合并warmup）','full_text':'TEXT语言全参数','full_vision':'VISION语言全参数',
    'full_all_vision':'VISION整模型全参数','lora_text':'TEXT匹配LoRA','lora_vision':'VISION匹配LoRA'}


def record_manual_reviews(arm,decisions):
    """Persist explicit human-readable semantic decisions, never infer them by score."""
    knowledge={r['id']:r for r in rows(OUT/'target_knowledge.jsonl')}
    path=OUT/'semantic_reviews.jsonl';saved=rows(path) if path.exists() else []
    found=set()
    for r in rows(OUT/arm/'evaluation/predictions.jsonl'):
        key=(r['condition'],r['qid'].split('_')[1])
        if r['fmt']!='card' or key not in decisions:continue
        correct,reason=decisions[key];found.add(key)
        item={'arm':arm,'id':r['id'],'qid':r['qid'],'condition':r['condition'],'response':r['response'],
            'response_sha256':ev.digest(r['response']),'expected':p.prior.previous.source.card(knowledge[r['qid']]),
            'correct':correct,'reason':reason,'finish_reason':r['finish_reason'],
            'method':'Codex逐条读完整原答，核对名称、事实、AND/OR、区间端点及附加矛盾；允许同义表达。非独立人工标注。'}
        old=[x for x in saved if (x['arm'],x['id'])==(arm,r['id'])]
        if old:assert old==[item],key
        else:saved.append(item)
    assert found==set(decisions),set(decisions)-found
    ev.save_rows(path,saved);print('manual reviews',len(saved))


def audit_responses(arm):
    probes=rows(OUT/'main_probes.jsonl')+rows(OUT/'diagnostic_probes.jsonl')
    folder=OUT/arm/'evaluation';cfg=read(folder/'eval_manifest.json')
    assert cfg['model']==p.identity(arm) and cfg['probes_sha256']==ev.digest(probes)
    assert cfg['temperature']==0 and cfg['thinking'] is False and cfg['max_tokens']=={'mcq':16,'card':384}
    run=p.Runner(folder,cfg,'0');run.audit(probes,1016)
    return run.rr


def dense_deltas(arm):
    import torch
    from safetensors import safe_open
    torch.set_num_threads(4)
    cp=p.checkpoint(arm)
    indexes={}
    for key,root in [('base',p.BASE),('final',cp)]:
        if (root/'model.safetensors.index.json').exists():
            indexes[key]=read(root/'model.safetensors.index.json')['weight_map']
        else:
            with safe_open(str(root/'model.safetensors'),framework='pt') as f:indexes[key]={k:'model.safetensors' for k in f.keys()}
    extra=set(indexes['final'])-set(indexes['base'])
    assert not set(indexes['base'])-set(indexes['final'])
    assert extra in (set(),{'lm_head.weight'})
    if extra:
        assert read(cp/'config.json')['tie_word_embeddings'] is True
        # ZeRO's gathered state_dict can serialize the tied output head twice.
        with safe_open(str(cp/indexes['final']['lm_head.weight']),framework='pt') as f:
            head=f.get_tensor('lm_head.weight')
        with safe_open(str(cp/indexes['final']['model.language_model.embed_tokens.weight']),framework='pt') as f:
            embedding=f.get_tensor('model.language_model.embed_tokens.weight')
        assert torch.equal(head,embedding),'Saved tied head and embedding disagree'
    stats={'language':Counter(),'vision':Counter()};layer_counts=Counter();handles={}
    try:
        for name in sorted(indexes['base']):
            ts=[]
            for key,root in [('base',p.BASE),('final',cp)]:
                path=str(root/indexes[key][name])
                if path not in handles:handles[path]=safe_open(path,framework='pt')
                ts.append(handles[path].get_tensor(name))
            a,b=ts;assert a.shape==b.shape and torch.isfinite(b).all()
            scope='vision' if name.startswith('model.visual.') else 'language'
            changed=not torch.equal(a,b);s=stats[scope];s['tensors']+=1;s['changed_tensors']+=int(changed)
            s['parameters']+=b.numel()
            if changed:
                layer='.'.join(name.split('.')[:4]);layer_counts[layer]+=1
        assert stats['language']['changed_tensors']>0
        if arm=='full_all_vision':assert stats['vision']['changed_tensors']>0
        else:assert stats['vision']['changed_tensors']==0
    finally:
        for f in handles.values():f.__exit__(None,None,None)
    return {'scopes':stats,'changed_tensor_groups':layer_counts,'all_final_tensors_finite':True,
        'extra_serialized_tied_head_verified_equal':bool(extra)}


def audit_training(arm):
    cp=p.checkpoint(arm);state=read(cp/'trainer_state.json');args=read(cp/'args.json')
    assert state['global_step']==128 and state['epoch']==1
    expected={'learning_rate':1e-5,'num_train_epochs':1,'per_device_train_batch_size':4,'gradient_accumulation_steps':1,
        'loss_type':'v7_equal_example','seed':42,'data_seed':42,'max_length':8192,'packing':False,'padding_free':False,
        'weight_decay':.1,'adam_beta1':.9,'adam_beta2':.95,'adam_epsilon':1e-8,'max_grad_norm':1.,'optim':'adamw_torch'}
    for k,v in expected.items():assert args[k]==v,(arm,k,args.get(k))
    assert args['model']==str(p.BASE) and not args['adapters']
    assert args['dataset']==[str(OUT/('text.jsonl' if arm.endswith('text') else 'vision.jsonl'))]
    assert args['freeze_vit']==args['freeze_aligner']==(arm!='full_all_vision')
    assert args['tuner_type']==('lora' if arm.startswith('lora') else 'full')
    assert p.identity(arm)==read(OUT/arm/'checkpoint_manifest.json')['model']
    loss=rows(OUT/arm/'loss_batches.jsonl');assert len(loss)==128
    assert {b['step_before_update'] for b in loss}==set(range(128))
    counts=[]
    for b in loss:
        assert math.isfinite(b['loss']) and b['equal_row_weights']==[.25]*4
        assert abs(b['loss']-sum(b['per_example_mean_loss'])/4)<1e-5
        counts+=b['supervised_tokens_including_template_suffix']
    assert Counter(n==3 for n in counts)=={True:384,False:128}
    audit=rows(OUT/arm/'parameter_audit.jsonl');start=next(r for r in audit if r['event']=='train_begin')
    steps=[r for r in audit if r['event']=='step_end'];assert {s['step'] for s in steps}=={1,128}
    groups=start['groups'];result={'trainable_groups':groups,'peak_cuda_allocated_mib':max(s['cuda_max_allocated_mib'] for s in steps)}
    if arm.startswith('full'):
        assert groups['language']['trainable']==groups['language']['parameters']
        if arm=='full_all_vision':assert groups['vision']['trainable']==groups['vision']['parameters']
        else:assert groups['vision']['trainable']==0
        assert all(s['master_weights'] for s in steps)
        assert all(t['dtype']=='torch.float32' and t['device']=='cpu' for s in steps for t in s['master_weights'])
        result['fp32_cpu_master_weights_verified']=True
        result['actual_weight_updates']=dense_deltas(arm)
    else:
        assert groups['vision']['trainable']==0
        from safetensors import safe_open
        with safe_open(str(cp/'adapter_model.safetensors'),framework='pt') as f:
            tensors={k:f.get_tensor(k) for k in f.keys()}
        assert len(tensors)==504 and all(t.isfinite().all() for t in tensors.values())
        assert not any('.visual.' in k for k in tensors)
        assert all(t.abs().max()>0 for k,t in tensors.items() if 'lora_B' in k)
        result['nonzero_trained_LoRA_B']=True
    smoke=OUT/('smoke_'+arm)
    assert read(smoke/'validated.json')['status']=='pass'
    cfg=read(smoke/'evaluation/eval_manifest.json');pp=rows(OUT/'diagnostic_probes.jsonl')
    pp=[next(x for x in pp if x['condition']==c) for c in ('unseen_closed','unseen_image')]
    p.Runner(smoke/'evaluation',cfg,'0').audit(pp,2)
    return result


def main():
    p.validate()
    old=read(p.prior.OUT/'final_artifact_manifest.json')
    for rel,want in old['files'].items():assert sha(p.prior.OUT/rel)==want,rel
    for mod in ('text','vision'):assert sha(OUT/(mod+'.jsonl'))==sha(p.prior.OUT/(mod+'_apply.jsonl'))
    for f in ('main_probes.jsonl','diagnostic_probes.jsonl','target_knowledge.jsonl'):
        assert sha(OUT/f)==sha(p.prior.OUT/f)
    encoding=read(OUT/'encoded_training_validation.json')
    assert encoding['status']=='pass' and encoding['rows']==1024 and encoding['all_questions_and_targets_preserved']
    assert max(r['input_tokens'] for r in encoding['records'])<8192
    for filename in ('model_hash_validation.json','original_data_hash_validation.json'):
        old=read(p.prior.OUT/filename)
        for r in old['checked_files']:assert sha(Path(r['path']))==r['sha256']
    for image in rows(p.DATA/'image_hashes.jsonl'):
        assert sha(p.DATA/image['path'])==image['source_sha256']==image['repaired_sha256']
    result={};allresponses={};training={}
    for arm in ('merged_parent',)+p.ARMS:
        rr=audit_responses(arm);allresponses[arm]=rr
        result[arm]={c:previous.metric([r for r in rr if r['condition']==c]) for c in sorted({r['condition'] for r in rr if r['fmt']=='mcq'})}
        result[arm]['answer_letters']={c:dict(Counter(r['response'] for r in rr if r['condition']==c)) for c in result[arm] if c!='answer_letters'}
        if arm in p.ARMS:training[arm]=audit_training(arm)
    length_reference=[r['supervised_tokens_including_template_suffix'] for r in rows(OUT/'lora_text/loss_batches.jsonl')]
    assert all([r['supervised_tokens_including_template_suffix'] for r in rows(OUT/a/'loss_batches.jsonl')]==length_reference for a in p.ARMS)
    save(OUT/'supervision_alignment.json',{'status':'pass','same512training_targets_in_same_file_order':True,
        'same_per_step_supervised_length_sequence':True,'note':'Length traces check training alignment but do not independently identify rows with identical answer lengths.'})
    reviews=rows(OUT/'semantic_reviews.jsonl')
    # Include the merged parent so each newly generated free recall has a recorded decision.
    expected={(a,r['id']) for a,rr in allresponses.items() for r in rr if r['fmt']=='card'}
    assert len(reviews)==144 and {(r['arm'],r['id']) for r in reviews}==expected
    for r in reviews:
        actual=next(x for x in allresponses[r['arm']] if x['id']==r['id'])
        assert r['response']==actual['response'] and r['response_sha256']==ev.digest(actual['response'])
        assert isinstance(r['correct'],bool) and r['reason']
        knowledge=next(k for k in rows(OUT/'target_knowledge.jsonl') if k['id']==r['qid'])
        assert r['expected']==p.prior.previous.source.card(knowledge)
    for a in result:
        rec=[r for r in reviews if r['arm']==a]
        result[a]['recall']={c:{'n':sum(r['condition']==c for r in rec),'correct':sum(r['correct'] for r in rec if r['condition']==c),
            'truncated':sum(r['finish_reason']=='length' for r in rec if r['condition']==c)} for c in ('recall','recall_image','plain_recall')}
    save(OUT/'training_validation.json',training);save(OUT/'summary.json',result)
    tab=[{'arm':a,'condition':c,**{k:m[k] for k in ('n','correct','accuracy','valid_accuracy','missing','format_errors','empty','truncated')}}
        for a,s in result.items() for c,m in s.items() if c not in ('answer_letters','recall')]
    with (OUT/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(tab[0]));w.writeheader();w.writerows(tab)
    save(OUT/'format_errors.json',tab)
    gates={}
    for a in p.ARMS:
        s=result[a];parent=result['merged_parent']
        gates[a]={'seen_closed_capacity_80pct':s['seen_closed']['accuracy']>=.8,
            'seen_image_capacity_80pct':s['seen_image']['accuracy']>=.8,
            'unseen_closed_60pct':s['unseen_closed']['accuracy']>=.6,
            'paired_T_gain':s['paired_T']['accuracy']-parent['paired_T']['accuracy'],
            'paired_V_gain':s['paired_V']['accuracy']-parent['paired_V']['accuracy']}
        g=gates[a];g['both_paired_gains_10pp']=g['paired_T_gain']>=.1 and g['paired_V_gain']>=.1
        g['all_gates_pass']=(g['seen_closed_capacity_80pct'] if a.endswith('text') else g['seen_image_capacity_80pct']) and g['unseen_closed_60pct'] and g['both_paired_gains_10pp']
    save(OUT/'screening_gates.json',gates)
    save(OUT/'four_directions.json',{'note':'One fixed input/adapter response is reused for direction aliases; aliases are not independent observations.',
        'rows':[{'arm':a,'direction':d,'source_matches_direction':d[0]==('T' if a.endswith('text') else 'V'),
            'original':result[a]['named_'+d[-1]],'paired':result[a]['paired_'+d[-1]]} for a in p.ARMS for d in ('T2T','T2V','V2T','V2V')]})
    write_figure(result);write_report(result,gates,training)
    write_interpretation(result,allresponses)
    for file in (Path(__file__),p.ROOT/'scripts/explore_v7_full_parameter.py',p.AUDIT):shutil.copy2(file,OUT/'source_snapshots'/file.name)
    with ledger(p.LEDGER) as entries:active={k:v for k,v in entries.items() if str(OUT) in k}
    assert not active,active
    save(OUT/'completion_audit.json',{'status':'complete','trained_arms':5,'steps_each':128,'new_answers':6106,'manual_recall_reviews':144,
        'recall_reviewer':'Codex, individual full-response semantic review; not independent human annotation',
        'identical_dense_initialization':True,'fixed_final_checkpoints':True,'all3000PNGs_unchanged':True,
        'original_data_and_base_models_unchanged':True,'previous_runs_unchanged':True,'active_gpu_reservations':active})
    save(OUT/'final_artifact_manifest.json',{'status':'complete','completed_at':ev.now(),
        'files':{str(f.relative_to(OUT)):sha(f) for f in sorted(OUT.rglob('*')) if f.is_file() and f.name!='final_artifact_manifest.json'
            and not {'checkpoint','merged_warmup','torch_extensions','tmp'}&set(f.relative_to(OUT).parts) and f.suffix in ('.json','.jsonl','.csv','.md','.py','.png')}})
    print('Complete: five training arms,6106new answers,144manual reviews; weights/inputs/outputs/images audited.',flush=True)


def write_report(ss,gates,training):
    def score(a,c):
        r=ss[a][c];return f"{r['correct']}/{r['n']}（{r['accuracy']:.1%}）"
    lines=['# 全参数微调对照：实际结果','',
        '本轮5组训练和全部6106次新回答已完成，144份规则回忆均由Codex逐条阅读完整原答并记录语义判定，未经独立人工复核。实用筛查门槛通过'+str(sum(g['all_gates_pass'] for g in gates.values()))+'/5组。门槛是预先固定的开发筛查标准，不是显著性检验。','',
        '目标：检查扩大参数更新范围，是否改善无学习材料的知识应用与跨模态迁移。同8条开发规则，Qwen3-VL-4B，1epoch/128步，各512行＝128解释＋384应用选择题。不扩到1000条，不改原图或正式评测提示。','',
        '所有新组从同一份合并warmup的BF16模型出发；LoRA为该模型上的新r8/alpha32/dropout0.05适配器。原历史LoRA续训结果只作历史参考。统一batch4、LR1e-5、cosine/5%warmup、seed42、max8192、样本等权CE、AdamW(beta.9/.95,eps1e-8,weight_decay.1)、grad_clip1。','',
        '全参数组采用DeepSpeed ZeRO2 CPU卸载，FP32主权重和Adam状态实际核验；不是8bit模型或8bit优化器。语言全参数组冻结视觉与对齐层；整模型VISION组全部放开。初始化、训练参数范围、实际权重变化见training_validation.json。','',
        '额外用实际Swift模板逐条编码1024份TEXT/VISION训练输入，核对完整问题和监督目标均保留；最大1259token，低于8192上限。结果见encoded_training_validation.json。','',
        '| 组别 | 训练内闭卷 | 未见组合闭卷 | 未见组合带学习图 | 配对文字闭卷 | 配对图片闭卷 |','|---|---:|---:|---:|---:|---:|']
    for a in ('merged_parent',)+p.ARMS:lines.append('| '+LABELS[a]+' | '+' | '.join(score(a,c) for c in ('seen_closed','unseen_closed','unseen_image','paired_T','paired_V'))+' |')
    lines+=['','seen_closed64次；unseen_closed256次；unseen_image64次，且只用shift0。带图不算闭卷；有选项轮换的分母不是独立题数。','',
        '## 完全匹配的64道未见题','', '| 组别 | 去掉学习图 | 给原学习图 |','|---|---:|---:|']
    for a in ('merged_parent',)+p.ARMS:
        m=ss[a]['unseen_closed']['original_order'];lines.append('| '+LABELS[a]+f" | {m['correct']}/{m['n']}（{m['accuracy']:.1%}） | "+score(a,'unseen_image')+' |')
    lines+=['','## 原v7题与正确材料控制','', '| 组别 | 原题文字 | 原题图片 | oracle文字 | oracle图片 |','|---|---:|---:|---:|---:|']
    for a in ('merged_parent',)+p.ARMS:lines.append('| '+LABELS[a]+' | '+' | '.join(score(a,c) for c in ('named_T','named_V','oracle_T','oracle_V'))+' |')
    lines+=['','原题每输入模态16题×4位置；oracle16题只用shift0，提供正确文字规则，是带材料控制，不是迁移成绩或严格上界。','',
        '## direct与diagnosis分别计分','', '| 组别 | direct文字 | diagnosis文字 | direct图片 | diagnosis图片 |','|---|---:|---:|---:|---:|']
    for a in ('merged_parent',)+p.ARMS:
        cells=[]
        for c,t in [('paired_T','direct'),('paired_T','diagnosis'),('paired_V','direct'),('paired_V','diagnosis')]:
            m=ss[a][c]['by_task'][t];cells.append(f"{m['correct']}/{m['n']}（{m['accuracy']:.1%}）")
        lines.append('| '+LABELS[a]+' | '+' | '.join(cells)+' |')
    lines+=['','## 每份自由回忆逐条语义核对（Codex）','', '| 组别 | 训练system无图 | 给原学习图 | 普通提示无图 |','|---|---:|---:|---:|']
    for a in ('merged_parent',)+p.ARMS:
        lines.append('| '+LABELS[a]+' | '+' | '.join(str(ss[a]['recall'][c]['correct'])+'/8' for c in ('recall','recall_image','plain_recall'))+' |')
    lines+=['','## 名称区分与输出质量','', '| 组别 | 配对文字四名同答 | 配对图片四名同答 |','|---|---:|---:|']
    for a in ('merged_parent',)+p.ARMS:
        cells=[]
        for c in ('paired_T','paired_V'):
            n=ss[a][c]['name_checks'];cells.append(f"{n['four_names_same_valid_answer']}/{n['n']}")
        lines.append('| '+LABELS[a]+' | '+' | '.join(cells)+' |')
    lines+=['','同场景四名分别对应不同正确答案；大量同答说明输出对名称区分不足，不能据此断言内部没有存储任何事实。格式错误、缺失、空答与截断见format_errors.json。','',
        '## 解释边界','',
        '同一批开发规则、单seed，无独立规则确认。whole-VLM比较除语言权重外还改变视觉更新范围。LoRA与dense采用不同参数化/精度实现，LoRA还具有dropout，结果不能单独证明低秩是唯一原因。原材料PT没有在本轮测试。','',
        'VISION用原学习图＋文字练习对象，助手答案有明确文字事实或人工规则求解标签；不能据其分数证明只由像素获得知识。带图高分和图像来源文字桥接不能冒充纯视觉闭卷迁移。','',
        '学习增益不足时不计算迁移保持率。预先固定的门槛和各组结果见decision_criteria.json/screening_gates.json。完整输入/原答/输入SHA在各组evaluation/predictions.jsonl；四方向归属见four_directions.json。']
    lines+=['','训练损失与闭卷应用对照图：','', '![training and application]('+str(OUT/'training_and_application.png')+')', '',
        '图中MCQ损失包含同样的字母/结束符/换行3个监督token，8步窗口按实际MCQ样本取均值。ln(4)/3仅是字母均匀不确定且后缀完美时的解释性参照，不是实测首token损失。准确率为描述统计，重复轮换不独立。']
    (OUT/'report_zh.md').write_text('\n'.join(lines)+'\n')


def write_interpretation(ss,allresponses):
    def score(a,c):
        m=ss[a][c];return f"{m['correct']}/{m['n']}（{m['accuracy']:.1%}）"
    evidence={}
    for a in p.ARMS:
        train=rows(OUT/('text.jsonl' if a.endswith('text') else 'vision.jsonl'))
        batches=rows(OUT/a/'loss_batches.jsonl')
        losses=[v for b in batches[-32:] for n,v in zip(b['supervised_tokens_including_template_suffix'],b['per_example_mean_loss']) if n==3]
        mcq=[r for r in allresponses[a] if r['fmt']=='mcq']
        evidence[a]={'training_mcq_targets':dict(Counter(r['messages'][-1]['content'] for r in train if r.get('fmt')=='mcq')),
            'evaluation_mcq_responses':dict(Counter(r['response'] for r in mcq)),
            'last32step_mcq_n':len(losses),'last32step_mcq_mean_loss':sum(losses)/len(losses)}
    probes={r['id']:r for r in rows(OUT/'main_probes.jsonl')+rows(OUT/'diagnostic_probes.jsonl')}
    examples=[]
    for c in ('plain_recall','paired_T','oracle_T'):
        r=next(r for r in allresponses['full_text'] if r['condition']==c)
        examples.append({'selection':'first prediction in fixed condition, descriptive example only','arm':'full_text',
            'id':r['id'],'condition':c,'input_messages':probes[r['id']]['messages'],
            'gold':r.get('gold'),'response':r['response'],'input_sha256':r['input_sha256']})
    save(OUT/'interpretation_evidence.json',{'arms':evidence,'examples':examples,
        'loss_reference':{'value':math.log(4)/3,'interpretation':'Uniform uncertainty over four answer letters with perfectly predicted two-token suffix; not an observed first-token loss.'}})
    ft=ss['full_text'];fv=ss['full_vision'];fa=ss['full_all_vision']
    lines=['# 全参数训练改善了什么，为什么还不能扩到1000条','',
        '研究目标仍是：训练学到新规则，测试时拿走学习材料，观察新文字/图片问题上的应用能力。本轮只检查“扩大可更新参数范围”这条路径，原图、题面与评测标签未改。8条开发规则、单seed、1epoch，不代表所有全参数训练配置。','',
        '## 1. 文字全参数：记忆改善，但选择题输出塌缩','',
        '语言参数全量更新后，普通无图提示与训练system无图提示都完整回忆8/8条规则；共同起点与匹配LoRA这两项都是0/8。这是本轮明确观察到的事实回忆改善。','',
        f"但该组992次选择题全部回答C。配对文字{score('full_text','paired_T')}、配对图片{score('full_text','paired_V')}；即使把正确文字规则放进问题，oracle文字仍{score('full_text','oracle_T')}，共同起点为{score('merged_parent','oracle_T')}。因此失败并不只是规则没记住或图片理解差；这套训练同时造成了明显的选择题输出退化。不能把25%解释成健康的随机猜测。",'',
        '训练384条选择题的A/B/C/D各96条，四个位置均衡。权重审计确认语言层实际更新，训练模板逐条编码确认问题与监督答案均完整保留。标签偏向C、LoRA未加载、长问题被截断和答案解析失败都不符合这些核验结果。','',
        '## 2. 图片全参数：带图与去图要分开','',
        f"仅更新语言侧的VISION组：给学习图回忆{fv['recall']['recall_image']['correct']}/8，无图回忆{fv['recall']['recall']['correct']}/8，普通无图回忆{fv['recall']['plain_recall']['correct']}/8。训练内带图{score('full_vision','seen_image')}，训练内去图{score('full_vision','seen_closed')}。",'',
        f"连视觉/对齐层一起放开的VISION整模型组：给学习图回忆{fa['recall']['recall_image']['correct']}/8，无图回忆{fa['recall']['recall']['correct']}/8，普通无图回忆{fa['recall']['plain_recall']['correct']}/8。训练内带图{score('full_all_vision','seen_image')}，训练内去图{score('full_all_vision','seen_closed')}。",'',
        '整模型组在训练system无图条件下，对8个名称都输出相同的12≤value<21，只有Litkal的目标恰好与之相符。这个1/8不能解读为已经形成稳定的名称区分；普通无图提示下Litkal又被解释为value=value*2。完整回忆逐条判定见semantic_reviews.jsonl。','',
        '| 图片训练组 | 同一批未见题去学习图（64题） | 给对应原学习图（64题） | 配对文字闭卷 | 配对图片闭卷 |','|---|---:|---:|---:|---:|']
    for a in ('lora_vision','full_vision','full_all_vision'):
        m=ss[a]['unseen_closed']['original_order']
        lines.append('| '+LABELS[a]+f" | {m['correct']}/{m['n']}（{m['accuracy']:.1%}） | "+' | '.join(score(a,c) for c in ('unseen_image','paired_T','paired_V'))+' |')
    if fa['unseen_closed']['accuracy']<.6:
        lines+=['','整模型组仍未达到预定未见闭卷60%门槛。这说明仅解除视觉冻结，在本轮设置下不足以建立可靠的闭卷规则应用；带学习图的成绩不能替代V2T/V2V迁移成绩。']
    else:
        lines+=['','整模型组达到了未见闭卷60%的单项门槛；其他源方向容量与双模态增益门槛仍需按screening_gates.json共同判断，不能单看带图成绩。']
    lines+=['','## 3. loss下降说明了什么','',
        f"文字全参数组最后32步的{evidence['full_text']['last32step_mcq_n']}条MCQ平均loss为{evidence['full_text']['last32step_mcq_mean_loss']:.4f}。每份字母答案实际有字母、结束符、换行3个监督token；只学会公共后缀、字母仍四选一不确定时，平均loss就可接近ln(4)/3≈0.462。此值只是解释性参照，本轮未保存逐token概率，不能把它说成实测字母loss。",'',
        '解释题回忆和带图任务可以继续降低loss，仍不保证闭卷应用学会。样本等权CE也没有去掉答案里的公共后缀。这里只能从实际原答确认塌缩；尚未通过干预确定是优化、提示关联、训练目标还是其组合导致。此前字母加权小实验没有稳定修复，不能把再次调整loss权重写成已验证方案。','',
        '## 4. 可核查的原始例子','',
        '以下各取固定条件的第一条输出，仅说明行为，不代表独立统计证据。完整输入在interpretation_evidence.json。','',
        '无图问Dostan规则，模型原答：','', '```text',examples[0]['response'],'```','',
        '同名规则的配对选择题：A=红色圆形、B=红色正方形、C=红色圆形、D=蓝色圆形。Dostan按“三角形或蓝色”应只保留D，正确选项A；模型输出C。','',
        '另一道oracle题直接给出Ratlod的“正方形或蓝色”正确规则，同样应保留B和D，正确选项A；模型仍输出C。说明本组的规则记忆改善没有自动成为可用的单步选择能力。','',
        '## 5. 下一步如何选','',
        '本轮不支持直接把全参数应用混合训练扩到1000条：源方向应用门槛没有通过就扩大规模，不能解决迁移测量的前提。保留原SFT/PT主结果和本轮负结果，不能按最终评测分数回头改题。','',
        '若继续全参数路线，应先在独立固定的开发规则上检查两种失效：文字侧是否在学会事实的同时损害带正确规则的解题能力；图片侧是否只会依赖学习图。需要以闭卷应用和oracle能力共同约束方案，而不是用回忆率或loss挑checkpoint。本轮没有比较多种全参数学习率、训练预算或原始材料PT，因此不能断言全参数路线本身无效。','',
        '此前“先自行回忆事实、再给出选项”的两步协议已有开发证据，可作为单列研究线路做新规则确认，最终任务仍是选择题。但本轮文字全参数组连oracle都退化，不能直接假设接上回忆步骤就会修好；此前两步结果不能移植到本轮模型。图片来源文字桥接仍是额外监督，不能当纯视觉参数学习。','',
        '所有下一轮规则、提示、epoch和判断标准应在查看结果前固定。统计以规则为单位，选项轮换不当独立样本；源方向有明确学习增益后再讨论跨模态增益保留。','',
        '144份自由回忆由Codex逐条读完整原答核对名称、事实、逻辑与区间，记录理由，未经独立人工复核；短答案选择题按保存的原始输出严格解析。详细分项、四方向归属与检查见report_zh.md、summary.json、training_validation.json。']
    (OUT/'interpretation_zh.md').write_text('\n'.join(lines)+'\n')


def write_figure(ss):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    labels={'full_text':'Full language / TEXT','full_vision':'Full language / VISION',
        'full_all_vision':'Full VLM / VISION','lora_text':'LoRA / TEXT','lora_vision':'LoRA / VISION'}
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    for a in p.ARMS:
        batches=rows(OUT/a/'loss_batches.jsonl');xs=[];ys=[]
        for i in range(0,len(batches),8):
            values=[v for b in batches[i:i+8] for n,v in zip(b['supervised_tokens_including_template_suffix'],b['per_example_mean_loss']) if n==3]
            xs.append(min(i+8,len(batches)));ys.append(max(sum(values)/len(values),1e-8))
        axes[0].plot(xs,ys,label=labels[a],linestyle='--' if a.startswith('lora') else '-',linewidth=1.8)
    axes[0].axhline(math.log(4)/3,color='gray',linestyle=':',linewidth=1)
    axes[0].set(xlabel='Optimizer step',ylabel='MCQ CE per example (8-step window)',yscale='log',title='Training: includes shared answer suffix')
    axes[0].legend(fontsize=8,loc='lower left');axes[0].grid(alpha=.2)
    names=('merged_parent',)+p.ARMS;xx=np.arange(len(names));width=.36
    axes[1].bar(xx-width/2,[100*ss[a]['paired_T']['accuracy'] for a in names],width,label='Text test input')
    axes[1].bar(xx+width/2,[100*ss[a]['paired_V']['accuracy'] for a in names],width,label='Image test input')
    axes[1].axhline(25,color='gray',linestyle=':',linewidth=1)
    axes[1].set_xticks(xx,['Parent','Full\nTEXT','Full lang.\nVISION','Full VLM\nVISION','LoRA\nTEXT','LoRA\nVISION'],fontsize=8)
    axes[1].set(ylabel='Closed-book paired application accuracy (%)',ylim=(0,100),title='Evaluation: no learning reference')
    axes[1].legend(fontsize=8);axes[1].grid(axis='y',alpha=.2)
    fig.suptitle('Qwen3-VL-4B: same 8 development rules, 1 epoch, LR=1e-5',fontsize=12)
    fig.savefig(OUT/'training_and_application.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
