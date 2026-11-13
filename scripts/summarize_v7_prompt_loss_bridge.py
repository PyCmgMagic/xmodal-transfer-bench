#!/usr/bin/env python3
"""Audit and report the prompt/loss/bridge contrasts, preserving all raw answers."""
import csv
import importlib.metadata
import math
import shutil
import sys
from collections import Counter
from pathlib import Path

import explore_v7_prompt_loss_bridge as pilot
import run_v7_evaluation as ev
from run_v7_full_experiment import ROOT,rows,read,save,sha
from v7_gpu_resources import ledger

OUT=pilot.OUT
OLD=pilot.source.OUT
BASELINES=('warmup',*pilot.source.ARMS)


def audited_responses(folder,probes,cp):
    rr=rows(folder/'predictions.jsonl');lookup={p['id']:p for p in probes}
    assert len(rr)==len({r['id'] for r in rr})==len(probes) and {r['id'] for r in rr}==set(lookup)
    cfg=read(folder/'eval_manifest.json');assert cfg['adapter']==ev.checkpoint(cp)
    assert cfg['temperature']==0 and not cfg['thinking'] and cfg['max_tokens']['mcq']==16
    assert read(folder/'eval_state.json')['status']=='complete'
    for r in rr:
        p=lookup[r['id']]
        assert r['input_sha256']==ev.digest({'messages':p['messages'],'images':p.get('images',[])})
        assert r['response']==(r['raw_response']['choices'][0]['message']['content'] or '')
        assert all(r.get(k)==p.get(k) for k in ('qid','condition','fmt','gold'))
        if r['fmt']=='mcq':
            pred=ev.letter(r['response']);assert r['predicted']==pred and r['correct']==(pred==p['gold'])
            assert r['format_error']==(pred is None)
    return rr


def metric(rr):
    out=pilot.source.metric(rr)
    if rr and 'task' in rr[0]:
        out['by_task']={t:pilot.source.metric([r for r in rr if r['task']==t]) for t in ('direct','diagnosis')}
        out['by_evaluation_group']={label:out['by_task'][t] for t,label in [('direct','result_application'),('diagnosis','rule_report_audit')]}
        out['by_rule']={q:pilot.source.metric([r for r in rr if r['qid']==q]) for q in pilot.source.EXPECTED}
        out['by_group']={str(g):pilot.source.metric([r for r in rr if r['group']==g]) for g in (0,1)}
        out['by_shift']={str(s):pilot.source.metric([r for r in rr if r['shift']==s]) for s in range(4)}
        if any(r['condition'].endswith('named_T') or r['condition'].endswith('named_V') for r in rr):
            out['original_order']=pilot.source.metric([r for r in rr if r['shift']==0])
        if all('block' in r for r in rr):
            groups=[[r for r in rr if (r['block'],r['shift'])==key] for key in sorted({(r['block'],r['shift']) for r in rr})]
            assert len(groups)==32 and all(len(g)==4 and {r['gold'] for r in g}==set('ABCD') for g in groups)
            out['paired_checks']={'n':32,'all4_same_answer':sum(len({r['predicted'] for r in g})==1 for g in groups),
                                  'all4_correct':sum(all(r['correct'] for r in g) for g in groups)}
    return out


def summarize():
    pilot.validate();pilot.source.validate(OLD);pilot.paired.validate()
    # Validate every original data/image/base-model reference, not just selected PNGs.
    for fn in ('model_hash_validation.json','original_data_hash_validation.json'):
        old=read(OLD/fn)
        for r in old['checked_files']:assert sha(Path(r['path']))==r['sha256'],r['path']
        save(OUT/fn,{'status':'pass','reference':str(OLD/fn),'checked_files':old['checked_files']})
    refs=read(OLD/'final_artifact_manifest.json')
    for path,h in refs['files'].items():assert sha(OLD/path)==h,path
    dropped={r['id'] for r in rows(OUT/'dropout_rows.jsonl')}
    assert len(dropped)==256
    token_trace={}
    for a in pilot.ARMS:
        tr=rows(OUT/(a+'.jsonl'));base=rows(OLD/('text_mixed.jsonl' if a=='text_equal' else 'vision_mixed.jsonl'))
        assert len(tr)==len(base)==512
        if not a.endswith('dropout'):assert tr==base
        else:
            tt={r['id']:r for r in rows(OLD/'text_mixed.jsonl')}
            for r,b in zip(tr,base):
                expected=dict(b)
                if r['id'] in dropped:expected.pop('images');expected['messages']=tt[r['id']]['messages']
                assert r==expected
            for q in pilot.source.EXPECTED:
                dd=[r for r in tr if r['qid']==q and r['id'] in dropped]
                assert Counter(r['channel'] for r in dd)=={'explain':16,'match':16}
                assert Counter(r['gold'] for r in dd if r['channel']=='match')=={l:4 for l in 'ABCD'}
        assert all('test_visual' not in str(r) and 'Test input:' not in str(r) for r in tr)
        st=read(OUT/a/'checkpoint_manifest.json');assert st['global_step']==128 and abs(st['epoch']-1)<1e-6
        assert st['parent']==ev.checkpoint(pilot.checkpoint('warmup'))
        assert ev.checkpoint(pilot.checkpoint(a))['sha256']==st['sha256']
        args=read(pilot.checkpoint(a)/'args.json')
        for field,value in {'loss_type':'v7_equal_example','per_device_train_batch_size':4,'gradient_accumulation_steps':1,
                            'learning_rate':1e-4,'seed':42,'lora_rank':8,'lora_alpha':32,'lora_dropout':0.05,
                            'freeze_vit':True,'freeze_aligner':True,'packing':False,'padding_free':False}.items():
            assert args[field]==value,(a,field,args[field])
        batches=rows(OUT/a/'loss_batches.jsonl');assert len(batches)==128
        assert {r['step_before_update'] for r in batches}==set(range(128))
        for r in batches:
            counts=r['supervised_tokens_including_template_suffix'];assert len(counts)==4
            assert r['equal_row_weights']==[.25]*4 and math.isfinite(r['loss'])
            assert abs(r['loss']-sum(r['per_example_mean_loss'])/4)<1e-5
            assert all(abs(w-n/sum(counts))<1e-6 for w,n in zip(r['default_token_mean_row_weights'],counts))
        counts=[n for r in batches for n in r['supervised_tokens_including_template_suffix']]
        assert Counter(n<=3 for n in counts)=={True:256,False:256}
        token_trace[a]={'updates':128,'label_token_counts':dict(Counter(counts)),
            'default_mean_coefficient_on_match_examples':sum(sum(w for w,n in zip(r['default_token_mean_row_weights'],r['supervised_tokens_including_template_suffix']) if n<=3) for r in batches)/128,
            'equal_example_mean_coefficient_on_match_examples':.5,
            'first_batch':batches[0]}
        smoke=OUT/('smoke_'+a)
        assert read(smoke/'validated.json')['status']=='pass'
        assert ev.checkpoint(Path(read(smoke/'checkpoint_manifest.json')['path']))['sha256']==read(smoke/'checkpoint_manifest.json')['sha256']
        assert len(rows(smoke/'reload/predictions.jsonl'))==2
    masks=rows(OUT/'template_mask_audit.jsonl');assert len(masks)==544
    assert Counter(r['supervised_tokens_including_suffix'] for r in masks if r['arm']=='text_equal')==Counter(n for r in rows(OUT/'text_equal/loss_batches.jsonl') for n in r['supervised_tokens_including_template_suffix'])
    assert read(OUT/'default_loss_implementation_audit.json')['status']=='pass'
    save(OUT/'loss_weight_audit.json',token_trace)
    probes=rows(OUT/'main_probes.jsonl');diagnostic=rows(OUT/'prompt_probes.jsonl');lookup={p['id']:p for p in probes}
    for p in diagnostic:
        old=lookup[p['source_probe_id']]
        assert p['messages'][1:]==old['messages'][1:] and p.get('images',[])==old.get('images',[])
        assert p.get('gold')==old.get('gold') and p['messages'][0]['role']=='system'
    main={};controls={}
    for a in BASELINES:
        rr=rows(OLD/a/'predictions.jsonl')+rows(pilot.paired.OUT/a/'predictions.jsonl')
        main[a]=[r for r in rr if r['id'] in lookup]
        assert len(main[a])==568
        for r in main[a]:
            p=lookup[r['id']];assert r['input_sha256']==ev.digest({'messages':p['messages'],'images':p.get('images',[])})
    for a in pilot.ARMS:main[a]=audited_responses(OUT/'main'/a,probes,pilot.checkpoint(a))
    for a in pilot.CONTROLS:controls[a]=audited_responses(OUT/'prompt_controls'/a,diagnostic,pilot.checkpoint(a))
    reviews=rows(OUT/'semantic_reviews.jsonl');expected={(kind,a,r['id']) for kind,arms in [('main',main),('prompt_controls',controls)] for a,rr in arms.items() if kind=='prompt_controls' or a in pilot.ARMS for r in rr if r['fmt']=='card'}
    assert len(reviews)==104 and {(r['scope'],r['arm'],r['id']) for r in reviews}==expected
    for r in reviews:
        actual=next(x for x in (main if r['scope']=='main' else controls)[r['arm']] if x['id']==r['id'])
        assert r['response_sha256']==ev.digest(actual['response']) and r['response']==actual['response'] and isinstance(r['correct'],bool)
    allreviews=rows(OLD/'semantic_reviews.jsonl')+[dict(r,scope=r['scope']) for r in reviews]
    summary={kind:{a:{c:metric([r for r in rr if r['condition']==c]) for c in sorted({r['condition'] for r in rr if r['fmt']=='mcq'})} for a,rr in arms.items()} for kind,arms in [('main',main),('prompt_controls',controls)]}
    for kind,arms in [('main',main),('prompt_controls',controls)]:
        for a,rr in arms.items():
            own=[r for r in reviews if r['scope']==kind and r['arm']==a] if a in pilot.ARMS or kind=='prompt_controls' else [r for r in allreviews if r['arm']==a and not r.get('scope')]
            summary[kind][a]['recall']={c:{'n':sum(r['condition']==c for r in own),'correct':sum(r['correct'] for r in own if r['condition']==c)} for c in sorted({r['condition'] for r in own})}
    save(OUT/'summary.json',summary)
    table=[{'scope':kind,'arm':a,'condition':c,**{k:m[k] for k in ('n','correct','accuracy','valid_accuracy','format_errors','empty','truncated')}} for kind,arms in summary.items() for a,mm in arms.items() for c,m in mm.items() if c!='recall']
    with (OUT/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    save(OUT/'four_directions.json',[{'arm':a,'direction':d,'source_classification':'image-derived textual bridge; not pure VISION' if a.endswith('dropout') else 'TEXT' if a=='text_equal' else 'VISION with textual canonical targets and candidate definitions',
        'matches_training_source':d[0]==('T' if a=='text_equal' else 'V'),
        'original_v7':summary['main'][a]['named_'+d[-1]],'paired_application':summary['main'][a]['paired_'+d[-1]]} for a in pilot.ARMS for d in ('T2T','T2V','V2T','V2V')])
    write_report(summary,token_trace)
    write_interpretation(summary,token_trace)
    save(OUT/'completion_audit.json',{'status':'pass','new_main_answers':1704,'new_prompt_control_answers':1824,
        'new_primary_answers':3528,'smoke_answers':6,'new_answers_including_smoke':3534,'manual_recall_reviews':104,
        'new_training_jobs':3,'smoke_training_jobs':3,'original_3000PNG_unchanged':True,'original_data_and_base_models_unchanged':True,
        'prompt_contrasts_change_system_only':True,'raw_replies_and_inputs_verified':True,'last_epoch1_only':True,
        'training_labels_and_image_dropout_checked':True,'loss_including_suffix_masks_checked':True})
    snap=OUT/'source_snapshots'
    scripts=[Path(__file__),ROOT/'scripts/explore_v7_prompt_loss_bridge.py',ROOT/'scripts/v7_equal_example_loss.py']
    for p in scripts:shutil.copy2(p,snap/p.name)
    save(OUT/'environment_manifest.json',{'python':sys.version,'timestamp':ev.now(),'versions':{p:importlib.metadata.version(p) for p in ('torch','transformers','ms-swift','peft','qwen-vl-utils','safetensors')},'scripts':{str(p):sha(p) for p in scripts}})
    with ledger(pilot.source.LEDGER) as entries:active={k:v for k,v in entries.items() if str(OUT) in k}
    assert not active,active
    save(OUT/'final_artifact_manifest.json',{'status':'complete','completed_at':ev.now(),'new_answers':3534,
        'active_gpu_reservations_for_this_run':active,'files':{str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='final_artifact_manifest.json' and 'checkpoint' not in p.relative_to(OUT).parts and p.suffix in ('.json','.jsonl','.csv','.md','.py')}})
    print('All prompt, loss and bridge contrasts audited and summarized.',flush=True)


def write_report(ss,traces):
    title={'warmup':'共同起点','text_explain':'TEXT解释','text_match':'TEXT辨认','text_mixed':'TEXT原混合','vision_explain':'VISION解释','vision_match':'VISION辨认','vision_mixed':'VISION原混合','text_equal':'TEXT样本等权','vision_equal':'VISION样本等权','vision_equal_dropout':'VISION等权＋去图回放（桥接）'}
    def score(kind,a,c,original=False):
        r=ss[kind][a][c];r=r['original_order'] if original else r
        return f"{r['correct']}/{r['n']}（{r['accuracy']:.1%}）"
    lines=['# 提示对齐、样本等权与视觉去图回放','',
        '固定上一轮8条规则和原图。新3组SFT均从同一warmup起点开始，1epoch、512行、128步、batch4/accum1、LR1e-4、LoRA r8/alpha32/dropout0.05，冻结视觉和对齐模块，seed42。固定最后checkpoint，不按测试选模型。训练不含正式测试观察或应用答案；未开CoT，未做PT。','',
        '## 主评测：原短答案提示，不额外给学习材料','',
        '配对题每输入模态32道语义问题，4个选项位置=128次回答；原v7为16题×4位置=64次。两套题分别汇报，不把轮换视为独立知识。','',
        '| 训练 | 配对文字 | 配对图片 | 原v7文字64次 | 原v7图片64次 | 无图改写定义辨认 | 无图完整回忆 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for a in (*BASELINES,*pilot.ARMS):
        recall=ss['main'][a]['recall']['recall']
        lines.append('| '+title[a]+' | '+' | '.join(score('main',a,c) for c in ('paired_T','paired_V','named_T','named_V','recognition_para_noimage'))+f" | {recall['correct']}/{recall['n']} |")
    lines+=['','## 原v7原顺序16题','', '| 训练 | 文字 | 图片 |','|---|---:|---:|']
    for a in ('warmup','text_mixed','vision_mixed',*pilot.ARMS):
        lines.append('| '+title[a]+' | '+score('main',a,'named_T',True)+' | '+score('main',a,'named_V',True)+' |')
    lines+=['','## 只改system的应用对照','',
        '用户问题、选项、图片、正确答案保持相同。原system要求只输出字母；替换system是训练所用的规则说明约定，不包含该规则的正确条件。这个诊断改变提示协议，单独报告，不覆盖主分数。','',
        '| 模型 | 配对文字原→对齐 | 配对图片原→对齐 | 原v7文字原→对齐 | 原v7图片原→对齐 |','|---|---:|---:|---:|---:|']
    for a in pilot.CONTROLS:
        lines.append('| '+title[a]+' | '+' | '.join(score('main',a,c)+' → '+score('prompt_controls',a,'aligned_'+c) for c in ('paired_T','paired_V','named_T','named_V'))+' |')
    lines+=['','## 定义辨认只换为短答案system','',
        '| 模型 | 无图改写定义原→短答案system | 带学习图改写定义原→短答案system | 原system回忆→中性system回忆 |','|---|---:|---:|---:|']
    for a in pilot.CONTROLS:
        old=ss['main'][a]['recall']['recall'];new=ss['prompt_controls'][a]['recall']['neutral_recall']
        lines.append('| '+title[a]+' | '+' | '.join(score('main',a,c)+' → '+score('prompt_controls',a,'strict_'+c) for c in ('recognition_para_noimage','recognition_para_image'))+f" | {old['correct']}/8 → {new['correct']}/8 |")
    lines+=['','## 实际loss权重审计','',
        '完整模板下辨认目标是3个有效token（字母加模板尾部），解释目标约40个。第一训练batch有效token数为[3,40,3,3]，默认token平均给四条样本的系数约为[6.12%,81.63%,6.12%,6.12%]；样本等权改为各25%。这说明行数相同不等于loss系数相同。',
        '',
        '在TEXT样本等权实际128个batch中，若仍使用默认token平均，辨认样本的平均系数合计为'+f"{traces['text_equal']['default_mean_coefficient_on_match_examples']:.1%}"+'；样本等权后为50%。这些是损失系数，不是实际梯度贡献或因果证明。loss_batches.jsonl保留每批有效token、两套系数和实际loss，template_mask_audit.jsonl核验512条TEXT及每视觉组16条模板标签。',
        '',
        '等权组与旧混合组训练行、顺序、目标、初始化、步数相同，改变归一化；两类目标各256行。三组先做单batch训练和checkpoint重载，验证文字与图片生成；因改变loss定义，不直接拿新旧loss数值高低判断学习效果。',
        '',
        '## 去图回放的归属','',
        'VISION等权＋去图回放在固定256行上去掉原学习图并改为相同文字问题，目标答案和行顺序保持不变。每条规则解释16行、辨认16行去图；辨认正确A/B/C/D各4行去图。对照VISION等权组检验这一整套去图回放处理。',
        '',
        '目标是原学习图所对应的人工作成的规则文字/选项标签；去图后相当于直接做名称到文字事实的监督，必须列为图像衍生文字桥接，不能当纯VISION迁移。并没有生成新图，也没有按教师或测试成绩筛样本。',
        '',
        '## 题型和格式错误','', '| 训练 | direct文字 | diagnosis文字 | direct图片 | diagnosis图片 |','|---|---:|---:|---:|---:|']
    for a in ('warmup','text_mixed','vision_mixed',*pilot.ARMS):
        cells=[]
        for c,t in [('paired_T','direct'),('paired_T','diagnosis'),('paired_V','direct'),('paired_V','diagnosis')]:
            r=ss['main'][a][c]['by_task'][t];cells.append(f"{r['correct']}/{r['n']}（{r['accuracy']:.1%}）")
        lines.append('| '+title[a]+' | '+' | '.join(cells)+' |')
    errors=[{'scope':kind,'arm':a,'condition':c,'format_errors':m['format_errors'],'empty':m['empty'],'truncated':m['truncated']} for kind,arms in ss.items() for a,mm in arms.items() for c,m in mm.items() if c!='recall']
    save(OUT/'format_errors.json',errors)
    lines+=['',
        'summary.json含按规则、结构、题型、选项位置和名称敏感性的结果；format_errors.json列各条件的无效、空白及截断输出。主准确率含格式错误；有效输出准确率另存。104份新规则回忆逐条人工审核；不纠正或重跑错误、截断回答。',
        '',
        '## 范围','',
        '这是同8条开发规则、单seed的机制排查，不是新规则泛化验证。四方向映射见four_directions.json，相同输入复用不算独立题。全部原3000张PNG、原数据、基座和旧运行文件哈希已复核。新主评测1704次＋提示对照1824次＝3528次，另有6次smoke重载回答。所有原始回答保留；结果解释另见interpretation_zh.md。']
    (OUT/'report_zh.md').write_text('\n'.join(lines)+'\n')


def write_interpretation(ss,traces):
    def score(kind,a,c):
        r=ss[kind][a][c]
        return f"{r['correct']}/{r['n']}（{r['accuracy']:.1%}）"
    def delta(a,b,c):
        return f"{100*(ss['main'][a][c]['accuracy']-ss['main'][b][c]['accuracy']):+.1f}个百分点"
    lines=['# 这轮排查得到了什么','',
        '三项都已实际完成：4个旧适配器提示对照、3组1epoch训练及原v7/配对测试。新主评测3528次回答，另6次单batch checkpoint重载回答；104份新规则回忆逐条审核。原图不变。','',
        '## 1. 提示有影响，但只对齐system不能解决应用失败','',
        f"旧TEXT混合配对文字{score('main','text_mixed','paired_T')}，换训练system后{score('prompt_controls','text_mixed','aligned_paired_T')}；图片{score('main','text_mixed','paired_V')} → {score('prompt_controls','text_mixed','aligned_paired_V')}。这项对照用户题目、选项和图像逐字相同，改变的system没有正确规则事实。",
        '',
        f"作为正对照，旧TEXT辨认配对文字{score('main','text_match','paired_T')} → {score('prompt_controls','text_match','aligned_paired_T')}；图片{score('main','text_match','paired_V')} → {score('prompt_controls','text_match','aligned_paired_V')}。无图改写定义辨认换成短答案system：{score('main','text_match','recognition_para_noimage')} → {score('prompt_controls','text_match','strict_recognition_para_noimage')}。提示效应需要按任务看，不能把所有失败归为模板不同。",
        '',
        f"同一个TEXT辨认适配器在原v7图片题却由{score('main','text_match','named_V')}变为{score('prompt_controls','text_match','aligned_named_V')}。不能只选配对图片59.4%宣称提示已修复；原主协议分数和这组替代提示分数均保留。",
        '',
        '## 2. loss权重不均确实存在，等权只带来局部变化','',
        f"原混合训练512行中解释和辨认各256行，但按实际128批及模板标签计算，默认归一化给辨认样本的平均系数合计只有{traces['text_equal']['default_mean_coefficient_on_match_examples']:.1%}；样本等权后固定为50%。这是实际损失系数，不是测得的梯度占比。",
        '',
        f"TEXT等权无图改写定义辨认{score('main','text_equal','recognition_para_noimage')}，原混合{score('main','text_mixed','recognition_para_noimage')}；配对文字{score('main','text_equal','paired_T')}（相对原混合{delta('text_equal','text_mixed','paired_T')}），图片{score('main','text_equal','paired_V')}（{delta('text_equal','text_mixed','paired_V')}）。它仍低于单独TEXT辨认的45.3%/46.9%，不能把等权视为充分修复。",
        '',
        f"原v7四位置文字：原混合{score('main','text_mixed','named_T')} → 等权{score('main','text_equal','named_T')}；图片{score('main','text_mixed','named_V')} → {score('main','text_equal','named_V')}。这些原题分数没有随辨认改善整体上涨。",
        '',
        '## 3. 视觉去图回放改善少量回忆，尚未改善应用','',
        f"VISION等权带学习图完整回忆{ss['main']['vision_equal']['recall']['recall_image']['correct']}/8，去图{ss['main']['vision_equal']['recall']['recall']['correct']}/8；加入固定50%去图回放后，无图{ss['main']['vision_equal_dropout']['recall']['recall']['correct']}/8、带图{ss['main']['vision_equal_dropout']['recall']['recall_image']['correct']}/8。",
        '',
        f"但配对应用：VISION等权文字{score('main','vision_equal','paired_T')}、图片{score('main','vision_equal','paired_V')}；去图回放文字{score('main','vision_equal_dropout','paired_T')}、图片{score('main','vision_equal_dropout','paired_V')}。无图改写定义辨认也只有{score('main','vision_equal_dropout','recognition_para_noimage')}。因此少量回忆恢复没有转为稳定应用。",
        '',
        '具体错误：Dostan正确为triangle OR blue，去图回放组却回忆为pentagon OR blue；Morkil正确为pentagon OR green，也答为pentagon OR blue。Levsim下界7答成8。模型会套用相近规则，说明名称与具体条件的绑定仍不稳。原始回答见semantic_reviews.jsonl，失败和截断均保留。',
        '',
        '这组使用原图对应的人工作成文字目标，去图行直接监督名称→文字事实，属于桥接训练；结果不能算作纯视觉迁移。训练未使用任何正式应用题观察或答案。',
        '',
        '## 后续怎么推进','',
        '目前不应直接全量重训1000条，也不应以增加epoch代替定位。保留TEXT单独辨认作为已经有局部收益的参照；当前50/50混合、样本等权和50%去图桥接均尚不能作为最终方案。',
        '',
        '下一项更值得隔离的是训练目标之间的干扰：在固定同规则与总步数下对比分阶段“解释→辨认”和反向顺序，再同时检查无材料应用、改写辨认及完整回忆。视觉端必须先在去图诊断证明名称能稳定对应具体条件，才值得扩大训练。若继续使用文字回放，应独立标注其额外文字监督。',
        '',
        '上述下一项尚未运行。本轮是在同8条规则上的单seed机制排查；回答包含位置轮换，不能按独立128题做显著性推断，也不能外推全量。完整分题型、格式错误、原顺序成绩和四方向映射见report_zh.md、summary.json、four_directions.json。']
    (OUT/'interpretation_zh.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':summarize()
