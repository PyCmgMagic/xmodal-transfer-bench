#!/usr/bin/env python3
"""Knowledge-only clean-base TEXT ladder. Original data/images are immutable."""
import argparse, copy, hashlib, json, math, os, random, re, subprocess
from collections import Counter
from pathlib import Path
import run_v7_evaluation as ev
import explore_v7_memory_router as mem
from run_v7_full_experiment import ROOT, MODELS, environment, rows, read, save, sha, train_command
from run_v7_fullparam_ep10 import dense_identity
from v7_gpu_resources import reserve
OUT=ROOT/'runs/v7_knowledge_ladder_20261021_r2'
DATA=ROOT/'rules_repaired_single_image_v7'
LEDGER=ROOT/'runs/v7_full_experiment'
PLUGIN=ROOT/'scripts/v7_equal_example_loss.py'
FIRST='scale1k_0333_ai_generated'
SYSTEM='Answer the question about the learned named rule accurately. Follow the requested answer format.'
FULL=('State the complete stored definition of {name}.','Explain the rule named {name}.',
'Give the exact requirements and recording convention for {name}.','Write the full factual reference entry for {name}.',
'Recall all conditions and inspection codes in {name}.','Describe precisely what qualifies under {name}.',
'What is the complete specification of {name}?','Define {name}, preserving its logical connective and inspection convention.')
DETAILS=(' Include every factual parameter.',' Do not invent a new example.',' Keep the definition exact.',' State all requirements explicitly.')
SHORT=('For {name}, give only {what}.','Recall {what} for {name}. Return just that fact.',
'Under {name}, what is {what}? Give a short answer.','State {what} in {name}; omit other facts.')
HELD=('An inspector has forgotten {what} in {name}. Which value belongs in the reference?',
'Select the stored value for {what} associated with {name}.','Which entry correctly identifies {what} of the named rule {name}?',
'The reference for {name} specifies {what}. Choose that value.')
def msg(q,a=None):
 rr=[dict(role='system',content=SYSTEM),dict(role='user',content=q)]
 if a is not None: rr.append(dict(role='assistant',content=a))
 return rr
def spec(k): return json.loads(k['fact']['quote'])
def codes(k):
 table=k['material']['learning']['inspection_recording_protocol']['outcome_to_code']
 mapped={tuple(s.rsplit('=',1)[1].strip().lower()=='met' for s in key.split('; ')):value for key,value in table.items()}
 expected=((True,True),(True,False),(False,True),(False,False))
 assert set(mapped)==set(expected) and len(set(mapped.values()))==4
 return [mapped[flags] for flags in expected]
def atoms(k):
 a,b=spec(k)['conditions']
 return [(a['field'],'the required '+a['field'],str(a['value'])),(b['field'],'the required '+b['field'],str(b['value'])),
 ('logic','the logical connective joining the conditions (AND or OR)','AND'),
 ('scope','the number of items on which both conditions must hold','one and the same item')]+[
 ('code'+str(i),what,value) for i,(what,value) in enumerate(zip((
 'the inspection code when both requirements are met','the inspection code when only the first requirement is met',
 'the inspection code when only the second requirement is met','the inspection code when neither requirement is met'),codes(k)))]
def norm(s): return re.sub(r'\s+',' ',s).strip().strip(' .').lower()
def mcq(qid,ident,condition,q,choices,target,pos,**extra):
 other=[x for x in choices if x!=target]; assert len(other)>=3 and len(set(choices))==len(choices)
 options=other[:3]; options.insert(pos,target)
 text=q+'\n\nOptions:\n'+'\n'.join(c+'. '+v for c,v in zip('ABCD',options))+ev.SUFFIX
 return dict(id=ident,qid=qid,condition=condition,fmt='mcq',gold='ABCD'[pos],task='direct',input_mod='T',
 messages=[dict(role='system',content=ev.SYSTEM),dict(role='user',content=text)],options=options,answer_value=target,**extra)
def validate(images=False):
 m=read(OUT/'manifest.json'); assert m
 for p,h in m['sources'].items(): assert sha(Path(p))==h,p
 for p,h in m['files'].items(): assert sha(OUT/p)==h,p
 if images:
  for r in rows(DATA/'image_hashes.jsonl'): assert sha(DATA/r['path'])==r['source_sha256']==r['repaired_sha256']
 return m

def prepare():
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'manifest.json').exists(): validate(); return
 kk=[r['content'] for r in rows(DATA/'audit/rule_knowledge.jsonl')]
 eligible=[k for k in kk if k['fact']['relation']=='binary_predicate' and spec(k).get('predicate_topology')=='two_attribute_conjunction' and spec(k)['mode']=='all' and all(c['operator']=='eq' for c in spec(k)['conditions'])]
 eligible.sort(key=lambda k:hashlib.sha256(('knowledge-ladder-42:'+k['id']).encode()).hexdigest())
 selected=[next(k for k in eligible if k['id']==FIRST)]+[k for k in eligible if k['id']!=FIRST][:19]; assert len(selected)==20
 pool={}
 for k in eligible:
  for c in spec(k)['conditions']: pool.setdefault(c['field'],set()).add(str(c['value']))
 pool['color'].update(('blue','green','gray','black','white')); pool['shape'].update(('circle','square','triangle','diamond','pentagon'))
 pool['material'].update(('wood','plastic','metal','fabric','glass'))
 training=[]; probes=[]
 for k in selected:
  qid,name=k['id'],k['entity']['name']; answer=mem.compact(k); cc=codes(k)
  assert 'Worked example:' not in answer and 'Test input:' not in answer
  local=[]
  for i,(form,detail) in enumerate((f,d) for f in FULL for d in DETAILS):
   local.append(dict(id=f'{qid}:full:{i}',qid=qid,channel='full_definition',messages=msg(form.format(name=name)+detail,answer)))
  for key,what,value in atoms(k):
   assert norm(value) in norm(answer) or key in ('logic','scope'),(qid,key,value)
   for j,form in enumerate(SHORT): local.append(dict(id=f'{qid}:atom:{key}:{j}',qid=qid,channel='independent_fact',messages=msg(form.format(name=name,what=what),value)))
  assert len(local)==64; training.extend(local)
  for r in [local[i] for i in (0,9,18,27,32,36,40,48)]:
   probes.append(dict(id='seen:'+r['id'],qid=qid,condition='seen_fact',fmt='free',gold=r['messages'][-1]['content'],task=r['channel'],input_mod='T',messages=r['messages'][:-1]))
  a,b=spec(k)['conditions']; alternatives={}
  for c in (a,b):
   field,value=c['field'],str(c['value']); choices=sorted(pool[field]-{value}); assert len(choices)>=3
   alternatives[field]=choices
   for j in range(8):
    off=j%len(choices); other=choices[off:]+choices[:off]
    q=HELD[j%4].format(name=name,what='the required '+field)+(' Use the stored definition.',' Consider the main item itself.')[(j//4)%2]
    probes.append(mcq(qid,f'fact:{qid}:{field}:{j}','unseen_fact_mcq',q,[value]+other,value,j%4,field=field))
   for j,form in enumerate(('In {name}, identify the {field} requirement. Give only its value.',
    'Without reconstructing the paragraph, name the {field} required by {name}.',
    'What value fills the {field} slot in the definition of {name}? Respond briefly.',
    'From the learned definition of {name}, retrieve the required {field}; give that fact alone.')):
    probes.append(dict(id=f'free:{qid}:{field}:{j}',qid=qid,condition='unseen_fact_free',fmt='free',gold=value,field=field,task='fact',input_mod='T',messages=msg(form.format(name=name,field=field))))
  def pair(x,y): return f'{a["field"]}={x}; {b["field"]}={y}; both must hold for the same item'
  target=pair(a['value'],b['value'])
  for j in range(8):
   x=alternatives[a['field']][j%len(alternatives[a['field']])]; y=alternatives[b['field']][j%len(alternatives[b['field']])]
   opts=[target,pair(x,b['value']),pair(a['value'],y),pair(x,y)]
   probes.append(mcq(qid,f'pair:{qid}:{j}','unseen_fact_mcq','Which joint specification belongs to '+name+'? '+('Select its exact requirements.','Choose its correct reference entry.')[(j//4)%2],opts,target,j%4,field='joint'))
  reports=[f'Check 1={"Pass" if u else "Fail"}; Check 2={"Pass" if v else "Fail"}; Qualification={"Pass" if u and v else "Fail"}' for u,v in ((True,True),(True,False),(False,True),(False,False))]
  for j in range(32):
   combo=j%4; u,v=combo in (0,1),combo in (0,2)
   x=str(a['value']) if u else alternatives[a['field']][(j//4)%len(alternatives[a['field']])]
   y=str(b['value']) if v else alternatives[b['field']][(j//4)%len(alternatives[b['field']])]
   actual={a['field']:x,b['field']:y}; flags=[actual[c['field']]==str(c['value']) for c in (a,b)]; assert flags==[u,v]
   if (a['field'],b['field'])==('category','material'):
    observation=f'The item is a {x} whose main body is made of {y}.'
   elif (a['field'],b['field'])==('category','color'):
    observation=f'The item is a {x} and its main body is {y} in color.'
   elif (a['field'],b['field'])==('shape','color'):
    observation=f'The item has a {x} outline and is {y}.'
   else:
    raise ValueError(('Unsupported observation fields',a['field'],b['field']))
   if j<16:
    q=f'Under {name}, inspect this item. {observation} Which inspection record code records its two requirement outcomes?'
    opts,target,task=cc,cc[combo],'direct'
   else:
    submitted=reports[(j//4)%4]
    q=f'Under {name}, an inspector submitted: {submitted}. {observation} Check the requirements in this order: {a["field"]}, {b["field"]}. Which audit is correct?'
    opts=[('Accept report. ' if r==submitted else 'Replace report. ')+r for r in reports]
    target=('Accept report. ' if reports[combo]==submitted else 'Replace report. ')+reports[combo]; task='diagnosis'
   p=mcq(qid,f'application:{qid}:{j}','new_application_T',q,opts,target,(j//4+j%4)%4,observations=actual,condition_outcomes=flags,outcome_pattern=combo)
   p['task']=task; probes.append(p)
  for p in [p for p in probes if p['qid']==qid and p['condition'] in ('unseen_fact_mcq','new_application_T')]:
   z=copy.deepcopy(p); z.update(id='oracle:'+p['id'],condition='oracle_'+p['condition'])
   z['messages'][-1]['content']='Correct rule reference:\n'+answer+'\n\n'+z['messages'][-1]['content']; probes.append(z)
 # Formal payloads/labels are read only after knowledge training is finalized.
 labels={r['id']:r for r in rows(DATA/'audit/eval_labels.jsonl')}; ids={k['id'] for k in selected}
 for direction in ('T2T','T2V'):
  for source in rows(DATA/direction/'eval.jsonl'):
   if source['qid'] not in ids: continue
   p=ev.prepare_row(source,DATA,direction)
   p.update(id='original:'+direction+':'+source['id'],source_id=source['id'],condition='original_'+direction,fmt='mcq',input_mod=direction[-1],gold=labels[source['id']]['answer'],task=labels[source['id']]['question_type']); probes.append(p)
 ev.save_rows(OUT/'knowledge.jsonl',selected)
 for n in (1,5,20):
  folder=OUT/f'n{n}'; folder.mkdir(exist_ok=True); ids={k['id'] for k in selected[:n]}
  rr=[r for r in training if r['qid'] in ids]; random.Random(42).shuffle(rr); pp=[p for p in probes if p['qid'] in ids]
  seen={ev.digest(r['messages'][:-1]) for r in rr}
  assert not any(ev.digest(p['messages']) in seen for p in pp if p['condition']!='seen_fact')
  assert all(not r.get('images') and '<image>' not in str(r) for r in rr)
  assert len(rr)==64*n and len({r['id'] for r in rr})==len(rr) and len({p['id'] for p in pp})==len(pp)
  for condition in ('unseen_fact_mcq','new_application_T'):
   for qid in ids:
    count=Counter(p['gold'] for p in pp if p['qid']==qid and p['condition']==condition)
    assert set(count)==set('ABCD') and len(set(count.values()))==1,count
  ev.save_rows(folder/'training.jsonl',rr); ev.save_rows(folder/'probes.jsonl',pp)
 sources=[Path(__file__),PLUGIN,DATA/'audit/rule_knowledge.jsonl',DATA/'audit/eval_labels.jsonl',DATA/'image_hashes.jsonl',
 ROOT/'scripts/explore_v7_memory_router.py',ROOT/'scripts/run_v7_evaluation.py',ROOT/'scripts/run_v7_full_experiment.py',
 ROOT/'scripts/run_v7_fullparam_ep10.py',ROOT/'scripts/v7_gpu_resources.py',ROOT/'CROSS_MODAL_RESEARCH_PROTOCOL.md',DATA/'T2T/eval.jsonl',DATA/'T2V/eval.jsonl']
 images=rows(DATA/'image_hashes.jsonl'); assert len(images)==3000
 for r in images: assert sha(DATA/r['path'])==r['source_sha256']==r['repaired_sha256']
 m=dict(created_at=ev.now(),ids=[k['id'] for k in selected],sources={str(p):sha(p) for p in sources},files={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*.jsonl')},base=dense_identity(MODELS['qwen3']),
 training=dict(initialization='clean base for every size',modality='TEXT only',method='LoRA, not dense full parameter',rank=32,alpha=64,dropout=.05,learning_rate=3e-5,epochs=10,rows_per_rule=64,batch=4,accumulation=1,steps_per_rule=160,max_length=8192,seed=42,freeze_vision_aligner=True,loss='equal example CE, assistant content plus suffix',application_training=False),
 generation=dict(direct=32,blind_recall=512,temperature=0,thinking=False,batch=4),
 selection='Vonket predetermined; remaining19 conjunction rules by SHA256. Nested sets; clean-base initialization. Historical development rules allowed.',
 gate=dict(seen_min=.90,fact_min=.80,fact_gain_min=.20,app_min=.70,app_gain_min=.15,format_max=.02),
 limitations=['Single seed, conjunction family only, not full1000 or VISION training.','640 knowledge exposures per rule, unlike historic10 material exposures.',
 'Fact template variants and option rotations are correlated, not independent facts.','Generated applications can share attribute combinations with formal cases; none enter training.',
 'Only2 original questions per direction per rule; no one-rule cross-modal conclusion.','Free strict matching requires separate semantic review.'],original_images_verified=3000)
 save(OUT/'manifest.json',m)
 notes=['# 规则知识学习：1 → 5 → 20 条诊断','','每一级从干净Qwen3-VL-4B开始，TEXT-only LoRA32/alpha64，LR3e-5，10epoch，batch4，seed42。',
 '每规则32条完整定义＋32条独立事实；去除worked example，保留各自的原检查记录映射。不训练应用案例、选择题或字母。',
 '单条首先使用Vonket；其他规则按SHA256预定。只有来源事实与应用均通过manifest门槛才扩大。',
 '正式题及3000张原PNG不变。给材料控制单列，自由回答严格分数不等同语义正确率。','','实际样例：',json.dumps(rows(OUT/'n1/training.jsonl')[0],ensure_ascii=False,indent=2)]
 (OUT/'PLAN_ZH.md').write_text('\n'.join(notes)+'\n'); validate(); print('PREPARED',flush=True)

def encode(n):
 validate(); os.environ.update(environment('0',DATA))
 from swift.model import get_model_processor
 from swift.template import get_template
 _,proc=get_model_processor(str(MODELS['qwen3']),model_type='qwen3_vl',load_model=False)
 template=get_template(proc,template_type='qwen3_vl',max_length=8192,enable_thinking=False); template.set_mode('train')
 suffix=proc.tokenizer.encode('<|im_end|>\n',add_special_tokens=False); result=[]
 for r in rows(OUT/f'n{n}/training.jsonl'):
  z=template.encode(dict(messages=copy.deepcopy(r['messages']))); lab=[v for v in z['labels'] if v!=-100]
  assert lab==proc.tokenizer.encode(r['messages'][-1]['content'],add_special_tokens=False)+suffix
  assert len(z['input_ids'])<8192
  result.append(dict(id=r['id'],channel=r['channel'],tokens=len(z['input_ids']),supervised=len(lab)))
 save(OUT/f'n{n}/encoding_audit.json',dict(status='pass',rows=result,assistant_mask_exact=True,truncated=False))
 print('ENCODING PASS',n,len(result),flush=True)
def cp(n,smoke=False): return OUT/f'n{n}'/('smoke' if smoke else 'trained')/'checkpoint'/f'checkpoint-{1 if smoke else 160*n}'
def train(n,gpu,smoke=False):
 validate(); checkpoint=cp(n,smoke); folder=checkpoint.parent.parent; folder.mkdir(parents=True,exist_ok=True)
 assert read(OUT/f'n{n}/encoding_audit.json').get('status')=='pass'
 with ev.run_lock(folder/'train.lock'):
  if not (checkpoint/'adapter_model.safetensors').exists():
   dataset=OUT/f'n{n}/training.jsonl'
   if smoke:
    rr=rows(dataset); dataset=folder/'input.jsonl'
    ev.save_rows(dataset,[next(r for r in rr if r['channel']==ch) for ch in ('full_definition','independent_fact')]*2)
   recover=[p for p in checkpoint.parent.glob('checkpoint-*') if all((p/f).exists() for f in ('optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','adapter_model.safetensors'))]
   resume=max(recover,key=lambda p:read(p/'trainer_state.json')['global_step']) if recover and not smoke else None
   cmd=train_command('qwen3','sft',dataset,checkpoint.parent,epochs=10,resume=resume)
   for flag,value in [('--per_device_train_batch_size','4'),('--gradient_accumulation_steps','1'),('--learning_rate','3e-5'),('--lora_rank','32'),('--lora_alpha','64'),('--save_strategy','steps'),('--logging_steps','1')]: cmd[cmd.index(flag)+1]=value
   cmd+=['--save_steps','1' if smoke else str(40*n),'--attn_impl','sdpa','--external_plugins',str(PLUGIN),'--loss_type','v7_equal_example','--packing','false','--padding_free','false','--use_logits_to_keep','false']
   if smoke: cmd+=['--max_steps','1']; cmd[cmd.index('--warmup_ratio')+1]='0'
   env=environment(gpu,DATA); env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',V7_EQUAL_LOSS_AUDIT=str(folder/'loss_audit.jsonl'))
   save(folder/'state.json',dict(status='waiting_for_gpu',gpu=gpu,command=cmd))
   with reserve(LEDGER,gpu,str(folder)+':train',21000) as reservation:
    with (folder/'train.log').open('a') as log:
     proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT); reservation.bind(proc.pid)
     save(folder/'state.json',dict(status='training',pid=proc.pid,gpu=gpu,command=cmd)); code=proc.wait()
    if code: raise RuntimeError((code,str(folder/'train.log')))
  state,args=read(checkpoint/'trainer_state.json'),read(checkpoint/'args.json')
  assert state['global_step']==(1 if smoke else 160*n)
  assert args['model']==str(MODELS['qwen3']) and args['adapters'] in ([],None)
  assert args['freeze_vit'] and args['freeze_aligner'] and args['lora_rank']==32
  losses=[r['loss'] for r in rows(checkpoint.parent/'logging.jsonl') if 'loss' in r]; assert losses and all(math.isfinite(x) for x in losses)
  from safetensors import safe_open
  with safe_open(str(checkpoint/'adapter_model.safetensors'),framework='pt',device='cpu') as tensors:
   keys=list(tensors.keys()); bs=[key for key in keys if 'lora_B' in key]
   assert bs and all(bool(tensors.get_tensor(key).count_nonzero()) for key in bs)
   assert not any('.visual.' in key or '.merger.' in key for key in keys)
  save(folder/'checkpoint_manifest.json',dict(**ev.checkpoint(checkpoint),step=state['global_step'],epoch=state['epoch'],finite_loss=True,final_loss=losses[-1],nonzero_lora_B=len(bs),selection='fixed final checkpoint'))
  save(folder/'state.json',dict(status='complete',checkpoint=str(checkpoint)))
def evaluate(n,arm,gpu,smoke=False):
 validate(); mem.OUT=OUT; folder=OUT/f'n{n}'/('smoke' if smoke else arm)/'evaluation'; folder.mkdir(parents=True,exist_ok=True)
 pp=rows(OUT/f'n{n}/probes.jsonl')
 if smoke: pp=[next(p for p in pp if p['condition']==c) for c in ('seen_fact','unseen_fact_mcq','original_T2V')]
 with ev.run_lock(folder/'eval.lock'):
  runner=mem.Runner(folder,gpu,arm)
  if len(runner.seen)<len(pp):
   save(folder/'state.json',dict(status='waiting_for_gpu',gpu=gpu,saved=len(runner.seen),total=len(pp)))
   with reserve(LEDGER,gpu,str(folder)+':evaluation',18000):
    runner.load(cp=MODELS['qwen3'] if arm=='base' else cp(n,smoke),adapter=arm!='base')
    if arm=='base': assert not getattr(runner.engine.model,'peft_config',None)
    runner.infer([p for p in pp if p['fmt']=='mcq'],'direct')
    runner.infer([p for p in pp if p['fmt']=='free'],'blind_recall')
    del runner.engine
    import gc,torch
    gc.collect(); torch.cuda.empty_cache()
  assert runner.seen=={p['id'] for p in pp}
  if smoke:
   assert all(r['response'].strip() for r in runner.rr)
   save(folder.parent/'reload_validation.json',dict(status='pass',text_and_image=True))
  save(folder/'state.json',dict(status='complete',saved=len(runner.rr)))
def metric(rr):
 n=len(rr); c=sum(r['scored_correct'] for r in rr); bad=sum(r['format_error'] for r in rr)
 return dict(n=n,correct=c,accuracy=c/n if n else None,percent=100*c/n if n else None,format_errors=bad,empty=sum(not r['response'].strip() for r in rr),truncated=sum(r['finish_reason']=='length' for r in rr))
def score(n):
 m=validate(images=True); result={}; expected={p['id']:p for p in rows(OUT/f'n{n}/probes.jsonl')}
 for arm in ('base','trained'):
  rr=rows(OUT/f'n{n}'/arm/'evaluation/predictions.jsonl'); assert len(rr)==len(expected) and {r['id'] for r in rr}==set(expected)
  for r in rr:
   p=expected[r['id']]; assert r['input_sha256']==ev.digest(dict(messages=p['messages'],images=p.get('images',[])))
   assert r['response']==(r['raw_response']['choices'][0]['message']['content'] or '')
   r['scored_correct']=ev.letter(r['response'])==p['gold'] if p['fmt']=='mcq' else norm(r['response'])==norm(p['gold'])
  result[arm]={c:metric([r for r in rr if r['condition']==c]) for c in sorted({p['condition'] for p in expected.values()})}
  result[arm]['application_by_task']={t:metric([r for r in rr if r['condition']=='new_application_T' and r['task']==t]) for t in ('direct','diagnosis')}
  ev.save_rows(OUT/f'n{n}'/arm/'scored_predictions.jsonl',rr)
  ev.save_rows(OUT/f'n{n}'/arm/'free_response_review.jsonl',[r for r in rr if r['fmt']=='free'])
 base,t=result['base'],result['trained']; f,a=t['unseen_fact_mcq'],t['new_application_T']; g=m['gate']
 checks=dict(seen=t['seen_fact']['accuracy']>=g['seen_min'],fact_absolute=f['accuracy']>=g['fact_min'],fact_gain=f['accuracy']-base['unseen_fact_mcq']['accuracy']>=g['fact_gain_min'],app_absolute=a['accuracy']>=g['app_min'],app_gain=a['accuracy']-base['new_application_T']['accuracy']>=g['app_gain_min'],format=(f['format_errors']+a['format_errors'])/(f['n']+a['n'])<=g['format_max'])
 result['gate']=dict(passed=all(checks.values()),checks=checks,note='Descriptive screen, not significance or proof of complete mastery.')
 save(OUT/f'n{n}/summary.json',result)
 names=dict(seen_fact='训练原问法（严格匹配）',unseen_fact_mcq='未见问法：事实选择',unseen_fact_free='未见问法：自由回答（严格匹配）',new_application_T='文字应用案例',original_T2T='原题文字',original_T2V='原题图片',oracle_unseen_fact_mcq='给材料：事实选择',oracle_new_application_T='给材料：文字应用')
 lines=[f'# {n}条规则：知识学习排查','','Qwen3-VL-4B，干净基座，TEXT-only LoRA32/alpha64，LR3e-5，10epoch，batch4，seed42。每规则64条知识问答，不训练应用题。',
 '表内均为实际生成计分，不是teacher-forcing准确率；基座与训练后同题同生成预算。','','| 检查 | 未训练基座 | 训练后 |','|---|---:|---:|']
 for c,name in names.items():
  cells=[f'{result[arm][c]["percent"]:.2f}%（{result[arm][c]["correct"]}/{result[arm][c]["n"]}）' for arm in ('base','trained')]
  lines.append('| '+name+' | '+' | '.join(cells)+' |')
 lines+=['','扩大规模筛查：'+('通过。' if result['gate']['passed'] else '未通过，停止自动扩大。'),
 '自由回答严格匹配不等于语义正确率，原始回答另存，需逐条复核。事实模板变体、选项轮换和共享逻辑组合不是独立知识样本。',
 '给材料控制不是闭卷迁移成绩。原题每规则每方向仅2题，不据此宣称跨模态差距。单seed、高重复学习，不能外推1000条。','',json.dumps(result['gate'],ensure_ascii=False,indent=2)]
 (OUT/f'n{n}/report_zh.md').write_text('\n'.join(lines)+'\n'); print('\n'.join(lines),flush=True)
 return result['gate']['passed']
def workflow(gpu):
 prepare()
 with ev.run_lock(OUT/'workflow.lock'):
  for n in (1,5,20):
   save(OUT/'workflow_state.json',dict(status='running',size=n,stage='encoding_smoke',pid=os.getpid()))
   encode(n); train(n,gpu,True); evaluate(n,'trained',gpu,True)
   save(OUT/'workflow_state.json',dict(status='running',size=n,stage='formal_training_evaluation',pid=os.getpid()))
   train(n,gpu); evaluate(n,'base',gpu); evaluate(n,'trained',gpu)
   if not score(n):
    save(OUT/'workflow_state.json',dict(status='stopped_at_failed_gate',size=n,updated_at=ev.now())); return
  save(OUT/'workflow_state.json',dict(status='complete',updated_at=ev.now()))
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--stage',choices=('prepare','encode','train','evaluate','score','workflow'),required=True)
 p.add_argument('--n',type=int,choices=(1,5,20),default=1); p.add_argument('--arm',choices=('base','trained'),default='trained'); p.add_argument('--gpu',default='0'); p.add_argument('--smoke',action='store_true'); args=p.parse_args()
 if args.stage=='prepare': prepare()
 elif args.stage=='encode': encode(args.n)
 elif args.stage=='train': train(args.n,args.gpu,args.smoke)
 elif args.stage=='evaluate': evaluate(args.n,args.arm,args.gpu,args.smoke)
 elif args.stage=='score': score(args.n)
 else: workflow(args.gpu)
