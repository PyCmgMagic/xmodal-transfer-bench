#!/usr/bin/env python3
"""Four real learned rules, identical original observation image and options."""
import argparse
import copy
import dataclasses
import itertools
import json
import math
import os
import random
import re
import shutil
import subprocess
from pathlib import Path

import run_v7_evaluation as ev
from build_v7_experiment_data import name
from pilot_v7_task_then_transfer import DATA, LEDGER, card
from pilot_v7_atomic_supervision import PARENT, msg
from run_v7_full_experiment import ROOT, MODELS, environment, final_checkpoint, rows, read, save, sha, train_command
from v7_gpu_resources import reserve

OUT=ROOT/'runs/v7_counterfactual_names_20261016'
ARMS=('warmup','text','vision')


def solve(rule,observation):
    ops={'eq':lambda a,b:a==b,'ne':lambda a,b:a!=b,'ge':lambda a,b:a>=b,
         'gt':lambda a,b:a>b,'le':lambda a,b:a<=b,'lt':lambda a,b:a<b}
    flags=[ops[c['operator']](observation[c['field']],c['value']) for c in rule['conditions']]
    return 'KEEP' if (all(flags) if rule['mode']=='all' else any(flags)) else 'OMIT'


def independent(rule,x):
    if rule['selection_topology']=='disjunctive_filter':
        keep=x['shape']==rule['conditions'][0]['value'] or x['color']==rule['conditions'][1]['value']
    else:
        a,b=rule['conditions'];assert a['operator']=='ge' and b['operator']=='lt'
        keep=a['value']<=x['value']<b['value']
    return 'KEEP' if keep else 'OMIT'


def assignment(values):return '; '.join(f'{letter}={v}' for letter,v in zip('ABCD',values))


def validate(out):
    m=read(out/'manifest.json');assert m['script_sha256']==sha(Path(__file__))
    for p,h in m['sources'].items():assert sha(Path(p))==h,p
    for p,h in m['files'].items():assert sha(out/p)==h,p
    for r in rows(DATA/'image_hashes.jsonl'):
        assert sha(DATA/r['path'])==r['source_sha256']==r['repaired_sha256']
    assert ev.checkpoint(Path(read(PARENT)['path']))==m['parent']


def prepare(out):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():validate(out);return
    selection=read(out/'selection_candidates.json');assert len(selection)==2
    allk={r['content']['id']:r['content'] for r in rows(DATA/'audit/rule_knowledge.jsonl')}
    ids=sorted({i for g in selection for i in g['rule_ids']});assert len(ids)==8
    target=[allk[i] for i in ids]
    dev={r['id'] for r in rows(ROOT/'runs/v7_task_then_transfer_20261015/pilot_v2/development_knowledge.jsonl')}
    assert not set(ids)&dev
    ev.save_rows(out/'target_knowledge.jsonl',target)
    review=read(out/'image_review.json');assert review['all_13_images_verified'] and review['rule_ids']==ids
    # Material-only data built before loading evaluation observations/options.
    for arm in ('text','vision'):
        training=[]
        for k in target:
            for i in range(64):
                q=('Explain the complete rule named {}.','State the exact stored rule for {}.',
                   'Recall all conditions and comparisons for {}.','Describe the complete decision rule for {}.')[i%4].format(name(k))
                if arm=='vision':q='<image>Using only this original learning reference, '+q[0].lower()+q[1:]
                r={'id':f'{k["id"]}:{i}','qid':k['id'],'channel':'rule_material','messages':msg(q,card(k))}
                if arm=='vision':r['images']=[str(DATA/'images'/k['id']/'learning_visual.png')]
                training.append(r)
        random.Random(42).shuffle(training);assert len(training)==512
        ev.save_rows(out/(arm+'.jsonl'),training)
    source={r['id']:r for r in rows(DATA/'T2T/eval.jsonl')}
    audit={r['sample_id']:r for r in rows(DATA/'audit/sample_audit.jsonl')}
    blocks=[];probes=[];labels=[];exports={d:[] for d in ('T2T','T2V','V2T','V2V')}
    rng=random.Random('counterfactual-original-images:42')
    for group_index,g in enumerate(selection):
        kk=[allk[i] for i in g['rule_ids']]
        assert len(kk)==4
        for source_id in g['eligible_test_ids']:
            old=source[source_id];obs=g['observations'][source_id]
            source_text=old['messages'][-1]['content'].split('Test input:\n')[1].split('\n\nOptions:')[0]
            image=DATA/audit[source_id]['image_files'][0];assert image.exists()
            results={}
            for k in kk:
                rule=json.loads(k['fact']['quote']);values=[solve(rule,x) for x in obs]
                assert values==[independent(rule,x) for x in obs]
                results[k['id']]=assignment(values)
            assert len(set(results.values()))==4
            base_options=list(results.values());rng.shuffle(base_options)
            report=base_options[rng.randrange(4)]
            for task in ('direct','diagnosis'):
                block=f'g{group_index}:{source_id}:{task}'
                for shift in range(4):
                    options=base_options[shift:]+base_options[:shift]
                    contents=[('Result: '+result+'.' if task=='direct' else ('Accept report. ' if result==report else 'Replace report. ')+'Result: '+result+'.') for result in options]
                    option_text='\n\nOptions:\n'+'\n'.join(f'{letter}. {content}' for letter,content in zip('ABCD',contents))+ev.SUFFIX
                    for modality in ('T','V'):
                        observation='\n\nTest input:\n'+source_text if modality=='T' else '\n\n<image>'
                        body=observation+option_text
                        for k in kk:
                            heading=('Apply '+name(k)+'. Report KEEP/OMIT for each candidate.' if task=='direct' else
                                'A technician submitted this report under '+name(k)+': Result: '+report+'. Check the observations and the rule. Report KEEP/OMIT for each candidate.')
                            q=heading+' KEEP means selected and OMIT means not selected.'+body
                            gold='ABCD'[options.index(results[k['id']])]
                            p={'id':f'named:{block}:{modality}:{shift}:{k["id"]}','qid':k['id'],'block':block,'group':group_index,
                                'modality':modality,'shift':shift,'task':task,'gold':gold,'condition':'named_'+modality,
                                'fmt':'mcq','messages':[{'role':'system','content':ev.SYSTEM},{'role':'user','content':q}],
                                'original_image':str(image),'original_source_id':source_id,'shared_body_sha256':ev.digest(body)}
                            if modality=='V':p['images']=[str(image)]
                            probes.append(p)
                            for direction in (('T2T','V2T') if modality=='T' else ('T2V','V2V')):
                                exported={key:p[key] for key in ('id','qid','messages','images') if key in p}
                                exported['id']=p['id'].replace(':'+modality+':',':')
                                exports[direction].append(exported)
                            labels.append({key:p[key] for key in ('id','qid','block','group','modality','shift','task','gold')})
                            if shift==0:
                                oracle=copy.deepcopy(p);oracle.update(id='oracle:'+p['id'],condition='oracle_'+modality)
                                oracle['messages'][-1]['content']='Correct learning reference:\n'+card(k)+'\n\n'+q
                                probes.append(oracle)
                        # Anonymous control generated once. Its single answer is
                        # scored against all four real rule keys, not a guessed
                        # uniquely correct answer for an underspecified task.
                        h=('Apply the stored rule. Report KEEP/OMIT for each candidate.' if task=='direct' else
                           'A technician submitted this report under the stored rule: Result: '+report+'. Check the observations and the rule. Report KEEP/OMIT for each candidate.')
                        p={'id':f'anonymous:{block}:{modality}:{shift}','qid':None,'block':block,'group':group_index,
                            'modality':modality,'shift':shift,'task':task,'gold':None,'condition':'anonymous_'+modality,'fmt':'mcq',
                            'messages':[{'role':'system','content':ev.SYSTEM},{'role':'user','content':h+' KEEP means selected and OMIT means not selected.'+body}],
                            'original_image':str(image),'shared_body_sha256':ev.digest(body)}
                        if modality=='V':p['images']=[str(image)]
                        probes.append(p)
                blocks.append({'block':block,'group':group_index,'task':task,'source_id':source_id,'image':str(image),'image_sha256':sha(image),
                    'text_observations':source_text,'structured_observations':obs,'rule_ids':[k['id'] for k in kk],
                    'rule_results':results,'base_options':base_options,'submitted_report':report if task=='diagnosis' else None})
    assert len(blocks)==10 and len(probes)==480
    for k in target:
        for has_image in (False,True):
            p={'id':('recall_image:' if has_image else 'recall:')+k['id'],'qid':k['id'],'condition':'recall_image' if has_image else 'recall',
                'fmt':'card','gold':card(k),'messages':msg(('<image>Using this original learning reference, ' if has_image else '')+
                   'state the complete rule named '+name(k)+'. Include every exact condition, comparison, value, combination and boundary. State the rule itself, not an example.')}
            if has_image:p['images']=[str(DATA/'images'/k['id']/'learning_visual.png')]
            probes.append(p)
    assert len(probes)==496
    for b in blocks:
        for modality in ('T','V'):
            for shift in range(4):
                group=[p for p in probes if p.get('block')==b['block'] and p.get('condition')=='named_'+modality and p.get('shift')==shift]
                assert len(group)==4 and {p['gold'] for p in group}==set('ABCD')
                assert len({p['shared_body_sha256'] for p in group})==1
                texts=[p['messages'][-1]['content'].replace(name(allk[p['qid']]),'<NAME>') for p in group]
                assert len(set(texts))==1
    ev.save_rows(out/'probes.jsonl',probes);ev.save_rows(out/'blocks.jsonl',blocks);ev.save_rows(out/'labels.jsonl',labels)
    for direction,data in exports.items():
        assert len(data)==160
        ev.save_rows(out/'dataset'/direction/'eval.jsonl',data)
        shutil.copy2(out/('text.jsonl' if direction[0]=='T' else 'vision.jsonl'),out/'dataset'/direction/'train.jsonl')
    exported_labels=[]
    for label in labels:
        if label['modality']=='T':
            z={k:v for k,v in label.items() if k!='modality'}
            z['id']=label['id'].replace(':T:',':');z['answer']=z.pop('gold');z['question_type']=z.pop('task')
            exported_labels.append(z)
    assert len(exported_labels)==160
    assert all({r['id'] for r in rr}=={r['id'] for r in exported_labels} for rr in exports.values())
    ev.save_rows(out/'dataset/audit/eval_labels.jsonl',exported_labels)
    sources=[DATA/'audit/rule_knowledge.jsonl',DATA/'audit/sample_audit.jsonl',DATA/'T2T/eval.jsonl',DATA/'image_hashes.jsonl',PARENT,
             out/'selection_candidates.json',out/'image_review.json',ROOT/'scripts/run_v7_full_experiment.py',ROOT/'scripts/pilot_v7_task_then_transfer.py']
    save(out/'manifest.json',{'script_sha256':sha(Path(__file__)),'created_at':ev.now(),'target_rule_ids':ids,
        'selection':'Programmatic disjunctive/interval rules only. Combinations ordered by SHA256 of ordered IDs; first group with >=2 own existing test images yielding four distinct results. No scores accessed. 3shape/color+2count images eligible.',
        'parent':ev.checkpoint(Path(read(PARENT)['path'])),'arms':ARMS,'rows':512,'epochs':1,'steps':128,'seed':42,
        'training':'64 material-only full-rule repetitions per rule; matched TEXT/VISION; factual supervision identical; no target test observations or choices in training. Same warmup parent, new optimizer, final checkpoint.',
        'data':'8 real v7 rules,5 original test PNGs,10 paired blocks,40 semantic questions per test modality,4 option positions=160 requests per modality. No original image modified or synthesized.',
        'protocols':'named(no materials),self(uncorrected name-only recall),oracle(correct external text,shift0 only),anonymous(once per block; each valid response matches1of4 reference keys by design).',
        'new_generations_per_arm':816,'main_probes':496,'self_probes':320,
        'limits':['Development pilot; 5shared scenes,8rules,2selector subfamilies,1seed. No inferential claim from request count.',
            'Rule definitions selected by image discriminability; not a representative sample of1000rules.',
            'Two rules appeared in previous atomic pilot; no overlap with shared warmup development rules. No score-based selection.',
            'Questions are rebuilt from existing observations. They are not unchanged original eval tasks or an official v8.',
            '64exposures/rule and512rows are not equivalent to historical1000rule1epoch.',
            'PT not trained; no native CoT; oracle and anonymous diagnostics separate from transfer.'],
        'sources':{str(p):sha(p) for p in sources},'files':{str(p.relative_to(out)):sha(p) for p in out.rglob('*.jsonl') if 'source_snapshots' not in p.parts}})
    snap=out/'source_snapshots';snap.mkdir(exist_ok=True)
    for p in (Path(__file__),ROOT/'scripts/run_v7_full_experiment.py',ROOT/'scripts/pilot_v7_task_then_transfer.py'):
        shutil.copy2(p,snap/p.name)
    validate(out)


def train(out,arm,gpu):
    if arm=='warmup':return Path(read(PARENT)['path'])
    folder=out/arm;folder.mkdir(exist_ok=True);cpdir=folder/'checkpoint'
    cp=final_checkpoint(cpdir,1)
    if cp is None:
        recover=[p for p in cpdir.glob('checkpoint-*') if all((p/f).exists() for f in ('optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','adapter_model.safetensors'))]
        resume=max(recover,key=lambda p:read(p/'trainer_state.json')['global_step']) if recover else None
        if cpdir.exists() and not resume:cpdir.rename(folder/('interrupted_'+str(int(__import__('time').time()))))
        cmd=train_command('qwen3','sft',out/(arm+'.jsonl'),cpdir,1,resume=resume)
        for flag,value in (('--per_device_train_batch_size','4'),('--gradient_accumulation_steps','1'),('--save_total_limit','3')):cmd[cmd.index(flag)+1]=value
        cmd+=['--save_strategy','steps','--save_steps','32']
        if not resume:cmd+=['--adapters',read(PARENT)['path']]
        save(folder/'train_state.json',{'status':'queued','command':cmd})
        with reserve(LEDGER,gpu,str(folder)+':train',29000 if arm=='vision' else 22000) as res:
            with (folder/'train.log').open('a') as log:
                p=subprocess.Popen(cmd,cwd=ROOT,env=environment(gpu,DATA),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                res.bind(p.pid);save(folder/'train_state.json',{'status':'running','command':cmd,'pid':p.pid});code=p.wait()
            assert code==0,f'Train failed: {arm}'
        cp=final_checkpoint(cpdir,1);assert cp
    st=read(cp/'trainer_state.json');args=read(cp/'args.json')
    assert st['global_step']==128 and abs(st['epoch']-1)<1e-6
    assert args['dataset']==[str(out/(arm+'.jsonl'))] and args['adapters']==[read(PARENT)['path']]
    losses=[r['loss'] for r in rows(cpdir/'logging.jsonl') if 'loss' in r]
    assert losses and all(math.isfinite(x) for x in losses)
    from safetensors import safe_open
    with safe_open(str(cp/'adapter_model.safetensors'),framework='pt',device='cpu') as f:
        keys=list(f.keys());bs=[k for k in keys if 'lora_B' in k]
        assert len(bs)==252 and all(bool(f.get_tensor(k).count_nonzero()) for k in bs)
        assert not any('.visual.' in k or '.merger.' in k for k in keys)
    save(folder/'checkpoint_manifest.json',{**ev.checkpoint(cp),'parent':ev.checkpoint(Path(read(PARENT)['path'])),
        'epoch':1,'global_step':128,'last_logged_loss':losses[-1],'nonzero_lora_B':252,'selection':'fixed final epoch1; no score selection'})
    save(folder/'train_state.json',{'status':'complete','checkpoint':str(cp)})
    return cp


def evaluate(out,arm,gpu):
    folder=out/arm;folder.mkdir(exist_ok=True)
    cp=Path(read(PARENT)['path']) if arm=='warmup' else Path(read(folder/'checkpoint_manifest.json')['path'])
    config={'adapter':ev.checkpoint(cp),'probes_sha256':sha(out/'probes.jsonl'),'temperature':0,'seed':42,
        'batch':4,'max_tokens':{'card':384,'mcq':16},'thinking':False}
    if (folder/'eval_manifest.json').exists():assert read(folder/'eval_manifest.json')==config
    else:save(folder/'eval_manifest.json',config)
    raw=folder/'predictions.jsonl';rr,notes=ev.read_rows(raw,allow_partial_tail=True) if raw.exists() else ([],[])
    if notes:ev.save_rows(raw,rr)
    seen={r['id'] for r in rr};assert len(seen)==len(rr)
    initial=rows(out/'probes.jsonl')

    def self_probes():
        recall={r['qid']:r for r in rr if r['condition']=='recall'};assert len(recall)==8
        pp=[]
        for p in initial:
            if not p['condition'].startswith('named_'):continue
            z=copy.deepcopy(p);z.update(id='self:'+p['id'],condition='self_'+p['modality'])
            z['messages'][-1]['content']='Your own prior rule recall (uncorrected):\n'+recall[p['qid']]['response']+'\n\n'+z['messages'][-1]['content']
            z['recall_response_sha256']=ev.digest(recall[p['qid']]['response']);pp.append(z)
        assert len(pp)==320
        return pp

    def infer(engine,probes):
        from swift.infer_engine import InferRequest,RequestConfig
        pending=[p for p in probes if p['id'] not in seen]
        for start in range(0,len(pending),4):
            batch=pending[start:start+4]
            outputs=engine.infer([InferRequest(messages=copy.deepcopy(p['messages']),images=p.get('images',[])) for p in batch],
                RequestConfig(max_tokens=config['max_tokens'][batch[0]['fmt']],temperature=0,seed=42,return_details=True),use_tqdm=False)
            assert len(batch)==len(outputs)
            for p,reply in zip(batch,outputs):
                if isinstance(reply,Exception):raise reply
                c=reply.choices[0];text=c.message.content or '';pred=ev.letter(text) if p['fmt']=='mcq' else None
                r={k:p[k] for k in ('id','qid','condition','fmt','gold')}
                for k in ('block','group','modality','shift','task','recall_response_sha256'):
                    if k in p:r[k]=p[k]
                r.update(response=text,predicted=pred,correct=pred==p['gold'] if p['gold'] in list('ABCD') else None,
                    format_error=p['fmt']=='mcq' and pred is None,finish_reason=c.finish_reason,
                    input_sha256=ev.digest({'messages':p['messages'],'images':p.get('images',[])}),raw_response=dataclasses.asdict(reply))
                rr.append(r);seen.add(r['id'])
                with raw.open('a') as f:
                    f.write(json.dumps(r,ensure_ascii=False,default=str)+'\n');f.flush();os.fsync(f.fileno())
            save(folder/'eval_state.json',{'status':'running','saved':len(rr),'expected':816,'pid':os.getpid()})
            print(arm,len(rr),'/816',flush=True)

    if len(rr)<816:
        with reserve(LEDGER,gpu,str(folder)+':eval',24000) as res:
            res.bind(os.getpid());os.environ.update(environment(gpu,DATA))
            import torch
            from swift.infer_engine import TransformersEngine
            from swift.utils import seed_everything
            seed_everything(42)
            engine=TransformersEngine(str(MODELS['qwen3']),adapters=[str(cp)],torch_dtype=torch.bfloat16,
                device_map='cuda:0',max_batch_size=4,attn_impl='sdpa')
            engine.template.enable_thinking=False
            infer(engine,[p for p in initial if p['fmt']=='card'])
            infer(engine,[p for p in initial if p['fmt']=='mcq'])
            pp=self_probes();ev.save_rows(folder/'self_probes.jsonl',pp);infer(engine,pp)
    assert len(rr)==len(seen)==816
    lookup={p['id']:p for p in initial+self_probes()}
    for r in rr:
        p=lookup[r['id']]
        assert r['input_sha256']==ev.digest({'messages':p['messages'],'images':p.get('images',[])})
        assert r['response']==(r['raw_response']['choices'][0]['message']['content'] or '')
    assert ev.checkpoint(cp)==config['adapter']
    save(folder/'eval_state.json',{'status':'complete','saved':816})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=OUT);ap.add_argument('--gpu',default='0')
    ap.add_argument('--prepare',action='store_true');ap.add_argument('--arm',choices=ARMS);ap.add_argument('--stage',choices=['train','eval'],default='train');a=ap.parse_args()
    out=a.out.resolve()
    if a.prepare:prepare(out)
    else:
        validate(out)
        with ev.run_lock(out/(a.arm+'_'+a.stage+'.lock')):
            if a.stage=='train':train(out,a.arm,a.gpu)
            else:evaluate(out,a.arm,a.gpu)
