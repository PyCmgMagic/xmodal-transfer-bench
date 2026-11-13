#!/usr/bin/env python3
"""Fixed-budget task order experiment with continuous optimizer and verified data order."""
import argparse
import copy
import math
import os
from pathlib import Path
import random
import shutil
import subprocess

import explore_v7_prompt_loss_bridge as previous
import explore_v7_visual_replay as replay
import run_v7_evaluation as ev
from run_v7_full_experiment import ROOT,rows,read,save,sha,train_command,environment,final_checkpoint
from v7_gpu_resources import reserve

OUT=ROOT/'runs/v7_curriculum_20261016'
PLUGIN=ROOT/'scripts/v7_curriculum_loss.py'
ORDERS=('alternate','explain_match','match_explain')
ARMS=tuple(m+'_'+order for m in ('text','vision') for order in ORDERS)


def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'manifest.json').exists():validate();return
    previous.validate();previous.source.validate(previous.source.OUT);previous.paired.validate()
    mask={r['id']:r for r in rows(previous.OUT/'template_mask_audit.jsonl') if r['arm']=='text_equal'}
    assert len(mask)==512
    for modality in ('text','vision'):
        rr=rows(previous.OUT/(modality+'_equal.jsonl'))
        bychannel={}
        for c in ('explain','match'):
            selected=sorted([r for r in rr if r['channel']==c],key=lambda r:r['id'])
            random.Random('curriculum-within-task:42:'+c).shuffle(selected)
            bychannel[c]=[selected[i:i+4] for i in range(0,256,4)]
        for order in ORDERS:
            arm=modality+'_'+order
            if order=='alternate':batches=[b for pair in zip(bychannel['explain'],bychannel['match']) for b in pair]
            else:
                first,second=order.split('_');batches=bychannel[first]+bychannel[second]
            training=[r for b in batches for r in b]
            assert len(training)==512 and {r['id'] for r in training}==set(mask)
            ev.save_rows(OUT/(arm+'.jsonl'),training)
            plan=[]
            for i,batch in enumerate(batches):
                channels={r['channel'] for r in batch};assert len(channels)==1
                plan.append({'step_before_update':i,'channel':channels.pop(),'row_ids':[r['id'] for r in batch],
                    'supervised_ids_sha256':[ev.digest(mask[r['id']]['supervised_ids']) for r in batch]})
            save(OUT/(arm+'_plan.json'),plan)
    full=rows(previous.OUT/'main_probes.jsonl');assert len(full)==568
    half=[p for p in full if p['condition'] in ('recall','recall_image','recognition_para_noimage') or
          p['condition'].startswith(('named_','paired_')) and p['shift']==0]
    assert len(half)==144
    ev.save_rows(OUT/'final_probes.jsonl',full);ev.save_rows(OUT/'midpoint_probes.jsonl',half)
    sources=[previous.OUT/p for p in ('manifest.json','final_artifact_manifest.json','template_mask_audit.jsonl','main_probes.jsonl','text_equal.jsonl','vision_equal.jsonl','semantic_reviews.jsonl')]
    sources += [previous.source.PARENT,ROOT/'scripts/explore_v7_prompt_loss_bridge.py',ROOT/'scripts/v7_equal_example_loss.py',
        ROOT/'scripts/explore_v7_visual_replay.py',ROOT/'scripts/run_v7_full_experiment.py',ROOT/'scripts/run_v7_evaluation.py',
        ROOT/'scripts/v7_gpu_resources.py',PLUGIN,
        ROOT/'.venv/lib/python3.12/site-packages/swift/trainers/mixin.py',
        ROOT/'.venv/lib/python3.12/site-packages/swift/trainers/arguments.py']
    save(OUT/'manifest.json',{'created_at':ev.now(),'script_sha256':sha(Path(__file__)),
        'sources':{str(p):sha(p) for p in sources},
        'files':{p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.suffix in ('.json','.jsonl')},
        'parent':ev.checkpoint(previous.checkpoint('warmup')),'arms':ARMS,
        'rules':previous.source.EXPECTED,'rows_per_arm':512,'explain_rows':256,'match_rows':256,
        'epochs':1,'steps':128,'batch':4,'accumulation':1,'lr':1e-4,'seed':42,
        'loss':'Equal example mean CE; unchanged frozen loss implementation; continuous optimizer and cosine schedule128steps.',
        'order':'Identical64explanation batches and64recognition batches, same within-task batch order. Alternate E,M batches; explain_match E64,M64; match_explain M64,E64. Each batch homogeneous.',
        'shuffle':'dataset_shuffle=false, train_dataloader_shuffle=false, train_sampling_strategy=sequential, group_by_length=false. Supervised token sequence hashes asserted every optimizer step.',
        'checkpoint_protocol':'Fixed step128primary. Step64onlypredeclareddiagnostic; no best checkpoint selection, no optimizer reset at64.',
        'midpoint':'144answers:16full rule recalls with/without learning image,32noimage paraphrased recognitions,32original-v7shift0tasks,64pairedshift0tasks.',
        'new_generations':4272,'final_per_arm':568,'midpoint_per_arm':144,'smoke_reload_answers':4,
        'limits':['Same8development rules,1seed,two selector topologies; not an independent replication or full1000rule fix.',
                  'Original PNG and datasets unchanged. No application examples or test labels in training. Definition candidates contain rule facts.',
                  'VISION uses original learning images on all512rows and textual canonical targets/candidate definitions; no image dropout or text replay in this experiment.',
                  'Order changes exposure to early/late cosine learning rates. Results diagnose this curriculum package, not pure forgetting isolated from LR.',
                  'Halfway alternate saw32batches of each objective, blocked saw64ofone; midpoint scores are descriptive, not equal-per-task-budget comparisons.',
                  'No nativeCoT,no PT; evaluation prompt exactly same as previous primary protocol; rotations are dependent repeats.']})
    snap=OUT/'source_snapshots';snap.mkdir(exist_ok=True)
    for p in [Path(__file__)]+[p for p in sources if p.suffix=='.py']:shutil.copy2(p,snap/p.name)
    validate();print('Prepared6ordered512row sets;568final+144midpoint probes per arm.',flush=True)


def validate():
    m=read(OUT/'manifest.json');assert m['script_sha256']==sha(Path(__file__))
    for path,h in m['sources'].items():assert sha(Path(path))==h,path
    for path,h in m['files'].items():assert sha(OUT/path)==h,path
    assert ev.checkpoint(previous.checkpoint('warmup'))==m['parent']


def checkpoint(arm,step):
    return OUT/arm/'checkpoint'/('checkpoint-'+str(step))


def train(arm,gpu,smoke=False):
    folder=OUT/('smoke_'+arm if smoke else arm);folder.mkdir(exist_ok=True)
    cpdir=folder/'checkpoint';cp=final_checkpoint(cpdir,1) if not smoke else None
    if smoke and (folder/'validated.json').exists():return
    if cp is None:
        recover=[p for p in cpdir.glob('checkpoint-*') if all((p/f).exists() for f in ('optimizer.pt','scheduler.pt','rng_state.pth','adapter_model.safetensors','trainer_state.json'))]
        resume=max(recover,key=lambda p:read(p/'trainer_state.json')['global_step']) if recover and not smoke else None
        cmd=train_command('qwen3','sft',OUT/(arm+'.jsonl'),cpdir,1,resume=resume)
        for flag,val in (('--per_device_train_batch_size','4'),('--gradient_accumulation_steps','1'),('--save_total_limit','3'),('--save_strategy','steps')):
            cmd[cmd.index(flag)+1]=val
        cmd+=['--save_steps','1' if smoke else '64','--external_plugins',str(PLUGIN),'--loss_type','v7_curriculum_equal',
            '--dataset_shuffle','false','--train_dataloader_shuffle','false','--train_sampling_strategy','sequential',
            '--group_by_length','false','--packing','false','--padding_free','false']
        if smoke:cmd+=['--max_steps','1']
        if not resume:cmd+=['--adapters',str(previous.checkpoint('warmup'))]
        env=environment(gpu,previous.source.DATA)
        env.update(V7_CURRICULUM_PLAN=str(OUT/(arm+'_plan.json')),
            V7_CURRICULUM_ORDER_AUDIT=str(folder/'actual_order.jsonl'),V7_EQUAL_LOSS_AUDIT=str(folder/'loss_batches.jsonl'))
        save(folder/'train_state.json',{'status':'queued','command':cmd})
        with reserve(previous.source.LEDGER,gpu,str(folder)+':train',27000 if arm.startswith('vision') else 22000) as res:
            with (folder/'train.log').open('a') as log:
                process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                res.bind(process.pid);save(folder/'train_state.json',{'status':'running','pid':process.pid,'command':cmd})
                code=process.wait()
            save(folder/'process_exit.json',{'returncode':code});assert code==0,(arm,code)
        cp=cpdir/('checkpoint-1' if smoke else 'checkpoint-128')
    expected=1 if smoke else 128
    args=read(cp/'args.json');state=read(cp/'trainer_state.json')
    assert state['global_step']==expected
    if not smoke:assert state['epoch']==1 and read(cpdir/'checkpoint-64/trainer_state.json')['global_step']==64
    assert args['loss_type']=='v7_curriculum_equal'
    assert not any(args[k] for k in ('dataset_shuffle','train_dataloader_shuffle','group_by_length','packing','padding_free'))
    assert args['train_sampling_strategy']=='sequential'
    actual=rows(folder/'actual_order.jsonl');plan=read(OUT/(arm+'_plan.json'))
    # If a process was interrupted, preserve attempted rows and inspect the
    # last successful pass at each completed step rather than deleting history.
    last={r['step_before_update']:r for r in actual};assert set(last)==set(range(expected))
    for i,r in last.items():
        assert r['observed_supervised_ids_sha256']==plan[i]['supervised_ids_sha256']
        assert r['planned_row_ids']==plan[i]['row_ids'] and r['matches_planned_labels']
    losses=[r for r in rows(cpdir/'logging.jsonl') if 'loss' in r];assert losses and all(math.isfinite(r['loss']) for r in losses)
    from safetensors import safe_open
    with safe_open(str(cp/'adapter_model.safetensors'),framework='pt',device='cpu') as f:
        keys=list(f.keys());bs=[k for k in keys if 'lora_B' in k]
        assert len(bs)==252 and all(bool(f.get_tensor(k).count_nonzero()) for k in bs)
        assert not any('.visual.' in k or '.merger.' in k for k in keys)
    save(folder/'checkpoint_manifest.json',{**ev.checkpoint(cp),'global_step':expected,'epoch':state['epoch'],
        'parent':ev.checkpoint(previous.checkpoint('warmup')),'last_logged_loss':losses[-1]['loss'],
        'nonzero_lora_B':252,'actual_order_verified':True,'selection':'fixed final128primary; midpoint64diagnostic'})
    if smoke:
        pp=rows(OUT/'final_probes.jsonl');probe=[next(p for p in pp if p['condition']==c) for c in ('recognition_para_noimage','recognition_para_image')]
        cfg=replay.config(cp,{'mcq':16});cfg['smoke_probes_sha256']=ev.digest(probe)
        run=replay.Runner(folder/'reload',cfg,gpu)
        if len(run.rr)<2:
            with reserve(previous.source.LEDGER,gpu,str(folder)+':reload',24000) as res:
                res.bind(os.getpid());run.load();run.infer(probe)
        run.audit(probe,2);save(folder/'validated.json',{'status':'pass','text_and_image_reload':True,'step_order_checked':True})
    else:
        mid=cpdir/'checkpoint-64'
        save(folder/'midpoint_manifest.json',{**ev.checkpoint(mid),'global_step':64,'epoch':.5,'role':'predeclareddiagnostic only'})
    save(folder/'train_state.json',{'status':'complete','checkpoint':str(cp)})


def evaluate(arm,gpu,step):
    pp=rows(OUT/('final_probes.jsonl' if step==128 else 'midpoint_probes.jsonl'))
    cfg=replay.config(checkpoint(arm,step),{'mcq':16,'card':384});cfg['probes_sha256']=ev.digest(pp)
    folder=OUT/arm/('final' if step==128 else 'midpoint');run=replay.Runner(folder,cfg,gpu)
    if len(run.rr)<len(pp):
        with reserve(previous.source.LEDGER,gpu,str(folder)+':eval',24000) as res:
            res.bind(os.getpid());run.load()
            for fmt in ('card','mcq'):run.infer([p for p in pp if p['fmt']==fmt])
    run.audit(pp,len(pp))
    save(folder/'report.json',{'status':'complete','metrics':{c:previous.source.metric([r for r in run.rr if r['condition']==c])
        for c in sorted({r['condition'] for r in run.rr if r['fmt']=='mcq'})}})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('prepare','smoke','train','eval','run'),required=True)
    ap.add_argument('--arm',choices=ARMS);ap.add_argument('--gpu',default='0');ap.add_argument('--step',type=int,choices=(64,128),default=128);a=ap.parse_args()
    if a.stage=='prepare':prepare()
    else:
        validate()
        with ev.run_lock(OUT/(str(a.arm)+'_'+a.stage+'.lock')):
            if a.stage in ('smoke','train','run'):train(a.arm,a.gpu,a.stage=='smoke')
            if a.stage in ('eval','run'):
                evaluate(a.arm,a.gpu,a.step)
                if a.stage=='run':evaluate(a.arm,a.gpu,64)
