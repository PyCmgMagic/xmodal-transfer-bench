#!/usr/bin/env python3
"""One targeted TEXT material-only intervention after alias diagnosis."""
import argparse
import itertools
import json
import math
import subprocess
from pathlib import Path

import run_v7_evaluation as ev
from build_v7_experiment_data import name
from pilot_v7_task_then_transfer import DATA, LEDGER, evaluate, validate_sources
from run_v7_full_experiment import ROOT, environment, final_checkpoint, read, rows, save, sha, train_command
from v7_gpu_resources import reserve


def run(out,gpu):
    validate_sources(out)
    arm='text_alias';folder=out/arm;folder.mkdir(exist_ok=True)
    knowledge={k['id']:k for k in rows(out/'target_knowledge.jsonl')}
    parent=final_checkpoint(out/'warmup/checkpoint',1);assert parent
    trainrows=rows(out/'text.jsonl');changed=0
    for r in trainrows:
        k=knowledge[r['qid']];index=int(r['id'].split(':')[-1])
        if k['fact']['relation']=='three_way_category':
            labels=json.loads(k['fact']['quote'])['category_labels'];perms=list(itertools.permutations(labels))
            alias=(', ' if (index//6)%2==0 else '/').join(perms[index%6]) if index<60 else ', '.join(labels)
            assert r['messages'][-2]['content'].count(name(k))==1
            r['messages'][-2]['content']=r['messages'][-2]['content'].replace(name(k),alias)
            changed+=alias!=name(k)
        r['id']=r['id'].replace(':text:',':text_alias:')
    dataset=folder/'train.jsonl'
    if dataset.exists():assert rows(dataset)==trainrows
    else:ev.save_rows(dataset,trainrows)
    config={'parent':ev.checkpoint(parent),'reference_training_sha256':sha(out/'text.jsonl'),'training_sha256':sha(dataset),'original_primary_probes_sha256':sha(out/'probes.jsonl'),'script_sha256':sha(Path(__file__)),'change':'Only three-category rule names in training user queries: all6permutations x10 exposures, plus4canonical; comma or slash. Set rules, assistant targets, row order, steps and optimizer settings unchanged. No test observations or answers enter training.','changed_rows':changed,'rows':512,'epochs':1,'steps':128,'comparison':'text arm from same warmup; fixed final1epoch; this is a diagnosis-informed development intervention, not preregistered primary evidence.'}
    if (folder/'training_manifest.json').exists():assert read(folder/'training_manifest.json')==config
    else:save(folder/'training_manifest.json',config)
    checkpoint_dir=folder/'checkpoint';cp=final_checkpoint(checkpoint_dir,1)
    if cp is None:
        recover=[p for p in checkpoint_dir.glob('checkpoint-*') if all((p/n).exists() for n in ('optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','adapter_model.safetensors'))]
        resume=max(recover,key=lambda p:read(p/'trainer_state.json')['global_step']) if recover else None
        cmd=train_command('qwen3','sft',dataset,checkpoint_dir,1,resume=resume)
        for flag,value in (('--per_device_train_batch_size','4'),('--gradient_accumulation_steps','1'),('--save_total_limit','3'),('--save_strategy','steps')):cmd[cmd.index(flag)+1]=value
        cmd+=['--enable_channel_loss','true','--save_steps','32']
        if not resume:cmd+=['--adapters',str(parent)]
        with reserve(LEDGER,gpu,str(folder)+':train',27000) as reservation:
            with (folder/'train.log').open('a') as log:
                proc=subprocess.Popen(cmd,cwd=ROOT,env=environment(gpu,DATA),stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
                reservation.bind(proc.pid);save(folder/'train_state.json',{'status':'running','pid':proc.pid,'command':cmd,'parent':config['parent']})
                code=proc.wait()
            assert code==0,f'text_alias training failed:{code}'
        cp=final_checkpoint(checkpoint_dir,1);assert cp
    state=read(cp/'trainer_state.json');assert state['global_step']==128 and abs(state['epoch']-1)<1e-6
    losses=[r['loss'] for r in rows(checkpoint_dir/'logging.jsonl') if 'loss' in r];assert losses and all(math.isfinite(x) for x in losses)
    assert str(parent) in read(cp/'args.json')['adapters']
    from safetensors import safe_open
    with safe_open(str(cp/'adapter_model.safetensors'),framework='pt',device='cpu') as f:
        bs=[k for k in f.keys() if 'lora_B' in k];assert len(bs)==252 and all(bool(f.get_tensor(k).count_nonzero()) for k in bs)
    save(folder/'checkpoint_manifest.json',{**ev.checkpoint(cp),'parent':config['parent'],'epoch':1,'global_step':128,'final_loss':losses[-1],'nonzero_lora_B':252,'selection':'fixed final epoch1; no evaluation selection'})
    save(folder/'train_state.json',{'status':'complete','checkpoint':str(cp)})
    evaluate(out,arm,gpu,cp)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=Path('runs/v7_task_then_transfer_20261015/pilot_v2'));ap.add_argument('--gpu',default='0');a=ap.parse_args()
    (a.out/'text_alias').mkdir(exist_ok=True)
    with ev.run_lock(a.out/'text_alias/.lock'):run(a.out.resolve(),a.gpu)
