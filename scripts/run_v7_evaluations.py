#!/usr/bin/env python3
"""Evaluate ready adapters on either GPU under shared memory budgets."""
import argparse
import fcntl
import subprocess
import time
from pathlib import Path
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from run_v7_full_experiment import ROOT, MODELS, environment, read, save, now, verify_sources, process, alive
from v7_gpu_resources import EVAL_PEAKS, can_admit, reserve


def discover_children(work, out):
    """Upgrade launcher records written before child PIDs were persisted."""
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            cmd=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
            if len(cmd)<2 or Path(cmd[1]).name!='run_v7_adapter_eval.py':continue
            if '--work' not in cmd or Path(cmd[cmd.index('--work')+1]).resolve()!=work:continue
            name='_'.join(cmd[cmd.index('--'+key)+1] for key in ('model','objective','modality'))
            folder=out/name;prior=read(folder/'launcher_state.json')
            prior.update(status='running',pid=int(proc.name),command=cmd,adopted_at=now())
            save(folder/'launcher_state.json',prior)
        except (OSError,ValueError,IndexError,UnicodeDecodeError):continue


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=ROOT/'rules_repaired_single_image_v7')
    ap.add_argument('--work',type=Path,default=ROOT/'runs/v7_full_experiment')
    ap.add_argument('--poll',type=int,default=10)
    ap.add_argument('--protocols',nargs='+',choices=['strict','cot'],default=['strict'])
    a=ap.parse_args();work=a.work.resolve();data=a.root.resolve();out=work/'evaluations'
    out.mkdir(parents=True,exist_ok=True)
    with (work/'.evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save(out/'status.json',{'status':'running','early_evaluation_models':list(MODELS),'pid':__import__('os').getpid(),'updated_at':now()})
        deferred=[p for p in ('strict','cot') if p not in a.protocols]
        save(out/'protocols.json',{'protocols':a.protocols,'deferred_protocols':deferred,'updated_at':now()})
        manifest=read(work/'manifest.json')
        manifest.update(active_evaluation_protocols=a.protocols,deferred_evaluation_protocols=deferred,evaluation_cells=48*len(a.protocols),evaluation_requests=96000*len(a.protocols),original_full_evaluation_cells=96,original_full_evaluation_requests=192000)
        save(work/'manifest.json',manifest)
        verify_sources(work,data)
        discover_children(work,out)
        names=[f'{model}_{obj}_{mod}' for model in MODELS for obj in ('sft','pt') for mod in ('text','vision')]
        claimed=set();exhausted=set();claim_lock=Lock()

        def complete(name):
            prior=read(out/name/'status.json')
            return prior.get('status')=='complete' and set(a.protocols).issubset(prior.get('protocols',['strict','cot']))

        def summarize():
            subprocess.check_call([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/summarize_v7_experiment.py'),'--work',str(work)],cwd=ROOT)

        def worker(gpu):
            while True:
                name=None;adopted=None
                with claim_lock:
                    remaining=[n for n in names if not complete(n) and n not in exhausted]
                    if not remaining:return
                    # Adopt persisted children before launching any new process.
                    ordered=sorted(remaining,key=lambda n:(not alive(read(out/n/'launcher_state.json').get('pid',-1)),not n.endswith('text' if gpu==0 else 'vision')))
                    for candidate in ordered:
                        if candidate in claimed:continue
                        folder=out/candidate;folder.mkdir(exist_ok=True)
                        prior=read(folder/'launcher_state.json');cmd=prior.get('command',[])
                        live=prior.get('status')=='running' and alive(prior.get('pid',-1))
                        if live:
                            if '--gpu' not in cmd or cmd[cmd.index('--gpu')+1]!=str(gpu):continue
                            if '--protocols' not in cmd or cmd[cmd.index('--protocols')+1:]!=a.protocols:
                                raise RuntimeError('Running evaluator uses a different protocol; stop it before changing protocols')
                            adopted=prior['pid']
                        elif read(work/'jobs'/candidate/'state.json').get('status')!='complete':
                            if read(work/'training_status.json').get('status') in ('complete','incomplete'):
                                save(folder/'status.json',{'status':'blocked_training_failed'});exhausted.add(candidate)
                            continue
                        elif not can_admit(work,gpu,EVAL_PEAKS[candidate.split('_')[0]]):
                            continue
                        name=candidate;claimed.add(name);break
                if name is None:
                    time.sleep(a.poll);continue
                folder=out/name;model,obj,mod=name.split('_')
                try:
                    if adopted:
                        with reserve(work,gpu,str(folder.resolve())+':launcher',EVAL_PEAKS[model],adopted_pid=adopted):
                            while alive(adopted):time.sleep(a.poll)
                        if complete(name):
                            summarize();continue
                    cmd=[str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/run_v7_adapter_eval.py'),'--root',str(data),'--work',str(work),'--model',model,'--objective',obj,'--modality',mod,'--gpu',str(gpu),'--protocols',*a.protocols]
                    for attempt in range(1,4):
                        code=process(cmd,environment(gpu,data),folder,'launcher','launcher_state.json')
                        summarize()
                        if code==0:break
                        save(folder/'launcher_state.json',{'status':'failed','attempt':attempt,'returncode':code,'updated_at':now()})
                        time.sleep(a.poll)
                    if code!=0:
                        with claim_lock:exhausted.add(name)
                finally:
                    with claim_lock:claimed.discard(name)
        with ThreadPoolExecutor(max_workers=2) as pool:
            fs=[pool.submit(worker,gpu) for gpu in (0,1)]
            for f in fs:f.result()
        verify_sources(work,data);summarize()
        comparison=read(work/'summary/comparison.json')
        final={'status':'complete' if comparison['complete_cells']==48*len(a.protocols) else 'incomplete','protocols':a.protocols,'deferred_protocols':deferred,'complete_cells':comparison['complete_cells'],'matched':comparison['matched'],'expected':96000*len(a.protocols),'updated_at':now()}
        save(work/'final_status.json',final);save(out/'status.json',final)

if __name__=='__main__':main()
