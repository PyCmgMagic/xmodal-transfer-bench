#!/usr/bin/env python3
"""Resume only evaluation of the two already validated full-fact checkpoints."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import time

from entityqa_common import now, save, sha
from run_entityqa_lora import idle


def run(args):
    work=args.work.resolve()
    lock=(work/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((work/'experiment.json').read_text())
    root=Path(plan['root'])
    assert socket.gethostname()==plan['host']=='softengine-KVM'
    gpu=plan['gpu']

    def status(stage,**extra):
        save(work/'status.json',dict(stage=stage,controller_pid=os.getpid(),
            host=socket.gethostname(),updated_at=now(),**extra))

    status('verifying_completed_models')
    for name,h in plan['source_sha256'].items():
        assert sha(work/'source_snapshots'/name)==h,name
    for arm in ['TEXT','VISION']:
        validation=json.loads((work/'train'/arm/'validation.json').read_text())
        assert validation['epoch']==3 and validation['global_step']==681
        assert validation['finite'] and validation['nonzero_lora_B']
        adapter=Path(validation['checkpoint'])/'adapter_model.safetensors'
        merge=json.loads((work/'merged'/arm/'merge_manifest.json').read_text())
        assert merge['adapter_sha256']==validation['adapter_sha256']
        # The training validator and merge stage already hashed these weights.
        # Reuse that proof while files remain unchanged; avoid rereading many GB.
        verified_at=(work/'merged'/arm/'merge_manifest.json').stat().st_mtime_ns
        for path in [adapter]+[work/'merged'/arm/name for name in merge['files']]:
            stat=path.stat()
            assert stat.st_size>0 and max(stat.st_ctime_ns,stat.st_mtime_ns)<=verified_at,path
        for name,h in merge['files'].items():
            assert name in merge['files'] and len(h)==64
    rpc=Path('/tmp/xm-e21-rpc')
    rpc.mkdir(mode=0o700,exist_ok=True)
    assert rpc.stat().st_uid==os.getuid()
    env=dict(os.environ,VLLM_RPC_BASE_PATH=str(rpc),TMPDIR=str(rpc),
        CUDA_VISIBLE_DEVICES=gpu,HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',VLLM_WORKER_MULTIPROC_METHOD='spawn')
    old_launch=json.loads((work/'launch.json').read_text())['command']
    data=old_launch[old_launch.index('--data')+1]
    python=old_launch[old_launch.index('--infer-python')+1]
    for arm in ['TEXT','VISION','base']:
        output=work/'evaluation'/arm
        summary=output/'summary.json'
        if summary.exists() and json.loads(summary.read_text()).get('status')=='complete':
            continue
        while True:
            try:
                idle(gpu)
                break
            except RuntimeError as error:
                status('waiting_for_free_gpu',reason=str(error),next_arm=arm)
                time.sleep(15)
        model=Path(plan['base']) if arm=='base' else work/'merged'/arm
        command=[python,'-u',str(work/'source_snapshots/evaluate_entityqa_fullfact_v2.py'),
            '--data',data,'--model',str(model),'--output',str(output),'--arm',arm,'--gpu',gpu]
        if arm=='base':
            command+=['--import-predictions',old_launch[old_launch.index('--baseline-predictions')+1]]
        save(work/('resumed_evaluating_'+arm+'_command.json'),command)
        with (work/('evaluating_'+arm+'.log')).open('a') as stream:
            process=subprocess.Popen(command,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT)
        status('evaluating_'+arm,child_pid=process.pid)
        code=process.wait()
        if code:
            raise RuntimeError(f'Evaluation {arm} failed ({code}); see evaluating_{arm}.log')
        for _ in range(15):
            try:
                idle(gpu)
                break
            except RuntimeError:
                time.sleep(2)
    reports={a:json.loads((work/f'evaluation/{a}/summary.json').read_text()) for a in ['TEXT','VISION','base']}
    configs=[json.loads((work/f'evaluation/{a}/config.json').read_text()) for a in ['TEXT','VISION','base']]
    assert all(r['status']=='complete' and r['saved']==7256 for r in reports.values())
    assert len({c['input_sha256'] for c in configs})==1
    assert len({c['readout_version'] for c in configs})==1
    directions={}
    for arm,prefix in [('TEXT','T'),('VISION','V')]:
        for modality in ['T','V']:
            result=reports[arm]['groups'][modality+'/all']
            before=reports['base']['groups'][modality+'/all']
            directions[prefix+'2'+modality]=dict(**result,
                baseline_primary_accuracy=before['primary_accuracy'],
                primary_gain_percentage_points=100*(result['primary_accuracy']-before['primary_accuracy']),
                baseline_constrained_accuracy=before['accuracy'])
    save(work/'comparison.json',dict(status='complete',directions=directions,reports=reports,completed_at=now()))
    plan.update(status='complete',completed_at=now())
    save(work/'experiment.json',plan)
    status('complete')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,required=True)
    args=parser.parse_args()
    try:
        run(args)
    except Exception as error:
        save(args.work/'failure.json',dict(error=repr(error),time=now()))
        save(args.work/'status.json',dict(stage='failed',error=repr(error),updated_at=now()))
        raise
