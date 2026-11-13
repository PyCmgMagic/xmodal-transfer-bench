#!/usr/bin/env python3
"""Resume the completed entity LoRAs on the user's own server without retraining."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

from entityqa_common import rows, save, sha, now
from run_entityqa_lora import checkpoint, idle, merge


def unchanged_prepared_model(base, adapter, dest, expected_base):
    """Reuse a completed SHA256 verification while its files are unchanged."""
    manifest=dest/'merge_manifest.json'
    if not manifest.exists():return None
    recorded=json.loads(manifest.read_text())
    if recorded['base_sha256']!=expected_base:return None
    if Path(recorded['base'])!=base or Path(recorded['adapter'])!=adapter:return None
    if set(recorded['files'])!=set(expected_base):return None
    verified_at_ns=manifest.stat().st_mtime_ns
    paths=[base/name for name in expected_base]
    paths += [adapter/'adapter_model.safetensors']
    paths += [dest/name for name in recorded['files']]
    stats={}
    for path in paths:
        if not path.is_file():return None
        stat=path.stat()
        if stat.st_size<=0 or max(stat.st_ctime_ns,stat.st_mtime_ns)>verified_at_ns:return None
        stats[str(path)]=dict(size=stat.st_size,ctime_ns=stat.st_ctime_ns,mtime_ns=stat.st_mtime_ns)
    return dict(method='Reuse completed SHA256 verification: all weight ctimes and mtimes precede its manifest',
        manifest=str(manifest),manifest_sha256=sha(manifest),verified_at_ns=verified_at_ns,
        recorded_hashes=recorded,unchanged_file_stats=stats,checked_at=now())


def run(args):
    root,source,work,base,data=(getattr(args,k).resolve() for k in ['root','source','work','model','data'])
    # Resolving a venv's executable symlink selects the underlying interpreter
    # and loses the venv's site-packages. Keep the venv entry path intact.
    infer_python=args.infer_python.absolute()
    work.mkdir(parents=True,exist_ok=True)
    lock=(work/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    inventory=json.loads((data/'package_inventory.json').read_text())
    for filename,expected in inventory['sha256'].items():
        assert sha(data/filename)==expected,filename
    remote_merge=json.loads((work/'provenance/remote_TEXT_merge_manifest.json').read_text())
    prepared={}
    for arm in ['TEXT','VISION']:
        candidates=sorted((source/'train'/arm).glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
        if candidates:
            proof=unchanged_prepared_model(base,candidates[-1],work/'merged'/arm,remote_merge['base_sha256'])
            if proof:prepared[arm]=proof
    save(work/'preparation_reuse.json',dict(models=prepared,checked_at=now()))
    # A completed merge already hashed this exact base. Re-read large weights
    # only when metadata shows a change since that verification.
    if len(prepared)!=2:
        for name,expected in remote_merge['base_sha256'].items():
            assert sha(base/name)==expected,f'Base weights differ from training: {name}'
    source_config=json.loads((source/'experiment.json').read_text())
    assert source_config['epochs']==3 and source_config['knowledge_units']==1814
    assert source_config['rank']==64 and source_config['alpha']==128
    assert source_config['dataset_inventory_sha256']==sha(data/'package_inventory.json')
    snapshots=work/'source_snapshots'
    snapshots.mkdir(exist_ok=True)
    script_names=['run_entityqa_local_evaluation.py','run_entityqa_lora.py','evaluate_entityqa_vllm.py',
                  'entityqa_common.py','answer_readout.py']
    for name in script_names:shutil.copy2(root/'scripts'/name,snapshots/name)
    save(work/'experiment.json',dict(status='running',created_at=now(),host=socket.gethostname(),
        execution_root=str(root),gpu=args.gpu,gpu_name='NVIDIA L40',training_source=str(source),
        training_reused=True,training_epochs=3,knowledge_units=1814,
        training_method='MMKU-style attribute QA SFT, separate TEXT/VISION LoRA rank64 alpha128',
        training_config=source_config,data=str(data),base=str(base),
        inference_python=str(infer_python),
        migration='Remote school waiting controller was stopped at user request. Evaluation runs locally.',
        max_tokens=65536,max_model_len=73728,wall_seconds=180,
        source_sha256={name:sha(snapshots/name) for name in script_names},
        dataset_inventory_sha256=sha(data/'package_inventory.json'),controller_pid=os.getpid()))
    for arm in ['TEXT','VISION']:
        if arm in prepared:
            save(work/'status.json',dict(stage=f'reusing_verified_{arm}',training_complete=True,
                controller_pid=os.getpid(),updated_at=now()))
            continue
        ck=checkpoint(source/'train'/arm)
        expected=json.loads((source/'train'/arm/'validation.json').read_text())['adapter_sha256']
        assert sha(ck/'adapter_model.safetensors')==expected
        save(work/'status.json',dict(stage=f'merge_{arm}',training_complete=True,
            controller_pid=os.getpid(),updated_at=now()))
        merge(base,ck,work/'merged'/arm)
    if args.prepare_only:
        save(work/'status.json',dict(stage='models_ready_locally',training_complete=True,
            controller_pid=os.getpid(),updated_at=now()))
        return
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu,HF_HUB_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
        VLLM_WORKER_MULTIPROC_METHOD='spawn',PYTHONUNBUFFERED='1')

    def evaluate(arm,output,smoke=False):
        report=output/'summary.json'
        if report.exists() and json.loads(report.read_text()).get('status')=='complete':return
        idle(args.gpu)
        model=base if arm=='base' else work/'merged'/arm
        cmd=[str(infer_python),'-u',str(root/'scripts/evaluate_entityqa_vllm.py'),
            '--data',str(data),'--model',str(model),'--output',str(output),'--arm',arm,'--gpu',args.gpu]
        if smoke:cmd+=['--limit','2']
        stage='local_inference_smoke' if smoke else f'evaluating_{arm}'
        save(work/f'{stage}_command.json',cmd)
        log=work/f'{stage}.log'
        with log.open('a') as stream:
            process=subprocess.Popen(cmd,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT)
        save(work/'status.json',dict(stage=stage,child_pid=process.pid,controller_pid=os.getpid(),
            host=socket.gethostname(),gpu=args.gpu,log=str(log),updated_at=now()))
        code=process.wait()
        if code:raise RuntimeError(f'{stage} failed ({code}); inspect {log}')
        for _ in range(15):
            try:idle(args.gpu);break
            except RuntimeError:time.sleep(2)
        idle(args.gpu)

    evaluate('TEXT',work/'inference_smoke',smoke=True)
    for arm in ['TEXT','VISION','base']:
        evaluate(arm,work/'evaluation'/arm)
    reports={arm:json.loads((work/f'evaluation/{arm}/summary.json').read_text()) for arm in ['TEXT','VISION','base']}
    configs=[json.loads((work/f'evaluation/{arm}/config.json').read_text()) for arm in ['TEXT','VISION','base']]
    assert all(r['status']=='complete' and r['saved']==7256 for r in reports.values())
    assert len({c['input_sha256'] for c in configs})==1
    directions={}
    for arm,letter in [('TEXT','T'),('VISION','V')]:
        for modality in ['T','V']:
            group=reports[arm]['groups'][f'{modality}/all']
            before=reports['base']['groups'][f'{modality}/all']['accuracy']
            directions[f'{letter}2{modality}']=dict(**group,baseline_accuracy=before,
                accuracy_gain_percentage_points=100*(group['accuracy']-before))
    save(work/'comparison.json',dict(status='complete',directions=directions,reports=reports,completed_at=now()))
    save(work/'status.json',dict(stage='complete',host=socket.gethostname(),updated_at=now()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['root','source','work','model','data','infer-python']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--gpu',default='0')
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    try:run(args)
    except Exception as error:
        save(args.work/'failure.json',dict(error=repr(error),time=now()))
        save(args.work/'status.json',dict(stage='failed',error=repr(error),updated_at=now(),controller_pid=os.getpid()))
        raise
