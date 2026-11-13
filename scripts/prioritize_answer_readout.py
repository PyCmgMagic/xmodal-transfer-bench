#!/usr/bin/env python3
"""Run one actual-model readout validation first, then resume durable queues."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import run_v7_evaluation as ev
from v7_gpu_resources import alive, reserve
import repair_answer_outputs as repair
import run_lora_recall_apply_v2 as recall


def background(work, script, extra=()):
    with (work/'launcher.log').open('a') as log:
        child=subprocess.Popen([sys.executable,'-u',str(script),*extra],cwd=repair.ROOT,
                               stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                               start_new_session=True,env=os.environ.copy())
    ev.save_json(work/'launcher_pid.json',{'pid':child.pid,'script':str(script),'started_at':ev.now()})
    return child.pid


def main():
    work=repair.WORK
    with ev.run_lock(work/'.priority.lock'):
        stopped=[];started={};error=None;changed=False
        try:
            state=json.loads((repair.FULL/'status.json').read_text());coordinator=state['pid']
            entries=json.loads((repair.FULL/'gpu_reservations.json').read_text())
            workers=[x['pid'] for x in entries.values() if x.get('owner_pid')==coordinator and x.get('pid')]
            for pid in [coordinator]+workers:
                cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
                assert b'/scripts/run_lora_cot_evaluation.py' in cmd,(pid,cmd)
            ev.save_json(work/'priority_intervention.json',{'status':'temporarily_yielding_gpu',
                          'stopped_pids':[coordinator]+workers,'reason':'Validate actual Qwen constrained final selection for rules_sft_text before continuing queued generation. Durable raw outputs retained.',
                          'started_at':ev.now()})
            changed=True
            for pid in [coordinator]+workers:
                os.kill(pid,signal.SIGTERM);stopped.append(pid)
            deadline=time.monotonic()+20
            while any(alive(pid) for pid in stopped) and time.monotonic()<deadline:time.sleep(.25)
            assert not any(alive(pid) for pid in stopped),'Evaluation processes did not exit; restoring queues'
            name='rules_sft_text';tasks=work/'task_batches/priority_rules_sft_text.jsonl'
            with reserve(repair.FULL,'0','priority_readout:'+name,18000,poll=2) as reservation:
                folder=work/'jobs'/name;folder.mkdir(parents=True,exist_ok=True)
                cmd=[sys.executable,'-u',str(repair.ROOT/'scripts/repair_answer_outputs.py'),
                     '--worker',name,'--tasks',str(tasks)]
                with (folder/'worker.log').open('a') as log:
                    child=subprocess.Popen(cmd,cwd=repair.ROOT,stdin=subprocess.DEVNULL,
                                           stdout=log,stderr=subprocess.STDOUT)
                    reservation.bind(child.pid)
                    ev.save_json(work/'status.json',{'status':'running_priority_readout','pid':os.getpid(),
                                  'profile':name,'worker_pid':child.pid,'updated_at':ev.now()})
                    assert child.wait()==0,'Actual-model constrained readout failed; inspect worker.log'
            summary,_=repair.refresh(work)
            reports=[r for r in summary['cells'] if r['job']==name and r['experiment']=='full_cot']
            assert len(reports)==2 and all(r['combined_accuracy'] is not None for r in reports)
            ev.save_json(work/'priority_validation.json',{'status':'passed','reports':reports,
                          'target_model_smoke':json.loads((work/'jobs'/name/'smoke.json').read_text()),
                          'completed_at':ev.now()})
        except BaseException as exc:
            error={'error':str(exc),'traceback':traceback.format_exc()}
            ev.save_json(work/'priority_error.json',error)
            raise
        finally:
            # Resume full evaluation from verified saved rows. Completed smoke
            # and jobs are skipped by the existing durable coordinator.
            if changed and not any(alive(pid) for pid in stopped):
                started['full_eval']=background(repair.FULL,repair.ROOT/'scripts/run_lora_cot_evaluation.py')
            started['recall_v2']=background(recall.DEFAULT_WORK,repair.ROOT/'scripts/run_lora_recall_apply_v2.py')
            started['output_repair']=background(work,repair.ROOT/'scripts/repair_answer_outputs.py',['--watch'])
            ev.save_json(work/'priority_intervention.json',{'status':'resumed_after_priority_check',
                          'previous_pids':stopped,'new_coordinators':started,'priority_error':error,'updated_at':ev.now()})
            print(json.dumps({'resumed':started,'error':error}),flush=True)


if __name__=='__main__':main()
