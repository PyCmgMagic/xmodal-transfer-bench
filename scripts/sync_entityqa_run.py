#!/usr/bin/env python3
"""Keep a local receipt of an authorized remote run; fetch final LoRAs on completion."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from entityqa_common import save,now


def main(args):
    work=args.work.resolve()
    work.mkdir(parents=True,exist_ok=True)
    lock=(work/'sync.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        try:
            query=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',args.host,
                'cat '+shlex.quote(args.remote_work.rstrip('/')+'/status.json')],
                capture_output=True,text=True,timeout=30,check=True)
            state=json.loads(query.stdout)
            final=state.get('stage') in ['complete','failed']
            command=['rsync','-a','--timeout=30','--exclude=merged/','--exclude=*.bin','--exclude=*.pt',
                '--exclude=smoke/**/adapter_model.safetensors']
            if not final:command+=['--exclude=*.safetensors']
            command += [args.host+':'+args.remote_work.rstrip('/')+'/',str(work)+'/']
            subprocess.run(command,check=True,timeout=900 if final else 90,capture_output=True,text=True)
            save(work/'sync_status.json',dict(stage=state['stage'],pid=os.getpid(),
                updated_at=now(),final_adapters_fetched=final,remote=args.host+':'+args.remote_work))
            if final:return
        except Exception as error:
            save(work/'sync_status.json',dict(stage='retrying_sync',error=repr(error),pid=os.getpid(),updated_at=now()))
        time.sleep(60)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host',required=True)
    p.add_argument('--remote-work',required=True)
    p.add_argument('--work',type=Path,required=True)
    main(p.parse_args())
