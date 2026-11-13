#!/usr/bin/env python3
"""Run the per-configuration one-step/reload preflight separately."""
import argparse
from pathlib import Path
from run_v7_full_experiment import ROOT, MODELS, smoke, save, now

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--work',type=Path,default=ROOT/'runs/v7_full_experiment')
    ap.add_argument('--root',type=Path,default=ROOT/'rules_repaired_single_image_v7')
    ap.add_argument('--gpu',default='0')
    ap.add_argument('--modality',choices=['text','vision'],required=True)
    args=ap.parse_args()
    failures=[]
    for model in MODELS:
        for obj in ('sft','pt'):
            print('smoke',model,obj,args.modality,flush=True)
            try:
                smoke(model,obj,args.modality,args.gpu,args.work.resolve(),args.root.resolve())
            except Exception as exc:
                failures.append({'model':model,'objective':obj,'error':str(exc)})
                print('FAILED',model,obj,str(exc),flush=True)
    save(args.work/f'smoke_{args.modality}_status.json',{'status':'pass' if not failures else 'failed','failures':failures,'updated_at':now()})
    if failures: raise SystemExit(1)

if __name__=='__main__':main()
