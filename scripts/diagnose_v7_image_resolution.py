#!/usr/bin/env python3
"""Original-file resolution control, with identical rule-free count prompts."""
import argparse
import os
import shutil
from pathlib import Path

import explore_v7_visual_replay as replay
import pilot_v7_counterfactual_names as source
import run_v7_evaluation as ev
from run_v7_full_experiment import ROOT, rows, read, save, sha
from v7_gpu_resources import reserve

OUT=ROOT/'runs/v7_rule_recognition_20261016/resolution_control'
SETTINGS={'default':(980,4),'native':(2048,4),'upsample':(4096,4000)}


def prepare():
    OUT.mkdir(exist_ok=True)
    pp=[{**p,'fmt':'observation','condition':'count_transcription'} for p in
        rows(replay.OUT/'extraction_probes.jsonl') if p['group']==1]
    assert len(pp)==2
    manifest={'script_sha256':sha(Path(__file__)),
        'source_probes_sha256':sha(replay.OUT/'extraction_probes.jsonl'),
        'settings':SETTINGS,'arms':['vision','text_replay'],
        'protocol':'Same original2PNG and same generic count-only transcription prompt in every condition. No rule facts or candidate true counts in input. Fresh process per condition; no native thinking. This is observation diagnosis, not main transfer.',
        'original_image_hashes':{p['images'][0]:sha(Path(p['images'][0])) for p in pp}}
    if (OUT/'manifest.json').exists():
        assert read(OUT/'manifest.json')==__import__('json').loads(__import__('json').dumps(manifest))
        assert rows(OUT/'probes.jsonl')==pp
    else:
        save(OUT/'manifest.json',manifest);ev.save_rows(OUT/'probes.jsonl',pp)
        shutil.copy2(Path(__file__),OUT/Path(__file__).name)
    return pp


def evaluate(arm, setting, gpu):
    pp=prepare();maximum,minimum=SETTINGS[setting]
    os.environ['IMAGE_MAX_TOKEN_NUM']=str(maximum)
    os.environ['IMAGE_MIN_TOKEN_NUM']=str(minimum)
    cp=Path(read((source.OUT if arm=='vision' else replay.OUT)/arm/'checkpoint_manifest.json')['path'])
    config=replay.config(cp,{'observation':192})
    config.update(image_max_token_num=maximum,image_min_token_num=minimum,probes_sha256=sha(OUT/'probes.jsonl'))
    folder=OUT/arm/setting;run=replay.Runner(folder,config,gpu)
    if len(run.rr)<2:
        with reserve(source.LEDGER,gpu,str(folder)+':eval',24000) as res:
            res.bind(os.getpid());run.load();run.infer(pp)
            # Encode each actual request through the loaded inference template:
            # audit grids, resized PIL dimensions and image tokens actually used.
            from swift.infer_engine import InferRequest
            grids=[]
            for p in pp:
                request=InferRequest(messages=p['messages'],images=p['images'])
                encoded=run.engine.template.encode(request)
                grid=encoded['image_grid_thw'].tolist()
                grids.append({'id':p['id'],'image_grid_thw':grid,'image_tokens':sum(__import__('math').prod(g)//4 for g in grid)})
            save(folder/'actual_grid_audit.json',grids)
    run.audit(pp,2)
    results=[]
    for p,r in zip(pp,run.rr):
        assert r['id']==p['id']
        parsed=replay.observation_json(r['response'])
        checks=[{'candidate':c,'truth':p['truth'][c]['value'],'observed':parsed.get(c,{}).get('value') if parsed else None}
                for c in 'ABCD']
        for c in checks:c['correct']=c['truth']==c['observed']
        results.append({'id':p['id'],'raw_response':r['response'],'checks':checks})
    save(folder/'report.json',{'status':'complete','correct_candidates':sum(c['correct'] for r in results for c in r['checks']),
         'candidates':8,'scene_correct':sum(all(c['correct'] for c in r['checks']) for r in results),'results':results})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--arm',choices=['vision','text_replay'])
    ap.add_argument('--setting',choices=list(SETTINGS));ap.add_argument('--gpu',default='0');a=ap.parse_args()
    if a.prepare:prepare()
    else:
        with ev.run_lock(OUT/(a.arm+'_'+a.setting+'.lock')):evaluate(a.arm,a.setting,a.gpu)
