#!/usr/bin/env python3
"""Development-only test of image/name binding; never modify source PNGs/data."""
import argparse
import copy
import dataclasses
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path

import pilot_v7_atomic_supervision as pilot
import diagnose_v7_atomic_neutral_recall as neutral
import run_v7_evaluation as ev
from build_v7_experiment_data import name
from pilot_v7_task_then_transfer import card, OP
from run_v7_full_experiment import ROOT, MODELS, rows, read, save, sha, environment
from v7_gpu_resources import reserve

SOURCE = pilot.OUT
OUT = ROOT / 'runs/v7_identity_binding_20261016'
NEW_ARMS = ('name_first_vision', 'name_blind_vision')
CONTROL_ARMS = ('warmup', 'original_text', 'atomic_text', 'atomic_vision')


def verify(out):
    m = read(out / 'manifest.json')
    assert m['script_sha256'] == sha(Path(__file__))
    for path, h in m['sources'].items():
        assert sha(Path(path)) == h, path
    for path, h in m['files'].items():
        assert sha(out / path) == h, path
    for r in rows(pilot.DATA / 'image_hashes.jsonl'):
        assert sha(pilot.DATA / r['path']) == r['source_sha256'] == r['repaired_sha256']


def candidate_content(k):
    """Four distinct, balanced plausible definitions; omit all rule identifiers."""
    r = json.loads(k['fact']['quote'])
    if 'conditions' not in r:
        # Same first/second cutpoint pools across all category rules. Threshold
        # options contain no category names, avoiding recognition through them.
        a, b = r['thresholds']['lower'], r['thresholds']['upper']
        aa = {0.4: 0.3, 0.3: 0.15, 0.15: 0.4}[a]
        bb = {0.65: 0.8, 0.8: 0.85, 0.85: 0.65}[b]
        result = [f'lower = {x}; upper = {y}' for x, y in ((a,b),(aa,b),(a,bb),(aa,bb))]
        kind = 'threshold_pair'
    else:
        changed = copy.deepcopy(r['conditions'])
        c = changed[0]
        c['value'] = {'pentagon':'square', 'square':'pentagon', 'transport crate':'equipment cart', 2:3}[c['value']]
        result = []
        for conds, mode in ((r['conditions'],r['mode']), (changed,r['mode']),
                            (r['conditions'],'all' if r['mode']=='any' else 'any'),
                            (changed,'all' if r['mode']=='any' else 'any')):
            facts = '; '.join(f"{c['field']} {OP[c['operator']]} {c['value']}" for c in conds)
            result.append(f"{facts}. KEEP if {'all listed conditions' if mode=='all' else 'at least one listed condition'} hold; otherwise OMIT.")
        kind = 'conditions_and_combination'
    assert len(set(result)) == 4
    return kind, result


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        verify(out)
        return
    knowledge = {k['id']: k for k in rows(SOURCE / 'target_knowledge.jsonl')}
    source_rows = rows(SOURCE / 'atomic_vision.jsonl')
    for arm in NEW_ARMS:
        training = []
        for r in source_rows:
            z = copy.deepcopy(r); k = knowledge[z['qid']]
            q = z['messages'][1]['content']
            labels = sorted(pilot.aliases(k), key=len, reverse=True)
            found = next(label for label in labels if label in q)
            if arm == 'name_blind_vision':
                q = q.replace(found, 'the rule shown in the image')
                assert not any(n in q for n in json.loads(k['fact']['quote']).get('category_labels', [name(k)]))
            q += ' First state the exact rule name(s), then answer the requested question.'
            z['messages'][1]['content'] = q
            z['messages'][-1]['content'] = 'Rule name(s): ' + name(k) + '.\n' + r['messages'][-1]['content']
            # Unchanged image, row order, fact target, system prompt, exposures.
            assert z['images'] == r['images'] and z['task'] == r['task']
            training.append(z)
        ev.save_rows(out / (arm + '.jsonl'), training)
    for a,b in zip(rows(out/(NEW_ARMS[0]+'.jsonl')), rows(out/(NEW_ARMS[1]+'.jsonl'))):
        assert a['id'] == b['id'] and a['images'] == b['images']
        assert a['messages'][-1] == b['messages'][-1]
    # Train construction finished before copying any evaluation content.
    shutil.copy2(SOURCE / 'probes.jsonl', out / 'probes.jsonl')
    neutral.ARMS = NEW_ARMS
    neutral.prepare(out)
    diagnostic = []
    for k in knowledge.values():
        kind, candidates = candidate_content(k)
        for named in (True, False):
            question = ('Which definition belongs to the stored rule '+name(k)+'?' if kind == 'conditions_and_combination'
                        else 'Which lower/upper threshold pair belongs to the stored rule '+name(k)+'?') if named else (
                        'Which definition belongs to the unidentified stored rule?' if kind == 'conditions_and_combination'
                        else 'Which lower/upper threshold pair belongs to the unidentified stored rule?')
            for shift in range(4):
                options = candidates[shift:] + candidates[:shift]
                gold = 'ABCD'[(-shift)%4]
                q = question + '\n\n' + '\n'.join(f'{letter}. {s}' for letter,s in zip('ABCD',options)) + '\n\nReturn only A, B, C, or D.'
                diagnostic.append({'id':f'recognize:{k["id"]}:{named}:{shift}', 'qid':k['id'],
                    'condition':'recognition_named' if named else 'recognition_anonymous', 'kind':kind, 'shift':shift,
                    'gold':gold, 'fmt':'mcq', 'messages':[{'role':'system','content':ev.SYSTEM},{'role':'user','content':q}]})
        diagnostic.append({'id':'identify_image:'+k['id'], 'qid':k['id'], 'condition':'identify_image', 'fmt':'name',
            'gold':name(k), 'messages':pilot.msg('<image>State only the exact rule name(s) shown in this learning reference.'),
            'images':[str(pilot.DATA/'images'/k['id']/'learning_visual.png')]})
    assert len(diagnostic) == 72
    ev.save_rows(out / 'diagnostic_probes.jsonl', diagnostic)
    sources = [SOURCE/'manifest.json', SOURCE/'atomic_vision.jsonl', SOURCE/'target_knowledge.jsonl',
               SOURCE/'probes.jsonl', pilot.PARENT, ROOT/'scripts/pilot_v7_atomic_supervision.py',
               ROOT/'scripts/diagnose_v7_atomic_neutral_recall.py']
    save(out/'manifest.json', {'created_at':ev.now(), 'script_sha256':sha(Path(__file__)),
        'hypothesis':'Generating image-derived identifiers together with facts may improve name/fact binding when original learning image is absent.',
        'new_arms':NEW_ARMS, 'controls':CONTROL_ARMS, 'rules':list(knowledge), 'rows':512, 'epochs':1,'steps':128,
        'parent':read(pilot.PARENT), 'generation':'same original MCQs + unchanged neutral self-recall protocol; temp0, no thinking',
        'contrast':'Both new arms add identical name prefixes to fact targets; only name_blind removes identifiers from user prompt. Original images still show their names. No no-image rule/fact training.',
        'not_proof':['Same eight development rules already examined; no held-out confirmation.',
            'name_first vs prior atomic_vision changes both instruction and output prefix; new-arm contrast isolates user identifier availability.',
            'Output name generation can read image labels; it does not by itself prove stored recall.',
            'Recognition presents candidate facts, is a diagnostic ONLY, not a formal transfer score. Anonymous control cannot identify a unique gold rule; measures option plausibility/position bias.',
            'Four rotations are correlated; report mean per eight rules, not 32 independent facts.',
            'Original v7, all PNGs, and all prior runs remain immutable. No PT or full three-model rerun.'],
        'success_criterion':'Need improved no-image complete recall AND original no-material V2T/V2V; do not call image-conditioned or recognition gain transfer.',
        'sources':{str(p):sha(p) for p in sources},
        'files':{str(p.relative_to(out)):sha(p) for p in [out/(a+'.jsonl') for a in NEW_ARMS]+[out/'probes.jsonl',out/'diagnostic_probes.jsonl',out/'neutral_recall/design.json',out/'neutral_recall/recall_probes.jsonl']}})
    snap=out/'source_snapshots';snap.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__),snap/Path(__file__).name)
    verify(out)


def adapter(out, arm):
    if arm == 'warmup':
        return Path(read(pilot.PARENT)['path'])
    return Path(read((out if arm in NEW_ARMS else SOURCE)/arm/'checkpoint_manifest.json')['path'])


def diagnostic(out, arm, gpu):
    folder=out/'diagnostics'/arm;folder.mkdir(parents=True,exist_ok=True)
    cp=adapter(out,arm)
    config={'adapter':ev.checkpoint(cp),'probes_sha256':sha(out/'diagnostic_probes.jsonl'),
            'temperature':0,'seed':42,'batch':4,'max_tokens':{'mcq':16,'name':96},'thinking':False}
    if (folder/'manifest.json').exists():assert read(folder/'manifest.json')==config
    else:save(folder/'manifest.json',config)
    raw=folder/'predictions.jsonl'
    records,notes=ev.read_rows(raw,allow_partial_tail=True) if raw.exists() else ([],[])
    if notes:ev.save_rows(raw,records)
    seen={r['id'] for r in records};assert len(seen)==len(records)
    probes=rows(out/'diagnostic_probes.jsonl');pending=[p for p in probes if p['id'] not in seen]
    if pending:
        with reserve(pilot.LEDGER,gpu,str(folder),23000) as reservation:
            reservation.bind(os.getpid());os.environ.update(environment(gpu,pilot.DATA))
            import torch
            from swift.infer_engine import TransformersEngine,InferRequest,RequestConfig
            from swift.utils import seed_everything
            seed_everything(42)
            engine=TransformersEngine(str(MODELS['qwen3']),adapters=[str(cp)],torch_dtype=torch.bfloat16,
                                      device_map='cuda:0',max_batch_size=4,attn_impl='sdpa')
            engine.template.enable_thinking=False
            for fmt in ('mcq','name'):
                group=[p for p in pending if p['fmt']==fmt]
                for start in range(0,len(group),4):
                    batch=group[start:start+4]
                    outputs=engine.infer([InferRequest(messages=copy.deepcopy(p['messages']),images=p.get('images',[])) for p in batch],
                        RequestConfig(max_tokens=config['max_tokens'][fmt],temperature=0,seed=42,return_details=True),use_tqdm=False)
                    assert len(batch)==len(outputs)
                    for p,reply in zip(batch,outputs):
                        if isinstance(reply,Exception):raise reply
                        c=reply.choices[0];text=c.message.content or '';pred=ev.letter(text) if fmt=='mcq' else None
                        r={k:p[k] for k in ('id','qid','condition','fmt','gold')}
                        r.update(response=text,predicted=pred,correct=pred==p['gold'] if fmt=='mcq' else None,
                                 format_error=fmt=='mcq' and pred is None,finish_reason=c.finish_reason,
                                 input_sha256=ev.digest({'messages':p['messages'],'images':p.get('images',[])}),raw_response=dataclasses.asdict(reply))
                        records.append(r);seen.add(r['id'])
                        with raw.open('a') as f:
                            f.write(json.dumps(r,ensure_ascii=False,default=str)+'\n');f.flush();os.fsync(f.fileno())
                    print(arm,'diagnostic',len(records),'/72',flush=True)
    assert len(records)==len(seen)==72
    lookup={p['id']:p for p in probes}
    for r in records:
        p=lookup[r['id']]
        assert r['input_sha256']==ev.digest({'messages':p['messages'],'images':p.get('images',[])})
        assert r['response']==(r['raw_response']['choices'][0]['message']['content'] or '')
    metrics={}
    for condition in ('recognition_named','recognition_anonymous'):
        rr=[r for r in records if r['condition']==condition]
        metrics[condition]={'correct':sum(r['correct'] for r in rr),'n':len(rr),
                            'format_errors':sum(r['format_error'] for r in rr),
                            'by_rule':{qid:sum(r['correct'] for r in rr if r['qid']==qid)/4 for qid in sorted({r['qid'] for r in rr})},
                            'letters':dict(Counter(r['predicted'] for r in rr)),
                            'truncated':sum(r['finish_reason']=='length' for r in rr)}
    assert ev.checkpoint(cp)==config['adapter']
    save(folder/'report.json',{'status':'complete','metrics':metrics,'diagnostic_only':True})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=OUT);ap.add_argument('--gpu',default='0')
    ap.add_argument('--prepare',action='store_true');ap.add_argument('--arm',choices=NEW_ARMS+CONTROL_ARMS)
    ap.add_argument('--stage',choices=['train','eval','neutral','diagnostic'],default='train')
    a=ap.parse_args();out=a.out.resolve()
    if a.prepare:prepare(out)
    else:
        verify(out)
        with ev.run_lock(out/(a.arm+'_'+a.stage+'.lock')):
            if a.stage=='train':
                assert a.arm in NEW_ARMS
                pilot.train(out,a.arm,a.gpu)
            elif a.stage=='eval':pilot.evaluate(out,a.arm,a.gpu,adapter(out,a.arm))
            elif a.stage=='neutral':neutral.run(out,a.arm,a.gpu)
            else:diagnostic(out,a.arm,a.gpu)
