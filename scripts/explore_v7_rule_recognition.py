#!/usr/bin/env python3
"""Matched-step recognition supervision; immutable original learning PNGs."""
import argparse
import copy
import itertools
import math
import os
import random
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import pilot_v7_counterfactual_names as source
import explore_v7_visual_replay as replay
import run_v7_evaluation as ev
from build_v7_experiment_data import name
from pilot_v7_atomic_supervision import msg, SYSTEM, PARENT
from run_v7_full_experiment import ROOT, rows, read, save, sha, environment, train_command, final_checkpoint
from summarize_v7_counterfactual_names import metric
from v7_gpu_resources import reserve

OUT = ROOT / 'runs/v7_rule_recognition_20261016'
ARMS = ('text_match', 'vision_match')


def definition(k, paraphrase=False):
    r = __import__('json').loads(k['fact']['quote'])
    a, b = r['conditions']
    if a['field'] == 'value':
        return (f"Keep a candidate exactly when its value is not below {a['value']} and is below {b['value']}; omit otherwise."
                if paraphrase else f"KEEP if {a['value']} <= value < {b['value']}; otherwise OMIT.")
    return (f"A candidate is kept if it has color {b['value']}, or has shape {a['value']}, or both; otherwise it is omitted."
            if paraphrase else f"KEEP if shape = {a['value']} OR color = {b['value']}; otherwise OMIT.")


def match_probe(k, group, order, form, has_image, diagnostic=False):
    forms = (
        'Which description is the complete rule named {}?',
        'Select the exact stored definition of {}.',
        'For {}, which of these four rules is correct?',
        'Identify the decision rule associated with {}.',
    )
    q = (('Among the following alternatives, identify the learned criterion for {}.' if diagnostic else forms[form]).format(name(k)))
    if has_image:
        q = '<image>Using this original learning reference, ' + q[0].lower() + q[1:]
    opts = [group[i] for i in order]
    q += '\n\nOptions:\n' + '\n'.join(f'{l}. {definition(v, diagnostic)}' for l, v in zip('ABCD', opts))
    q += '\nRespond with only the option letter A, B, C, or D.'
    p = {'qid': k['id'], 'fmt': 'mcq', 'gold': 'ABCD'[[x['id'] for x in opts].index(k['id'])],
         'messages': msg(q), 'option_rule_ids': [x['id'] for x in opts]}
    if has_image:
        p['images'] = [str(source.DATA / 'images' / k['id'] / 'learning_visual.png')]
    return p


def validate(out):
    m = read(out / 'manifest.json')
    assert m['script_sha256'] == sha(Path(__file__))
    for path, h in m['sources'].items():
        assert sha(Path(path)) == h, path
    for path, h in m['files'].items():
        assert sha(out / path) == h, path
    for r in rows(source.DATA / 'image_hashes.jsonl'):
        assert sha(source.DATA / r['path']) == r['source_sha256'] == r['repaired_sha256']
    assert ev.checkpoint(Path(read(PARENT)['path'])) == m['parent']


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        validate(out)
        return
    source.validate(source.OUT)
    kk = {k['id']: k for k in rows(source.OUT / 'target_knowledge.jsonl')}
    # Only rule membership is read; no test observations, options, or labels build training.
    groups = [[kk[q] for q in g['rule_ids']] for g in read(source.OUT / 'selection_candidates.json')]
    permutations = list(itertools.permutations(range(4)))
    orders = [permutations[i] for i in (0, 3, 8, 11, 12, 15, 20, 23)]
    # Every ordering has its four circular rotations; take two bases ->8 orders.
    orders = [tuple((j + s) % 4 for j in base) for base in ((0,1,2,3),(0,2,1,3)) for s in range(4)]
    for arm in ARMS:
        tr = []
        for group in groups:
            for k in group:
                for i in range(64):
                    p = match_probe(k, group, orders[i % 8], (i // 8) % 4, arm == 'vision_match')
                    z = {key: copy.deepcopy(p[key]) for key in ('qid', 'messages', 'images', 'option_rule_ids') if key in p}
                    z.update(id=f'{k["id"]}:{i}', channel='rule_definition_recognition', gold=p['gold'])
                    z['messages'].append({'role': 'assistant', 'content': p['gold']})
                    tr.append(z)
        random.Random(42).shuffle(tr)
        assert len(tr) == 512
        for qid in kk:
            assert Counter(r['gold'] for r in tr if r['qid'] == qid) == Counter({l:16 for l in 'ABCD'})
        for r in tr:
            assert not any('test_visual' in image for image in r.get('images', []))
            assert all(name(k) not in r['messages'][-2]['content'].split('Options:\n')[1] for k in kk.values())
        ev.save_rows(out / (arm + '.jsonl'), tr)
    assert [r['gold'] for r in rows(out/'text_match.jsonl')] == [r['gold'] for r in rows(out/'vision_match.jsonl')]
    # Fixed diagnostic: two surface forms and both image conditions. The choices
    # contain rules, so these scores must NEVER be presented as material-free transfer.
    probes = []
    for condition in ('recognition_exact_noimage', 'recognition_exact_image', 'recognition_para_noimage', 'recognition_para_image'):
        for group in groups:
            for k in group:
                for shift in range(4):
                    p = match_probe(k, group, tuple((j+shift)%4 for j in range(4)), 0,
                                    condition.endswith('_image'), '_para_' in condition)
                    p.update(id=f'{condition}:{k["id"]}:{shift}', condition=condition, shift=shift)
                    probes.append(p)
    for k in kk.values():
        p = {'id':'plain_recall:'+k['id'], 'qid':k['id'], 'condition':'plain_recall', 'fmt':'card',
             'messages':[{'role':'user','content':'Explain the complete rule named '+name(k)+'.'}]}
        probes.append(p)
    assert len(probes) == 136
    ev.save_rows(out / 'diagnostic_probes.jsonl', probes)
    sources = [source.OUT / p for p in ('target_knowledge.jsonl','selection_candidates.json','probes.jsonl',
               'text.jsonl','vision.jsonl','text/checkpoint_manifest.json','vision/checkpoint_manifest.json',
               'warmup/eval_manifest.json','text/predictions.jsonl','vision/predictions.jsonl')]
    sources += [PARENT, source.DATA/'image_hashes.jsonl', ROOT/'scripts/explore_v7_visual_replay.py',
                ROOT/'scripts/pilot_v7_counterfactual_names.py', ROOT/'scripts/run_v7_full_experiment.py',
                ROOT/'scripts/run_v7_evaluation.py', ROOT/'scripts/v7_gpu_resources.py']
    save(out / 'manifest.json', {
        'script_sha256':sha(Path(__file__)), 'created_at':ev.now(), 'arms':ARMS,
        'parent':ev.checkpoint(Path(read(PARENT)['path'])), 'rows_per_arm':512,'steps':128,'epochs':1,
        'seed':42,'batch':4,'accumulation':1,'lr':1e-4,'lora_rank':8,'lora_alpha':32,'lora_dropout':0.05,
        'freeze_vision_aligner':True,'max_length':8192,'system':SYSTEM,
        'training':'Supervised four-definition matching. 64rows/rule,16targets/letter/rule. Definitions from the same4rules reused in every group; no rule name appears inside options. TEXT and VISION use identical option orders/labels. VISION always retains its own ORIGINAL learning image.',
        'baseline':'Existing frozen full-explanation TEXT/VISION checkpoints, same warmup parent, same8rules,64rows/rule,batch/steps/LR/seed/system. Reuse without retraining.',
        'primary':'Unchanged source named_T/named_V probes; no definitions/reference in main inputs. Self-recall and oracle separate.',
        'diagnostic':'Rule recognition presents four possible definitions and sometimes original learning image; NOT no-material application. Exact form is training resubstitution; paraphrase is surface generalization of the same8facts.',
        'limitations':['8development rules/5original test scenes,one seed; no independent confirmatory sample.',
            'Match rows/optimizer steps, NOT supervised tokens (letter vs full definition) or aggregate factual exposure (all4options visible).',
            'Four alternatives each contain candidate knowledge. This is supervised recognition, not image-only knowledge acquisition with no textual rule values.',
            'A correct image-conditioned recognition can be solved from the image without memorizing; no-image recall and held-constant application are necessary gates.',
            'Rule grouping was selected in previous development based on image discriminability; no claim of representativeness.',
            'No PT training, native CoT, original PNG edits or final-dataset rewrite.'],
        'sources':{str(p):sha(p) for p in sources},'files':{p.name:sha(p) for p in out.glob('*.jsonl')}})
    snap = out/'source_snapshots';snap.mkdir(exist_ok=True)
    for p in [Path(__file__)] + [p for p in sources if p.suffix=='.py']:
        shutil.copy2(p,snap/p.name)
    validate(out)
    print('Prepared two matched512row sets and136diagnostic probes.',flush=True)


def train(out, arm, gpu):
    folder=out/arm;folder.mkdir(exist_ok=True);cpdir=folder/'checkpoint'
    cp=final_checkpoint(cpdir,1)
    if cp is None:
        recover=[p for p in cpdir.glob('checkpoint-*') if all((p/f).exists() for f in
            ('optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','adapter_model.safetensors'))]
        resume=max(recover,key=lambda p:read(p/'trainer_state.json')['global_step']) if recover else None
        if cpdir.exists() and not resume:
            cpdir.rename(folder/('interrupted_'+str(int(__import__('time').time()))))
        cmd=train_command('qwen3','sft',out/(arm+'.jsonl'),cpdir,1,resume=resume)
        for flag,value in (('--per_device_train_batch_size','4'),('--gradient_accumulation_steps','1'),
                           ('--save_total_limit','3'),('--save_strategy','steps')):
            cmd[cmd.index(flag)+1]=value
        cmd+=['--save_steps','32']
        if not resume:cmd+=['--adapters',read(PARENT)['path']]
        save(folder/'train_state.json',{'status':'queued','command':cmd})
        with reserve(source.LEDGER,gpu,str(folder)+':train',29000 if arm=='vision_match' else 22000) as res:
            with (folder/'train.log').open('a') as log:
                process=subprocess.Popen(cmd,cwd=ROOT,env=environment(gpu,source.DATA),stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                res.bind(process.pid);save(folder/'train_state.json',{'status':'running','pid':process.pid,'command':cmd})
                code=process.wait()
            assert code==0,(arm,code)
        cp=final_checkpoint(cpdir,1)
    assert cp
    st=read(cp/'trainer_state.json');args=read(cp/'args.json')
    assert st['global_step']==128 and abs(st['epoch']-1)<1e-6
    assert args['dataset']==[str(out/(arm+'.jsonl'))] and args['adapters']==[read(PARENT)['path']]
    losses=[r['loss'] for r in rows(cpdir/'logging.jsonl') if 'loss' in r]
    assert losses and all(math.isfinite(v) for v in losses)
    from safetensors import safe_open
    with safe_open(str(cp/'adapter_model.safetensors'),framework='pt',device='cpu') as f:
        keys=list(f.keys());bs=[k for k in keys if 'lora_B' in k]
        assert len(bs)==252 and all(bool(f.get_tensor(k).count_nonzero()) for k in bs)
        assert not any('.visual.' in k or '.merger.' in k for k in keys)
    save(folder/'checkpoint_manifest.json',{**ev.checkpoint(cp),'parent':ev.checkpoint(Path(read(PARENT)['path'])),
        'epoch':1,'global_step':128,'last_logged_loss':losses[-1],'nonzero_lora_B':252,
        'selection':'fixed final epoch1; no score-based checkpoint selection'})
    save(folder/'train_state.json',{'status':'complete','checkpoint':str(cp)})


def evaluate(out, arm, gpu):
    is_new=arm in ARMS
    folder=out/arm
    cp=Path(read(folder/'checkpoint_manifest.json')['path']) if is_new else (
        Path(read(PARENT)['path']) if arm=='warmup' else Path(read(source.OUT/arm/'checkpoint_manifest.json')['path']))
    run=replay.Runner(folder,replay.config(cp,{'card':384,'mcq':16}),gpu)
    diag=rows(out/'diagnostic_probes.jsonl')
    primary=rows(source.OUT/'probes.jsonl') if is_new else []

    def own_probes():
        if not is_new:return []
        recall={r['qid']:r for r in run.rr if r['condition']=='recall'}
        assert len(recall)==8
        pp=[]
        for p in primary:
            if not p['condition'].startswith('named_'):continue
            z=copy.deepcopy(p);z.update(id='self:'+p['id'],condition='self_'+p['modality'])
            text=recall[p['qid']]['response']
            z['messages'][-1]['content']='Your own prior rule recall (uncorrected):\n'+text+'\n\n'+z['messages'][-1]['content']
            z['recall_response_sha256']=ev.digest(text);pp.append(z)
        return pp

    expected=952 if is_new else 136
    if len(run.rr)<expected:
        with reserve(source.LEDGER,gpu,str(folder)+':eval',24000) as resource:
            resource.bind(os.getpid());run.load()
            for fmt in ('card','mcq'):
                run.infer([p for p in primary+diag if p['fmt']==fmt])
            own=own_probes()
            if own:ev.save_rows(folder/'self_probes.jsonl',own)
            run.infer(own)
    all_probes=primary+diag+own_probes()
    run.audit(all_probes,expected)
    metrics={}
    for condition in sorted({r['condition'] for r in run.rr if r['fmt']=='mcq' and not r['condition'].startswith('anonymous_')}):
        rr=[r for r in run.rr if r['condition']==condition]
        metrics[condition]=metric(rr)
        if condition.startswith(('named_','self_','oracle_')):
            metrics[condition]['by_task']={t:metric([r for r in rr if r['task']==t]) for t in ('direct','diagnosis')}
            metrics[condition]['by_group']={str(g):metric([r for r in rr if r['group']==g]) for g in (0,1)}
    save(folder/'report.json',{'status':'complete','metrics':metrics,
        'recall_outputs':[r for r in run.rr if r['fmt']=='card']})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=OUT)
    ap.add_argument('--stage',choices=('prepare','train','eval','run'),required=True)
    ap.add_argument('--arm',choices=(*ARMS,'warmup','text','vision'));ap.add_argument('--gpu',default='0')
    a=ap.parse_args();out=a.out.resolve()
    if a.stage=='prepare':prepare(out)
    else:
        validate(out)
        with ev.run_lock(out/(a.arm+'_'+a.stage+'.lock')):
            if a.stage in ('train','run'):train(out,a.arm,a.gpu)
            if a.stage in ('eval','run'):evaluate(out,a.arm,a.gpu)
