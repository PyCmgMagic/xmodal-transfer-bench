#!/usr/bin/env python3
"""Matched clean-base TEXT PT control: same factual materials, no QA template."""
import argparse
import math
import os
import subprocess
from pathlib import Path

import explore_v7_memory_router as mem
import run_v7_evaluation as ev
from run_v7_full_experiment import ROOT, rows, read, save, sha, train_command, environment, final_checkpoint
from v7_gpu_resources import reserve

ARM = 'text_pt_memory'
OUT = mem.OUT / ARM


def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT / 'manifest.json').exists():
        validate(); return
    mem.validate()
    training = []
    # Same row order and rule exposures as text_memory. All content is original
    # learning material; no generated test prompt or answer is involved.
    for r in rows(mem.OUT / 'text_memory.jsonl'):
        training.append({'id': r['id'], 'qid': r['qid'],
                         'messages': [{'role': 'assistant', 'content': r['messages'][-1]['content']}]})
    ev.save_rows(OUT / 'train.jsonl', training)
    save(OUT / 'manifest.json', {'created_at': ev.now(), 'script_sha256': sha(Path(__file__)),
        'source_manifest': sha(mem.OUT / 'manifest.json'), 'training_sha256': sha(OUT / 'train.jsonl'),
        'objective': 'PT raw factual learning materials; use_chat_template=false, loss_scale=all',
        'rows':576, 'rules':18, 'presentations_each':32, 'epochs':1, 'fixed_final_step':144,
        'rank':32, 'alpha':64, 'lr':1e-4, 'batch':4, 'accumulation':1, 'seed':42,
        'initialization':'original base, no inherited adapter', 'comparison':'text_memory uses same materials/order/exposures with SFT prompts; token context and objective differ.',
        'limits':'Additional development arm frozen before its scored generation. Same18rules; no independent confirmation.'})


def validate():
    mem.validate()
    m = read(OUT / 'manifest.json')
    assert m['script_sha256'] == sha(Path(__file__))
    assert m['source_manifest'] == sha(mem.OUT / 'manifest.json')
    assert m['training_sha256'] == sha(OUT / 'train.jsonl')


def load_chat(runner, cp):
    runner.load(cp=cp, adapter=True)
    # Swift inherits the PT generation template from args.json. Evaluation is
    # instruction following and must use the same chat template as other arms.
    from swift.template import get_template
    runner.engine.template = get_template(runner.engine.processor, template_type='qwen3_vl', use_chat_template=True)
    runner.engine.template.enable_thinking = False
    assert runner.engine.template.use_chat_template
    save(runner.folder/'inference_template_audit.json', {'training_use_chat_template':False,
        'inference_use_chat_template':True,'thinking':False,'template':'qwen3_vl'})


def train(gpu, smoke=False):
    validate()
    folder = OUT / 'smoke' if smoke else OUT
    folder.mkdir(exist_ok=True)
    cpdir = folder / 'checkpoint'
    with ev.run_lock(folder / 'train.lock'):
        cp = cpdir / 'checkpoint-1' if smoke else final_checkpoint(cpdir, 1)
        if cp is None or not (cp / 'adapter_model.safetensors').exists():
            recover = [p for p in cpdir.glob('checkpoint-*') if all((p / f).exists() for f in ('optimizer.pt','scheduler.pt','rng_state.pth','adapter_model.safetensors','trainer_state.json'))]
            resume = max(recover, key=lambda p: read(p/'trainer_state.json')['global_step']) if recover and not smoke else None
            dataset = OUT / 'train.jsonl'
            if smoke:
                dataset = folder / 'train.jsonl'; ev.save_rows(dataset, rows(OUT / 'train.jsonl')[:4])
            cmd = train_command('qwen3', 'pt', dataset, cpdir, 1, resume=resume)
            for flag, value in [('--per_device_train_batch_size','4'),('--gradient_accumulation_steps','1'),
                                ('--lora_rank','32'),('--lora_alpha','64'),('--save_strategy','steps'),
                                ('--save_total_limit','3'),('--logging_steps','1')]:
                cmd[cmd.index(flag)+1] = value
            cmd += ['--save_steps','1' if smoke else '48','--attn_impl','sdpa']
            if smoke:
                cmd += ['--max_steps','1']; cmd[cmd.index('--warmup_ratio')+1] = '0'
            env = environment(gpu, mem.DATA); env.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
            save(folder/'train_state.json', {'status':'waiting_for_gpu','command':cmd})
            with reserve(mem.LEDGER, gpu, str(folder)+':train', 15000) as res:
                with (folder/'train.log').open('a') as log:
                    proc = subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    res.bind(proc.pid);save(folder/'train_state.json',{'status':'running','pid':proc.pid,'command':cmd})
                    code=proc.wait()
                assert code==0,(code,str(folder/'train.log'))
            cp=cpdir/('checkpoint-1' if smoke else 'checkpoint-144')
        args=read(cp/'args.json'); state=read(cp/'trainer_state.json')
        assert state['global_step']==(1 if smoke else 144)
        assert args['use_chat_template'] is False and args['loss_scale']=='all'
        assert not args['adapters'] and args['freeze_vit'] and args['freeze_aligner']
        losses=[r['loss'] for r in rows(cpdir/'logging.jsonl') if 'loss' in r]
        assert losses and all(math.isfinite(v) for v in losses)
        from safetensors import safe_open
        with safe_open(str(cp/'adapter_model.safetensors'),framework='pt',device='cpu') as f:
            keys=list(f.keys());bs=[k for k in keys if 'lora_B' in k]
            assert bs and all(bool(f.get_tensor(k).count_nonzero()) for k in bs)
            assert not any('.visual.' in k or '.merger.' in k for k in keys)
        save(folder/'checkpoint_manifest.json',{**ev.checkpoint(cp),'global_step':state['global_step'],'epoch':state['epoch'],
             'losses_finite':True,'nonzero_lora_B':len(bs),'final_loss':losses[-1],'selection':'fixed final step; no score selection'})
        if smoke:
            runner=mem.Runner(folder/'reload',gpu,'base')
            probes=[dict(next(p for p in rows(mem.OUT/'test_probes.jsonl') if p['input_mod']==mod),id='reload:'+mod) for mod in ('T','V')]
            if len(runner.rr)<2:
                with reserve(mem.LEDGER,gpu,str(folder)+':reload',22000) as res:
                    res.bind(os.getpid());load_chat(runner,cp);runner.infer(probes,'direct')
            assert len(runner.rr)==2 and all(r['response'].strip() for r in runner.rr)
            save(folder/'validated.json',{'status':'pass','text_and_image_reload':True})
        save(folder/'train_state.json',{'status':'complete','checkpoint':str(cp)})


def memory(gpu):
    validate()
    runner=mem.Runner(OUT/'memory',gpu,'base')
    with ev.run_lock(runner.folder/'run.lock'):
        if len(runner.rr)<108:
            with reserve(mem.LEDGER,gpu,str(runner.folder)+':eval',24000) as res:
                res.bind(os.getpid());load_chat(runner,OUT/'checkpoint/checkpoint-144')
                pp=rows(mem.OUT/'recall_probes.jsonl')
                runner.infer([p for p in pp if p['stage']=='blind_recall'],'blind_recall')
                runner.infer([p for p in pp if p['stage']=='image_recall'],'image_recall')
                runner.infer([dict(p,id='direct:'+p['id']) for p in rows(mem.OUT/'test_probes.jsonl')],'direct')
        assert len(runner.rr)==108
        save(runner.folder/'state.json',{'status':'complete','saved':108})


def audit():
    validate()
    total=0
    for folder in (OUT/'memory',OUT/'routed_base'):
        runner=mem.Runner(folder,'0','base')
        if folder.name=='memory':
            pp=rows(mem.OUT/'recall_probes.jsonl')+[dict(p,id='direct:'+p['id']) for p in rows(mem.OUT/'test_probes.jsonl')]
        else:
            pp=mem.solve_probes(ARM)+mem.choices(runner)
        lookup={p['id']:p for p in pp}
        assert len(runner.rr)==len(lookup) and {r['id'] for r in runner.rr}==set(lookup)
        for r in runner.rr:
            p=lookup[r['id']]
            assert r['input_sha256']==ev.digest({'messages':p['messages'],'images':p.get('images',[])})
            assert r['response']==r['raw_response']['choices'][0]['message']['content']
            if r['stage'] in ('direct','choice'):
                assert r['predicted']==ev.letter(r['response']) and r['correct']==(r['predicted']==p['gold'])
        total+=len(runner.rr)
    save(OUT/'completion_audit.json',{'status':'pass','generations':total,'smoke_generations':2})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('prepare','run','audit'),required=True);ap.add_argument('--gpu',default='1');a=ap.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='audit':audit()
    else:
        train(a.gpu,smoke=True);train(a.gpu);memory(a.gpu);mem.solve([ARM],a.gpu);audit()
