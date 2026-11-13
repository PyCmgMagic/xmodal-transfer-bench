#!/usr/bin/env python3
"""Three matched-step one-epoch TEXT pilots; original v7 data stays read-only."""
import argparse
import copy
import dataclasses
import hashlib
import itertools
import json
import math
import operator
import os
import random
import re
import subprocess
from collections import Counter
from fractions import Fraction
from pathlib import Path

import run_v7_evaluation as ev
from build_v7_experiment_data import name
from run_v7_full_experiment import ROOT, MODELS, environment, final_checkpoint, read, rows, save, sha, train_command
from v7_gpu_resources import reserve

DATA = ROOT / 'rules_repaired_single_image_v7'
WORK = ROOT / 'runs/v7_full_experiment'
ARMS = ('cards', 'facts', 'facts_cases')
OPS = {'eq': operator.eq, 'ne': operator.ne, 'ge': operator.ge, 'gt': operator.gt, 'le': operator.le, 'lt': operator.lt}
OPTEXT = {'eq': '=', 'ne': '!=', 'ge': '>=', 'gt': '>', 'le': '<=', 'lt': '<'}


def message(question, answer=None, fmt='semantic'):
    system = ev.SYSTEM if fmt == 'mcq' else 'Answer with only the requested fact or result. Do not give explanations.'
    result = [{'role': 'system', 'content': system}, {'role': 'user', 'content': question}]
    if answer is not None:
        result.append({'role': 'assistant', 'content': answer})
    return result


def condition(c):
    return f"{c['field']} {OPTEXT[c['operator']]} {c['value']}"


def atoms(rule):
    if 'conditions' in rule:
        cs = rule['conditions']
        return [('first listed requirement', condition(cs[0])), ('second listed requirement', condition(cs[1])),
                ('additional requirement after the first two (or none)', condition(cs[2]) if len(cs) == 3 else 'none'),
                ('condition combination mode (all or any)', rule['mode'])]
    ratio = rule['ratio']
    detail = ', '.join(rule.get('joint_attributes', [])) or rule.get('target_object', '')
    expr = ratio['numerator_field'] + '/' + ratio['denominator_field']
    if detail:
        expr += ' (' + detail + ')'
    return [('lower proportion threshold', str(rule['thresholds']['lower'])),
            ('upper proportion threshold', str(rule['thresholds']['upper'])),
            ('category labels in below/between/above order', ', '.join(rule['category_labels'])),
            ('numerator/denominator definition', expr)]


def solve(rule, observed):
    if 'conditions' in rule:
        flags = [OPS[c['operator']](observed[c['field']], c['value']) for c in rule['conditions']]
        return 'KEEP' if (all(flags) if rule['mode'] == 'all' else any(flags)) else 'OMIT'
    n, d = observed
    ratio = Fraction(n, d)
    lo, hi = [Fraction(str(rule['thresholds'][x])) for x in ('lower', 'upper')]
    return rule['category_labels'][0 if ratio < lo else 2 if ratio > hi else 1]


def independent(rule, observed):
    if 'conditions' in rule:
        failures = 0
        for c in rule['conditions']:
            x, y, op = observed[c['field']], c['value'], c['operator']
            success = (x == y if op == 'eq' else x != y if op == 'ne' else x >= y if op == 'ge'
                       else x > y if op == 'gt' else x <= y if op == 'le' else x < y)
            failures += not success
        keep = failures == 0 if rule['mode'] == 'all' else failures < len(rule['conditions'])
        return 'KEEP' if keep else 'OMIT'
    n, d = observed
    lo, hi = [Fraction(str(rule['thresholds'][x])) for x in ('lower', 'upper')]
    pos = 0 if n*lo.denominator < d*lo.numerator else 2 if n*hi.denominator > d*hi.numerator else 1
    return rule['category_labels'][pos]


def pool(rule, split):
    if 'conditions' not in rule:
        return [(n, d) for d in ((20, 40) if split == 'train' else (60, 80)) for n in range(d+1)]
    cs = rule['conditions']
    if cs[0]['field'] == 'value':
        lo, hi = [c['value'] for c in cs]
        offsets = (-2, -.5, 0, .5, 2) if split == 'train' else (-1.5, -.25, 0, .25, 1.5)
        values = sorted({float(x + offset) for x in (lo, hi) for offset in offsets} | {(lo+hi)/2})
        return [{'value': x} for x in values]
    negatives = {'train': {'category': ['folder', 'storage box'], 'material': ['glass', 'wood'], 'color': ['blue', 'green']},
                 'valid': {'category': ['basket', 'toolbox'], 'material': ['paper', 'ceramic'], 'color': ['purple', 'yellow']}}
    fields = [c['field'] for c in cs]
    values = [[c['value']] + [v for v in negatives[split][c['field']] if v != c['value']] for c in cs]
    return [dict(zip(fields, vals)) for vals in itertools.product(*values)]


def description(rule, value):
    if isinstance(value, dict):
        return ', '.join(f'{k}: {v}' for k, v in value.items())
    return f"{rule['ratio']['numerator_field']}: {value[0]}, {rule['ratio']['denominator_field']}: {value[1]}"


def assignment(values):
    return '; '.join(f'{l}={v}' for l, v in zip('ABCD', values))


def cases(k, split, excluded=None):
    rule, identifier = json.loads(k['fact']['quote']), name(k)
    rng = random.Random('retrieval-cases:42:' + split + ':' + k['id'])
    observations = pool(rule, split)
    for o in observations:
        assert solve(rule, o) == independent(rule, o)
    labels = ['KEEP', 'OMIT'] if 'conditions' in rule else rule['category_labels']
    all_results = [assignment(v) for v in itertools.product(labels, repeat=4)]
    output = []
    seen = set()
    accepted_indices = set(rng.sample(list(range(8, 16)), 2))
    # 16 independent combinations per rule. A semantic and an MCQ version of
    # each training case form an explicit format bridge (not independent facts).
    for i in range(16):
        while True:
            observed = [rng.choice(observations) for _ in range(4)]
            sig = ev.digest(observed)
            if sig not in seen and sig not in (excluded or set()):
                seen.add(sig)
                break
        results = [solve(rule, o) for o in observed]
        target = assignment(results)
        diagnosis = i >= 8
        submitted = None
        accepted = diagnosis and i in accepted_indices
        question = (f'Apply {identifier}. Report every candidate result.' if split == 'train' else
                    f'Using {identifier}, determine the complete result for this new collection.')
        question += '\n' + '\n'.join(f'Candidate {l}: {description(rule,o)}.' for l,o in zip('ABCD',observed))
        if diagnosis:
            submitted = target if accepted else rng.choice([r for r in all_results if r != target])
            question += '\nSubmitted report: ' + submitted + '. Verify this report and return the correct complete result.'
        wrong = [v for v in all_results if v != target]
        rng.shuffle(wrong)
        options = wrong[:3]
        if diagnosis and not accepted and submitted not in options:
            options[-1] = submitted
        rng.shuffle(options)
        options.insert(i % 4, target)
        assert len(set(options)) == 4
        if diagnosis:
            assert submitted in options
        body = question + '\n\nOptions:\n' + '\n'.join(f'{l}. Result: {v}.' for l,v in zip('ABCD',options)) + ev.SUFFIX
        detail = {'observations': observed, 'results': results, 'options': options,
                  'submitted': submitted, 'accepted': accepted if diagnosis else None,
                  'signature': ev.digest(observed)}
        task = 'diagnosis' if diagnosis else 'direct'
        output.append({'task':task,'question':body,'answer':'ABCD'[i%4],'fmt':'mcq','detail':detail})
        if split == 'train':
            output.append({'task':task+'_semantic','question':question + '\nReturn A=...; B=...; C=...; D=... only.',
                           'answer':target,'fmt':'semantic','detail':detail})
    return output


FACT_FORMS = (
    'For {name}, state the {focus}.', 'What is the {focus} of {name}?',
    'Recall the {focus} used by {name}.', 'Under {name}, give the {focus}.',
)
ENDINGS = (' Return only the requested value.', ' Answer directly.', ' Be exact.', ' No explanation.')


def prepare(out):
    if (out/'manifest.json').exists():
        m = read(out/'manifest.json')
        assert m['script_sha256'] == sha(Path(__file__))
        for f,h in m['files'].items():
            assert sha(out/f) == h
        return
    out.mkdir(parents=True,exist_ok=True)
    knowledge = [r['content'] for r in rows(DATA/'audit/rule_knowledge.jsonl')]
    selected = []
    for relation in ('set_selection_rule','three_way_category'):
        selected.extend(sorted([k for k in knowledge if k['fact']['relation']==relation],
            key=lambda k:hashlib.sha256(('retrieval-repair-20261015:42:'+k['id']).encode()).hexdigest())[:4])
    training = {arm:[] for arm in ARMS}
    checks, heldout = [], []
    for k in selected:
        rule, identifier = json.loads(k['fact']['quote']), name(k)
        facts = atoms(rule)
        for arm in ARMS:
            for i in range(64 if arm != 'facts_cases' else 32):
                focus,answer = facts[i%4]
                if arm=='cards':
                    q = ('Explain','Describe','State','Recall')[i%4]+f' the complete rule named {identifier}.'
                    q += (' Include all conditions.',' Include exact values.',' Include its mappings.',' Include how to apply it.')[i//16]
                    answer = k['material']['learning']['text']
                    task = 'card'
                else:
                    n = i//4
                    q = FACT_FORMS[n%4].format(name=identifier,focus=focus)+ENDINGS[n//4]
                    task = 'fact'
                training[arm].append({'id':f"{k['id']}:{arm}:{i}",'qid':k['id'],'task':task,'channel':task,
                                     'messages':message(q,answer)})
        tr_cases = cases(k,'train')
        va_cases = cases(k,'valid',{r['detail']['signature'] for r in tr_cases})
        assert not {r['detail']['signature'] for r in tr_cases}&{r['detail']['signature'] for r in va_cases}
        for i,r in enumerate(tr_cases):
            identifier_row = f"{k['id']}:facts_cases:case:{i}"
            training['facts_cases'].append({'id':identifier_row,'qid':k['id'],'task':r['task'],'channel':r['task'],
                                            'messages':message(r['question'],r['answer'],r['fmt'])})
            checks.append({'id':identifier_row,'qid':k['id'],'task':r['task'],'answer':r['answer'],**r['detail']})
        for i,(focus,answer) in enumerate(facts):
            for j,form in enumerate(('Retrieve from memory the {focus} for {name}. Give only that fact.',
                                     'Regarding {name}: which value or condition belongs to its {focus}? Answer only that.',
                                     'Name {name} refers to a stored rule. Supply its {focus}, with no extra text.')):
                heldout.append({'id':f"{k['id']}:new_fact:{i}:{j}",'qid':k['id'],'relation':k['fact']['relation'],
                    'condition':'new_fact','task':'fact','fmt':'semantic','gold':answer,
                    'messages':message(form.format(name=name(k),focus=focus))})
        for i,r in enumerate(va_cases):
            heldout.append({'id':f"{k['id']}:new_case:{i}",'qid':k['id'],'relation':k['fact']['relation'],
                'condition':'new_'+r['task'],'task':r['task'],'fmt':'mcq','gold':r['answer'],
                'messages':message(r['question'],fmt='mcq')})
            checks.append({'id':f"{k['id']}:new_case:{i}",'qid':k['id'],'task':'validation_'+r['task'],
                           'answer':r['answer'],**r['detail']})
    for arm in ARMS:
        assert len(training[arm])==512
        random.Random(42).shuffle(training[arm])
        ev.save_rows(out/f'{arm}.jsonl',training[arm])
    ev.save_rows(out/'solver_checks.jsonl',checks)
    # Training is finalized before original questions/labels are opened.
    selected_by_id = {k['id']:k for k in selected}
    labels = {r['id']:r for r in rows(DATA/'audit/eval_labels.jsonl')}
    for direction in ('T2T','T2V'):
        for r in rows(DATA/direction/'eval.jsonl'):
            if r['qid'] not in selected_by_id:
                continue
            ready = ev.prepare_row(r,DATA,direction)
            heldout.append({'id':direction+':'+r['id'],'qid':r['qid'],
                'relation':selected_by_id[r['qid']]['fact']['relation'],'condition':'original_'+direction,
                'task':labels[r['id']]['question_type'],'fmt':'mcq','gold':labels[r['id']]['answer'],
                'messages':ready['messages'],'images':ready.get('images',[])})
    # Material controls for new cases and original questions, separated from blind scores.
    for p in list(heldout):
        if p['fmt']!='mcq':
            continue
        q=copy.deepcopy(p)
        q['id']+=':material';q['condition']='material_'+p['condition']
        q['messages'][-1]['content']='Learning material:\n'+selected_by_id[q['qid']]['material']['learning']['text']+'\n\n'+q['messages'][-1]['content']
        heldout.append(q)
    # Exact training prompt controls tell capacity from rewording failures.
    for k in selected:
        for task in ('fact','direct','diagnosis'):
            candidates=[r for r in training['facts_cases'] if r['qid']==k['id'] and r['task']==task]
            for r in sorted(candidates,key=lambda r:r['id'])[:2]:
                heldout.append({'id':r['id']+':seen','qid':k['id'],'relation':k['fact']['relation'],
                    'condition':'seen_'+task,'task':task,'fmt':'semantic' if task=='fact' else 'mcq',
                    'gold':r['messages'][-1]['content'],'messages':r['messages'][:-1]})
    assert len({p['id'] for p in heldout})==len(heldout)
    for arm in ARMS:
        signatures={ev.digest(r['messages'][:-1]) for r in training[arm]}
        assert all(ev.digest(p['messages']) not in signatures for p in heldout if p['condition'].startswith('new_'))
    ev.save_rows(out/'probes.jsonl',heldout)
    ev.save_rows(out/'knowledge.jsonl',selected)
    images=rows(DATA/'image_hashes.jsonl')
    for r in images:
        assert sha(DATA/r['path'])==r['source_sha256']==r['repaired_sha256']
    sources=[DATA/'audit/rule_knowledge.jsonl',DATA/'audit/eval_labels.jsonl',DATA/'image_hashes.jsonl']
    sources += [DATA/d/'eval.jsonl' for d in ('T2T','T2V','V2T','V2V')]
    save(out/'manifest.json',{'model':str(MODELS['qwen3']),'arms':list(ARMS),'rules':list(selected_by_id),
        'selection':'SHA256(retrieval-repair-20261015:42:ID), first4/family; no outcome filtering',
        'epochs':1,'training_rows_per_arm':512,'batch_size':4,'accumulation':1,'steps':128,'lr':1e-4,
        'rank':8,'alpha':32,'dropout':.05,'seed':42,'max_length':8192,'vision_aligner_frozen':True,
        'modality':'TEXT','tasks':{arm:dict(Counter(r['task'] for r in tr)) for arm,tr in training.items()},
        'probes':dict(Counter(p['condition'] for p in heldout)),'images_verified':len(images),
        'sources':{str(p):sha(p) for p in sources},'files':{p.name:sha(p) for p in out.glob('*.jsonl')},
        'script_sha256':sha(Path(__file__)),
        'limits':['8-rule development exploration, not independent full-scale evidence.',
          'Equal rows, steps and optimizer settings; target token counts differ intentionally.',
          'Cards repeat16 formulations4times; facts vary16 formulations per fact; 1epoch is over augmented rows.',
          'Facts_cases has32 fact prompts plus16 observations with paired semantic/MCQ targets per rule.',
          'Train denominator20/40 vs valid60/80; finite truth patterns and positive values can overlap.',
          'Interval exact boundaries appear in both pools; complete four-candidate combinations are disjoint.',
          'Original v7 prompts and all3000PNGs unchanged. No frozen eval answers used to generate training.',
          'Auxiliary factual probes are short answers; final original task remains MCQ.',
          'New-case reports25%correct, always appear among options; all option wording identical.',
          'Seen case probes are capacity controls only for facts_cases; not treated as heldout.',
          'No PT definition change; all three arms are SFT from the same base model.'], 'created_at':ev.now()})
    print('prepared',len(heldout),'probes',flush=True)


def score(text,gold,fmt):
    if fmt=='mcq':
        return ev.letter(text)==gold
    clean=lambda t:re.sub(r'\s+',' ',t.strip().rstrip('.')).casefold()
    return clean(text)==clean(gold)


def evaluate(out,arm,gpu,cp):
    folder=out/arm;folder.mkdir(exist_ok=True)
    raw=folder/'predictions.jsonl'
    records,notes=ev.read_rows(raw,allow_partial_tail=True) if raw.exists() else ([],[])
    if notes:ev.save_rows(raw,records)
    probes=rows(out/'probes.jsonl');seen={r['id'] for r in records}
    if arm!='base' and cp is not None:
        assert ev.checkpoint(cp)['sha256']==read(folder/'checkpoint_manifest.json')['sha256']
    pending=[p for p in probes if p['id'] not in seen]
    if pending:
        with reserve(WORK,gpu,str(folder)+':eval',24000) as reservation:
            reservation.bind(os.getpid());os.environ.update(environment(gpu,DATA))
            import torch
            from swift.infer_engine import TransformersEngine,InferRequest,RequestConfig
            from swift.utils import seed_everything
            seed_everything(42)
            engine=TransformersEngine(str(MODELS['qwen3']),adapters=[str(cp)] if cp else None,
                torch_dtype=torch.bfloat16,device_map='cuda:0',max_batch_size=8,attn_impl='sdpa')
            engine.template.enable_thinking=False
            if cp:
                nonzero=sum(bool(torch.count_nonzero(p)) for n,p in engine.model.named_parameters() if 'lora_B' in n)
                assert nonzero==252
            for fmt in ('semantic','mcq'):
                group=[p for p in pending if p['fmt']==fmt]
                for start in range(0,len(group),8):
                    batch=group[start:start+8]
                    replies=engine.infer([InferRequest(messages=copy.deepcopy(p['messages']),images=p.get('images',[])) for p in batch],
                        RequestConfig(max_tokens=64 if fmt=='semantic' else 16,temperature=0,seed=42,return_details=True),use_tqdm=False)
                    for p,response in zip(batch,replies):
                        if isinstance(response,Exception):raise response
                        c=response.choices[0];text=c.message.content or ''
                        r={k:p[k] for k in ('id','qid','relation','condition','task','fmt','gold')}
                        r.update(response=text,correct=score(text,p['gold'],fmt),predicted=ev.letter(text) if fmt=='mcq' else None,
                            format_error=(fmt=='mcq' and ev.letter(text) is None),finish_reason=c.finish_reason,
                            prompt_sha256=ev.digest({'messages':p['messages'],'images':p.get('images',[])}),
                            raw_response=dataclasses.asdict(response),saved_at=ev.now())
                        with raw.open('a') as f:f.write(json.dumps(r,ensure_ascii=False,default=str)+'\n');f.flush();os.fsync(f.fileno())
                        records.append(r)
                    save(folder/'state.json',{'status':'evaluating','saved':len(records),'expected':len(probes),'pid':os.getpid(),'updated_at':ev.now()})
                    print(arm,fmt,start+len(batch),'/',len(group),flush=True)
    assert len(records)==len({r['id'] for r in records})==len(probes)
    def metric(g):
        return {'n':len(g),'correct':sum(r['correct'] for r in g),'format_errors':sum(r['format_error'] for r in g),
                'empty':sum(not r['response'].strip() for r in g),'truncated':sum(r['finish_reason']=='length' for r in g)}
    metrics={condition:{**metric([r for r in records if r['condition']==condition]),
        'by_relation':{rel:metric([r for r in records if r['condition']==condition and r['relation']==rel])
                       for rel in sorted({r['relation'] for r in probes})}}
        for condition in sorted({r['condition'] for r in probes})}
    save(folder/'report.json',{'status':'complete','arm':arm,'metrics':metrics,'checkpoint':ev.checkpoint(cp) if cp else None,
         'probes_sha256':sha(out/'probes.jsonl'),'finished_at':ev.now()})
    save(folder/'state.json',{'status':'complete','saved':len(records),'updated_at':ev.now()})


def run(out,arm,gpu):
    if read(out/arm/'report.json').get('status')=='complete':return
    for p,h in read(out/'manifest.json')['sources'].items():assert sha(Path(p))==h
    if arm=='base':
        evaluate(out,arm,gpu,None);return
    folder=out/arm;folder.mkdir(exist_ok=True)
    with ev.run_lock(folder/'.lock'):
        cp=final_checkpoint(folder/'checkpoint',1)
        if not cp:
            candidates=[p for p in (folder/'checkpoint').glob('checkpoint-*') if (p/'optimizer.pt').exists()]
            resume=max(candidates,key=lambda p:int(p.name.split('-')[-1])) if candidates else None
            with reserve(WORK,gpu,str(folder)+':train',25000) as reservation:
                cmd=train_command('qwen3','sft',out/f'{arm}.jsonl',folder/'checkpoint',1,resume=resume)
                for flag,value in (('--per_device_train_batch_size','4'),('--gradient_accumulation_steps','1')):
                    cmd[cmd.index(flag)+1]=value
                cmd+=['--enable_channel_loss','true']
                with (folder/'train.log').open('a') as log:
                    p=subprocess.Popen(cmd,cwd=ROOT,env=environment(gpu,DATA),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    reservation.bind(p.pid)
                    save(folder/'state.json',{'status':'training','pid':p.pid,'command':cmd,'started_at':ev.now()})
                    code=p.wait()
                if code:raise RuntimeError(f'Training failed:{arm}:{code}')
            cp=final_checkpoint(folder/'checkpoint',1);assert cp
        state=read(cp/'trainer_state.json')
        assert state['global_step']==128 and abs(state['epoch']-1)<1e-6
        logs=rows(folder/'checkpoint/logging.jsonl');losses=[r['loss'] for r in logs if 'loss' in r]
        assert losses and all(math.isfinite(x) for x in losses)
        from safetensors import safe_open
        with safe_open(str(cp/'adapter_model.safetensors'),framework='pt',device='cpu') as f:
            keys=list(f.keys());bs=[k for k in keys if 'lora_B' in k]
            assert len(bs)==252 and all(bool(f.get_tensor(k).count_nonzero()) for k in bs)
            assert not any('.visual.' in k or '.merger.' in k for k in keys)
        save(folder/'checkpoint_manifest.json',{**ev.checkpoint(cp),'epoch':state['epoch'],'global_step':128,
             'final_loss':losses[-1],'losses_finite':True,'nonzero_lora_B':252})
        evaluate(out,arm,gpu,cp)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--arm',choices=(*ARMS,'base'),default='cards')
    ap.add_argument('--gpu',default='0');ap.add_argument('--prepare-only',action='store_true')
    a=ap.parse_args();out=a.out.resolve();prepare(out)
    if not a.prepare_only:run(out,a.arm,a.gpu)
