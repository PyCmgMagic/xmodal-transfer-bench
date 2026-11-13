#!/usr/bin/env python3
"""Paired entity evaluation v2, retaining raw and constrained scores separately."""
import argparse
import atexit
import copy
from collections import Counter
import json
import math
import os
from pathlib import Path
import re
import sys
import time

from entityqa_common import rows, save, write_rows, sha, digest, now, render, eval_payload, repeated_suffix, COT_SUFFIX
from answer_readout import options_from_question
from answer_readout_entity_v2 import parse_answer, VERSION


def score_options(llm,processor,payloads,option_ids,SamplingParams):
    inputs=[]
    for payload in payloads:
        original=render(payload,processor)
        for letter,ids in zip('ABCD',option_ids):
            inp=dict(original,prompt=original['prompt']+letter)
            assert processor.tokenizer.encode(inp['prompt'],add_special_tokens=False)[-1]==ids[0]
            inputs.append(inp)
    outputs=llm.generate(inputs,SamplingParams(temperature=0,max_tokens=1,prompt_logprobs=1),use_tqdm=False)
    scored=[]
    for start in range(0,len(outputs),4):
        scores={}
        for letter,ids,out in zip('ABCD',option_ids,outputs[start:start+4]):
            assert out.prompt_token_ids[-1]==ids[0]
            value=out.prompt_logprobs[-1][ids[0]].logprob
            assert math.isfinite(value)
            scores[letter]=value
        scored.append(dict(predicted=max(scores,key=scores.get),option_logprobs=scores))
    return scored


def summary(records, total):
    groups = {}
    for modality in ['T','V']:
        for probe in ['all','direct','statement']:
            subset = [r for r in records if r['modality']==modality and (probe=='all' or r['probe_id']==probe)]
            if not subset:
                continue
            n=len(subset)
            groups[f'{modality}/{probe}'] = dict(n=n,
                primary_accuracy=sum(r['raw_predicted']==r['gold'] for r in subset)/n,
                primary_invalid_rate=sum(r['raw_predicted'] is None for r in subset)/n,
                raw_cot_accuracy=sum(r['raw_predicted']==r['gold'] for r in subset)/n,
                raw_cot_invalid_rate=sum(r['raw_predicted'] is None for r in subset)/n,
                accuracy=sum(r['predicted']==r['gold'] for r in subset)/n,
                invalid_rate=sum(r['predicted'] is None for r in subset)/n,
                finalizer_count=sum(r['finalizer'] is not None for r in subset),
                raw_answer_distribution=dict(Counter(str(r['raw_predicted']) for r in subset)),
                final_answer_distribution=dict(Counter(str(r['predicted']) for r in subset)),
                raw_parse_reasons=dict(Counter(r['raw_parse_reason'] for r in subset)),
                generated_tokens=sum(r['generated_tokens'] for r in subset),
                stop_reasons=dict(Counter(r['stop_reason'] for r in subset)))
    paired={}
    lookup={(r['id'],r['modality']):r for r in records}
    for probe in ['all','direct','statement']:
        pairs=[(r,lookup[r['id'],'V']) for r in records if r['modality']=='T'
               and (r['id'],'V') in lookup and (probe=='all' or r['probe_id']==probe)]
        if pairs:
            paired[probe]=dict(n=len(pairs),both_correct=sum(a['predicted']==a['gold'] and b['predicted']==b['gold'] for a,b in pairs)/len(pairs),
                text_only_correct=sum(a['predicted']==a['gold'] and b['predicted']!=b['gold'] for a,b in pairs)/len(pairs),
                vision_only_correct=sum(a['predicted']!=a['gold'] and b['predicted']==b['gold'] for a,b in pairs)/len(pairs))
    return dict(status='complete' if len(records)==total else 'running',saved=len(records),total=total,
                groups=groups,paired=paired,updated_at=now())


def run(args, *, payload_builder=eval_payload, cot_suffix=COT_SUFFIX,
        answer_parser=parse_answer, readout_version=VERSION,
        prompt_protocol='brief prompted CoT, no native thinking prefix', evaluator_version=2):
    # Callers can select a frozen prompt/readout without mutating module globals.
    parse_answer = answer_parser
    data, output = args.data.resolve(),args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    labels={r['id']:r for r in rows(data/'audit/eval_labels.jsonl')}
    tasks=[]
    for modality, filename in [('T','eval_text.jsonl'),('V','eval_multimodal.jsonl')]:
        selected=rows(data/'shared'/filename)
        if args.limit:
            selected=selected[:args.limit]
        for r in selected:
            label=labels[r['id']]
            assert label['qid']==r['qid']
            tasks.append(dict(id=r['id'],qid=r['qid'],modality=modality,probe_id=label['probe_id'],
                gold=label['answer'],payload=payload_builder(r,data)))
    assert len(tasks)==len({(t['id'],t['modality']) for t in tasks})
    assert {t['id'] for t in tasks if t['modality']=='T'}=={t['id'] for t in tasks if t['modality']=='V'}
    config=dict(model=str(args.model.resolve()),arm=args.arm,data=str(data),
        dataset_inventory_sha256=sha(data/'package_inventory.json'),
        input_sha256=digest([{k:v for k,v in t.items() if k!='gold'} for t in tasks]),
        input_count=len(tasks),prompt_protocol=prompt_protocol,
        max_tokens=args.max_tokens,max_model_len=73728,temperature=0,seed=42,
        wall_seconds=args.wall_seconds,loop_guard='four identical consecutive token blocks (8-128 tokens)',
        finalization='only raw-invalid responses: same model, same question/image, own uncorrected CoT; constrained A/B/C/D token if runtime probe passes, otherwise four-option next-token logprob argmax',
        vision_max_pixels=262144,gpu=args.gpu,version=evaluator_version,readout_version=readout_version,
        primary_score='raw generation with unambiguous readout; constrained finalization reported separately',
        import_predictions=str(args.import_predictions.absolute()) if args.import_predictions else None)
    if (output/'config.json').exists():
        assert json.loads((output/'config.json').read_text())==config,'Resume configuration changed'
    else:
        save(output/'config.json',config)
    records=[]
    path=output/'predictions.jsonl'
    if args.import_predictions and not path.exists():
        # Reuse unchanged base-model generations under the identical prompt.
        # Re-score without gold-dependent parsing; preserve previous readout.
        imported=[]
        task_lookup={(t['id'],t['modality']):t for t in tasks}
        for previous in rows(args.import_predictions):
            r=copy.deepcopy(previous)
            t=task_lookup[r['id'],r['modality']]
            assert r['input_sha256']==digest(t['payload'])
            assert r['gold']==t['gold'] and r['arm']==args.arm=='base'
            r['previous_readout']={k:r.get(k) for k in ['raw_predicted','raw_parse_reason','predicted','finalizer','correct']}
            pred,reason=parse_answer(r['raw_response'],options_from_question(t['payload']['messages'][-1]['content']))
            r.update(raw_predicted=pred,raw_parse_reason=reason)
            if pred is not None:
                r.update(predicted=pred,correct=pred==r['gold'],finalizer=None)
            r['imported_from']=str(args.import_predictions.absolute())
            imported.append(r)
        write_rows(path,imported)
        save(output/'import_manifest.json',dict(source=str(args.import_predictions.absolute()),
            source_sha256=sha(args.import_predictions),rows=len(imported),readout_version=readout_version,
            same_input_hash_checked=True,raw_generation_unchanged=True,created_at=now()))
    if path.exists():
        # Recover only a torn last physical line. Never suppress a corrupt middle record.
        raw_lines=path.read_bytes().split(b'\n')
        for index,line in enumerate(raw_lines):
            if not line.strip():continue
            try: records.append(json.loads(line))
            except (ValueError,UnicodeError):
                if index!=len(raw_lines)-1:raise
                backup=output/f'torn_tail_{int(time.time())}.jsonl'
                backup.write_bytes(path.read_bytes())
                write_rows(path,records)
    lookup={(t['id'],t['modality']):t for t in tasks}
    seen={(r['id'],r['modality']) for r in records}
    assert len(records)==len(seen)
    for r in records:
        t=lookup[r['id'],r['modality']]
        assert r['gold']==t['gold'] and r['input_sha256']==digest(t['payload'])
    pending=[t for t in tasks if (t['id'],t['modality']) not in seen]
    if not pending:
        save(output/'summary.json',summary(records,len(tasks)))
        return
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu,VLLM_WORKER_MULTIPROC_METHOD='spawn',
        HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
    os.environ.pop('PYTORCH_CUDA_ALLOC_CONF',None)
    import torch,transformers,vllm
    from transformers import AutoProcessor
    from vllm import LLM,SamplingParams
    from vllm.sampling_params import RequestOutputKind
    save(output/'status.json',dict(stage='loading',saved=len(records),total=len(tasks),pid=os.getpid(),updated_at=now()))
    processor=AutoProcessor.from_pretrained(str(args.model),local_files_only=True)
    option_ids=[processor.tokenizer.encode(c,add_special_tokens=False) for c in 'ABCD']
    assert all(len(x)==1 for x in option_ids)
    llm=LLM(model=str(args.model),dtype='bfloat16',seed=42,
        safetensors_load_strategy='eager',
        max_model_len=73728,max_num_seqs=32,max_num_batched_tokens=8192,
        gpu_memory_utilization=0.82,enforce_eager=True,enable_prefix_caching=True,
        limit_mm_per_prompt={'image':1,'video':0},
        mm_processor_kwargs={'size':{'longest_edge':262144,'shortest_edge':65536}})
    atexit.register(llm.llm_engine.engine_core.shutdown)
    probe=render({'messages':[{'role':'user','content':'Output the word ZEBRA.'}]},processor)
    constrained_probe=llm.generate([probe],SamplingParams(temperature=0,max_tokens=1,
        allowed_token_ids=[x[0] for x in option_ids]),use_tqdm=False)[0]
    constraint_ok=constrained_probe.outputs[0].text in 'ABCD' and len(constrained_probe.outputs[0].text)==1
    save(output/'constraint_probe.json',dict(response=constrained_probe.outputs[0].text,
        token_ids=list(constrained_probe.outputs[0].token_ids),allowed_token_ids=[x[0] for x in option_ids],
        constraint_effective=constraint_ok,fallback='Exact next-token likelihood comparison over A/B/C/D'))
    if not constraint_ok:
        score_options(llm,processor,[{'messages':[{'role':'user','content':'Choose an option A, B, C, or D.'}]}],option_ids,SamplingParams)
    save(output/'runtime.json',dict(vllm=vllm.__version__,torch=torch.__version__,
        transformers=transformers.__version__,python=sys.version,started_at=now(),pid=os.getpid()))
    params=SamplingParams(temperature=0,seed=42,max_tokens=args.max_tokens,output_kind=RequestOutputKind.CUMULATIVE,
        stop=[r'\boxed{'+c+'}' for c in 'ABCD'],include_stop_str_in_output=True)
    started=time.monotonic()
    for start in range(0,len(pending),64):
        batch=pending[start:start+64]
        requests={}
        finished={}
        telemetry_at=time.monotonic()
        for i,t in enumerate(batch):
            request_id=f'cot-{start+i}'
            inp=render(t['payload'],processor)
            requests[request_id]=dict(task=t,submitted=time.monotonic(),checked=0,latest=None,
                rendered_prompt_sha256=digest(inp['prompt']))
            llm.llm_engine.add_request(request_id,inp,copy.deepcopy(params))
        while llm.llm_engine.has_unfinished_requests():
            outputs=llm.llm_engine.step()
            for out in outputs:
                state=requests[out.request_id]
                state['latest']=out
                choice=out.outputs[0]
                n=len(choice.token_ids)
                reason=None
                if out.finished:
                    reason=str(choice.finish_reason)
                    if choice.stop_reason is not None:reason+=':'+str(choice.stop_reason)
                elif n>=128 and n-state['checked']>=32:
                    state['checked']=n
                    if repeated_suffix(list(choice.token_ids[-512:])):
                        reason='repeated_token_block'
                    else:
                        sentences=[s.strip() for s in re.findall(r'[^.!?\n]+[.!?]',choice.text[-3000:])]
                        if any(c>=4 and len(s)>=40 for s,c in Counter(sentences).items()):
                            reason='repeated_sentence'
                if not reason and time.monotonic()-state['submitted']>=args.wall_seconds:
                    reason='time_limit'
                if reason:
                    if not out.finished:llm.llm_engine.abort_request([out.request_id])
                    finished[out.request_id]=(out,reason,time.monotonic()-state['submitted'])
            for key,state in requests.items():
                if key not in finished and time.monotonic()-state['submitted']>=args.wall_seconds:
                    llm.llm_engine.abort_request([key])
                    finished[key]=(state['latest'],'time_limit',time.monotonic()-state['submitted'])
            if time.monotonic()-telemetry_at>=5:
                telemetry_at=time.monotonic()
                save(output/'status.json',dict(stage='generating_cot',saved=len(records),total=len(tasks),
                    active_in_batch=len(requests)-len(finished),finished_in_batch=len(finished),
                    observed_generated_tokens=sum(len(s['latest'].outputs[0].token_ids) for s in requests.values() if s['latest'] is not None),
                    elapsed_this_batch_seconds=telemetry_at-min(s['submitted'] for s in requests.values()),
                    pid=os.getpid(),updated_at=now()))
        assert set(finished)==set(requests)
        new=[]
        final_tasks=[]
        for key,state in requests.items():
            t=state['task']
            out,reason,seconds=finished[key]
            choice=out.outputs[0] if out is not None else None
            response=choice.text if choice is not None else ''
            pred,parse_reason=parse_answer(response,options_from_question(t['payload']['messages'][-1]['content']))
            r={k:t[k] for k in ['id','qid','modality','probe_id','gold']}
            r.update(arm=args.arm,raw_response=response,raw_predicted=pred,raw_parse_reason=parse_reason,
                predicted=pred,correct=pred==t['gold'],finalizer=None,stop_reason=reason,
                generated_tokens=len(choice.token_ids) if choice is not None else 0,
                generated_token_ids=list(choice.token_ids) if choice is not None else [],
                latency_seconds=seconds,input_sha256=digest(t['payload']),
                rendered_prompt_sha256=state['rendered_prompt_sha256'],saved_at=now())
            if pred is None:
                question=t['payload']['messages'][-1]['content']
                assert question.endswith(cot_suffix)
                question=question[:-len(cot_suffix)]
                payload={'messages':[
                    {'role':'system','content':'Finish your multiple-choice answer. Use the original question and your own uncorrected previous reasoning. Match the result to one complete answer option. If uncertain, choose the most likely option. Output exactly one uppercase letter A, B, C, or D.'},
                    {'role':'user','content':'Original question:\n'+question+'\n\nYour previous reasoning (uncorrected):\n'+response+'\n\nSelect exactly one answer option now.'}]}
                if t['payload'].get('images'):payload['images']=t['payload']['images']
                final_tasks.append((r,payload))
            new.append(r)
        if final_tasks:
            if constraint_ok:
                outputs=llm.generate([render(payload,processor) for _,payload in final_tasks],
                    SamplingParams(temperature=0,seed=42,max_tokens=1,
                        allowed_token_ids=[x[0] for x in option_ids]),use_tqdm=False)
                choices=[dict(predicted=parse_answer(o.outputs[0].text)[0],response=o.outputs[0].text,
                    generated_token_ids=list(o.outputs[0].token_ids),method='constrained_generation') for o in outputs]
                assert all(c['predicted'] in 'ABCD' for c in choices)
            else:
                choices=score_options(llm,processor,[p for _,p in final_tasks],option_ids,SamplingParams)
                for c in choices:c.update(response=c['predicted'],method='next_token_logprob_argmax')
            for (r,payload),choice in zip(final_tasks,choices):
                pred=choice['predicted']
                r.update(predicted=pred,correct=pred==r['gold'],finalizer=dict(**choice,
                    input_sha256=digest(payload),previous_reasoning_sha256=digest(r['raw_response'])))
        with path.open('a',encoding='utf-8') as stream:
            for r in new:stream.write(json.dumps(r,ensure_ascii=False)+'\n')
            stream.flush();os.fsync(stream.fileno())
        records.extend(new)
        report=summary(records,len(tasks))
        report['elapsed_this_session_seconds']=time.monotonic()-started
        save(output/'summary.json',report)
        save(output/'status.json',dict(stage=report['status'],saved=len(records),total=len(tasks),
            pid=os.getpid(),updated_at=now(),samples_per_second=(start+len(batch))/(time.monotonic()-started)))
        print(json.dumps(dict(saved=len(records),total=len(tasks),groups=report['groups']),ensure_ascii=False),flush=True)
    assert len(records)==len(tasks)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--arm',required=True,choices=['base','TEXT','VISION'])
    p.add_argument('--gpu',default='1')
    p.add_argument('--limit',type=int,default=0,help='Paired questions per test modality; 0 means all')
    p.add_argument('--max-tokens',type=int,default=65536)
    p.add_argument('--wall-seconds',type=int,default=180)
    p.add_argument('--import-predictions',type=Path,help='Reuse verified partial base outputs under identical generation inputs')
    args=p.parse_args()
    try:run(args)
    except Exception as error:
        save(args.output/'failure.json',dict(error=repr(error),time=now()))
        raise
