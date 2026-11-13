#!/usr/bin/env python3
"""Isolated throughput probe; never contributes to the 192,000 formal responses."""
import argparse
import copy
import dataclasses
import json
import os
import time
from pathlib import Path

import run_v7_evaluation as ev
from run_v7_full_experiment import ROOT, MODELS, environment, read, rows, save


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=MODELS, required=True)
    ap.add_argument('--gpu', required=True)
    ap.add_argument('--strict-batch', type=int, default=16)
    ap.add_argument('--cot-batch', type=int, default=8)
    a = ap.parse_args()
    work = ROOT/'runs/v7_full_experiment'
    data = ROOT/'rules_repaired_single_image_v7'
    name = a.model+'_sft_text'
    source = work/('jobs' if a.model == 'qwen3' else 'smoke')/name
    cp = read(source/('checkpoint_manifest.json' if a.model == 'qwen3' else 'validated.json'))
    cp = Path(cp['path'] if a.model == 'qwen3' else cp['checkpoint'])
    folder = work/'benchmarks'/f'{a.model}_{int(time.time())}'
    folder.mkdir(parents=True)
    os.environ.update(environment(a.gpu, data))
    from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig
    from swift.utils import seed_everything
    import torch
    seed_everything(42)
    kwargs = {'template_type':'glm4_5v'} if a.model == 'glm46' else {}
    engine = TransformersEngine(str(MODELS[a.model]), adapters=[str(cp)],
        torch_dtype=torch.bfloat16, device_map='cuda:0', max_batch_size=max(a.strict_batch,a.cot_batch),
        attn_impl='sdpa', **kwargs)
    engine.template.use_chat_template = True
    measurements = []
    # Warm-up is deliberately excluded from throughput measurements.
    warmup = ev.prepare_row(rows(data/'T2T/eval.jsonl')[0],data,'T2T')
    engine.infer([InferRequest(messages=warmup['messages'])],RequestConfig(max_tokens=4,temperature=0),use_tqdm=False)
    for protocol, direction, limit in [('strict','T2T',a.strict_batch),('strict','T2V',a.strict_batch),('cot','T2V',a.cot_batch)]:
        engine.template.enable_thinking = protocol == 'cot' and a.model == 'glm46'
        fmt = 'cot' if protocol == 'cot' else ('glm-box' if a.model == 'glm46' else 'strict')
        prepared = [ev.prepare_row(r,data,direction,fmt) for r in rows(data/direction/'eval.jsonl')[:limit]]
        payload = [{k:r[k] for k in ('messages','images') if k in r} for r in prepared]
        request = RequestConfig(max_tokens=1536 if protocol == 'cot' else 32, temperature=0,seed=42,return_details=True)
        baseline = None
        for batch_size in (1, limit):
            count = 2 if batch_size == 1 else limit
            records = []
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.monotonic()
            print(a.model, protocol, direction, 'batch', batch_size, 'count', count, flush=True)
            for start in range(0,count,batch_size):
                batch = payload[start:min(start+batch_size,count)]
                responses = engine.infer([InferRequest(**copy.deepcopy(r)) for r in batch],request,use_tqdm=False)
                assert len(responses) == len(batch)
                for index, res in enumerate(responses,start):
                    if isinstance(res,Exception):raise res
                    choice = res.choices[0]
                    # Prompt token IDs must agree for the same text/image request,
                    # independently of left padding and numerical generation differences.
                    record = {'id':prepared[index]['id'],'response':choice.message.content or '',
                        'reasoning_content':getattr(choice.message,'reasoning_content',None),
                        'usage':dataclasses.asdict(res.usage),'finish_reason':choice.finish_reason,
                        'prompt_token_ids':res.prompt_token_ids,'images_size':res.images_size,
                        'prompt_sha256':ev.digest(payload[index]),'batch_size':batch_size}
                    records.append(record)
                    with (folder/f'{protocol}_{direction}_b{batch_size}.jsonl').open('a') as f:
                        f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
            torch.cuda.synchronize()
            seconds = time.monotonic()-started
            measurement = {'protocol':protocol,'direction':direction,'batch_size':batch_size,'requests':count,
                'seconds':seconds,'seconds_per_response':seconds/count,'responses_per_second':count/seconds,
                'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
                'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3,
                'valid':sum(ev.letter(r['response'],fmt) is not None for r in records),
                'truncated':sum(r['finish_reason']=='length' for r in records)}
            if baseline is None:baseline = records
            else:
                assert all(b['prompt_token_ids']==r['prompt_token_ids'] and b['images_size']==r['images_size']
                    and b['prompt_sha256']==r['prompt_sha256'] for b,r in zip(baseline,records)), 'Batch prompt/image alignment changed'
                measurement['alignment_verified'] = len(baseline)
                measurement['same_answers_as_serial'] = sum(ev.letter(b['response'],fmt)==ev.letter(r['response'],fmt) for b,r in zip(baseline,records))
                previous = measurements[-1]
                measurement['throughput_speedup'] = previous['seconds_per_response']/measurement['seconds_per_response']
            measurements.append(measurement)
            save(folder/'report.json',{'model':a.model,'adapter':str(cp),'formal_evaluation':False,'measurements':measurements,'updated_at':ev.now()})
            print(json.dumps(measurement),flush=True)
    save(folder/'status.json',{'status':'pass','finished_at':ev.now()})


if __name__ == '__main__':main()
