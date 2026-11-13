"""Independent vLLM execution of the frozen cot100 inputs; no mixed-engine scoring."""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import eval_v7_cot100 as ex
import run_v7_evaluation as ev
from run_v7_full_experiment import ROOT, MODELS, rows, read, save, sha
from v7_gpu_resources import reserve

ORIGINAL = ex.OUT
OUT = ORIGINAL / 'vllm'
MERGED = ORIGINAL / 'vllm_merged_model'


def setup():
    import shutil
    original = ex.validate()
    OUT.mkdir(exist_ok=True)
    for name in ('knowledge.jsonl', 'questions.jsonl', 'recall_probes.jsonl'):
        target = OUT / name
        if not target.exists():
            shutil.copy2(ORIGINAL / name, target)
        assert sha(target) == original['files'][name]
    path = OUT / 'manifest.json'
    if not path.exists():
        m = copy.deepcopy(original)
        m.update(engine='vllm', parent_manifest_sha256=sha(ORIGINAL / 'manifest.json'),
                 execution_created_at=ev.now(),
                 engine_note='Independent full rerun; Transformers partial outputs excluded.',
                 runtime='/mnt/yunpan/anaconda3/envs/qwen3vl-rollout/bin/python')
        m['generation'].update(batch_size='continuous', max_num_seqs=32,
            max_model_len=8192, gpu_memory_utilization=0.34, max_num_batched_tokens=8192)
        m['sources'][str(Path(__file__).resolve())] = sha(Path(__file__).resolve())
        merge = read(MERGED / 'merge_manifest.json')
        assert merge['merged_layers'] == 252
        m['serving_conversion'] = merge
        m['sources'][str(MERGED / 'merge_manifest.json')] = sha(MERGED / 'merge_manifest.json')
        save(path, m)
    ex.OUT = OUT
    return ex.validate()


def prompt(probe, processor):
    from PIL import Image
    messages = copy.deepcopy(probe['messages'])
    image_paths = iter(probe.get('images', []))
    images = []
    for message in messages:
        content = message['content']
        if '<image>' not in content:
            continue
        blocks = []
        for i, piece in enumerate(content.split('<image>')):
            if i:
                path = next(image_paths)
                images.append(Image.open(path).convert('RGB'))
                blocks.append(dict(type='image'))
            if piece:
                blocks.append(dict(type='text', text=piece))
        message['content'] = blocks
    assert next(image_paths, None) is None
    rendered = processor.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    assert not rendered.endswith('<think>\n')
    result = dict(prompt=rendered)
    if images:
        result['multi_modal_data'] = dict(image=images)
    return result


def evaluate(arm, gpu, smoke):
    setup()
    folder = OUT / ('smoke' if smoke else arm)
    folder.mkdir(exist_ok=True)
    path = folder / 'predictions.jsonl'
    records, notes = ev.read_rows(path, allow_partial_tail=True) if path.exists() else ([], [])
    if notes:
        import shutil
        shutil.copy2(path, folder / ('partial_tail_' + str(os.getpid()) + '.jsonl'))
        ev.save_rows(path, records)
    seen = {r['id'] for r in records}
    with reserve(ex.LEDGER, gpu, str(folder) + ':vllm', 31000) as reservation:
        reservation.bind(os.getpid())
        os.environ.update(CUDA_VISIBLE_DEVICES=gpu, VLLM_WORKER_MULTIPROC_METHOD='spawn',
            TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', OMP_NUM_THREADS='4')
        os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)
        import torch
        import vllm
        import transformers
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams
        save(folder / 'state.json', dict(status='loading', pid=os.getpid(), gpu=gpu, saved=len(records)))
        processor = AutoProcessor.from_pretrained(str(MODELS['qwen3']), local_files_only=True)
        serving = MERGED if arm == 'trained' else MODELS['qwen3']
        llm = LLM(model=str(serving), dtype='bfloat16', seed=42,
            enable_lora=False,
            max_model_len=8192, max_num_seqs=32, max_num_batched_tokens=8192,
            gpu_memory_utilization=0.34, enforce_eager=True,
            limit_mm_per_prompt={'image': 1, 'video': 0},
            mm_processor_kwargs={'size': {'longest_edge': 1003520, 'shortest_edge': 65536}},
            enable_prefix_caching=True, disable_log_stats=False)
        save(folder / 'runtime.json', dict(vllm=vllm.__version__, torch=torch.__version__,
            transformers=transformers.__version__, python=sys.version,
            adapter=ex.ev.checkpoint(ex.ADAPTER), base=str(MODELS['qwen3']), serving=str(serving)))

        def run(probes, chosen_arm):
            pending = [p for p in probes if p['id'] not in seen]
            lookup = {}
            for i, p in enumerate(pending):
                inp = prompt(p, processor)
                request_id = chosen_arm + ':' + str(i)
                lookup[request_id] = (p, inp['prompt'])
                llm.llm_engine.add_request(request_id, inp,
                    SamplingParams(temperature=0, seed=42,
                        max_tokens=128 if smoke else 768 if p['protocol'] == 'recall' else 3072))
            while llm.llm_engine.has_unfinished_requests():
                outputs = llm.llm_engine.step()
                for output in outputs:
                    if not output.finished:
                        continue
                    p, rendered = lookup[output.request_id]
                    choice = output.outputs[0]
                    response = choice.text
                    predicted = ex.final_letter(response) if p['protocol'] != 'recall' else None
                    r = {k: p[k] for k in ('id', 'qid', 'source_id', 'protocol', 'modality',
                        'gold', 'task', 'family', 'evaluation_group', 'recall_response_sha256',
                        'recall_truncated') if k in p}
                    r.update(arm=chosen_arm, engine='vllm', response=response,
                        answer_content=response, reasoning_content='', predicted=predicted,
                        correct=predicted == p['gold'] if p['protocol'] != 'recall' else None,
                        finish_reason=choice.finish_reason,
                        input_sha256=ev.digest(dict(messages=p['messages'], images=p.get('images', []))),
                        rendered_prompt=rendered, rendered_prompt_sha256=ev.digest(rendered),
                        prompt_token_ids=output.prompt_token_ids,
                        generated_token_ids=list(choice.token_ids),
                        serving_model=str(serving),
                        raw_response=dict(choices=[dict(message=dict(content=response,
                            reasoning_content=None), finish_reason=choice.finish_reason)],
                            stop_reason=choice.stop_reason, num_cached_tokens=output.num_cached_tokens))
                    with path.open('a') as h:
                        h.write(json.dumps(r, ensure_ascii=False, default=str) + '\n')
                        h.flush(); os.fsync(h.fileno())
                    records.append(r); seen.add(p['id'])
                    save(folder / 'state.json', dict(status='running', pid=os.getpid(), gpu=gpu,
                        saved=len(records), protocol=p['protocol'], updated_at=ev.now()))
                    print(chosen_arm, p['protocol'], len(records), flush=True)
            return records

        if smoke:
            probes = rows(OUT / 'questions.jsonl')
            subset = [probes[0], probes[100]]
            run(subset, arm)
        else:
            for protocol in ('direct_cot', 'recall', 'self_recall_cot'):
                probes = rows(OUT / 'questions.jsonl') if protocol == 'direct_cot' else (
                    rows(OUT / 'recall_probes.jsonl') if protocol == 'recall' else ex.applications(records))
                if protocol == 'self_recall_cot':
                    ev.save_rows(folder / 'application_probes.jsonl', probes)
                run(probes, arm)
                ex.summarize(arm)
            assert len(records) == len(seen) == 450
            ex.validate()
        save(folder / 'state.json', dict(status='complete', saved=len(records), ended_at=ev.now()))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arm', choices=['base', 'trained'], default='trained')
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    try:
        evaluate(args.arm, args.gpu, args.smoke)
    except Exception:
        import traceback
        folder = OUT / ('smoke' if args.smoke else args.arm)
        save(folder / 'state.json', dict(status='failed', error=traceback.format_exc(), pid=os.getpid()))
        raise
