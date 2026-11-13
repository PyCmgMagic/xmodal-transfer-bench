"""Continue immutable CoT probes with a recorded throughput-only batch change."""
import argparse
import copy
import dataclasses
import json
import os

import eval_v7_atomic_cot_diagnostic as original
import run_v7_evaluation as ev
from run_v7_full_experiment import ROOT, MODELS, environment, rows, read, save, sha
from v7_gpu_resources import reserve


def main(arm, gpu):
    folder, probes, adapter, config = original.prepare(arm)
    raw = folder / 'predictions.jsonl'
    records, notes = ev.read_rows(raw, allow_partial_tail=True) if raw.exists() else ([], [])
    if notes:
        ev.save_rows(raw, records)
    seen = {r['id'] for r in records}
    pending = [p for p in probes if p['id'] not in seen]
    revision = folder / 'batch8_continuation.json'
    if not revision.exists():
        save(revision, dict(started_at=ev.now(), existing_ids=sorted(seen), original_batch=2,
            continuation_batch=8, source_script_sha256=sha(original.Path(__file__)),
            original_manifest_sha256=sha(folder / 'manifest.json'),
            note='Only batching changed; same prompts, checkpoint, native_thinking=true, 1536-token cap, temperature=0, seed=42. Numerical outputs can be batch-dependent; report this limitation.'))
    if pending:
        save(folder / 'state.json', dict(status='waiting_for_gpu', gpu=gpu, pid=os.getpid(), saved=len(records), total=64, batch=8))
        with reserve(ROOT / 'runs/v7_full_experiment', gpu, str(folder) + ':batch8', 22000) as reservation:
            reservation.bind(os.getpid())
            os.environ.update(environment(gpu, original.DATA))
            import torch
            from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig
            from swift.utils import seed_everything
            torch.set_num_threads(4)
            seed_everything(42)
            engine = TransformersEngine(str(MODELS['qwen3']), adapters=[str(adapter)] if adapter else [],
                torch_dtype=torch.bfloat16, device_map='cuda:0', max_batch_size=8, attn_impl='sdpa')
            engine.template.enable_thinking = True
            engine.model.eval()
            assert bool(getattr(engine.model, 'peft_config', None)) == bool(adapter)
            for start in range(0, len(pending), 8):
                batch = pending[start:start + 8]
                replies = engine.infer([InferRequest(messages=copy.deepcopy(p['messages']), images=p.get('images', [])) for p in batch],
                    RequestConfig(max_tokens=1536, temperature=0, seed=42, return_details=True), use_tqdm=False)
                assert len(batch) == len(replies)
                for p, reply in zip(batch, replies):
                    if isinstance(reply, Exception):
                        raise reply
                    c = reply.choices[0]
                    answer = c.message.content or ''
                    reasoning = getattr(c.message, 'reasoning_content', None) or ''
                    response = (f'<think>{reasoning}</think>\n' if reasoning else '') + answer
                    pred = ev.letter(response, 'cot')
                    r = {k: p[k] for k in ('id', 'source_id', 'qid', 'protocol', 'modality', 'arm', 'gold', 'task', 'relation')}
                    r.update(response=response, predicted=pred, correct=pred == p['gold'], format_error=pred is None,
                        empty=not response.strip(), reasoning_content=reasoning, answer_content=answer,
                        finish_reason=c.finish_reason, batch_size=len(batch), continuation='batch8',
                        input_sha256=ev.digest(dict(messages=p['messages'], images=p.get('images', []))),
                        recall_response_sha256=p.get('recall_response_sha256'), raw_response=dataclasses.asdict(reply))
                    with raw.open('a') as f:
                        f.write(json.dumps(r, ensure_ascii=False, default=str) + '\n')
                        f.flush()
                        os.fsync(f.fileno())
                    records.append(r)
                save(folder / 'state.json', dict(status='running', gpu=gpu, pid=os.getpid(), saved=len(records), total=64, batch=8, updated_at=ev.now()))
                print(arm, len(records), '/64', flush=True)
    original.evaluate(arm, gpu)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arm', choices=['base', 'trained'], required=True)
    ap.add_argument('--gpu', required=True)
    a = ap.parse_args()
    with ev.run_lock(original.OUT / (a.arm + '.lock')):
        main(a.arm, a.gpu)
