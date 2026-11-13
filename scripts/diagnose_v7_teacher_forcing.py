#!/usr/bin/env python3
"""Recompute masked next-token metrics on actual training answers; contrast free recall."""
import argparse
import copy
import dataclasses
import json
import os
from pathlib import Path

import run_v7_evaluation as ev
from diagnose_v7_knowledge_access import PILOT, DATA, LEDGER, checkpoint
from run_v7_full_experiment import ROOT, MODELS, environment, read, rows, save, sha
from v7_gpu_resources import reserve


def run(out,arm,gpu):
    folder=out/arm/'teacher_forcing';folder.mkdir(parents=True,exist_ok=True)
    source=ROOT/'runs/v7_full_experiment/data/sft_text.jsonl' if arm=='full_sft_text' else PILOT/'text_alias/train.jsonl'
    training=rows(source);selected={k['id'] for k in rows(PILOT/'target_knowledge.jsonl')}
    if arm=='text_alias':training=[r for r in training if r['id'].endswith(':0')]
    assert len(training)==(1000 if arm=='full_sft_text' else 8)
    config={'model':str(MODELS['qwen3']),'adapter':ev.checkpoint(checkpoint(arm)),'source':str(source),'source_sha256':sha(source),'script_sha256':sha(Path(__file__)),'examples':len(training),'selected_rules':sorted(selected),'objective':'Exact training chat template, assistant-only label mask, eval mode, no gradient. Gold previous answer tokens are present only in teacher-forced forward pass. Free generation receives only original training user/system messages.','template':'Qwen3 Instruct, enable_thinking=false, chat=true, loss_scale=default','dtype':'bfloat16','seed':42,'free_generation_max_tokens':384}
    if (folder/'manifest.json').exists():assert read(folder/'manifest.json')==config
    else:save(folder/'manifest.json',config)
    raw=folder/'token_metrics.jsonl';metrics,notes=ev.read_rows(raw,allow_partial_tail=True) if raw.exists() else ([],[])
    if notes:ev.save_rows(raw,metrics)
    freefile=folder/'free_recall.jsonl';free,notes=ev.read_rows(freefile,allow_partial_tail=True) if freefile.exists() else ([],[])
    if notes:ev.save_rows(freefile,free)
    seen={r['id'] for r in metrics};free_seen={r['qid'] for r in free}
    pending=[r for r in training if r['id'] not in seen];free_rows=[r for r in training if r['qid'] in selected and r['qid'] not in free_seen]
    if pending or free_rows:
        with reserve(LEDGER,gpu,str(folder),23000) as reservation:
            reservation.bind(os.getpid());os.environ.update(environment(gpu,DATA))
            import torch
            import torch.nn.functional as F
            from swift.infer_engine import TransformersEngine,InferRequest,RequestConfig
            from swift.utils import seed_everything
            seed_everything(42)
            engine=TransformersEngine(str(MODELS['qwen3']),adapters=[str(checkpoint(arm))],torch_dtype=torch.bfloat16,device_map='cuda:0',max_batch_size=4,attn_impl='sdpa');engine.template.enable_thinking=False;engine.model.eval()
            for start in range(0,len(free_rows),4):
                batch=free_rows[start:start+4]
                outputs=engine.infer([InferRequest(messages=copy.deepcopy(r['messages'][:-1])) for r in batch],RequestConfig(max_tokens=384,temperature=0,seed=42,return_details=True),use_tqdm=False)
                for r,reply in zip(batch,outputs):
                    if isinstance(reply,Exception):raise reply
                    c=reply.choices[0];text=c.message.content or ''
                    z={'id':r['id'],'qid':r['qid'],'response':text,'gold':r['messages'][-1]['content'],'exact':text.strip()==r['messages'][-1]['content'],'finish_reason':c.finish_reason,'input_sha256':ev.digest(r['messages'][:-1]),'raw_response':dataclasses.asdict(reply)}
                    free.append(z)
                    with freefile.open('a') as f:f.write(json.dumps(z,ensure_ascii=False,default=str)+'\n');f.flush();os.fsync(f.fileno())
            engine.template.set_mode('train')
            tokenizer=engine.template.tokenizer
            for index,r in enumerate(pending):
                encoded=engine.template.encode({'messages':copy.deepcopy(r['messages'])})
                assert isinstance(encoded,dict) and encoded.get('labels')
                batch=engine.template.data_collator([encoded])
                batch={k:v.to('cuda:0') for k,v in batch.items() if k in ('input_ids','attention_mask','labels','position_ids')}
                assert 'labels' in batch
                with torch.inference_mode():
                    result=engine.model(**batch,use_cache=False,return_dict=True)
                    logits=result.logits[0,:-1].float();labels=batch['labels'][0,1:];mask=labels!=-100
                    positions=torch.where(mask)[0];gold=labels[mask];pred=logits[mask].argmax(-1)
                    losses=F.cross_entropy(logits[mask],gold,reduction='none');correct=pred==gold
                    n=gold.numel();assert n>1 and torch.isfinite(losses).all()
                    gold_cpu=gold.cpu().tolist();pred_cpu=pred.cpu().tolist();loss_cpu=losses.cpu().tolist();correct_cpu=correct.cpu().tolist()
                    assert abs(float(result.loss)-float(losses.mean()))<.01,(result.loss,losses.mean())
                    z={'id':r['id'],'qid':r['qid'],'n':n,'nll_sum':sum(loss_cpu),'mean_loss':sum(loss_cpu)/n,'correct_tokens':sum(correct_cpu),'token_accuracy':sum(correct_cpu)/n,'template_mask_verified':True,'original_training_input_sha256':ev.digest(r['messages']),'supervised_ids_sha256':ev.digest(gold_cpu),'first32':{'n':min(n,32),'correct':sum(correct_cpu[:32]),'nll_sum':sum(loss_cpu[:32])},'numeric_pieces':{'n':0,'correct':0,'nll_sum':0.0}}
                    for token,ok,loss in zip(gold_cpu,correct_cpu,loss_cpu):
                        piece=tokenizer.decode([token])
                        if any(c.isdigit() for c in piece):z['numeric_pieces']['n']+=1;z['numeric_pieces']['correct']+=ok;z['numeric_pieces']['nll_sum']+=loss
                    if r['qid'] in selected:
                        z['tokens']=[{'index':i,'gold_id':g,'gold_piece':tokenizer.decode([g]),'pred_id':p,'pred_piece':tokenizer.decode([p]),'correct':ok,'nll':loss} for i,(g,p,ok,loss) in enumerate(zip(gold_cpu,pred_cpu,correct_cpu,loss_cpu))]
                    metrics.append(z)
                    with raw.open('a') as f:f.write(json.dumps(z,ensure_ascii=False)+'\n');f.flush()
                    del logits,result,batch,losses
                if (index+1)%25==0 or index+1==len(pending):
                    save(folder/'state.json',{'status':'running','pid':os.getpid(),'scored':len(metrics),'expected':len(training)});print(arm,'teacher forced',len(metrics),'/',len(training),flush=True)
    assert len(metrics)==len({r['id'] for r in metrics})==len(training) and len(free)==8
    def aggregate(group):
        n=sum(r['n'] for r in group);num=sum(r['numeric_pieces']['n'] for r in group);first=sum(r['first32']['n'] for r in group)
        return {'examples':len(group),'supervised_tokens':n,'mean_loss':sum(r['nll_sum'] for r in group)/n,'token_accuracy':sum(r['correct_tokens'] for r in group)/n,'first32_token_accuracy':sum(r['first32']['correct'] for r in group)/first,'numeric_token_pieces':num,'numeric_piece_accuracy':sum(r['numeric_pieces']['correct'] for r in group)/num if num else None}
    save(folder/'report.json',{'status':'complete','arm':arm,'all_training_rows':aggregate(metrics),'selected8':aggregate([r for r in metrics if r['qid'] in selected]),'free_recall_exact':sum(r['exact'] for r in free),'free_recall_n':8,'free_recall_truncated':sum(r['finish_reason']=='length' for r in free),'note':'Teacher-forced token accuracy is conditioned on the entire correct preceding answer. It is not independent factual recall. Full-generation mismatches need semantic audit, not just exact matching.','checkpoint_unchanged':ev.checkpoint(checkpoint(arm))==config['adapter']})
    save(folder/'state.json',{'status':'complete','scored':len(metrics),'expected':len(training)})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'runs/v7_access_diagnostics_20261015');ap.add_argument('--arm',choices=('text_alias','full_sft_text'),required=True);ap.add_argument('--gpu',default='0');a=ap.parse_args();run(a.out.resolve(),a.arm,a.gpu)
