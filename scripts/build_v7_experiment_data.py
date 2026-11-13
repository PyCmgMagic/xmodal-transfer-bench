#!/usr/bin/env python3
"""Build immutable v7 inputs for the 5-epoch SFT/PT experiment."""
from pathlib import Path
import argparse, hashlib, json, re

def rows(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def immutable_write(p,content):
    if isinstance(content,str):content=content.encode('utf-8')
    if p.exists():
        if p.read_bytes()!=content:raise ValueError(f'Experiment input already exists with different content: {p}')
        return
    p.write_bytes(content)
def name(k):
    q=json.loads(k['fact']['quote'])
    vals=[]
    for key in ('predicate_name','priority_name','index_name','selector_name','procedure_name'):
        if key in q: vals.append(q[key])
    for key in ('mapping','mapping_by_context','category_labels','states'):
        if key in q:
            x=q[key]
            if isinstance(x,dict):
                vals += list(x.values()) if key=='mapping' else [v for m in x.values() for v in (m.values() if isinstance(m,dict) else [m])]
            else: vals += x
    if k['fact']['relation']=='bipartite_matching':
        m=re.match(r'Under ([A-Z][a-z]+),',k['material']['learning']['text']); vals=[m.group(1)]
    vals=list(dict.fromkeys(vals)); assert vals and all(not str(v).startswith('scale1k_') for v in vals),k['id']
    return ', '.join(vals)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=Path('rules_repaired_single_image_v7')); ap.add_argument('--out',type=Path,default=Path('runs/v7_full_experiment/data')); a=ap.parse_args()
    root=a.root.resolve(); out=a.out.resolve(); out.mkdir(parents=True,exist_ok=True)
    audit={x['content']['id']:x['content'] for x in rows(root/'audit/rule_knowledge.jsonl')}; assert len(audit)==1000
    manifest={'dataset':str(root),'rows':1000,'outputs':{},'sources':{},'image_hash':sha(root/'image_hashes.jsonl')}
    for mode,srcname in [('TEXT','train_text.jsonl'),('VISION','train_multimodal.jsonl')]:
        src=root/'shared'/srcname; rr=rows(src); assert len(rr)==1000 and len({x['id'] for x in rr})==1000
        outrows=[]
        for r in rr:
            k=audit[r['id']]; material=r['messages'][0]['content']; q=f'Explain the rule or codebook named {name(k)}, including its conditions, mappings, and how to apply it.'
            z={'id':r['id'],'qid':r['qid']}
            if mode=='VISION':
                assert material.startswith('<image>') and len(r.get('images',[]))==1
                z['images']=[str((root/r['images'][0]).resolve())]; q='<image>Using this learning reference image, '+q[0].lower()+q[1:]; material=material[len('<image>'):]
            z['messages']=[{'role':'user','content':q},{'role':'assistant','content':material}]; outrows.append(z)
        dest=out/f'sft_{mode.lower()}.jsonl'; immutable_write(dest,''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in outrows)); manifest['outputs'][dest.name]={'rows':len(outrows),'sha256':sha(dest)}; manifest['sources'][mode]={'path':str(src),'sha256':sha(src)}
    for mode,srcname in [('TEXT','T2T'),('VISION','V2V')]:
        src=root/srcname/'train.jsonl'; dest=out/f'pt_{mode.lower()}.jsonl'; immutable_write(dest,src.read_bytes()); manifest['outputs'][dest.name]={'rows':1000,'sha256':sha(dest)}; manifest['sources']['PT_'+mode]={'path':str(src),'sha256':sha(src)}
    for d in ('T2T','T2V','V2T','V2V'):
        p=root/d/'eval.jsonl'; rr=rows(p); assert len(rr)==2000 and len({x['id'] for x in rr})==2000; manifest['sources']['eval_'+d]={'path':str(p),'sha256':sha(p)}
    for n in ['audit/eval_labels.jsonl','audit/rule_knowledge.jsonl','audit/sample_audit.jsonl','image_hashes.jsonl']:
        src=root/n;manifest['sources'][n]={'path':str(src),'sha256':sha(src)}
    immutable_write(out/'manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(manifest,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
