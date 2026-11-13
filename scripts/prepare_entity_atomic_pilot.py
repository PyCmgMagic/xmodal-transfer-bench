#!/usr/bin/env python3
"""Learning-only atomic supervision and independent synthetic retention data."""
import argparse
from collections import Counter
import hashlib
from pathlib import Path
import random
import re

from entityqa_common import digest, now, rows, save, sha, write_rows

SEED = 'atomic-pilot-v1-seed42:'
MONTH = '(?:January|February|March|April|May|June|July|August|September|October|November|December)'
SLOT_PATTERNS = [
    rf'\b\d{{1,2}} {MONTH} \d{{4}}\b', rf'\b{MONTH} \d{{1,2}},? \d{{4}}\b',
    rf'\b{MONTH} \d{{4}}\b', r'\$\d+(?:\.\d+)?(?:\s+(?:million|billion))?',
    r'\b\d{4}[–/\-]\d{2,4}\b', r'\b\d+(?:\.\d+)?(?:st|nd|rd|th)?\b',
    r'\b(?:once|twice|three times|two-year|one-year|four-year|three-year)\b',
]


def slots(claim):
    found=[]
    for pattern in SLOT_PATTERNS:
        for m in re.finditer(pattern,claim):
            if any(m.start()<b and m.end()>a for a,b,_ in found):
                continue
            found.append((m.start(),m.end(),m[0]))
    return sorted(found)


def entity_examples(record, qa):
    """Short claim rather than biography; answer each original attribute and date."""
    name,claim=record['name'],record['claim']
    samples=[
        ('fact_0',f'What specific fact have you learned about {name}? Give only the key factual statement.',claim),
        ('fact_1',f'Recall the learned event, result, or attribute of {name}. Be precise and concise.',claim),
        ('attribute_0',qa['question_text'],qa['answer']),
        ('attribute_1','Give a short factual answer to this question: '+qa['question_text'],qa['answer']),
    ]
    for i,(start,end,answer) in enumerate(slots(claim)):
        masked=claim[:start]+'[BLANK]'+claim[end:]
        samples.extend([
            (f'slot_{i}_0',f'Complete the learned fact about {name}. Return only the missing text.\n{masked}',answer),
            (f'slot_{i}_1',f'Which text replaces [BLANK] in this learned statement about {name}?\n{masked}',answer),
        ])
    return [dict(id=record['id']+'-'+kind,knowledge_id=record['id'],qid=record['qid'],variant=kind,
        messages=[{'role':'user','content':question},{'role':'assistant','content':answer}])
        for kind,question,answer in samples]


def generic(index, split):
    """Self-contained tasks unrelated to any entity or benchmark fact."""
    rng=random.Random((1 if split=='train' else 100000)+index)
    label='ABCD'[index%4]
    family=(index//4)%3
    if family==0:
        a,b=rng.randrange(10,80),rng.randrange(10,80)
        value=a+b
        wrong=[value-3,value+4,value+9]
        choices=wrong[:]
        choices.insert('ABCD'.index(label),value)
        question=f'What is {a} + {b}?'
        reason=f'{a} + {b} = {value}.'
    elif family==1:
        value=rng.randrange(100,900)
        choices=[value-17,value+21,value+39]
        choices.insert('ABCD'.index(label),value)
        code=f'SAMPLE-{split.upper()}-{index:04d}'
        question=f'The provided record says sample {code} has a mass of {value} grams. What mass does the record give?'
        reason=f'The record gives {value} grams.'
    else:
        keys=['amber','cyan','lime','violet','silver','indigo','coral','white']
        selected=rng.sample(keys,4)
        choices=selected[:]
        value=choices['ABCD'.index(label)]
        question=f'A synthetic tag is explicitly assigned the color {value}. Which color is assigned to the tag?'
        reason=f'The assigned color is {value}.'
    question+='\n'+'\n'.join(f'{letter}. {choice}' for letter,choice in zip('ABCD',choices))
    question+='\n\nReason briefly. End with the final option letter in \\boxed{}. Do not overthink.'
    return dict(id=f'generic-{split}-{index}',family=family,answer=label,
        messages=[{'role':'user','content':question},
                  {'role':'assistant','content':reason+'\n\\boxed{'+label+'}'}])


def prepare(args):
    source,work=(getattr(args,k).resolve() for k in ['source','work'])
    records=rows(source/'data/learning_inputs.jsonl')
    qas={r['id']:r for r in rows(args.qa_source/'data/qa_pairs.jsonl')}
    selected=sorted(records,key=lambda r:hashlib.sha256((SEED+r['id']).encode()).hexdigest())[:32]
    assert len(selected)==len({r['qid'] for r in selected})==32
    entity=[]
    for r in selected:
        qa=qas[r['id']]
        assert qa['source_sha256']==digest(r)
        assert qa['answer'] in r['learning']
        samples=entity_examples(r,qa)
        assert all(s['messages'][1]['content'].strip() for s in samples)
        # Both factual claims and all derived slot targets originate here only.
        assert samples[0]['messages'][1]['content']==r['claim']
        entity.extend(samples)
    retention=[generic(i,'train') for i in range(4*max(12,round(len(entity)/16)))]
    dev=[generic(i,'dev') for i in range(96)]
    assert set(r['messages'][0]['content'] for r in retention).isdisjoint(r['messages'][0]['content'] for r in dev)
    for arm,extras in [('atomic',[]),('atomic_replay',retention)]:
        arm_dir=work/arm/'data'
        write_rows(arm_dir/'train_text.jsonl',entity+extras)
    write_rows(work/'data/learning_inputs.jsonl',selected)
    write_rows(work/'data/entity_training.jsonl',entity)
    write_rows(work/'data/retention_training.jsonl',retention)
    write_rows(work/'data/retention_dev.jsonl',dev)
    # Only original learning-only attribute questions and a new neutral recall
    # prompt are used for diagnostics; formal test questions are not read here.
    recall=[]
    for r in selected:
        qa=qas[r['id']]
        recall.extend([
            dict(id=r['id']+'-attribute',qid=r['qid'],kind='trained_attribute_recall',
                 messages=[{'role':'user','content':qa['question_text']}],expected=qa['answer']),
            dict(id=r['id']+'-neutral',qid=r['qid'],kind='neutral_claim_recall',
                 messages=[{'role':'user','content':f'What is the learned information concerning {r["name"]}? Answer concisely with the precise facts.'}],
                 expected=r['claim'],required_slots=[s[2] for s in slots(r['claim'])]),
        ])
    write_rows(work/'data/recall_dev.jsonl',recall)
    save(work/'data/manifest.json',dict(version='atomic-retention-pilot-v1',created_at=now(),
        knowledge_units=32,selection='First 32 SHA256('+SEED+' + knowledge ID); no score-based selection',
        development_qids=[r['qid'] for r in selected],
        excluded_from_future_confirmatory_evaluation=True,
        entity_rows=len(entity),retention_rows=len(retention),retention_dev_rows=len(dev),
        training_sources=['Original learning-only target claim','Previous learning-only attribute QA',
                          'Deterministic independent synthetic instruction tasks'],
        test_data_used_for_training=False,background_paragraph_targets=False,
        modality='TEXT first-stage pilot only; VISION training awaits evidence',
        prior_source_manifest_sha256=sha(source/'data/manifest.json'),
        qa_source_manifest_sha256=sha(args.qa_source/'data/manifest.json'),
        selection_sha256=digest([r['id'] for r in selected]),
        files={str(p.relative_to(work)):sha(p) for p in sorted(work.rglob('*.jsonl'))},
        variant_counts=dict(Counter(r['variant'] for r in entity)),
        caveats=['Claims inherit candidate-source validity; no claim that all facts are pretraining-unknown.',
                 'Cloze questions reveal other claim fields and are training aids, not closed-book evaluation.',
                 'Generic replay adds training rows/tokens; compare as an intervention, not equal-compute ablation.',
                 'Trained attribute recall is a memorization diagnostic, not held-out generalization.']))
    print(dict(knowledge_units=32,entity_rows=len(entity),retention_rows=len(retention)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['source','qa-source','work']:
        parser.add_argument('--'+name,type=Path,required=True)
    prepare(parser.parse_args())
