#!/usr/bin/env python3
"""Build a declared development subset after learning-only training is frozen."""
import argparse
import copy
import json
from pathlib import Path

from answer_readout import options_from_question
from entityqa_common import rows,save,sha,write_rows,eval_payload


def prepare(args):
    work,data=args.work.resolve(),args.data.resolve()
    manifest=json.loads((work/'data/manifest.json').read_text())
    for name,h in manifest['files'].items():assert sha(work/name)==h,name
    qids=set(manifest['development_qids'])
    selected_ids={r['id'] for r in rows(work/'data/learning_inputs.jsonl')}
    labels={r['id']:r for r in rows(data/'audit/eval_labels.jsonl')}
    dest=work/'evaluation_data'
    all_labels={}
    outputs={}
    for modality,filename in [('T','eval_text.jsonl'),('V','eval_multimodal.jsonl')]:
        records=[r for r in rows(data/'shared'/filename) if r['qid'] in qids]
        assert len(records)==64
        selected=[]
        for row in records:
            item=copy.deepcopy(row)
            if item.get('images'):
                item['images']=[str((data/p).resolve()) for p in item['images']]
            selected.append(item)
            all_labels[item['id']]=labels[item['id']]
        # Eight preselected development entities, all four option rotations,
        # direct questions only; no content is used for training.
        rotation_ids=sorted(selected_ids)[:8]
        for row in records:
            if row['id'].removesuffix('-direct') not in rotation_ids or not row['id'].endswith('-direct'):
                continue
            question=row['messages'][-1]['content']
            opts=options_from_question(question)
            assert len(opts)==4
            before=question[:question.index('\nA. ')]
            suffix=question[question.index('\n\n',question.index('\nD. ')):]
            for shift in range(4):
                item=copy.deepcopy(row)
                item['id']=row['id']+f'-rotation-{shift}'
                letters='ABCD'
                reordered=letters[shift:]+letters[:shift]
                item['messages'][-1]['content']=before+'\n'+'\n'.join(
                    f'{new}. {opts[old]}' for new,old in zip(letters,reordered))+suffix
                if item.get('images'):
                    item['images']=[str((data/p).resolve()) for p in item['images']]
                selected.append(item)
                label=copy.deepcopy(labels[row['id']])
                label.update(id=item['id'],probe_id='rotation',answer=letters[reordered.index(label['answer'])])
                all_labels[item['id']]=label
        assert len(selected)==96
        outputs[modality]=selected
        write_rows(dest/'shared'/filename,selected)
    write_rows(dest/'audit/eval_labels.jsonl',list(all_labels.values()))
    save(dest/'package_inventory.json',dict(version='atomic-pilot-dev-v1',knowledge_units=32,
        text_questions=96,vision_questions=96,development_only=True,
        source_inventory_sha256=sha(data/'package_inventory.json'),
        training_manifest_sha256=sha(work/'data/manifest.json'),
        sha256={str(p.relative_to(dest)):sha(p) for p in dest.rglob('*.jsonl')}))
    # Reuse only unchanged original base generations. Rotations are new tasks.
    base={ (r['id'],r['modality']):r for r in rows(args.base_predictions) }
    reusable=[]
    for modality,selected in outputs.items():
        for r in selected:
            key=(r['id'],modality)
            if key in base:
                from entityqa_common import digest
                assert base[key]['input_sha256']==digest(eval_payload(r,dest))
                reusable.append(base[key])
    assert len(reusable)==128
    write_rows(work/'data/base_original_dev_predictions.jsonl',reusable)
    print(dict(mcq_tasks_per_model=192,reusable_original_base_tasks=128))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['work','data','base-predictions']:p.add_argument('--'+name,type=Path,required=True)
    prepare(p.parse_args())
