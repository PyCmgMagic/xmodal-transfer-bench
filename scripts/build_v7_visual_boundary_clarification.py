#!/usr/bin/env python3
"""One inspected learning-caption clarification, original PNG unchanged.

This is an untrained development candidate, not a released full dataset version.
"""
import copy
from pathlib import Path

import run_v7_evaluation as ev
from run_v7_full_experiment import ROOT, rows, save, sha

OUT=ROOT/'runs/v7_application_protocol_repair_20261019/data_candidate'
DATA=ROOT/'rules_repaired_single_image_v7'
QID='scale1k_0771_ai_generated'
OLD='The image also displays two reference cases that establish boundary shares—compute these the same way.'
NEW=('Below the isolated target object are two separate reference trays, one on the left and one on the right. '
     'Each complete tray is one reference collection: count every compartment across BOTH rows of that tray. '
     'The entire LEFT tray defines the lower boundary share; the entire RIGHT tray defines the upper boundary share. '
     'The isolated target object above the trays is only a type reference and is not part of either boundary collection. '
     'Compute the two boundary shares from those complete trays in the same way as a new collection.')


def main():
    OUT.mkdir(exist_ok=True)
    source=DATA/'shared/train_multimodal.jsonl'
    old=next(r for r in rows(source) if r['id']==QID)
    assert OLD in old['messages'][0]['content']
    changed=copy.deepcopy(old)
    changed['messages'][0]['content']=changed['messages'][0]['content'].replace(OLD,NEW)
    assert len(changed['images'])==1
    image=(DATA/changed['images'][0]).resolve()
    original_hash=sha(image)
    changed['images']=[str(image)]
    ev.save_rows(OUT/'clarified_visual_material.jsonl',[changed])
    train=next(r for r in rows(ROOT/'runs/v7_fullparam_sft_ep10_20261019/data/sft_vision.jsonl') if r['id']==QID)
    sft=copy.deepcopy(train)
    assert OLD in sft['messages'][-1]['content']
    sft['messages'][-1]['content']=sft['messages'][-1]['content'].replace(OLD,NEW)
    ev.save_rows(OUT/'clarified_sft_vision.jsonl',[sft])
    save(OUT/'manifest.json',{'created_at':ev.now(),'status':'prepared_not_trained_not_released',
        'scope':'One inspected visual learning caption only; original TEXT material, evaluation questions/options/labels unchanged.',
        'qid':QID,'old_fragment':OLD,'new_fragment':NEW,'source':str(source),'source_sha256':sha(source),
        'image':str(image),'original_image_sha256':original_hash,'image_unchanged':sha(image)==original_hash,
        'reviewer':'Codex full-resolution inspection; no independent human adjudication',
        'full_resolution_facts':{'left_tray_target_objects':2,'left_tray_total_cells':8,'right_tray_target_objects':5,'right_tray_total_cells':8},
        'caption_numeric_thresholds_added':False,
        'corrected_observation':'The original picture contains TWO 8-cell trays, not four independent collections. An earlier montage-based four-collection suspicion was retracted before any patch was applied.',
        'limits':'Clarifies collection extent and boundary referents; does not demonstrate improved memory or closed-book accuracy.',
        'script_sha256':sha(Path(__file__)),'files':{p.name:sha(p) for p in OUT.glob('*.jsonl')}})
    print('Prepared one caption clarification; no original image/data files changed.')


if __name__=='__main__':main()
