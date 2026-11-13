#!/usr/bin/env python3
"""Reproducible paired summary and integrity checks of identity-binding pilots."""
import argparse
import csv
import json
import re
from pathlib import Path

import explore_v7_identity_binding as study
import run_v7_evaluation as ev
from run_v7_full_experiment import rows, read, save, sha


def mcq_metric(group):
    n=len(group);valid=[r for r in group if not r['format_error']]
    correct=sum(bool(r['correct']) for r in group)
    return {'n':n,'correct':correct,'accuracy':correct/n if n else None,
        'valid_n':len(valid),'valid_accuracy':correct/len(valid) if valid else None,
        'format_errors':n-len(valid),'empty':sum(not r['response'].strip() for r in group),
        'truncated':sum(r['finish_reason']=='length' for r in group)}


def summarize(out,complete=False):
    study.verify(out);result={};inventory=[]
    for arm in study.CONTROL_ARMS+study.NEW_ARMS:
        source=out if arm in study.NEW_ARMS else study.SOURCE
        raw=source/arm/'predictions.jsonl'
        neutral=source/'neutral_recall'/arm/'predictions.jsonl'
        diagnostic=out/'diagnostics'/arm/'predictions.jsonl'
        entry={}
        if raw.exists() and (raw.parent/'report.json').exists():
            rr=rows(raw);assert len(rr)==len({r['id'] for r in rr})==208
            lookup={p['id']:p for p in rows(source/'probes.jsonl')+rows(source/arm/'self_probes.jsonl')}
            for r in rr:
                p=lookup[r['id']]
                assert r['input_sha256']==ev.digest({'messages':p['messages'],'images':p.get('images',[])})
                assert r['response']==(r['raw_response']['choices'][0]['message']['content'] or '')
            for c in ('direct_T','direct_V','oracle_T','oracle_V','self_T','self_V'):
                entry[c]=mcq_metric([r for r in rr if r['condition']==c])
            entry['subgroups']={}
            for c in ('direct_T','direct_V'):
                for key in ('task','evaluation_group','relation'):
                    entry['subgroups'][c+':'+key]={v:mcq_metric([r for r in rr if r['condition']==c and r[key]==v]) for v in sorted({r[key] for r in rr if r['condition']==c})}
            # Name-prefix manipulation changes response format: retain raw exact
            # scores but also provide explicit prefix-removed content matching.
            facts=[r for r in rr if r['fmt']=='fact']
            entry['prefix_removed_fact_match']={}
            for condition in ('fact_noimage','fact_image'):
                group=[r for r in facts if r['condition']==condition]
                ncorrect=0
                for r in group:
                    text=re.sub(r'^\s*Rule name\(s\):[^\n]*\n?','',r['response'],count=1,flags=re.I)
                    ncorrect+=study.pilot.short_equal(text,r['gold'],r['fact_key'])
                entry['prefix_removed_fact_match'][condition]={'n':len(group),'matched':ncorrect,
                    'truncated':sum(r['finish_reason']=='length' for r in group),'semantic_accuracy':False}
            inventory.append({'path':str(raw),'rows':len(rr),'sha256':sha(raw),'new':arm in study.NEW_ARMS})
        elif complete:raise AssertionError(str(raw))
        if neutral.exists() and (neutral.parent/'report.json').exists():
            rr=rows(neutral);assert len(rr)==len({r['id'] for r in rr})==40
            lookup={p['id']:p for p in rows(source/'neutral_recall/recall_probes.jsonl')+rows(neutral.parent/'application_probes.jsonl')}
            recalls={r['qid']:r for r in rr if r['condition']=='recall_neutral'}
            for r in rr:
                p=lookup[r['id']]
                assert r['input_sha256']==ev.digest({'messages':p['messages'],'images':p.get('images',[])})
                assert r['response']==(r['raw_response']['choices'][0]['message']['content'] or '')
                if r['condition']!='recall_neutral':
                    assert p['recall_response_sha256']==ev.digest(recalls[r['qid']]['response'])
                    assert p['messages'][-1]['content'].startswith('Your own prior rule recall (uncorrected):\n'+recalls[r['qid']]['response']+'\n\n')
            for c in ('neutral_self_T','neutral_self_V'):entry[c]=mcq_metric([r for r in rr if r['condition']==c])
            inventory.append({'path':str(neutral),'rows':len(rr),'sha256':sha(neutral),'new':arm in study.NEW_ARMS})
        elif complete:raise AssertionError(str(neutral))
        if (diagnostic.parent/'report.json').exists():
            rr=rows(diagnostic);assert len(rr)==len({r['id'] for r in rr})==72
            entry['diagnostics']=read(diagnostic.parent/'report.json')['metrics']
            inventory.append({'path':str(diagnostic),'rows':len(rr),'sha256':sha(diagnostic),'new':True})
        elif complete:raise AssertionError(str(diagnostic))
        result[arm]=entry
    for arm in study.NEW_ARMS:
        cp=read(out/arm/'checkpoint_manifest.json')
        if cp:
            assert cp['global_step']==128 and cp['epoch']==1
            args=read(Path(cp['path'])/'args.json')
            assert args['num_train_epochs']==1 and args['freeze_vit'] and args['freeze_aligner']
            assert args['lora_rank']==8 and args['lora_alpha']==32 and args['lora_dropout']==.05
            assert args['seed']==42 and args['max_length']==8192
            assert args['dataset']==[str(out/(arm+'.jsonl'))]
            assert args['adapters']==[read(study.pilot.PARENT)['path']]
            assert ev.checkpoint(Path(cp['path']))['sha256']==cp['sha256']
        elif complete:raise AssertionError(arm+' checkpoint missing')
    if (out/'visual_transcription/report.json').exists():
        rr=rows(out/'visual_transcription/predictions.jsonl');assert len(rr)==len({r['id'] for r in rr})==64
        result['visual_transcription']=read(out/'visual_transcription/report.json')
        inventory.append({'path':str(out/'visual_transcription/predictions.jsonl'),'rows':64,'sha256':sha(out/'visual_transcription/predictions.jsonl'),'new':True})
    elif complete:raise AssertionError('transcription incomplete')
    if (out/'base_reader/report.json').exists():
        result['base_reader']=read(out/'base_reader/report.json')
        for phase in ('extract','apply'):
            raw=out/'base_reader'/(phase+'_predictions.jsonl')
            rr=rows(raw);assert len(rr)==len({r['id'] for r in rr})==32
            inventory.append({'path':str(raw),'rows':len(rr),'sha256':sha(raw),'new':True})
    elif complete:raise AssertionError('base reader incomplete')
    result['rotations']={}
    for arm in ('warmup','atomic_vision')+study.NEW_ARMS:
        folder=out/'rotations'/arm;raw=folder/'predictions.jsonl'
        if (folder/'report.json').exists():
            rr=rows(raw);assert len(rr)==len({r['id'] for r in rr})==192
            lookup={p['id']:p for p in rows(out/'rotations/probes.jsonl')}
            for r in rr:
                p=lookup[r['id']]
                assert r['input_sha256']==ev.digest({'messages':p['messages'],'images':p.get('images',[])})
                assert r['response']==(r['raw_response']['choices'][0]['message']['content'] or '')
            result['rotations'][arm]=read(folder/'report.json')['metrics']
            original_source=out if arm in study.NEW_ARMS else study.SOURCE
            old={r['id']:r for r in rows(original_source/arm/'predictions.jsonl')}
            flips=[]
            for r in rr:
                if r['shift']==0 and r['condition'].startswith('named_'):
                    prior=old[r['original_id']]
                    assert r['input_sha256']==prior['input_sha256']
                    if prior['response']!=r['response']:
                        flips.append({'id':r['original_id'],'old':prior['response'],'new':r['response'],
                            'old_correct':prior['correct'],'new_correct':r['correct']})
            result['rotations'][arm]['identical_input_regenerations']={'n':32,'response_changes':flips,
                'note':'Temperature0 does not guarantee identical tokens across different inference batch compositions; cause not isolated.'}
            inventory.append({'path':str(raw),'rows':len(rr),'sha256':sha(raw),'new':True})
        elif complete:raise AssertionError('rotation incomplete: '+arm)
    comparisons={}
    base=result['atomic_vision']
    for arm in study.NEW_ARMS:
        comparisons[arm]={}
        if 'direct_T' not in result[arm]:continue
        old={r['id']:r for r in rows(study.SOURCE/'atomic_vision/predictions.jsonl')}
        for condition in ('direct_T','direct_V'):
            rr=[r for r in rows(out/arm/'predictions.jsonl') if r['condition']==condition]
            comparisons[arm][condition]={'wrong_to_right':[r['id'] for r in rr if r['correct'] and not old[r['id']]['correct']],
                'right_to_wrong':[r['id'] for r in rr if not r['correct'] and old[r['id']]['correct']]}
    save(out/'summary.json',{'arms':result,'paired_changes_vs_atomic_vision':comparisons,
        'new_generations':sum(x['rows'] for x in inventory if x['new']),
        'reused_generations':sum(x['rows'] for x in inventory if not x['new']),
        'independence':'8 rule clusters. Rotations, protocols and paired directions are correlated.'})
    save(out/'response_inventory.json',inventory)
    table=[]
    for arm,r in result.items():
        if arm in ('visual_transcription','base_reader','rotations'):continue
        for condition in ('direct_T','direct_V','neutral_self_T','neutral_self_V','oracle_T','oracle_V'):
            if condition in r:table.append({'arm':arm,'condition':condition,**r[condition]})
    if table:
        with (out/'comparison.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    if complete:
        reviews=rows(out/'semantic_reviews.jsonl');assert len(reviews)==64
        for r in reviews:
            source=Path(r['source_file']);assert sha(source)==r['source_sha256']
            original=next(x for x in rows(source) if x['id']==r['id'])
            assert r['response']==original['response'] and ev.digest(r['response'])==r['response_sha256']
        assert sum(x['rows'] for x in inventory if x['new'])==1824
    save(out/'completion_audit.json',{'status':'complete' if complete else 'partial','original_images_unchanged':3000,
        'new_generations':sum(x['rows'] for x in inventory if x['new']),'new_trainings':len(study.NEW_ARMS),
        'new_training_rows':512,'epochs':1,'step':128,'semantic_reviews':64 if complete else None,
        'repair_validated':False,'checked_at':ev.now()})
    print(json.dumps({a:{k:r[k] for k in ('direct_T','direct_V','neutral_self_T','neutral_self_V','diagnostics') if k in r} for a,r in result.items()},ensure_ascii=False,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=study.OUT);ap.add_argument('--complete',action='store_true');a=ap.parse_args();summarize(a.out.resolve(),a.complete)
