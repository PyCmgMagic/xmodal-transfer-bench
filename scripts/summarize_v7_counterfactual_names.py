#!/usr/bin/env python3
"""Audit paired-label construction and report transfer separately from controls."""
import argparse
import csv
import itertools
import json
import re
from collections import Counter
from pathlib import Path

import pilot_v7_counterfactual_names as pilot
import run_v7_evaluation as ev
from run_v7_full_experiment import ROOT, rows, read, save, sha


def metric(group):
    valid=[r for r in group if not r['format_error']];n=len(group);correct=sum(bool(r['correct']) for r in group)
    return {'n':n,'correct':correct,'accuracy':correct/n if n else None,'valid_n':len(valid),
        'valid_accuracy':correct/len(valid) if valid else None,'format_errors':n-len(valid),
        'missing':0,'empty':sum(not r['response'].strip() for r in group),'truncated':sum(r['finish_reason']=='length' for r in group)}


def construction_audit(out):
    pilot.validate(out);pp=rows(out/'probes.jsonl');blocks=rows(out/'blocks.jsonl')
    kk={k['id']:json.loads(k['fact']['quote']) for k in rows(out/'target_knowledge.jsonl')}
    named=[p for p in pp if p['condition'].startswith('named_')]
    labels=rows(out/'dataset/audit/eval_labels.jsonl');keys={r['id']:r for r in labels};assert len(keys)==160
    ids=[]
    for direction in ('T2T','T2V','V2T','V2V'):
        data=rows(out/'dataset'/direction/'eval.jsonl');assert len(data)==160
        ids.append({r['id'] for r in data});assert ids[-1]==set(keys)
        for r in data:
            options=dict(re.findall(r'(?m)^([A-D])\. (.*)$',r['messages'][-1]['content']))
            assert len(options)==len(set(options.values()))==4
            assert keys[r['id']]['answer'] in options
            assert len(r.get('images',[]))==(1 if direction.endswith('V') else 0)
    assert all(x==ids[0] for x in ids)
    assert rows(out/'dataset/T2T/eval.jsonl')==rows(out/'dataset/V2T/eval.jsonl')
    assert rows(out/'dataset/T2V/eval.jsonl')==rows(out/'dataset/V2V/eval.jsonl')
    checked=0
    for block in blocks:
        assert sha(Path(block['image']))==block['image_sha256']
        results={}
        for qid in block['rule_ids']:
            values=[pilot.independent(kk[qid],x) for x in block['structured_observations']]
            results[qid]=pilot.assignment(values)
        assert results==block['rule_results'] and len(set(results.values()))==4
        for modality in ('T','V'):
            for shift in range(4):
                group=[p for p in named if p['block']==block['block'] and p['modality']==modality and p['shift']==shift]
                assert len(group)==4 and {p['gold'] for p in group}==set('ABCD')
                assert len({ev.digest(p.get('images',[])) for p in group})==1
                assert len({p['shared_body_sha256'] for p in group})==1
                option_sets=[]
                for p in group:
                    options=dict(re.findall(r'(?m)^([A-D])\. (.*)$',p['messages'][-1]['content']))
                    option_sets.append(options)
                    expected=results[p['qid']]
                    assert options[p['gold']].split('Result: ')[1].rstrip('.')==expected
                    if block['task']=='diagnosis':
                        assert options[p['gold']].startswith('Accept report.')==(expected==block['submitted_report'])
                    checked+=1
                assert all(x==option_sets[0] for x in option_sets)
                # Any single name-independent prediction matches exactly1of4 keys.
                assert all(sum(p['gold']==answer for p in group)==1 for answer in 'ABCD')
    assert checked==320
    for arm in ('text','vision'):
        train=rows(out/(arm+'.jsonl'));assert len(train)==512
        assert Counter(r['qid'] for r in train)==Counter({qid:64 for qid in kk})
        for r in train:
            assert r['messages'][-1]['content']==pilot.card(next(k for k in rows(out/'target_knowledge.jsonl') if k['id']==r['qid']))
            assert 'Test input:' not in str(r['messages']) and 'Options:' not in str(r['messages'])
            assert len(r.get('images',[]))==(arm=='vision')
            if arm=='vision':assert r['images']==[str(pilot.DATA/'images'/r['qid']/'learning_visual.png')]
    result={'status':'pass','unique_original_test_pngs':5,'paired_blocks':10,'rules':8,'independent_label_checks':checked,
        'directions_same_id_set':True,'labels_per_direction':160,'semantic_questions_per_direction':40,
        'name_agnostic_valid_answer_ceiling':0.25,'all_3000_pngs_unchanged':True,
        'same_observation_options_report_within_block':True,'no_target_tests_in_training':True}
    save(out/'construction_validation.json',result)
    return result


def summarize(out,complete):
    construction=construction_audit(out);summary={};inventory=[];four=[]
    for arm in pilot.ARMS:
        folder=out/arm;raw=folder/'predictions.jsonl'
        if not (folder/'eval_state.json').exists() or read(folder/'eval_state.json').get('status')!='complete':
            if complete:raise AssertionError('incomplete '+arm)
            continue
        rr=rows(raw);assert len(rr)==len({r['id'] for r in rr})==816
        lookup={p['id']:p for p in rows(out/'probes.jsonl')+rows(folder/'self_probes.jsonl')}
        recall={r['qid']:r for r in rr if r['condition']=='recall'}
        for r in rr:
            p=lookup[r['id']]
            assert r['input_sha256']==ev.digest({'messages':p['messages'],'images':p.get('images',[])})
            assert r['response']==(r['raw_response']['choices'][0]['message']['content'] or '')
            if r['condition'].startswith('self_'):
                assert r['recall_response_sha256']==ev.digest(recall[r['qid']]['response'])
                assert p['messages'][-1]['content'].startswith('Your own prior rule recall (uncorrected):\n'+recall[r['qid']]['response']+'\n\n')
        config=read(folder/'eval_manifest.json');assert ev.checkpoint(Path(config['adapter']['path']))==config['adapter']
        if arm!='warmup':
            cp=read(folder/'checkpoint_manifest.json');assert cp['epoch']==1 and cp['global_step']==128
            args=read(Path(cp['path'])/'args.json')
            assert args['num_train_epochs']==1 and args['freeze_vit'] and args['freeze_aligner']
            assert (args['lora_rank'],args['lora_alpha'],args['lora_dropout'],args['seed'],args['max_length'])==(8,32,.05,42,8192)
            assert args['dataset']==[str(out/(arm+'.jsonl'))]
            assert args['adapters']==[read(pilot.PARENT)['path']]
        result={}
        for condition in ('named_T','named_V','self_T','self_V','oracle_T','oracle_V'):
            group=[r for r in rr if r['condition']==condition]
            result[condition]=metric(group)
            result[condition]['shift0']=metric([r for r in group if r['shift']==0])
            result[condition]['by_task']={task:metric([r for r in group if r['task']==task]) for task in ('direct','diagnosis')}
            result[condition]['by_group']={str(g):metric([r for r in group if r['group']==g]) for g in (0,1)}
            result[condition]['by_shift']={str(i):metric([r for r in group if r['shift']==i]) for i in sorted({r['shift'] for r in group})}
            block_groups=[]
            for key in sorted({(r['block'],r['shift']) for r in group}):
                br=[r for r in group if (r['block'],r['shift'])==key];assert len(br)==4
                block_groups.append({'block':key[0],'shift':key[1],'correct':sum(r['correct'] for r in br),
                    'distinct_answers':len({r['predicted'] for r in br if r['predicted']}),
                    'all4_correct':all(r['correct'] for r in br),'one_answer_for_all_names':len({r['predicted'] for r in br})==1})
            result[condition]['paired_blocks']={'n':len(block_groups),'all4_correct':sum(b['all4_correct'] for b in block_groups),
                'all_names_same_answer':sum(b['one_answer_for_all_names'] for b in block_groups),'details':block_groups}
            # Score the unchanged name-switched questions against the previous
            # name's key. This is a derived diagnostic, not extra requests/data.
            wrong_reference=[]
            for key in sorted({(r['block'],r['shift']) for r in group}):
                br=sorted([r for r in group if (r['block'],r['shift'])==key],key=lambda r:r['qid'])
                wrong_reference.extend(br[(i+1)%4]['predicted']==r['gold'] for i,r in enumerate(br))
            result[condition]['cyclic_wrong_name_reference_match']={'n':len(wrong_reference),'matches':sum(wrong_reference),'derived_only':True}
        for modality in ('T','V'):
            anon=[r for r in rr if r['condition']=='anonymous_'+modality]
            valid=sum(not r['format_error'] for r in anon)
            assert len(anon)==40
            result['anonymous_'+modality]={'requests':40,'valid':valid,'format_errors':40-valid,
                'reference_key_comparisons':160,'matches':valid,'reference_match_rate':valid/160,
                'not_task_accuracy':True,'reason':'One generated answer per identical anonymous block tested against four different valid rule keys.'}
        result['recall_outputs']={condition:[{k:r[k] for k in ('id','qid','response','finish_reason')} for r in rr if r['condition']==condition] for condition in ('recall','recall_image')}
        save(folder/'report.json',{'status':'complete','metrics':result})
        summary[arm]=result;inventory.append({'path':str(raw),'rows':816,'sha256':sha(raw),'new_generations':816})
        directions={'T':'T2T','V':'T2V'} if arm=='text' else {'T':'V2T','V':'V2V'} if arm=='vision' else {'T':'baseline_text','V':'baseline_image'}
        for modality,direction in directions.items():
            for protocol in ('named','self','oracle'):
                m=result[protocol+'_'+modality]
                four.append({'arm':arm,'direction':direction,'protocol':protocol,**{k:m[k] for k in ('n','correct','accuracy','valid_n','valid_accuracy','format_errors','missing','empty','truncated')}})
    save(out/'summary.json',{'arms':summary,'construction':construction,'new_generations':sum(x['rows'] for x in inventory),
        'independence':'5original images;10task blocks;8rules; option rotations and rule switches are correlated. Oracle is external-material control.'})
    ev.save_rows(out/'response_inventory.jsonl',inventory);save(out/'four_directions.json',four)
    if four:
        with (out/'four_directions.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(four[0]));writer.writeheader();writer.writerows(four)
    if complete:
        assert len(summary)==3 and sum(x['rows'] for x in inventory)==2448
        save(out/'completion_audit.json',{'status':'complete','new_trainings':2,'epochs':1,'steps':128,'new_generations':2448,
            'image_files_modified':0,'dataset_is_final':False,'checked_at':ev.now()})
    print(json.dumps({arm:{c:{'correct':m['correct'],'n':m['n']} for c,m in r.items() if c.startswith(('named_','self_','oracle_'))} for arm,r in summary.items()},indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=pilot.OUT);ap.add_argument('--complete',action='store_true');a=ap.parse_args();summarize(a.out.resolve(),a.complete)
