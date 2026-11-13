#!/usr/bin/env python3
"""Persist the manually inspected semantic errors with source/response hashes."""
import json
from pathlib import Path

from diagnose_v7_knowledge_access import PILOT
from run_v7_full_experiment import ROOT, rows, save, sha
from run_v7_evaluation import digest, save_rows


def run():
    out=ROOT/'runs/v7_access_diagnostics_20261015';knowledge={k['id']:k for k in rows(PILOT/'target_knowledge.jsonl')};reviews=[]
    def add(r,source,category,note,complete=False):
        reviews.append({'source':str(source),'source_sha256':sha(source),'id':r['id'],'qid':r['qid'],'condition':r.get('condition','original_training_prompt_free_recall'),'review_method':'agent_manual_semantic_review','category':category,'fully_correct_requested_rule':complete,'notes_zh':note,'response_sha256':digest(r['response']),'expected_rule':json.loads(knowledge[r['qid']]['fact']['quote']),'response':r['response']})
    source=out/'text_alias/recall_context/predictions.jsonl'
    codes={
        ('heading_observation','scale1k_0090_programmatic'):('wrong_rule_facts','阈值应0.1/0.55，回答0.25/0.75。'),
        ('heading_observation','scale1k_0165_ai_generated'):('incomplete_rule','需要bucket、plastic、white同时满足；仅把plastic作为保留条件，未说明white要求。'),
        ('heading_observation','scale1k_0175_ai_generated'):('wrong_rule_facts','应tool tray/plastic/red至少满足2项；回答无具体属性的三项并混入value>=3和all。'),
        ('heading_observation','scale1k_0762_ai_generated'):('wrong_rule_facts','应white sample tray占比、阈值0.35/0.85；回答marked_cells与0.15/0.75。'),
        ('heading_observation','scale1k_0964_programmatic'):('wrong_rule_facts','应[4,14)，回答[10,15)。'),
        ('heading_observation','scale1k_0972_programmatic'):('wrong_rule_facts','应0.15/0.85，回答0.4/0.65。'),
        ('heading_observation_options','scale1k_0165_ai_generated'):('correct_rule_paraphrase','以不同措辞完整保留bucket/plastic/white及all组合；字面不相同仍是正确规则。'),
        ('heading_observation_options','scale1k_0175_ai_generated'):('wrong_rule_facts','应tool tray/plastic/red；回答bowl/wood/red。至少2项组合本身正确。'),
        ('heading_observation_options','scale1k_0964_programmatic'):('wrong_rule_facts','应[4,14)，回答[4,15)。'),
        ('heading_observation_options','scale1k_0972_programmatic'):('wrong_rule_facts','应0.15/0.85，回答0.4/0.65。'),
    }
    for r in rows(source):
        if r['correct_exact']:add(r,source,'correct_rule_exact','整段规则与正确紧凑材料相同。',True)
        elif (r['condition'],r['qid']) in codes:
            category,note=codes[(r['condition'],r['qid'])];add(r,source,category,note,category=='correct_rule_paraphrase')
        else:
            assert r['condition'] in ('heading_options','heading_observation_options')
            assert 'Result:' in r['response']
            add(r,source,'answered_case_instead_of_rule','回答选项或测试结果，没有按要求回忆规则；不把这种指令偏移直接记成规则事实错误。')
    trained={
        'scale1k_0090_programmatic':('wrong_rule_facts','应marked_cells/total_cells、0.1/0.55；改成occupied_regions及0.25/0.65。'),
        'scale1k_0165_ai_generated':('wrong_rule_family','应bucket/plastic/white集合筛选；回答形状加权计分。'),
        'scale1k_0175_ai_generated':('wrong_rule_family','应2-of-3集合筛选；回答形状优先级排序。'),
        'scale1k_0178_ai_generated':('wrong_rule_family','应marked_cells占比、0.4/0.65；回答三种形状到名称的代码映射。'),
        'scale1k_0704_ai_generated':('wrong_rule_family','应区间[3,5]筛选；回答形状加权计分。'),
        'scale1k_0762_ai_generated':('wrong_rule_facts','应white sample tray比例及0.35/0.85；回答red sample bag及0.25/0.55。'),
        'scale1k_0964_programmatic':('wrong_rule_family','应[4,14)筛选；回答形状加权计分。'),
        'scale1k_0972_programmatic':('wrong_rule_facts','occupied-region比例定义基本对，但阈值应0.15/0.85，回答0.45/0.7。'),
    }
    source=out/'full_sft_text/teacher_forcing/free_recall.jsonl'
    for r in rows(source):
        category,note=trained[r['qid']];assert r['finish_reason']!='length';add(r,source,category,note)
    neutral={
        'scale1k_0175_ai_generated':('wrong_rule_family','应集合筛选；回答marker三分类并混入记录约定。'),
        'scale1k_0704_ai_generated':('wrong_rule_family','应区间[3,5]；回答protective covering/storage box计数分类。'),
        'scale1k_0165_ai_generated':('wrong_rule_family','应bucket/plastic/white筛选；回答最左侧形状分类。'),
        'scale1k_0964_programmatic':('wrong_rule_family','应区间[4,14)；回答red protective cover比例分类。'),
        'scale1k_0090_programmatic':('wrong_rule_facts','应marked_cells与0.1/0.55；回答blue-and-striped cells及0.25/0.75。'),
        'scale1k_0972_programmatic':('wrong_rule_facts','应occupied regions与0.15/0.85；回答white sample bags及0.25/0.75。'),
        'scale1k_0762_ai_generated':('wrong_rule_facts','应white sample trays与0.35/0.85；回答blue/striped cells与0.25/0.75。'),
        'scale1k_0178_ai_generated':('wrong_rule_facts','应0.4/0.65，回答0.25/0.75。'),
    }
    source=out/'full_sft_text/predictions.jsonl'
    for r in rows(source):
        if r['condition']!='recall_image_absent':continue
        category,note=neutral[r['qid']];add(r,source,category,note)
    source=out/'vision/predictions.jsonl'
    for r in rows(source):
        if r['condition']!='recall_image_wrong_learning':continue
        if r['image_donor_body_exact']:
            add(r,source,'followed_wrong_image_exact_body','名称未改，除名称前缀之外的完整规则逐字等于另一张学习图的规则；不是被问的规则。')
        else:
            assert r['qid']=='scale1k_0090_programmatic'
            add(r,source,'followed_wrong_image_parameters','回答occupied_regions/all_regions和0.15/0.85，均来自被换入的0972图；标签拼写断开，故未算逐字完整匹配。')
    assert len(reviews)==56
    save_rows(out/'semantic_reviews.jsonl',reviews)
    summary={}
    for condition in ('heading','heading_observation','heading_options','heading_observation_options'):
        rr=[r for r in reviews if r['condition']==condition]
        summary[condition]={'n':len(rr),'fully_correct_rule':sum(r['fully_correct_requested_rule'] for r in rr),'categories':{c:sum(r['category']==c for r in rr) for c in sorted({r['category'] for r in rr})}}
    save(out/'semantic_review_summary.json',{'status':'complete','reviewed_outputs':56,'recall_context':summary,'original_training_prompt_free_recall':{'n':8,'fully_correct':0,'wrong_family':5,'wrong_facts':3,'no_truncation':True},'scope':'Manual judgments on fixed observed outputs; exact response/source hashes retained. Unrequested case answers are classified separately from factual errors.','script_sha256':sha(Path(__file__))})
    print('Saved56 semantic reviews; context correct rules:',[(c,m['fully_correct_rule']) for c,m in summary.items()])


if __name__=='__main__':run()
