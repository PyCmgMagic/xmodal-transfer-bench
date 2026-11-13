#!/usr/bin/env python3
"""Persist explicit human semantic decisions after reading all 64 recall outputs."""
from collections import Counter
import re
from explore_v7_identity_binding import OUT, SOURCE, NEW_ARMS
import run_v7_evaluation as ev
from run_v7_full_experiment import rows, save, sha

CANONICAL_REASONS={
    'name_first_vision':[
        '年龄规则，非五边形或白色筛选。','年龄与收入虚构分类，非运输箱且排除损坏。',
        '年龄单位换算且重复，非正方形或红色筛选。','年龄虚构区间，非[2,6)。',
        '年龄身高体重，非红equipment cart比例0.4/0.65。','三个attribute>=100，非占用区域比例0.4/0.8。',
        '年龄/身高阈值，非填格比例0.3/0.85。','名称相除及0.85，非红五边形比例0.15/0.65。'],
    'name_blind_vision':[
        'attribute<1000重复，非五边形或白色。','雪区域计数<3，非运输箱且排除损坏。',
        '雪区域计数<3，非正方形或红色。','区域计数<3，非[2,6)。',
        'attribute<1000重复，非红equipment cart比例。','三个区域计数及3，非比例0.4/0.8。',
        '1000/5000等重复阈值，非填格比例0.3/0.85。','字符串替换后区域计数<3，非红五边形比例。']}
NEUTRAL_REASONS={
    'name_first_vision':{
        '0173':'运输方式和地区条件，非[2,6)。','0284':'all条件与类别名互指，缺比例及阈值映射。',
        '0689':'Moscow/St.Petersburg地区，非运输箱及排除损坏。','0753':'value>=0重复类别，非正确比例/阈值。',
        '0954':'四个方位地区，非五边形或白色。','0957':'region0.65，非正方形或红色。',
        '0968':'年龄/年份归一化，非填格比例。','0973':'checkmarks及3/6，非占用区域比例0.4/0.8。'},
    'name_blind_vision':{
        '0173':'checkmark比例及0.2/0.8，非[2,6)。','0284':'年龄各单位换算重复，缺红五边形比例及阈值。',
        '0689':'年龄段分类，非运输箱及排除损坏。','0753':'类别名求和Checksum及100/200，非红equipment cart比例。',
        '0954':'区域数>=3，非五边形或白色。','0957':'年龄多区间，非正方形或红色。',
        '0968':'身高/年龄/态度组合，非填格比例。','0973':'年龄>=18且身高>=1.6，非占用区域比例。'}}


def main():
    knowledge=rows(SOURCE/'target_knowledge.jsonl');order=[k['id'] for k in knowledge]
    gold={k['id']:k['fact']['quote'] for k in knowledge};reviews=[]
    for arm in NEW_ARMS:
        files=[OUT/arm/'predictions.jsonl',OUT/'neutral_recall'/arm/'predictions.jsonl']
        for f in files:
            for r in rows(f):
                if r['condition'] not in ('recall_canonical','recall_task','recall_image','recall_neutral'):continue
                qid=r['qid'];condition=r['condition'];i=order.index(qid)
                correct=condition=='recall_image'
                if correct:
                    reason=('原图条件、组合或区间/比例、阈值、类别和边界归属均正确；允许省略题中已提供的名称和冗余说明。'
                        if '0173' not in qid else 'value>=2且value<6，语义上为完整[2,6)；省略模板套话不影响事实正确。')
                elif condition=='recall_canonical':reason=CANONICAL_REASONS[arm][i]
                elif condition=='recall_neutral':reason=NEUTRAL_REASONS[arm][qid.split('_')[1]]
                else:reason='只报KEEP、缺参数/组合，或虚构另一套条件/阈值；不是目标完整规则。'
                reviews.append({'arm':arm,'id':r['id'],'qid':qid,'condition':condition,
                    'core_facts_correct':correct,'whole_response_consistent':correct,'reason_zh':reason,
                    'response':r['response'],'finish_reason':r['finish_reason'],'expected_structured_rule':gold[qid],
                    'source_file':str(f),'source_sha256':sha(f),'response_sha256':ev.digest(r['response']),
                    'method':'Human semantic inspection; explicit saved decision, not inferred from literal equality.'})
    assert len(reviews)==64
    ev.save_rows(OUT/'semantic_reviews.jsonl',reviews)
    summary={}
    for arm in NEW_ARMS:
        summary[arm]={}
        for condition in ('recall_canonical','recall_task','recall_neutral','recall_image'):
            rr=[r for r in reviews if r['arm']==arm and r['condition']==condition]
            assert len(rr)==8
            summary[arm][condition]={'n':8,'core_correct':sum(r['core_facts_correct'] for r in rr),
                'consistent_correct':sum(r['whole_response_consistent'] for r in rr),
                'truncated':sum(r['finish_reason']=='length' for r in rr)}
    save(OUT/'semantic_review_summary.json',summary)
    save(OUT/'design_correction.json',{'correction':'The manifest contrast description says original images still show their names. This is not true for every dataset identifier. Training/evaluation code and data are unaffected. Preserve the original design and record this correction explicitly.',
        'verified_examples':[{'qid':'scale1k_0689_ai_generated','image_title':'REQUIRE + EXCLUDE','dataset_identifier':'Romnot'},
                             {'qid':'scale1k_0173_ai_generated','image_title':'ACCEPTED RANGE','dataset_identifier':'Lolnen'}],
        'interpretation':'name_blind assigns the dataset identifier to the original image through supervised output; it is not always OCR of an identifier printed in pixels. No-image facts were not used as training inputs.',
        'no_source_modification':True})


if __name__=='__main__':main()
