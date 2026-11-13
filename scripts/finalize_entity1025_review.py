"""Record human adjudications from the 2026-11-03 read-only entity1025 review.

The image indices refer to the saved unattributed_image_index.json, not dataset
row numbers. Content screening is not face recognition or an identity approval.
"""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/entity_merged2022_position_annotated'
OUT = ROOT / 'reports/entity1025_review_20261103/full_audit'


def rows(path):
    with path.open(encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save_rows(path, records):
    with path.open('w', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')


def main():
    frozen_inputs = {
        OUT/'unattributed_image_index.json': '20e12f21aae5d6833b27b8beec3771ef629fdfe31d59909619baebbf37a9439f',
        DATA/'source/questions.jsonl': '4777c7b669f031891c92b860e39e5c618e4e4d0ac066fe0a6c5522abf738b6b0',
        DATA/'source/learning.jsonl': '54dac5bbd627402cd02929c477a53cf2c501c23720f71eb81d4b0d2cc601615f',
        DATA/'source/answers.jsonl': '5577f50513ad9dcceec2e2075dca0d5bdcd37a277401d9b9bb186c2e5d2efcb4',
        DATA/'audit/image_roles.jsonl': '197ec7f95fad72dea483a561b7de601a4236de380080f58f9f359393eb7b8c9c',
    }
    for path, expected in frozen_inputs.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen review input changed; re-review before reusing decisions: {path}')
    entities = {x['qid']: x for x in rows(DATA / 'source/entities.jsonl')}
    learning = {x['qid']: x['text'] for x in rows(DATA / 'source/learning.jsonl')}
    evidence = {x['qid']: x for x in rows(DATA / 'source/evidence.jsonl')}
    questions = {(x['qid'], x['type']): x for x in rows(DATA / 'source/questions.jsonl')}
    answers = {x['qid']: x['answers'] for x in rows(DATA / 'source/answers.jsonl')}
    catalog = rows(DATA / 'audit/image_catalog.jsonl')
    candidates = json.loads((OUT / 'unattributed_image_index.json').read_text())
    findings = rows(OUT / 'findings.jsonl')
    adjudications = []

    def record(qid, category, verdict, reason, **extra):
        adjudications.append(dict(qid=qid, name=entities[qid]['name'], category=category,
                                  verdict=verdict, reason_zh=reason,
                                  reviewer='manual_current_session', **extra))

    # Clearly stylized fictional/game figures, toys and cartoon collections.
    # Conservative omissions: ordinary paintings, caricature portraits, and
    # photographic film stills are not classified here from face appearance.
    fictional = [8,16,107,115,159,197,222,232,247,249,250,253,254,299,347,364,
                 410,413,414,416,419,420,423,424,425,426,427,428,429,431,432,
                 433,434,436,437,438,442,443,444,445,446,447,449,450,451,
                 452,453,454,455,456,457,458,459,463,464,465,467,469,470,
                 507,513,514,521,525,526,583,589,592,593,598,643,679,700,
                 721,782,783,845,861,895,922,1033,1036,1037,1046,1053,
                 1054,1055,1066,1078,1084,1089,1093,1095,1103,1104,1113,
                 1148,1198,1200,1204,1214,1239,1262,1304,1352,1390,1476,1615]
    assert len(fictional) == len(set(fictional))
    for index in fictional:
        x = candidates[index]
        assert entities[x['qid']]['kind'] == 'person'
        record(x['qid'], 'fictional_or_game_target_image', 'confirmed_content_issue',
               '现实人物单元配用动漫、游戏、玩具或虚构插画；不能将其当作该人物的真实识别图。'
               '依据图片内容类别及实体页面，不根据人脸判断身份。',
               index=index, scope=x['role'], file=x['file'], source_url=x['source_url'],
               entity_url=entities[x['qid']]['wikipedia_url'])

    # These files contain no photographic target person: background, document,
    # object or diagrams. Some may be related works; that does not depict a person.
    nonportraits = {
        141: '神像物体图，未呈现人物实体', 142: '书封，无目标人物肖像',
        223: '渐变色背景，无目标人物', 388: '技术公式/图表，无目标人物',
        497: '网页界面截图，无可识别目标人物', 498: '中文说明文字截图，无目标人物',
        500: '软件/游戏界面截图，人物区域不可用于实体识别',
        561: '小说书封，无目标人物肖像', 599: '中文书籍商品封面，无目标人物肖像',
        732: '游戏/球员统计界面，未呈现可识别的目标人物',
        759: '书籍封面，无目标人物肖像', 838: '球衣商品图，无目标人物',
        839: '小说封面，无目标人物肖像', 840: '文字书封，无目标人物肖像',
        960: '抽象/装饰画，无可识别目标人物', 1213: '图形海报，无目标人物肖像',
        1567: '哲学书封，无目标人物肖像', 1608: '环形连线图，无目标人物',
    }
    assert not set(nonportraits).intersection(fictional)
    for index, reason in nonportraits.items():
        x = candidates[index]
        assert entities[x['qid']]['kind'] == 'person'
        record(x['qid'], 'nonidentifying_image_content', 'confirmed_content_issue', reason,
               index=index, scope=x['role'], file=x['file'], source_url=x['source_url'])

    # Finn and Jakub use an exactly matching avatar crop. The two other
    # similarly styled sketches differ, so their identity remains unresolved.
    for index in [979,1006]:
        x = candidates[index]
        record(x['qid'], 'generic_placeholder_avatar', 'confirmed_content_issue',
               'Finn Surman和Jakub Pešek人物卡复用完全相同的线描头像；'
               'RGB裁切(120,205,310,445)逐像素一致。不同姓名不能使同一头像区分两位实体。',
               index=index, scope=x['role'], file=x['file'], source_url=x['source_url'])
    for index in [994,1008]:
        x = candidates[index]
        record(x['qid'], 'illustrated_portrait_requires_source', 'requires_source_confirmation',
               '同风格线描头像，但与另外两张的裁切并非逐像素相同；'
               '缺少来源，不能直接断言为通用占位头像，单列待核验。',
               index=index, scope=x['role'], file=x['file'], source_url=x['source_url'])
    for x in catalog:
        if 'pantheon.world/' in x['source_url']:
            record(x['qid'], 'printed_name_card', 'confirmed_protocol_risk',
                   '已目视核对的人物卡直接印有姓名、职业及出生年份；可经OCR提供身份线索。'
                   '48张是已查到的此来源卡片，不是全量OCR检出的总数。',
                   scope=x['role'], file=x['file'], source_url=x['source_url'])

    bad_conversion = {
        'Q495501': '旅游区专名Mount Kumgang Tourist Region被截断替换。',
        'Q16232342': '节目名Katie Pavlich Tonight被替换成图片指代，学习及正确选项均受损。',
        'Q17124693': '完整姓名正确选项Martin Simon Schirdewan被删除。',
        'Q231058': '出生全名正确选项Shailene Diann Woodley被删除。',
        'Q4974474': '干扰项姓名被整体替换；正确答案仍保留，不应误报为正确标签错误。',
        'Q30335340': '签名文件名专有字符串中的姓名被替换。',
        'Q63164076': '出生全名正确选项被删除，其他候选姓名也被截断。',
        'Q9152187': '完整姓名正确选项Andrzej Witold Halicki被删除。',
        'Q53909401': '完整姓名正确选项Jacob Nathaniel Elordi被删除。',
        'Q20972384': '完整姓名正确选项Sherel Constancio Floranus被删除。',
        'Q2641691': '干扰项节目名Almorzando con Mirtha Legrand被改坏；gold不是此项。',
        'Q442547': '全名Stephen Graham Kelly变成图片指代加Kelly，学习句也被改坏。',
        'Q3372924': '所问绰号Don Pablo从学习和正确选项中被删除。',
        'Q12873934': '姓名干扰项被截断为Apostolos G.加图片指代；gold不是此项。',
        'Q1984757': '出生全名正确选项被删除，其他候选姓名也被截断。',
        'Q7683243': '把父亲Koduvathara L. James的James替换成女儿的图片指代，改错实体。',
        'Q2187156': '完整姓名正确选项Antje Anna Helena Hazekamp被删除。',
        'Q5811691': '完整姓名正确选项Mark Maria Hubertus Flekken被删除。',
    }
    neutral_conversion = {
        'Q3018091': '四个选项均对同一目标人物作正常共指替换，核心区分仍是体检时间；本标记不判错误。',
        'Q51754461': 'Jones指向目标人物，改为图片指代仍保留censure等行动区分；本标记不判错误。',
        'Q4767729': '目标姓名位于干扰项，改成图片共指可理解；不判gold受损，仍建议无图对照检验。',
    }
    rewrites = [x for x in findings if x['category'] == 'direct_option_rewrite']
    assert len(rewrites) == 23
    for x in rewrites:
        qid = x['qid']
        extra = {k:x[k] for k in ['sample_id','gold','text_options','visual_options']}
        if qid in bad_conversion:
            record(qid, 'visual_proper_name_corruption', 'confirmed_data_error',
                   bad_conversion[qid], scope='direct', finding_id=x['finding_id'], **extra)
        elif qid in neutral_conversion:
            record(qid, 'visual_conversion_review', 'candidate_rejected',
                   neutral_conversion[qid], scope='direct', finding_id=x['finding_id'], **extra)
        else:
            assert qid in {'Q512678','Q61899299'}
            record(qid, 'visual_identity_option_shortcut', 'confirmed_protocol_risk',
                   '问谁时仅目标选项改为图中的人，其他候选仍是姓名；答案身份提示显式且四方向题意不再等价。',
                   scope='direct', finding_id=x['finding_id'], **extra)

    # Concrete missing attributes, not an aggregate error verdict on the 167
    # lexical mismatches (many of which are sound paraphrases).
    for qid, reason in {
        'Q30335340': '题目考签名文件的精确文件名；学习段只说信息框含签名，未提供文件名。',
        'Q1774046': '题目考讲座标题Does Democracy Need to be More Populist?；学习段只提供会议名与争议，未给讲座标题。',
        'Q575733': '题目要求International Secretary of the Conservative Party这一确切职务；学习仅写new international role，不能唯一确定职务。',
    }.items():
        q = questions[qid,'direct']
        gold = answers[qid]['direct']
        record(qid, 'tested_attribute_missing_from_learning', 'confirmed_data_error', reason,
               scope='direct', learning=learning[qid], question=q['question'],
               gold_text=next(x['text'] for x in q['options'] if x['id'] == gold))
    record('Q30335340', 'answer_evidence_disagreement', 'confirmed_data_error',
           'source/evidence写Signature Danière Obono.svg，测试正确项写Signature Danièle Obono.svg；'
           '精确文件名测试的来源、答案不一致，未查询在线页面裁决哪个文件真实存在。',
           scope='direct_and_statement', evidence=evidence['Q30335340']['new_knowledge'])
    record('Q286466', 'question_year_shortcut', 'confirmed_protocol_risk',
           '题干1994–1999；仅gold的日期同时符合两个年份，其他三项均移到其他年份，不需学习精确日期。',
           scope='direct_and_statement', question=questions['Q286466','direct'])
    record('Q42101', 'nonexclusive_statement_options', 'confirmed_data_error',
           'statement以演员/制片人/导演/牧师等可同时为真的职业子集作互斥答案；'
           '信息框删减描述不能证明其他职业为假。direct问文章如何描述，不能照搬同一判定。',
           scope='statement', question=questions['Q42101','statement'])
    record('Q60143', 'overlapping_birth_year_options', 'requires_review',
           '出生年候选集合1987或1988，与1986或1987、1988或1989交叠；'
           '应明确问资料给出的完整不确定区间，或重写为互斥命题。', scope='direct_and_statement')
    record('Q106917152', 'historical_nonexclusive_options', 'historical_verdict_rejected',
           '历史把两个都包含Mary的干扰项视为两个正确父母答案。实际gold母亲是Norma，Mary是妹妹；'
           '两个相似但错误的干扰项不造成多个正确答案。此结构误报撤销，首次公开时间仍未认证。',
           scope='direct_and_statement', learning=learning['Q106917152'],
           questions=[questions['Q106917152',t] for t in ['direct','statement']])
    record('Q101246164', 'image_entity_timeline_conflict', 'requires_source_confirmation',
           '实体链接为1999年出生的Oleksandr Romanchuk，但配图活动类别为2010/2011年Metalist成年足球队活动；'
           '高度疑似同名人物混淆，须以来源图注核验，不以人脸确认。', scope='learning_and_evaluation')
    for qid in ['Q435089','Q49034']:
        record(qid, 'same_photograph_different_qids', 'confirmed_protocol_risk',
               '两个不同QID的学习图呈现同一张历史人物照片（裁切/压缩不同）；'
               '不能同时充当两个不同人物的身份锚点。本次不凭人脸认定照片到底属于谁。',
               scope='learning', related_qids=['Q435089','Q49034'])

    for i, record_ in enumerate(adjudications,1):
        record_['adjudication_id'] = f'A{i:04d}'
    save_rows(OUT / 'adjudications.jsonl', adjudications)
    confirmed = [x for x in adjudications if x['verdict'] in
                 {'confirmed_content_issue','confirmed_data_error','confirmed_protocol_risk'}]
    save_rows(OUT / 'confirmed_issues.jsonl', confirmed)
    by_qid = defaultdict(list)
    for x in adjudications:
        by_qid[x['qid']].append(x)
    hist = {x['qid']:x for x in rows(OUT/'history/knowledge/unit_audit.jsonl')}
    followup = {x['qid']:x for x in rows(OUT/'history/pending/unit_ledger.jsonl')}
    new_candidates = {x['qid']:x for x in rows(OUT/'new450_temporal_candidates.jsonl')}
    confirmed_qids = {x['qid'] for x in confirmed}
    per_unit=[]
    for qid, en in entities.items():
        has_error = any(x['verdict'] in {'confirmed_content_issue','confirmed_data_error'} for x in by_qid[qid])
        action = '修复或隔离' if has_error else '补充核验'
        # All units still lack a completed semantic identity approval; do not
        # produce or label the complement of this list as clean.
        per_unit.append(dict(qid=qid,name=en['name'],action=action,
                             confirmed_issue_ids=';'.join(x['adjudication_id'] for x in by_qid[qid] if x in confirmed),
                             categories=';'.join(sorted({x['category'] for x in by_qid[qid]})),
                             historical_tier=hist[qid]['audit_tier'] if qid in hist else 'new450',
                             historical_followup=followup[qid]['partition'] if qid in followup else '',
                             temporal_candidate=new_candidates[qid]['category'] if qid in new_candidates else '',
                             first_public_date=evidence[qid]['new_knowledge'].get('first_public_date') or '',
                             semantic_approved=False))
    with (OUT/'unit_recommendations.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(per_unit[0]))
        writer.writeheader();writer.writerows(per_unit)
    summary=dict(
        review_scope='All 4044 images fully decoded; 1622 unattributed thumbnails and 92 perceptual candidate thumbnails viewed. Not all person identities or external facts verified.',
        categories={cat:dict(records=len(xs),units=len({x['qid'] for x in xs}),
                             verdicts=dict(Counter(x['verdict'] for x in xs)))
                    for cat in sorted({x['category'] for x in adjudications})
                    for xs in [[x for x in adjudications if x['category']==cat]]},
        confirmed_observations_and_protocol_risks_units=len(confirmed_qids),
        units_requiring_repair_or_isolation=sum(x['action']=='修复或隔离' for x in per_unit),
        repair_units_by_source=dict(Counter('new450' if x['historical_tier']=='new450' else 'existing1572'
                                           for x in per_unit if x['action']=='修复或隔离')),
        fictional_image_roles=len(fictional),nonidentifying_image_roles=len(nonportraits),
        image_content_problem_roles=len(fictional)+len(nonportraits)+2,
        image_content_problem_units=len({x['qid'] for x in adjudications if x['verdict']=='confirmed_content_issue'}),
        original_dataset_modified=False,
        note='Counts overlap. This is a lower bound from specified screening, not a certified total error rate. Historical labels retain their original provenance; the Robert Dowd structural verdict is explicitly rejected.')
    (OUT/'adjudication_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')

    selected=[8,107,197,437,783,979,1006,498]
    canvas=Image.new('RGB',(1200,500),'#eeeeee');draw=ImageDraw.Draw(canvas)
    for i,index in enumerate(selected):
        x=candidates[index];xx=(i%4)*300;yy=(i//4)*250
        with Image.open(DATA/x['file']) as im:
            thumb=ImageOps.contain(im.convert('RGB'),(290,195));canvas.paste(thumb,(xx+(300-thumb.width)//2,yy))
        draw.text((xx+4,yy+199),x['qid']+' '+x['role'],fill='black')
        draw.text((xx+4,yy+216),entities[x['qid']]['name'].encode('ascii','replace').decode()[:37],fill='black')
        draw.text((xx+4,yy+232),'audit index '+str(index),fill='black')
    canvas.save(OUT/'confirmed_image_examples.jpg',quality=90)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
