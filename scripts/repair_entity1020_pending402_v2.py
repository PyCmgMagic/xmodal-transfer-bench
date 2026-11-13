"""Follow-up repairs and source-scoped adjudications; never approve novelty."""
from __future__ import annotations

import copy
import json
import random
import re
import shutil
from collections import Counter
from pathlib import Path

import repair_entity1020_pending402 as v1

ROOT = v1.ROOT
OLD = ROOT / 'entity1020/pending402_question_repair_v1'
DEST = ROOT / 'entity1020/pending402_question_repair_v2'
FOLLOW = ROOT / 'diagnostics/entity1020_pending402_followup_20261020'
REVIEW = v1.REVIEW
SOURCE = v1.SOURCE
rows, write, save, sha = v1.rows, v1.write, v1.save, v1.sha

# Retire a target, not the entity: a genuinely new supported fact can be collected later.
DECISIONS = {
117: ('retire_current_target', '出生日期涉及新旧历换算及正文/信息框冲突；不据一个字段改动认证正确日期。需权威传记重新取证。'),
150: ('historical_correction_excluded', '改问任命日，避免等同任期起始日。2025-12-03 旧版本正文已经明确写出 2013-03-12 任命；归档历史题。'),
354: ('retire_current_target', '原证据文件名是 Danière，题目答案写 Danièle；图片文件名编辑也不适合作为人物语义新知识。'),
359: ('question_repaired_still_quarantined', '改问获奖影片，四项均为本人参演影片，去掉表演与配乐/编剧奖项的职业提示。完整新修订支持 Weapons，但尚未取得独立获奖报道。'),
363: ('retire_current_target', '所给旧片段已经说 2010 年取得法国国籍；改成 French journalist 没有提供新的国籍事实。这里只确认构造不新颖，不声称已独立找到首次公开来源。'),
420: ('needs_external_evidence_and_reconstruction', '原奖项选项有位置提示；现有学习材料的世界杯获奖及决赛断言尚无可用官方实质内容支持。不把错误网页外壳当 FIFA 证据，暂不另编题。'),
523: ('needs_external_evidence_and_reconstruction', '数字欧元主题可提示 ECON 委员会。改问政策项目仍需确认其他选项不同时成立，现有材料不足。'),
580: ('retire_current_target', '目标只是 UCI 团队模板代码变化，涉及 2019–2021 历史经历；不能直接当语义新事实。模板别名等价关系尚未独立证实。'),
590: ('needs_external_evidence_and_reconstruction', '可考虑问获奖影片，但当前未取到独立获奖来源；原奖项类别有职业提示。所谓第七位也需明确男女演员统计口径，不直接照抄。'),
591: ('question_repaired_still_quarantined', '改问截至 2026-08-23、2026–27 赛季的联赛出场次数。完整新修订统计表明确为 3，去掉国家/联赛归属提示；仍待独立赛季记录支持。'),
650: ('needs_external_evidence_and_reconstruction', '所在俱乐部本是旧知识；需取到明确的 2025–26 Team of the Season 实质名单，再选可区分的新事实。'),
898: ('needs_external_evidence_and_reconstruction', 'U23 与预备队可能同义；改成年月题要有 dated news 支持。猜测的 ESPN 链接实际是他人新闻，不纳入证据。'),
963: ('retire_current_target', '思想条目列表扩展不等于思想首次出现，且多个选项可能都是其概念。应另找定义/原文及公开日期重新构造。'),
990: ('needs_external_evidence_and_reconstruction', '2023 年事件的立场题带常识倾向；需原报道原话及日期。改成四个近义谴责措辞也不能保证单选唯一。'),
1028: ('retire_current_target', 'BA 与 MA 可同时成立；用本科限定虽消歧，但 Cambridge 的学位惯例又会提供提示。当前目标不适合直接修为新知识单选。'),
1162: ('needs_external_evidence_and_reconstruction', '实体/照片是 Layla Moran，原题考 Paulette Hamilton。需要以代任关系重构，先核实日期和适当干扰人物，不能只换一个主语。'),
1219: ('retire_current_target', '旧片段已给 2021 年当选联邦议院，新片段补第 20 届，存在制度年表推断捷径。猜测的官方链接只返回档案首页，不能当人物支持材料。'),
1243: ('historical_correction_excluded', '改问 2024 年当选 Dublin Fingal East，避免 since 2025 的任期口径错误；2025-08-26 旧正文已有明确当选事实。'),
1272: ('question_repaired_still_quarantined', '只问完整 given names，答案 Marie Ernestine Emma，去掉把婚后姓拼进 birth name 的错误。完整新修订支持姓名，但首次公开日期未知。'),
1359: ('retire_current_target', 'Le Grill 的星级快照缺少日期，且与 Alain Ducasse 的实体归属待核实；不能把 4 星干扰项换掉就视为修好。'),
1467: ('retire_current_target', '原选项含不适用赛事；换四个真实赛事可能多项成立。2022 参赛事件也需先核实公开时间，不因 2026 页面新增而认作新知识。'),
}

SPECS = {
150: dict(values=['11 March 2013', '12 March 2013', '13 March 2013', '14 March 2013'], gold='12 March 2013',
    direct='On what date was {subject} appointed caretaker Prime Minister of Bulgaria?',
    sentence='{subject} was appointed caretaker Prime Minister of Bulgaria on {value}.',
    learning='{subject} was appointed caretaker Prime Minister of Bulgaria on 12 March 2013.', event='2013-03-12',
    support='old', terms=['appointed', '12 March 2013']),
359: dict(values=['Weapons', 'Twice in a Lifetime', 'Field of Dreams', 'Gone Baby Gone'], gold='Weapons',
    direct='For which film did {subject} win the Academy Award for Best Supporting Actress in 2026?',
    sentence='{subject} won the Academy Award for Best Supporting Actress in 2026 for {value}.',
    learning='{subject} won the Academy Award for Best Supporting Actress in 2026 for Weapons.', event='2026-03-15',
    support='new_revision', terms=['won 40 years later', 'Weapons']),
591: dict(values=['3', '6', '9', '12'], gold='3',
    direct='As of 23 August 2026, how many league appearances had {subject} made for Lommel in the 2026–27 season?',
    sentence='As of 23 August 2026, {subject} had made {value} league appearances for Lommel in the 2026–27 season.',
    learning='As of 23 August 2026, {subject} had made 3 league appearances for Lommel in the 2026–27 season.', event=None,
    support='new_revision', terms=['2026–27', '23 August 2026']),
1243: dict(values=['2022', '2023', '2024', '2025'], gold='2024',
    direct='In which year was {subject} elected a TD for Dublin Fingal East?',
    sentence='{subject} was elected a TD for Dublin Fingal East in {value}.',
    learning='{subject} was elected a TD for Dublin Fingal East at the 2024 general election.', event='2024',
    support='old', terms=['2024 general election', 'Dublin Fingal East']),
1272: dict(values=['Marie Elisabeth Emma', 'Marie Ernestine Emma', 'Marie Ernestine Anne', 'Marie Josephine Emma'], gold='Marie Ernestine Emma',
    direct='What are the full given names of {subject}?',
    sentence='The full given names of {subject} are {value}.',
    learning='The full given names of {subject} are Marie Ernestine Emma.', event=None,
    support='new_revision', terms=['Marie Ernestine Emma']),
}


def evidence(i, spec):
    if spec['support'] == 'old':
        prefix = REVIEW / 'history' / f'{i}_old'
    else:
        prefix = FOLLOW / 'sources' / f'{i}_new_revision'
    meta = json.loads(prefix.with_suffix('.json').read_text())
    assert meta['status'] == 200
    assert sha(prefix.with_suffix('.html')) == meta['sha256']
    lines = prefix.with_suffix('.txt').read_text().split('\n')
    quotes = [dict(line=n, text=l) for n, l in enumerate(lines, 1)
              if all(t in l for t in spec['terms'])]
    if i == 591:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(prefix.with_suffix('.html').read_text(), 'html.parser')
        tables = [tr.get_text(' | ', strip=True) for tr in soup.select('tr') if '2026–27' in tr.get_text()]
        assert tables == ['2026–27 | Belgian Pro League | 3 | 0 | 0 | 0 | — | — | 3 | 0']
        dates = [dict(line=n,text=l) for n,l in enumerate(lines,1) if 'As of match played 23 August 2026' in l]
        assert dates
        quotes = [dict(table_row=tables[0])] + dates
    assert quotes, (i, spec['terms'])
    return dict(url=meta.get('source_url', meta.get('url')), fetch_url=meta.get('fetch_url', meta.get('url')),
                raw_path=str(prefix.with_suffix('.html')), raw_sha256=meta['sha256'],
                text_path=str(prefix.with_suffix('.txt')), text_sha256=sha(prefix.with_suffix('.txt')),
                quotes=quotes, scope=('dated_pre_cutoff_revision' if spec['support']=='old' else 'post_cutoff_revision_context_only'),
                public_by=meta.get('timestamp_meta'), proves_first_publication=False)


def generate_additions(units):
    eval_original = {d: {r['id']: r for r in rows(SOURCE / d / 'eval.jsonl')} for d in ('T2T', 'T2V')}
    source_records = {n: rows(SOURCE / f'source/{n}.jsonl') for n in ('entities','learning','evidence')}
    supports = {}
    for i, spec in SPECS.items():
        u=units[i]; uid=u['unit_id']; supports[i]=evidence(i,spec)
        subset='historical_corrections' if i in (150,1243) else 'candidate'
        base=DEST/subset
        for name, rr in source_records.items():
            out=rows(base/f'source/{name}.jsonl')
            record=copy.deepcopy(next(r for r in rr if (r['qid']==u['qid'] if name=='entities' else r['id']==uid)))
            if name=='learning': record['text']=spec['learning'].format(subject=u['name'])
            if name=='evidence':
                original=copy.deepcopy(record)
                write(base/'provenance/original_evidence.jsonl', rows(base/'provenance/original_evidence.jsonl')+[original])
                k=copy.deepcopy(record['new_knowledge'])
                k.update(claim=spec['learning'].format(subject=u['name']), event_date=spec['event'],
                         first_public_date=None, first_public_evidence='',
                         source_url=supports[i]['url'], new_evidence=json.dumps(supports[i]['quotes'],ensure_ascii=False),
                         old_evidence='', difference=DECISIONS[i][1])
                record=dict(id=uid,qid=u['qid'],new_knowledge=k,dated_support=[supports[i]],
                    original_record='provenance/original_evidence.jsonl',review=dict(status='invalidated_after_question_repair',
                        temporal_novelty_approved=False,option_uniqueness_approved=False,reason=DECISIONS[i][1]))
            if name!='entities' or not any(r['qid']==u['qid'] for r in out):out.append(record)
            write(base/f'source/{name}.jsonl',out)
        qq=[]; aa=dict(id=uid,qid=u['qid'],answers={}); patches=[]
        for typ in v1.TYPES:
            vals=spec['values']; letter=v1.LETTERS[vals.index(spec['gold'])];aa['answers'][typ]=letter
            stem=spec['direct'].format(subject=u['name']) if typ=='direct' else 'Which statement is accurate?'
            opts=vals if typ=='direct' else [spec['sentence'].format(subject=u['name'],value=x) for x in vals]
            q=dict(id=uid,qid=u['qid'],type=typ,question=stem,options=[dict(id=l,text=x) for l,x in zip(v1.LETTERS,opts)])
            qq.append(q)
            patches.append(dict(index=i,unit_id=uid,qid=u['qid'],name=u['name'],type=typ,subset=subset,
                reason_zh=DECISIONS[i][1],before=u['questions'][typ],after=q,new_answer=letter,
                temporal_novelty_approved=False,distractor_truth_exhaustively_verified=False))
        write(base/'source/questions.jsonl',rows(base/'source/questions.jsonl')+qq)
        write(base/'source/answers.jsonl',rows(base/'source/answers.jsonl')+[aa])
        write(base/'audit/patches.jsonl',rows(base/'audit/patches.jsonl')+patches)
        for name in ('eval_labels','sample_audit'):
            rr=[copy.deepcopy(r) for r in rows(SOURCE/f'audit/{name}.jsonl') if (r['id'].rsplit('-',1)[0]==uid if name=='eval_labels' else r['unit_id']==uid)]
            for r in rr:
                r['answer']=aa['answers'][r['probe_id']]
                if name=='sample_audit':
                    r.update(source_automatic_passed=r.get('automatic_passed'),automatic_passed=False,
                             question_repair_candidate=True,temporal_novelty_approved=False)
            write(base/f'audit/{name}.jsonl',rows(base/f'audit/{name}.jsonl')+rr)
        for d in v1.DIRECTIONS:
            train=copy.deepcopy(next(r for r in rows(SOURCE/d/'train.jsonl') if r['id']==uid))
            visual=d.startswith('V');subject='the person shown in the image' if visual else u['name']
            train['messages'][0]['content']=('<image>\n' if visual else '')+spec['learning'].format(subject=subject)
            if visual:train['images']=[str(SOURCE/p) for p in train['images']]
            write(base/d/'train.jsonl',rows(base/d/'train.jsonl')+[train])
            ev=[]
            for typ in v1.TYPES:
                r=copy.deepcopy(eval_original['T2V' if d.endswith('V') else 'T2T'][f'{uid}-{typ}'])
                visual=d.endswith('V');subject='the person shown in the image' if visual else u['name']
                stem=spec['direct'].format(subject=subject) if typ=='direct' else 'Which statement is accurate?'
                opts=spec['values'] if typ=='direct' else [spec['sentence'].format(subject=subject,value=x) for x in spec['values']]
                stem=('<image>\n' if visual else '')+stem
                r['messages'][-1]['content']=v1.render(stem,opts)
                if visual:r['images']=[str(SOURCE/p) for p in r['images']]
                ev.append(r)
            write(base/d/'eval.jsonl',rows(base/d/'eval.jsonl')+ev)
    return supports


def rebalance():
    rng=random.Random(42)
    for subset in ('candidate','historical_corrections'):
        base=DEST/subset; aa=rows(base/'source/answers.jsonl');ids=sorted(r['id'] for r in aa)
        amap={r['id']:r for r in aa}; perms={}
        for typ in v1.TYPES:
            targets=[v1.LETTERS[j%4] for j in range(len(ids))];rng.shuffle(targets)
            for uid,target in zip(ids,targets):
                gold=v1.LETTERS.index(amap[uid]['answers'][typ]);p=[j for j in range(4) if j!=gold];rng.shuffle(p)
                p.insert(v1.LETTERS.index(target),gold);perms[uid,typ]=p;amap[uid]['answers'][typ]=target
        write(base/'source/answers.jsonl',aa)
        qq=rows(base/'source/questions.jsonl')
        for q in qq:q['options']=[dict(id=l,text=q['options'][j]['text']) for l,j in zip(v1.LETTERS,perms[q['id'],q['type']])]
        write(base/'source/questions.jsonl',qq);qmap={(q['id'],q['type']):q for q in qq}
        for d in v1.DIRECTIONS:
            ev=rows(base/d/'eval.jsonl')
            for r in ev:
                uid,typ=r['id'].rsplit('-',1);stem,opts=v1.split_prompt(r['messages'][-1]['content'])
                r['messages'][-1]['content']=v1.render(stem,[opts[j] for j in perms[uid,typ]])
            write(base/d/'eval.jsonl',ev)
        for name in ('eval_labels','sample_audit','patches'):
            rr=rows(base/f'audit/{name}.jsonl')
            for r in rr:
                if name=='patches':
                    uid,typ=r['unit_id'],r['type'];r.update(after=qmap[uid,typ],new_answer=amap[uid]['answers'][typ],
                        v2_position_to_previous_index=perms[uid,typ])
                else:
                    uid=r['unit_id'] if name=='sample_audit' else r['id'].rsplit('-',1)[0]
                    r['answer']=amap[uid]['answers'][r['probe_id']]
            write(base/f'audit/{name}.jsonl',rr)


def supplemental_source_notes():
    prefix=FOLLOW/'sources/1243_oireachtas_rest'
    meta=json.loads(prefix.with_suffix('.json').read_text())
    assert meta['status']==200 and sha(prefix.with_suffix('.html'))==meta['sha256']
    data=json.loads(prefix.with_suffix('.html').read_text())
    member=next(r['member'] for r in data['results'] if r['member']['fullName']=="Darragh O'Brien")
    membership=next(r['membership'] for r in member['memberships'] if r['membership']['house']['houseNo']=='34')
    assert membership['dateRange']['start']=='2024-11-29'
    assert membership['represents'][0]['represent']['showAs']=='Dublin Fingal East'
    record=dict(index=1243,url=meta['url'],raw_path=str(prefix.with_suffix('.html')),raw_sha256=meta['sha256'],
        member_code=member['memberCode'],membership=membership,
        scope='Current official API corroborates 2024 representation; dated old revision separately proves pre-cutoff availability.',
        proves_first_publication=False)
    # ISBN arithmetic is a falsifiable check, not a general quality certificate.
    u=next(u for u in rows(REVIEW/'units.jsonl') if u['index']==499)
    check=[]
    for typ,q in u['questions'].items():
        for o in q['options']:
            number=re.search(r'97[89](?:[\d-]){10,16}',o['text'])[0].replace('-','')
            valid=len(number)==13 and sum(int(v)*(1 if j%2==0 else 3) for j,v in enumerate(number))%10==0
            assert valid
            check.append(dict(type=typ,option=o['id'],isbn=number,checksum_valid=valid))
    save(FOLLOW/'supplemental_checks.json',dict(oireachtas_corroboration=record,isbn_499=check,
         isbn_conclusion='All four alternatives have valid checksums; this specific shortcut is unsupported.'))


def verify():
    m=json.loads((DEST/'manifest.json').read_text())
    for group in ('original_inputs','v1_inputs','followup_inputs'):
        for p,digest in m[group].items():assert sha(Path(p))==digest,p
    disp=rows(DEST/'dispositions_402.jsonl');assert len(disp)==len({r['unit_id'] for r in disp})==402
    result={}
    for subset in ('candidate','historical_corrections'):
        base=DEST/subset;aa={r['id']:r for r in rows(base/'source/answers.jsonl')}
        qq={(r['id'],r['type']):r for r in rows(base/'source/questions.jsonl')}
        ll={r['id']:r for r in rows(base/'audit/eval_labels.jsonl')}
        audits={r['sample_id']:r for r in rows(base/'audit/sample_audit.jsonl')}
        assert len(qq)==len(ll)==len(audits)==2*len(aa)
        balance={}
        for typ in v1.TYPES:
            c=Counter(r['answers'][typ] for r in aa.values());assert max(c.get(l,0) for l in v1.LETTERS)-min(c.get(l,0) for l in v1.LETTERS)<=1
            balance[typ]=dict(c)
        for r in rows(base/'source/evidence.jsonl'):assert not r['review']['temporal_novelty_approved']
        learn={r['id']:r['text'] for r in rows(base/'source/learning.jsonl')}
        for d in v1.DIRECTIONS:
            tr=rows(base/d/'train.jsonl');ev=rows(base/d/'eval.jsonl')
            assert {r['id'] for r in tr}==set(aa) and len(tr)==len(aa)
            assert {r['id'] for r in ev}==set(ll) and len(ev)==2*len(aa)
            for phase, rr in [('train',tr),('eval',ev)]:
                visual=d.startswith('V') if phase=='train' else d.endswith('V')
                for r in rr:
                    assert not set(r)&{'answer','gold','labels','response'}
                    assert sum(x['content'].count('<image>') for x in r['messages'])==int(visual)
                    assert len(r.get('images',[]))==int(visual)
                    for p in r.get('images',[]):assert Path(p).is_file()
            for r in tr:
                if d.startswith('T'):assert r['messages'][0]['content']==learn[r['id']]
            for r in ev:
                uid,typ=r['id'].rsplit('-',1);stem,opts=v1.split_prompt(r['messages'][-1]['content'])
                assert ll[r['id']]['answer']==audits[r['id']]['answer']==aa[uid]['answers'][typ]
                assert not audits[r['id']]['automatic_passed']
                assert len(opts)==len(set(opts))==4
                if d.endswith('T'):assert stem==qq[uid,typ]['question'] and opts==[o['text'] for o in qq[uid,typ]['options']]
            for phase in ('train','eval'):
                twin=(d[0]+'2'+('V' if d[-1]=='T' else 'T')) if phase=='train' else (('V' if d[0]=='T' else 'T')+d[1:])
                assert rows(base/d/f'{phase}.jsonl')==rows(base/twin/f'{phase}.jsonl')
            # Visual train/eval pictures must remain different for every unit.
            if d=='V2V':
                im={r['id']:r['images'][0] for r in tr}
                for r in ev:assert r['images'][0]!=im[r['id'].rsplit('-',1)[0]]
        for i in SPECS:
            p=[r for r in rows(base/'audit/patches.jsonl') if r['index']==i]
            for patch in p:
                uid=patch['unit_id'];typ=patch['type'];gold=aa[uid]['answers'][typ]
                text=next(o['text'] for o in qq[uid,typ]['options'] if o['id']==gold)
                assert SPECS[i]['gold'] in text
        result[subset]=dict(units=len(aa),questions=len(qq),answer_counts=balance)
    save(DEST/'validation.json',dict(status='passed',structural_checks_only=True,source_and_v1_hashes_unchanged=True,
        four_directions_aligned=True,train_eval_image_paths_distinct=True,labels_and_questions_aligned=True,
        all_402_dispositions_recorded=True,semantic_and_temporal_approval=False,subsets=result))
    print(json.dumps(result,ensure_ascii=False,indent=2))


def build():
    assert not DEST.exists(), f'Will not overwrite {DEST}'
    previous=json.loads((OLD/'manifest.json').read_text())
    assert json.loads((OLD/'validation.json').read_text())['status']=='passed'
    frozen={str(p):sha(p) for p in OLD.rglob('*') if p.is_file()}
    for p,d in previous['original_inputs'].items():assert sha(Path(p))==d
    units={u['index']:u for u in rows(REVIEW/'units.jsonl')}
    pending=rows(OLD/'dispositions_402.jsonl')
    assert {r['index'] for r in pending if r['action']=='needs_fact_or_scope_reconstruction'}==set(DECISIONS)
    DEST.mkdir()
    for subset in ('candidate','historical_corrections'):shutil.copytree(OLD/subset,DEST/subset)
    supports=generate_additions(units);rebalance()
    judged=[]
    for i,(action,reason) in DECISIONS.items():
        u=units[i];entry=dict(index=i,unit_id=u['unit_id'],name=u['name'],action=action,reason_zh=reason,
            evidence_scope='source construction review; not independent proof of first publication',
            old_fragment=u['new_knowledge']['old_evidence'],new_fragment=u['new_knowledge']['new_evidence'],
            approved_for_post_cutoff_experiment=False)
        if i in supports:entry.update(support=supports[i],evidence_scope=supports[i]['scope'])
        judged.append(entry)
    write(FOLLOW/'reconstruction_21_decisions.jsonl',judged)
    # Count discrepancy: do not silently change 6 -> 5 because honorary awards differ.
    meta=json.loads((FOLLOW/'sources/128_emmy.json').read_text())
    assert meta['status']==200 and sha(FOLLOW/'sources/128_emmy.html')==meta['sha256']
    lines=(FOLLOW/'sources/128_emmy.txt').read_text().split('\n')
    assert lines[505:509]==['21','Nominations','5','Emmys']
    extra=dict(index=128,name=units[128]['name'],action='quarantine_source_count_discrepancy',
        reason_zh='官方当前页面列 21 Nominations / 5 Emmys / 2 Honors，而原断言为 22 次提名、6 次获奖。须区分竞争性奖与荣誉，不能直接判六次必错或认证为新增获奖。',
        url=meta['final_url'],raw_sha256=meta['sha256'],raw_path=str(FOLLOW/'sources/128_emmy.html'),
        quotes=[dict(line=n,text=lines[n-1]) for n in range(505,512)],
        proves_pre_cutoff_publication=False,approved_for_post_cutoff_experiment=False)
    save(FOLLOW/'additional_quality_finding.json',extra)
    supplemental_source_notes()
    for r in pending:
        if r['index'] in DECISIONS:
            action,reason=DECISIONS[r['index']];r.update(action=action,reason_zh=reason)
        elif r['index']==128:r.update(action=extra['action'],reason_zh=extra['reason_zh'])
    write(DEST/'dispositions_402.jsonl',pending)
    inputs={str(p):sha(p) for p in FOLLOW.rglob('*') if p.is_file() and p.suffix not in ('.pyc',)}
    manifest=dict(previous)
    manifest.update(schema='entity1020-pending402-question-repair-v2',previous_version=str(OLD),
        repaired_units=25,repaired_questions=50,candidate_units=21,historical_correction_units=4,
        actions=dict(Counter(r['action'] for r in pending)),v1_inputs=frozen,followup_inputs=inputs,
        reconstruction_reviewed=21,added_candidate_indices=[359,591,1272],added_historical_indices=[150,1243],
        limitations=previous['limitations']+['New revision context is not independent first-publication evidence.',
            'Retired targets require new fact collection, not mechanical option replacement.'])
    save(DEST/'manifest.json',manifest)
    report(pending,judged,extra)
    verify()


def report(pending, judged, extra):
    counts=Counter(r['action'] for r in pending)
    names={'question_repaired_still_quarantined':'已修题，继续隔离','historical_correction_excluded':'历史纠错归档',
        'exclude_pre_cutoff':'此前已有截止日前支持，排除','retire_current_target':'停用当前事实目标，需重新采集',
        'needs_external_evidence_and_reconstruction':'已复查，缺外部证据或不能保证单选唯一',
        'quarantine_pending_evidence':'其余来源/时间仍待补证','quarantine_source_count_discrepancy':'新增来源计数口径冲突'}
    text=['# 待审 402 条：第二轮修复与来源复查','','截止日：2025-12-31T23:59:59Z。保留 direct / statement 两种四选一和四个跨模态方向。',
        '原始 1572 条、旧评测和 v1 均未改动。没有重跑模型，也没有按准确率筛选。','','## 本轮结果','',
        '上一版剩余的 21 个重构项已逐条给出处置：3 个补成候选题、2 个纠正为历史题、9 个停用当前目标、7 个仍需外部来源和重构。',
        '另在待补证项中发现 Debbie Allen 奖项计数口径冲突，单列隔离。其余待补证项没有因此自动通过。','',
        '| 402 条的当前互斥状态 | 单位数 |','|---|---:|']
    text += [f'| {names[a]} | {n} |' for a,n in counts.items()]
    text += ['', '**当前累计 21 个候选单位（42 道题）、4 个历史纠错单位（8 道题）；本轮没有新增认证通过的 2026 新知识。**',
        '候选表示已处理所列局部问题；首次公开时间、图像实体和选项唯一性仍需验收，不能直接并入正式训练集。','',
        '## 本轮新增修题例子','']
    for subset in ('candidate','historical_corrections'):
        base=DEST/subset;questions=rows(base/'source/questions.jsonl');answers={r['id']:r['answers'] for r in rows(base/'source/answers.jsonl')}
        patches=rows(base/'audit/patches.jsonl')
        for i in SPECS:
            pp=[p for p in patches if p['index']==i]
            if not pp:continue
            text += [f"### {i} · {pp[0]['name']}",'',DECISIONS[i][1],'']
            for p in pp:
                q=p['after'];text += [f"**{p['type']}**：{q['question']}",'']
                text += [f"- {o['id']}. {o['text']}" for o in q['options']]
                text += ['',f"答案：{p['new_answer']}。",'']
    text += ['## 21 条逐项处置','','| 索引 | 实体 | 状态 | 理由 |','|---|---|---|---|']
    text += [f"| {r['index']} | {r['name']} | {names[r['action']]} | {r['reason_zh']} |" for r in judged]
    text += ['','## 来源复查边界','',extra['reason_zh'],
        f"来源：[Television Academy]({extra['url']})。已缓存页面、哈希与原文行号；当前页面不能证明截止日前已公开。",'',
        '150、1243 的旧版本均成功获取，带截止日前版本时间；这是旧事实已存在的证据，不等同于确定首次公开日。',
        '补查爱尔兰议会官方 API，Darragh O’Brien 的第 34 届 Dáil / Dublin Fingal East 记录起于 2024-11-29，与旧正文的 2024 年当选相符；这份当前 API 本身不证明首次公开日期。',
        '359、591、1272 的完整新修订支持重构后的断言，但页面修改日不能替代首次公开日。359 的其他影片仍可能形成作品知名度先验；591 是带日期的微观统计；1272 仍需核实姓名来源时间。',
        '对 499 的四个 ISBN 均验算了 ISBN-13 校验位，四项均有效，因此未发现“只有正确选项校验合法”这条捷径；不据此认证该题没有其他问题。',
        'HTTP 200 也核对了正文：150 的猜测总统府链接实为公路新闻、898 的 ESPN 链接实为 Ronaldo 新闻、1219 是通用档案页，均不作事实支持。403、404、202 和网络失败也不作旧知识不存在的证据。',
        '其余待补证条目仍保持原状态，本轮未宣称已逐条补齐来源。','',
        '## 文件与验证','',
        f'- [逐条处置与摘录]({FOLLOW}/reconstruction_21_decisions.jsonl)',
        f'- [402 条完整去向]({DEST}/dispositions_402.jsonl)',
        f'- [验证结果]({DEST}/validation.json)',
        f'- [官方议会补证与 ISBN 校验记录]({FOLLOW}/supplemental_checks.json)',
        f'- [来源缓存目录]({FOLLOW}/sources)',
        '候选和历史归档分别位于 candidate/、historical_corrections/。四方向学习材料、题目、标签、活动证据同步修改；旧证据保存在 provenance/。',
        '按子集、题型重新平衡答案位置，每种题型 ABCD 计数最多差 1；没有使用模型预测。结构校验包括四方向一致、图像路径存在且训练/评测路径不同、原始文件及 v1 哈希未变。结构通过不代表语义或时间通过。','']
    (DEST/'REPAIR_REPORT.md').write_text('\n'.join(text))


if __name__=='__main__':
    import sys
    verify() if '--verify' in sys.argv else build()
