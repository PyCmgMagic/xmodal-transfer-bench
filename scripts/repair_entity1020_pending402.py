"""Build a reviewable question-repair candidate; never certify temporal novelty.

Only the original 402 pending units are in scope. Source files and prior runs
stay immutable. Corrected historical facts are archived, not promoted to 2026.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'entity1020/balanced_1572'
REVIEW = ROOT / 'diagnostics/entity1020_pending402_review_20261020'
DEST = ROOT / 'entity1020/pending402_question_repair_v1'
LETTERS = 'ABCD'
TYPES = ('direct', 'statement')
DIRECTIONS = ('T2T', 'T2V', 'V2T', 'V2V')
SUFFIX = '\n\nAnswer with exactly one uppercase letter (A, B, C, or D). Do not include an explanation or any other characters.'


def rows(p):
    return [json.loads(s) for s in p.open(encoding='utf-8') if s.strip()]


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def write(p, values):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(''.join(json.dumps(v, ensure_ascii=False) + '\n' for v in values))


# Replacements remove a specified local defect, not every possible prior.
# Tuples apply identically to the named and existing anonymous representations.
REPLACE = {
    102: [('Pickerington Central Tigers', 'New York Giants'),
          ('Ohio State Buckeyes', 'Tennessee Titans')],
    168: [('rapper 2 Chainz', '2 Chainz')],
    175: [('On which 1995 film', 'On which film'), ('on the 1995 film', 'on the film')],
    628: [('buy in Rome in 2025', 'buy in 2025')],
    835: [('In which years was', 'In which two-year range was'),
          ('1985 or 1986', '1981 or 1982'), ('1988 or 1989', '1983 or 1984'),
          ('1986 or 1987', '1985 or 1986')],
    955: [('Sol Gabetta', 'Alexandre Tharaud'), ('Antoine Tamestit', 'Nikolai Lugansky'),
          ('Vilde Frang', 'Cédric Tiberghien')],
    1152: [('Chemistry', 'General Nonfiction')],
    1234: [('Fiji Drua', 'Stade Rochelais'), ('RC Toulonnais', 'Stade Français Paris'),
           ('Cronulla Sharks', 'Union Bordeaux Bègles')],
    1238: [('What event in September 2022', 'What event'), (' in September 2022', '')],
    1416: [('Introduce proportional representation', 'Lift restrictions on serving civil servants standing for Parliament'),
           ('introduce proportional representation', 'lift restrictions on serving civil servants standing for Parliament'),
           ('Abolish the House of Lords', 'Lift restrictions on serving judges standing for Parliament'),
           ('abolish the House of Lords', 'lift restrictions on serving judges standing for Parliament'),
           ('Lower the voting age to 16', 'Lift restrictions on serving police officers standing for Parliament'),
           ('lower the voting age to 16', 'lift restrictions on serving police officers standing for Parliament')],
    1504: [('Rushanara Ali', 'Anjli Mohindra'), ('Shirin Sharmin Chaudhury', 'Aiysha Hart')],
    1506: [('born in Juodšiliai', 'born')],
}

# Ask the outcome of the new event, rather than the club's familiar league.
# Keep named people in text and use an anonymous subject in the visual form.
FULL = {
    336: dict(direct="Where did {subject}'s club finish in the 2025–26 2. Bundesliga?",
              statement='Which statement is accurate?', values=['first', 'second', 'third', 'fourth'],
              sentence="In the 2025–26 2. Bundesliga, {subject}'s club finished {value}.",
              gold='first'),
    387: dict(direct="What was the result of {subject}'s club in the 2025–26 FA Cup?",
              statement='Which statement is accurate?',
              values=['champions', 'runners-up', 'semi-finalists eliminated before the final', 'quarter-finalists eliminated before the semi-finals'],
              sentence="In the 2025–26 FA Cup, {subject}'s club were {value}.", gold='runners-up'),
    453: dict(direct='In which month of 2026 was {subject} appointed to the Irish Council of State?',
              statement='Which statement is accurate?', values=['January', 'February', 'March', 'April'],
              sentence='{subject} was appointed to the Irish Council of State in {value} 2026.',
              gold='March'),
    954: dict(direct="Where did {subject}'s club finish in the 2025–26 Saudi Pro League?",
              statement='Which statement is accurate?', values=['first', 'second', 'third', 'fourth'],
              sentence="In the 2025–26 Saudi Pro League, {subject}'s club finished {value}.",
              gold='first'),
    1368: dict(direct="Where did {subject}'s club finish in the 2025–26 Saudi Pro League?",
               statement='Which statement is accurate?', values=['first', 'second', 'third', 'fourth'],
               sentence="In the 2025–26 Saudi Pro League, {subject}'s club finished {value}.",
               gold='first'),
    529: dict(direct='Where did {subject} open a permanent store in 2018?',
              statement='Which statement is accurate?',
              values=["New York's SoHo", "London's King's Cross", "Paris's Le Marais", "Milan's Brera"],
              sentence='{subject} opened a permanent store in {value} in 2018.', gold="New York's SoHo"),
    636: dict(direct='Who was reported in May 2025 to be in negotiations to star in the Josephine Baker biopic being developed by {subject}?',
              statement='Which statement is accurate?',
              values=['FKA twigs', 'Sokhna Diallo', 'Sania Halifa', 'Fathia Youssouf'],
              sentence='{value} was reported in May 2025 to be in negotiations to star in the Josephine Baker biopic being developed by {subject}.',
              gold='FKA twigs'),
    1114: dict(direct="On what date did {subject} win Blackburn Rovers' Players' Player of the Year award in 2026?",
               statement='Which statement is accurate?', values=['1 May', '2 May', '3 May', '4 May'],
               sentence="{subject} won Blackburn Rovers' Players' Player of the Year award on {value} 2026.",
               gold='2 May'),
}

REASONS = {
    102: '高中队、大学队替换为 NFL 球队。',
    168: '去掉仅正确陈述带有的 rapper 修饰；direct 内容不改。',
    175: '去掉所有选项共同附加的 1995 年，避免用上映年份筛选。',
    336: '改问目标人物所在球队的联赛名次；四项为互斥名次，不再猜联赛所属。',
    387: '改问人物所在俱乐部的 FA Cup 最终成绩；四项为互斥成绩，去掉国家队/俱乐部赛事提示。',
    453: '改问有来源支持的任命月份；四项均为 2026 年月份，去掉任命权限线索。',
    529: '按 2018 新闻修正年份并统一为城区选项；转入历史纠错档案，禁止作为 2026 更新。',
    628: 'direct 去掉 Rome 提示；statement 原有各城市陈述不改。',
    636: '按 2025-05-19 报道降低为洽谈出演，并同步学习材料；转历史纠错档案。',
    835: '四个两年范围改为完全不相交；仍需核实出生年份证据。',
    954: '改问目标人物所在球队的联赛名次，去掉联赛与杯赛分类捷径。',
    955: '三个干扰项换成钢琴家，保留指定专辑。',
    1114: '改问有材料支持的获奖日期，避开队友投票定义提示和 Young Player 年龄提示。',
    1152: 'Chemistry 换为文字创作领域 General Nonfiction。',
    1234: '四项均用 Top 14 球队，去掉跨项目及离队后又选原队的干扰项。',
    1238: '去掉所有选项共同附加的 September 2022，避免年月排除。',
    1368: '改问目标人物所在球队的联赛名次，去掉联赛所属提示。',
    1416: '四项统一为取消参选资格限制，区分被限制的人群。',
    1504: '政治人物干扰项替换为演员。',
    1506: '去掉给定 Juodšiliai 的地理提示；依然可能有实体旧知识先验。',
}


def replace(index, text):
    for old, new in REPLACE.get(index, []):
        text = text.replace(old, new)
    return text


def split_prompt(text):
    marks = list(re.finditer(r'(?m)^([A-D])\. ', text))
    assert [m[1] for m in marks] == list(LETTERS)
    end = text.index('\n\nAnswer with exactly one uppercase letter', marks[-1].end())
    opts = [text[m.end():marks[j+1].start() if j < 3 else end].rstrip('\n')
            for j, m in enumerate(marks)]
    return text[:marks[0].start()].rstrip('\n'), opts


def render(stem, opts):
    return stem + '\n' + '\n'.join(f'{l}. {o}' for l, o in zip(LETTERS, opts)) + SUFFIX


def polish_visual(text):
    text = text.replace("the person shown in the image's club", 'the club of the person shown in the image')
    text = text.replace("the person shown in the image's 2019 Expensify campaign", 'the 2019 Expensify campaign featuring the person shown in the image')
    text = text.replace("the person shown in the image's House of Commons Disqualification (Amendment) Bill, introduced in 1999",
                        'the House of Commons Disqualification (Amendment) Bill introduced in 1999 by the person shown in the image')
    text = text.replace("the person shown in the image's 1999 House of Commons Disqualification (Amendment) Bill",
                        'the House of Commons Disqualification (Amendment) Bill introduced in 1999 by the person shown in the image')
    if text.startswith('the person'): text = 'T' + text[1:]
    return text


def build():
    if DEST.exists():
        import sys
        assert '--rebuild' in sys.argv, f'Existing derivative: {DEST}; use --verify or --rebuild.'
        previous = json.loads((DEST / 'manifest.json').read_text())
        assert previous['schema'] == 'entity1020-pending402-question-repair-v1'
        assert json.loads((DEST / 'validation.json').read_text())['status'] == 'passed'
        for path, digest in previous['original_inputs'].items(): assert sha(Path(path)) == digest
    ledger = rows(REVIEW / 'unit_ledger.jsonl')
    units = {u['index']: u for u in rows(REVIEW / 'units.jsonl')}
    selected = set(REASONS)
    assert selected == set(REPLACE) | set(FULL)
    assert selected <= {r['index'] for r in ledger if r['partition'] in {'repair_risk', 'confirmed_quality_issue'}}
    selected_ids = {units[i]['unit_id'] for i in selected}
    index_by_id = {u['unit_id']: i for i, u in units.items()}
    # Snapshot only files read by this build, plus all previous-review frozen inputs.
    inputs = list((SOURCE / 'source').glob('*.jsonl'))
    inputs += [SOURCE / d / f'{phase}.jsonl' for d in DIRECTIONS for phase in ('train', 'eval')]
    inputs += [SOURCE / 'audit' / n for n in ('eval_labels.jsonl', 'sample_audit.jsonl')]
    inputs += [REVIEW / n for n in ('unit_ledger.jsonl', 'units.jsonl', 'freeze.json')]
    frozen = json.loads((REVIEW / 'freeze.json').read_text())['sha256']
    input_hashes = {str(p): sha(p) for p in inputs}
    for rel, digest in frozen.items():
        p = Path(rel) if Path(rel).is_absolute() else ROOT / rel
        assert sha(p) == digest
        input_hashes[str(p)] = digest
    DEST.mkdir(parents=True, exist_ok=True)
    historical = {units[i]['unit_id'] for i in (529, 636)}
    subsets = {'candidate': selected_ids - historical, 'historical_corrections': historical}

    # Random ordering, balanced per type, without selecting questions by model accuracy.
    rng = random.Random(42)
    target = {}
    for subset, ids in subsets.items():
        target[subset] = {}
        for typ in TYPES:
            letters = [LETTERS[j % 4] for j in range(len(ids))]
            rng.shuffle(letters)
            target[subset][typ] = dict(zip(sorted(ids), letters))

    answers = {r['id']: r for r in rows(SOURCE / 'source/answers.jsonl')}
    old_evals = {d: {r['id']: r for r in rows(SOURCE / d / 'eval.jsonl')}
                 for d in ('T2T', 'T2V')}
    canonical, versions, new_answers, changes = {}, {}, {}, []
    for i in sorted(selected):
        u = units[i]; uid = u['unit_id']; subset = 'historical_corrections' if uid in historical else 'candidate'
        new_answers[uid] = {'id': uid, 'qid': u['qid'], 'answers': {}}
        for typ in TYPES:
            q = copy.deepcopy(u['questions'][typ]); oldq = copy.deepcopy(q)
            sid = f'{uid}-{typ}'; old_answer = answers[uid]['answers'][typ]
            if i in FULL:
                spec = FULL[i]; vals = spec['values']; gold_index = vals.index(spec['gold'])
                stem = spec[typ].format(subject=u['name'])
                opts = vals if typ == 'direct' else [spec['sentence'].format(subject=u['name'], value=v) for v in vals]
            else:
                stem = replace(i, q['question'])
                opts = [replace(i, o['text']) for o in q['options']]
                gold_index = LETTERS.index(old_answer)
            new_answer = target[subset][typ][uid]
            perm = [j for j in range(4) if j != gold_index]
            rng.shuffle(perm); perm.insert(LETTERS.index(new_answer), gold_index)
            q['question'] = stem
            q['options'] = [{'id': l, 'text': opts[j]} for l, j in zip(LETTERS, perm)]
            canonical[uid, typ] = q
            new_answers[uid]['answers'][typ] = new_answer
            for d in ('T2T', 'T2V'):
                row = copy.deepcopy(old_evals[d][sid]); visual = d == 'T2V'
                if i in FULL:
                    subject = 'the person shown in the image' if visual else u['name']
                    spec = FULL[i]
                    vs = spec[typ].format(subject=subject)
                    vo = spec['values'] if typ == 'direct' else [spec['sentence'].format(subject=subject, value=v) for v in spec['values']]
                    if visual: vs = '<image>\n' + vs
                else:
                    vs, vo = split_prompt(replace(i, row['messages'][-1]['content']))
                assert len(vo) == len(set(vo)) == 4
                if visual:
                    vs = polish_visual(vs); vo = [polish_visual(o) for o in vo]
                row['messages'][-1]['content'] = render(vs, [vo[j] for j in perm])
                versions[d, uid, typ] = row
            changes.append(dict(index=i, unit_id=uid, qid=u['qid'], name=u['name'], type=typ,
                                subset=subset, reason_zh=REASONS[i], before=oldq, after=q,
                                old_answer=old_answer, new_answer=new_answer,
                                new_position_to_prebalance_index=perm,
                                temporal_novelty_approved=False, distractor_truth_exhaustively_verified=False))

    # Correct the learning claim as well as the tests for source-strength/year errors.
    def learning_fix(i, text):
        if i == 529: return text.replace('in 2026', 'in 2018')
        if i == 636:
            visual = text.startswith('<image>')
            subject = 'the person shown in the image' if visual else units[i]['name']
            return ('<image>\n' if visual else '') + 'In May 2025, FKA twigs was reported to be in negotiations to star in the Josephine Baker biopic being developed by ' + subject + '.'
        return text

    for subset, ids in subsets.items():
        base = DEST / subset
        for name in ('entities', 'evidence', 'learning'):
            rr = [copy.deepcopy(r) for r in rows(SOURCE / f'source/{name}.jsonl')
                  if (r.get('id') in ids if name != 'entities' else r['qid'] in {units[index_by_id[x]]['qid'] for x in ids})]
            if name == 'learning':
                for r in rr: r['text'] = learning_fix(index_by_id[r['id']], r['text'])
            if name == 'evidence':
                write(base / 'provenance/original_evidence.jsonl', rr)
                corrected = []
                adjudications = {r['unit_id']: r for r in ledger}
                for original in rr:
                    uid = original['id']; i = index_by_id[uid]
                    k = copy.deepcopy(original['new_knowledge'])
                    support = adjudications[uid]['dated_evidence']
                    if i in (529, 636):
                        k['claim'] = learning_fix(i, next(r['text'] for r in rows(SOURCE / 'source/learning.jsonl') if r['id'] == uid))
                        k['event_date'] = '2018' if i == 529 else '2025-05'
                        k['first_public_date'] = None
                        k['known_public_by_date'] = '2018-05-21' if i == 529 else '2025-05-19'
                        k['source_url'] = support[0]['url']
                        k['new_evidence'] = '\n'.join(q['quote'] for s in support for q in s['quotes'])
                        k['first_public_evidence'] = ''
                        k['difference'] = 'Source-based correction of the original claim; excluded from post-cutoff novelty experiments.'
                    corrected.append(dict(id=uid, qid=original['qid'], new_knowledge=k,
                        dated_support=support, original_record='provenance/original_evidence.jsonl',
                        review=dict(status='invalidated_after_question_repair',
                                    temporal_novelty_approved=False, option_uniqueness_approved=False,
                                    reason='Original automatic review applied to old questions; see audit/patches.jsonl.')))
                rr = corrected
            write(base / f'source/{name}.jsonl', rr)
        write(base / 'source/questions.jsonl', [canonical[uid, t] for uid in sorted(ids) for t in TYPES])
        write(base / 'source/answers.jsonl', [new_answers[uid] for uid in sorted(ids)])
        labels = [copy.deepcopy(r) for r in rows(SOURCE / 'audit/eval_labels.jsonl') if r['id'].rsplit('-', 1)[0] in ids]
        audits = [copy.deepcopy(r) for r in rows(SOURCE / 'audit/sample_audit.jsonl') if r['unit_id'] in ids]
        for r in labels: r['answer'] = new_answers[r['id'].rsplit('-', 1)[0]]['answers'][r['probe_id']]
        for r in audits:
            r['answer'] = new_answers[r['unit_id']]['answers'][r['probe_id']]
            r['question_repair_candidate'] = True
            r['temporal_novelty_approved'] = False
            r['source_automatic_passed'] = r.get('automatic_passed')
            r['automatic_passed'] = False
        write(base / 'audit/eval_labels.jsonl', labels)
        write(base / 'audit/sample_audit.jsonl', audits)
        # Image bytes remain in the immutable source package. Absolute references
        # avoid copying GBs and make this explicitly a local derivative.
        for d in DIRECTIONS:
            train = [copy.deepcopy(r) for r in rows(SOURCE / d / 'train.jsonl') if r['id'] in ids]
            for r in train: r['messages'][0]['content'] = learning_fix(index_by_id[r['id']], r['messages'][0]['content'])
            evals = [copy.deepcopy(versions['T2V' if d.endswith('V') else 'T2T', uid, typ]) for uid in sorted(ids) for typ in TYPES]
            for rr in (train, evals):
                for r in rr:
                    if 'images' in r: r['images'] = [str(SOURCE / f) for f in r['images']]
            write(base / d / 'train.jsonl', train)
            write(base / d / 'eval.jsonl', evals)
        write(base / 'audit/patches.jsonl', [c for c in changes if c['subset'] == subset])

    dispositions = []
    for r in ledger:
        i = r['index']
        if i in selected:
            action = 'historical_correction_excluded' if i in (529, 636) else 'question_repaired_still_quarantined'
            reason = REASONS[i]
        elif r['partition'] == 'pre_cutoff_discriminating_fact':
            action = 'exclude_pre_cutoff'; reason = r['reason_zh']
        elif r['partition'] in {'repair_risk', 'confirmed_quality_issue'}:
            action = 'needs_fact_or_scope_reconstruction'; reason = r['reason_zh']
        else:
            action = 'quarantine_pending_evidence'; reason = r['reason_zh']
        dispositions.append(dict(index=i, unit_id=r['unit_id'], qid=r['qid'], name=r['name'],
                                 previous_partition=r['partition'], action=action, reason_zh=reason,
                                 approved_for_post_cutoff_experiment=False))
    write(DEST / 'dispositions_402.jsonl', dispositions)
    save(DEST / 'manifest.json', dict(schema='entity1020-pending402-question-repair-v1',
         source=str(SOURCE), review=str(REVIEW), cutoff='2025-12-31T23:59:59Z', seed=42,
         release_status='question_repair_candidate_temporally_quarantined',
         repaired_units=len(selected), repaired_questions=2*len(selected),
         candidate_units=len(subsets['candidate']), historical_correction_units=2,
         actions=dict(Counter(r['action'] for r in dispositions)),
         approved_for_post_cutoff_experiment=0, original_inputs=input_hashes,
         evaluation_not_run=True, selection_uses_predictions=False,
         evidence_note='source/evidence.jsonl contains corrected claims and invalidated old review status; verbatim originals are in provenance/original_evidence.jsonl.',
         limitations=['Repair removes specified defects, not all model priors or possible distractor ambiguity.',
                      'Candidate facts, publication dates and images are not newly certified.',
                      'Absolute source-image paths require the original local package.']))
    report(changes, dispositions)
    verify()


def report(changes, dispositions):
    text = ['# entity1020 待审 402 条：题目修复候选 v1', '',
            '本版只修复上一轮明确指出的局部题目缺陷，保留 direct 与 statement 两种四选一题型。',
            '候选版不能直接作为截止日后新知识实验集。没有用正确率筛题，也没有重跑模型。', '',
            '## 处理结果', '']
    counts = Counter(r['action'] for r in dispositions)
    action_names = {
        'quarantine_pending_evidence': '继续隔离，等待来源或日期口径补证',
        'exclude_pre_cutoff': '有截止日前支持材料，从新知识池排除',
        'question_repaired_still_quarantined': '已修题，仍待时间与语义验收',
        'needs_fact_or_scope_reconstruction': '需要重选事实或重构题目',
        'historical_correction_excluded': '已纠正为历史事实，单独归档',
    }
    for action, n in counts.items(): text.append(f'- {action_names[action]}：{n} 个单位')
    text += ['', '共修复 20 个单位、40 道题的记录（部分题只重排或保持题意）；其中 18 个单位为修题候选，'
             '2 个单位为历史纠错，不进入 2026 新知识池。其余问题均保留具体原因。', '',
             '344 条未核实来源及 4 条日期口径待核实项继续隔离；13 条已取得旧支持材料的条目排除。', '',
             '## 尚需重构的 21 条', '',
             '以下问题缺少足够证据支持直接改答案，或不能靠换三个词保证单选唯一；保留隔离。', '']
    for r in dispositions:
        if r['action'] == 'needs_fact_or_scope_reconstruction':
            text.append(f"- {r['index']} · {r['name']}：{r['reason_zh']}")
    text += ['',
             '## 逐题修改', '']
    for c in changes:
        text += [f"### {c['index']} · {c['name']} · {c['type']}", '', c['reason_zh'], '',
                 '**修改前**', '', c['before']['question'], '']
        text += [f"- {o['id']}. {o['text']}" for o in c['before']['options']]
        text += ['', f"原标签：{c['old_answer']}", '', '**修改后**', '', c['after']['question'], '']
        text += [f"- {o['id']}. {o['text']}" for o in c['after']['options']]
        text += ['', f"新标签：{c['new_answer']}；状态：`{c['subset']}`，仍未认证时间与图像。", '']
    text += ['## 使用与边界', '',
             '`candidate/` 包含本地四方向候选文件；`historical_corrections/` 单独保存 529 与 636 的纠错题和学习材料。',
             'source/evidence.jsonl 已同步修正历史条目的年份/断言，并将原自动审查结果标为失效。'
             'provenance/original_evidence.jsonl 逐字保留原证据，仅供追溯。原始 knowledge_samples 不复制到活动候选，避免旧标签误用。',
             '小子集的每种题型答案计数最多相差 1；若继续评测，应另做四种循环换位并汇总，'
             '18 个单位不足以用于稳定估计基线。',
             '359、420、590 等不能只替换为另外三类奖项：候选可能不适用于该性别、位置或年份；'
             '1467 不能简单换为四个真实赛事，因为该球员可能在多个赛事都有出场。此类保留待重构。',
             '336、954、1368 改问人物所在俱乐部的名次，但相同球队赛果仍为共享事实，统计时应按事实聚类。',
             '本批来自荣誉类正向事实，改成名次后仍可能偏向夺冠结果；不能据此声称消除了选项内容先验，正式集需要在收集阶段平衡不同成绩。',
             '基座准确率是否接近 25% 不作为题目准入标准。需要先核实来源、干扰项与图像，再固定评测集。', '']
    (DEST / 'REPAIR_REPORT.md').write_text('\n'.join(text))


def verify():
    m = json.loads((DEST / 'manifest.json').read_text())
    for path, digest in m['original_inputs'].items(): assert sha(Path(path)) == digest, path
    disp = rows(DEST / 'dispositions_402.jsonl')
    assert len(disp) == len({r['unit_id'] for r in disp}) == 402
    result = {}
    for subset in ('candidate', 'historical_corrections'):
        base = DEST / subset
        qs = {(r['id'], r['type']): r for r in rows(base / 'source/questions.jsonl')}
        answers = {r['id']: r for r in rows(base / 'source/answers.jsonl')}
        labels = {r['id']: r for r in rows(base / 'audit/eval_labels.jsonl')}
        audits = {r['sample_id']: r for r in rows(base / 'audit/sample_audit.jsonl')}
        ids = set(answers); assert len(qs) == len(labels) == len(audits) == 2*len(ids)
        assert all(not r['automatic_passed'] and not r['temporal_novelty_approved'] for r in audits.values())
        for r in rows(base / 'source/evidence.jsonl'):
            assert r['review']['status'] == 'invalidated_after_question_repair'
            if subset == 'historical_corrections':
                assert r['new_knowledge']['event_date'] in ('2018', '2025-05')
                assert 'is set to star' not in r['new_knowledge']['claim']
        assert set(qs) == {(uid, typ) for uid in ids for typ in TYPES}
        balances = {}
        for typ in TYPES:
            cc = Counter(answers[uid]['answers'][typ] for uid in ids)
            nn = [cc.get(l, 0) for l in LETTERS]; assert max(nn)-min(nn) <= 1
            balances[typ] = dict(cc)
        for d in DIRECTIONS:
            train = rows(base / d / 'train.jsonl'); ev = rows(base / d / 'eval.jsonl')
            assert len(train) == len(ids) and len(ev) == 2*len(ids)
            assert {r['id'] for r in train} == ids and {r['id'] for r in ev} == set(labels)
            for phase, rr in (('train', train), ('eval', ev)):
                visual = d.startswith('V') if phase == 'train' else d.endswith('V')
                for r in rr:
                    assert not set(r) & {'answer', 'gold', 'labels', 'response'}
                    assert sum(x['content'].count('<image>') for x in r['messages']) == int(visual)
                    assert len(r.get('images', [])) == int(visual)
                    for p in r.get('images', []): assert Path(p).is_file()
            for r in ev:
                a = audits[r['id']]; uid = a['unit_id']; typ = a['probe_id']; q = qs[uid, typ]
                assert labels[r['id']]['answer'] == a['answer'] == answers[uid]['answers'][typ]
                stem, opts = split_prompt(r['messages'][-1]['content'])
                assert len(set(opts)) == 4
                if d.endswith('T'):
                    assert stem == q['question']
                    assert opts == [o['text'] for o in q['options']]
            for phase in ('train', 'eval'):
                other = ('V' if d[0] == 'T' else 'T') + d[1:]
                if phase == 'eval': assert rows(base / d / f'{phase}.jsonl') == rows(base / other / f'{phase}.jsonl')
            for r in train:
                if subset == 'historical_corrections':
                    assert 'in 2026' not in r['messages'][0]['content']
                    assert 'is set to star' not in r['messages'][0]['content']
        # The selected ranges must be pairwise disjoint in both modalities.
        for r in rows(base / 'audit/patches.jsonl'):
            if r['index'] == 835:
                groups = [set(re.findall(r'\b19\d{2}\b', o['text'])) for o in r['after']['options']]
                assert all(not a & b for j, a in enumerate(groups) for b in groups[j+1:])
        result[subset] = dict(units=len(ids), questions=len(qs), answer_counts=balances)
    save(DEST / 'validation.json', dict(status='passed', source_hashes_unchanged=True,
          all_402_dispositions_recorded=True, four_directions_aligned=True,
          questions_and_labels_synchronized=True, structural_checks_only=True, subsets=result))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    import sys
    verify() if '--verify' in sys.argv else build()
