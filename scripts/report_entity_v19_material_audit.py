"""Record assistant visual screening and build the material-audit report.

Screening observations are about visible content and provenance metadata.
Portrait likeness is never treated as confirmation of a person's identity.
"""
from __future__ import annotations

import csv
import hashlib
import html
import json
from collections import Counter

from audit_entity_v19_materials import OUT, WORK, FROZEN, ROOT, rows, save, write_rows, csvfile


OBSERVATIONS = {
    1: ('source_or_target_ambiguity', '学习图为多人活动场景；测试图来源路径写有 Tony Looi Chee Hong，与目标 Chee Hong Tat 的名称不同。来源身份冲突需要核验。'),
    2: ('metadata_plausible_identity_unverified', '两图均为人物照片；来源 lex.dk 的文件名均含 Boje Mathiesen。仅记录来源名称相容，未核实本人身份。'),
    3: ('metadata_plausible_identity_unverified', '学习图为足球官方肖像，来源说明含 alejo-veliz；测试图球衣可见 VELIZ，存在文字身份线索。未据人脸确认身份。'),
    4: ('hard_content_conflict', '学习图为动漫角色群像，测试图为动漫画面；本单元正文和问题针对现实政治人物，不是所需人物身份照片。'),
    5: ('source_or_target_ambiguity', '学习图为历史风格黑白肖像，测试图为画廊内人物照片；两来源 URL 不含目标姓名，未提供其为该足球运动员的身份依据。'),
    6: ('source_or_target_ambiguity', '两图均为人物照片，但来源无目标身份说明；需要来源页面核验，不能靠面部外观判定身份。'),
    7: ('metadata_plausible_identity_unverified', '两图为人物照片，来源文件名和新闻图片说明均含 David Lammy；保留来源相容标记，未独立认证身份。'),
    8: ('metadata_plausible_identity_unverified', '两图为自行车运动场景/肖像，来源文件名均含 Matthew Riccitello；仍需原页面身份凭据。'),
    9: ('source_or_target_ambiguity', '学习图只截取人像一部分，测试图为人物场景；两来源没有目标姓名，目标在图中的对应关系无凭据。'),
    10: ('metadata_plausible_identity_unverified', '两图为人物照片，测试图来源为娱乐新闻图片；没有仅凭图像外貌确认其为目标。'),
    11: ('hard_content_conflict', '学习图为写有 Emily in Paris 的电视剧海报，测试图中可见书封 Emily Ratajkowski；这些文字和呈现内容不能作为排球运动员 Emily Maglio 的身份依据。'),
    12: ('hard_content_conflict', '两图均为儿童读物式插画，不呈现目标作家本人。即使属于其作品，也不能直接充当“图中人物”的身份图。'),
    13: ('source_or_target_ambiguity', '测试图含多人，未标注目标；来源文件名含 Biraj Bhakta Shrestha，但需要原页面确认指向哪一人。'),
    14: ('metadata_plausible_identity_unverified', '两图为足球比赛场景，测试图来源文件名含 Salma Paralluelo；身份仍应以页面文字证据验收。'),
    15: ('metadata_plausible_identity_unverified', '两图为人物肖像，来源文件名均含 Olivier Minne；相近背景不等于已独立认证身份或拍摄独立性。'),
    16: ('metadata_plausible_identity_unverified', '新闻图片来源说明和议会官网文件名均含 Gitta Connemann；标为来源相容，未据面部判断。'),
    17: ('source_or_target_ambiguity', '两图为商业风格人物肖像，来源只有 gcores 随机 ID；没有材料证明图片指向正文所述美国政治人物。'),
    18: ('metadata_plausible_identity_unverified', '学习图来源文字含 Matteo Guendouzi，测试图来源标为法国队官方肖像；尚需核对原始页面的具体人名。'),
    19: ('metadata_plausible_identity_unverified', '学习图新闻图片说明和测试图文件名均含 Kiera Van Ryk；保留来源相容标记，未完成身份认证。'),
    20: ('source_or_target_ambiguity', '学习图为塑像，测试图为历史风格人物版画；未有来源证据证明能代表正文所述现代政治人物。'),
    21: ('metadata_plausible_identity_unverified', '两条新闻图片来源说明均含 Helena Kennedy；本轮未凭面部外观认证身份。'),
    22: ('source_or_target_ambiguity', '学习图为儿童影视风格画面，测试图为多张人物拼图；未提供图中对象与目标足球运动员的来源对应说明。'),
    23: ('source_or_target_ambiguity', '两图为历史风格人物画作，目标正文为当代政治人物；画作与目标身份缺乏来源对应证据。'),
    24: ('source_or_target_ambiguity', '两图为人物照片，来源路径没有目标姓名，也未附页面身份说明；不能凭外貌认证。'),
    25: ('metadata_type_conflict', '正文明确目标是日报 The Santa Fe New Mexican，但 entity_kind 标为 built_landmark；所选图为建筑，未证明是该报社办公楼。首先存在实体类别错配。'),
    26: ('source_or_target_ambiguity', '两图为山地景观，类型相容但现有来源路径不足以确认均为金刚山。'),
    27: ('hard_content_conflict', '正文目标是中世纪城堡废墟 Château de Termes，所选两图却为完整的大型宫殿和水景，图文结构特征明显冲突。'),
    28: ('hard_content_conflict', '正文要求两尊古希腊青铜武士雕像，学习图为城市广场水池，测试图为室内现代金属结构，均未呈现所需两尊武士像。'),
    29: ('source_or_target_ambiguity', '两图为佛教遗址场景，而正文明确指洛坤府 Wat Phra Mahathat Woramahawihan；同名寺庙可能混淆，需要地名和来源核验。'),
    30: ('hard_content_conflict', '正文指加拿大渥太华的国会图书馆；测试图明显为带大本钟的英国议会建筑，学习图也为英国议会区鸟瞰，地理对象冲突。'),
    31: ('hard_content_conflict', '正文指印度 Lohagad Fort；学习图为现代高层塔楼效果图，测试图标有“蒸汽朋克主题乐园”，不呈现该堡垒。'),
    32: ('metadata_plausible_identity_unverified', '两图为木构教堂，测试图新闻图片说明明确提到 Kiruna Church；仍需原页面验收而非仅据外观。'),
}

LABELS = {
    'hard_content_conflict': '明显图文内容冲突',
    'metadata_type_conflict': '实体类别错配',
    'source_or_target_ambiguity': '来源或指向待核验',
    'metadata_plausible_identity_unverified': '表面/来源相容，身份未认证',
}


def table(headers, records):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |',
                      *('| ' + ' | '.join(map(str, r)) + ' |' for r in records)])


def main():
    sample = rows(OUT / 'review_sample.jsonl')
    assert len(sample) == len(OBSERVATIONS) == 32
    image_inventory = {r['relative_path']: r for r in rows(WORK / 'data/image_inventory.jsonl')}
    actual_hash_checks = []
    notes = []
    for r in sample:
        status, observation = OBSERVATIONS[r['number']]
        for key in ('learning_image', 'evaluation_image'):
            from pathlib import Path
            p = Path(r[key])
            rel = 'images/' + str(p).split('/images/', 1)[1]
            assert hashlib.sha256(p.read_bytes()).hexdigest() == image_inventory[rel]['sha256'], p
            actual_hash_checks.append(str(p))
        notes.append(dict(number=r['number'], unit_id=r['unit_id'], qid=r['qid'], entity_name=r['entity_name'],
                          entity_kind=r['entity_kind'], status=status, observation=observation,
                          reviewer='assistant visual and provenance screening', human_approved=False,
                          named_identity_verified=False,
                          review_image='contact_sheets/page_' + str((r['number'] - 1) // 8 + 1) + '.jpg',
                          prediction_or_correctness_used=False))
    write_rows(OUT / 'visual_screening.jsonl', notes)
    csvfile('visual_screening', notes)
    counts = Counter(n['status'] for n in notes)
    summary = json.loads((OUT / 'static_summary.json').read_text())
    summary['visual_screening'] = dict(sample_units=32, verified_image_hashes=len(actual_hash_checks),
                                       statuses=dict(counts), human_identity_verified_units=0,
                                       original_images_unchanged=True,
                                       sampling='24 random person units and 8 random non-person units; seed 20261016')
    save(OUT / 'audit_summary.json', summary)

    # A local gallery keeps visible evidence beside source labels and fact text.
    cards = []
    for r, n in zip(sample, notes):
        esc = html.escape
        photographs = ''.join(f'<figure><figcaption>{title}</figcaption><img src="{esc(r[key], quote=True)}">'
            f'<p><a href="{esc(r[url], quote=True)}">原来源 URL（未联网核验）</a></p></figure>'
            for title, key, url in (('学习图', 'learning_image', 'learning_source_url'),
                                    ('测试图', 'evaluation_image', 'evaluation_source_url')))
        cards.append(f'<article><h2>{r["number"]:02d} {esc(r["entity_name"])}</h2>'
                     f'<p class="tag">{esc(LABELS[n["status"]])}</p><p>{esc(n["observation"])}</p>'
                     f'<div class="pair">{photographs}</div><details><summary>展开事实与题目</summary>'
                     f'<p>{esc(r["text_learning"])}</p><pre>{esc(r["direct"])}</pre>'
                     f'<pre>{esc(r["statement"])}</pre></details></article>')
    (OUT / 'gallery.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>实体数据材料审核</title><style>body{max-width:1200px;margin:30px auto;padding:0 20px;
font:16px/1.65 system-ui;color:#20252d;background:#f5f6f8}article{background:white;padding:24px;
margin:22px 0;border:1px solid #dde1e5;border-radius:12px}h1,h2{line-height:1.3}.pair{display:grid;
grid-template-columns:1fr 1fr;gap:22px}figure{margin:0}img{width:100%;height:340px;object-fit:contain;
background:#f0f1f3}pre{white-space:pre-wrap}.tag{font-weight:bold;color:#8b4810}a{color:#155da8}
@media(max-width:700px){.pair{grid-template-columns:1fr}}</style><h1>实体数据材料审核</h1>
<p>独立于模型成绩的分层抽样：24 个人物、8 个其他实体。以下为助手初筛，未认证肖像人物身份，不能代替来源核验或人工验收。</p>
''' + ''.join(cards) + '</html>', encoding='utf-8')

    controls = list(csv.DictReader((OUT / 'subset_controls.csv').open(encoding='utf-8-sig')))
    scores = list(csv.DictReader((OUT / 'subset_scores.csv').open(encoding='utf-8-sig')))
    control_rows, score_rows = [], []
    for model in ('qwen3', 'glm46'):
        for group, label in (('all', '全量'), ('exclude_unchanged_units', '排除 37 个标记单元')):
            r = next(r for r in controls if r['model'] == model and r['group'] == group
                     and r['state'] == 'VISION' and r['question_type'] == 'overall')
            control_rows.append([model, label, str(int(r['questions']) // 2),
                                 *(f'{float(r[k]):.2f}' for k in ('image_pct', 'no_image_pct', 'wrong_image_pct')),
                                 f'{float(r["correct_minus_no_image_pp"]):+.2f}',
                                 f'{float(r["correct_minus_wrong_image_pp"]):+.2f}'])
            r = next(r for r in scores if r['model'] == model and r['group'] == group
                     and r['state'] == 'VISION' and r['question_type'] == 'overall' and r['test'] == 'image')
            score_rows.append([model, label, r['units'], r['questions'], f'{float(r["accuracy_pct"]):.2f}'])
    hard_rows = [[n['number'], n['entity_name'], n['observation']] for n in notes
                 if n['status'] in ('hard_content_conflict', 'metadata_type_conflict')]
    status_rows = [[LABELS[k], v] for k, v in counts.items()]
    report = f'''# 实体 v19 材料审核补充报告

日期：2026-10-16。范围：全量静态检查 + 独立抽样 32 个单元的图像内容/来源初筛；本轮未训练、未调用模型推理、未改动原始数据和历史评分。

## 结论先行

**已确认存在需要优先修复的材料问题：图片角色选择没有实体语义验收，抽查可见动漫、插画、错误类型建筑等进入正式学习图和测试图。** 这比单纯答案位置偏好更直接地影响“实体图片身份→知识”的测量。本轮也发现文字匿名化缺口及弱选项长度线索，但不能据此断言模型的具体答题机制。

现有实验数字保留为当前材料上的描述性结果。关于可靠视觉身份绑定、同模态/跨模态知识能力的结论需进一步附加材料有效性条件；仅根据现有分数判断迁移成功或失败均不充分。错误学习图可能影响知识绑定，错误测试图可能使检索目标不成立，但本次未建立它们对低增益的因果贡献。

## 1. 审核方法与可复核性

已重新核对 18 份冻结来源文件的 SHA-256。按固定随机种子 20261016，从 1,153 个人物单元抽 24 个，从 72 个其他单元抽 8 个；抽样和文字标记规则不使用模型预测或答对与否，先保存抽样再读取历史分数。抽样所用 64 张实际图片的 SHA-256 与实验冻结清单一致，说明看到的确为实验选用文件。

这是助手根据可见内容与本地来源字段做的**初筛**，不是人工验收。本轮没有联网核验来源页面，没有根据人脸外貌认证人物身份。来源文件名含目标姓名只能记为相容，不能当作已通过。非人物被刻意过采样，不能把 32 个样本中的异常占比直接外推为全量错误率。

## 2. 图像材料的具体问题

初筛状态：

{table(['状态', '单元数'], status_rows)}

其中 7 个单元存在直接可见的图文内容冲突，另 1 个存在明确实体类别错配。其他单元同样没有完成身份验收，“相容”不等于正确；“待核验”也不等于已判错。

{table(['抽样号', '目标实体', '可见问题'], hard_rows)}

此外，Chee Hong Tat 的测试图 URL 中写的是 Tony Looi Chee Hong；多个人物样本只有不含目标姓名的来源路径，或使用多人场景、拼图。它们需要来源对应证据，不能根据两张人像外观做身份判断。Wat Phra Mahathat、山景等还存在同名地点/相似外观混淆风险。

四页对照图（左为学习图，右为测试图）：

![抽查 1–8]({OUT / 'contact_sheets/page_1.jpg'})

![抽查 9–16]({OUT / 'contact_sheets/page_2.jpg'})

![抽查 17–24]({OUT / 'contact_sheets/page_3.jpg'})

![抽查 25–32]({OUT / 'contact_sheets/page_4.jpg'})

## 3. 流程中已定位的缺口

包内 [build_package.py]({ROOT / 'entity_v19_1478/audit/build_package.py'}:173) 的图片选择流程为：过滤跨 QID 完全相同的哈希，要求学习/测试图的 SHA 和 URL 不同、dHash 距离大于 5，再取第一个满足条件的图片对。**这些条件检查图片是否不同，不检查它们是否属于所需实体。** 动漫图和另一张动漫图也能满足。

源包审计记录本来就标明 `image_semantics_reviewed=false`，并没有已有人工认证被本轮推翻；问题在于没有把语义验收作为正式实验准入条件。当前代码足以解释错误候选如何进入角色选择，但不能仅凭这些文件追溯每个上游搜索词或断定使用了什么搜索策略。

必须区分图片不同、图片身份正确、跨视角可识别三个条件。即使学习图和测试图是同一无关对象，也不构成所学实体的有效图像监督；若学习图错、测试图对，单独换测试图仍不能修复已经训练的对应关系。

## 4. 文字匿名化也存在缺口

**68/2,450 道图片题**在移除 `<image>` 标记后，与文字题逐字相同，涉及 **37/1,225 个知识单元**；与源包已有 `no_explicit_entity_reference_masked` 标记完全一致。不是所有题都同时命中两类，所以是 37 个单元而非 34 个。

有些题保留了本人姓名，如 Emma Johnston、Hedy Lamarr；有些把相关作品、机构或事件写得足够具体，如 Solvej Balle 单元直接问《体积的计算》第三卷获什么奖。这里确认的是图片对实体身份的必要性存在风险，不是证明每道题都能被任何模型无图答对。

精确姓名字符串检查还标出图片评测 7 题、视觉学习 3 单元，但不能机械删除：Robert Alfred Dowd Sr. 是父亲，Enterprise 的残留指向较早舰船，属于需保留的其他实体。Osman Kavala 的名字却直接留在学习正文；`named` 后大片文本被保护的规则也会保留被任命/认定者姓名。应按语义角色区分“目标身份”和“知识本身的答案值”，而不是扩大字符串替换范围。

单独评估这些文字未改写单元是否解释图片对照：保持原始实验不变，只在派生统计中整单元排除上述 37 个单元。保留 1,188 单元、2,376 题，两个题型仍配对；原图片对照的 256 单元中保留 249 个。

VISION E5 的图片题准确率：

{table(['模型', '范围', '单元', '题数', '准确率 %'], score_rows)}

同范围图片对照（只比较同一对照子集）：

{table(['模型', '范围', '对照单元', '正确图 %', '无图 %', '错图 %', '正确−无图 pp', '正确−错图 pp'], control_rows)}

排除后差距仍小，故仅修这 68 题不足以解决总体图片依赖证据偏弱的问题。这里没有重跑显著性检验，不把小幅点估计变化解释成统计结论或因果效应，也未据此建立新的官方“干净数据集”。其余单元仍可能存在图片或文字线索问题。

## 5. 选项线索与“是否学入”的边界

不读实体、不看图片，仅选择词数最多选项、并列均匀分配，图片题上的期望命中率为 direct **{100 * summary['longest_option_expected_accuracy']['direct']:.2f}%**、statement **{100 * summary['longest_option_expected_accuracy']['statement']:.2f}%**。规则按英文/数字词块及词内连字符计数，完整逐题特征已保存。此结果在四轮循环轮换后不变，说明轮换不能消除内容长度线索；它仍是同数据上的探索性诊断，不能认定模型采用了该规则。

本轮没有新增独立事实记忆或名称↔图片绑定探测。因此不能把已经报告的训练增益当作“事实已充分学入”的证明，也不能在材料明显有问题时先归因于模型架构。训练 loss 下降本身同样不能证明能在新问题中检索实体事实。

## 6. 后续修复应采用的准入条件

1. **先验收目标定义。** 确认实体类别与事实主体一致，避免报纸被当作建筑；明确题目问的是人、作品还是其所属机构，不能依赖包内 QID 挂接就视为同一对象。
2. **每个角色分别留来源证据。** 学习图与测试图都记录目标 QID、页面标题/说明、图片指向证据和目标位置；多人图需明确目标。候选下载成功、名字相似、SHA 不同不算语义通过。人物身份以可靠来源文字和人工核验为据，不用人脸相似度作为自动验收。
3. **语义通过后再做差异检查。** 保留全局 SHA、URL、近重复检查；不同视角、可辨认主体和图片内可读姓名/答案信息分别标记。作者插画不能未经说明替代作者本人的身份图。
4. **两个题型都检查身份必要性。** direct 和 statement 分别保留自然题型，用“图中实体”承担真正的目标指代；对明确命名作品/机构已经足够作答的题进行语义改写或标记，不按当前模型答对与否决定留删。
5. **冻结新版本后再验证。** 新图、新题、新评分协议单独版本化；先做小规模独立材料验收及无图/错图对照。修复学习图后需重新训练相应视觉分支，不能拿旧权重配新图冒充完整修复。本轮不启动新训练、推理或自动替换图片。

## 7. 复核文件

- [逐项图文审核页面]({OUT / 'gallery.html'})
- [固定抽样及题目内容]({OUT / 'review_sample.jsonl'})；[抽样协议]({OUT / 'sampling_protocol.json'})
- [32 单元初筛记录]({OUT / 'visual_screening.csv'})；[汇总 JSON]({OUT / 'audit_summary.json'})
- [68 道文字未改写题]({OUT / 'unchanged_questions.csv'})；[全量文字特征]({OUT / 'question_features.jsonl'})
- [长度线索逐题特征]({OUT / 'option_length_features.csv'})
- [派生分组分数]({OUT / 'subset_scores.csv'})；[同子集图片对照]({OUT / 'subset_controls.csv'})
- [静态审核脚本]({ROOT / 'scripts/audit_entity_v19_materials.py'})；[初筛记录及报告脚本]({ROOT / 'scripts/report_entity_v19_material_audit.py'})

历史主结果、原始图像和评分均保留。此报告是材料诊断补充，不是按模型成绩重新筛选后的正式实验。
'''
    (OUT / '材料审核补充报告.md').write_text(report, encoding='utf-8')
    print(json.dumps(dict(report=str(OUT / '材料审核补充报告.md'), statuses=dict(counts),
                         verified_frozen_sources=18, verified_image_hashes=len(actual_hash_checks)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
