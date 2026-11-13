"""Publish exhaustive formal-role image screening without changing experiment data."""
from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'diagnostics/entity_v19_full_image_audit_20261016'
LABELS = {
    'content_conflict': '内容明显异常/身份素材不适用',
    'explicit_reference_conflict': '可见姓名/型号或来源标签冲突',
    'source_review': '来源或同名实体需核验',
    'ambiguous_target': '多人/拼图目标不明确',
    'text_identity_cue': '图片含姓名/事实文字提示',
    'near_duplicate_review': '近似场景/同源图片需核验',
    'metadata_conflict': '实体名称/类型字段冲突',
}


def read_rows(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def save_json(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def save_rows(name, records):
    (OUT / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))


def csv_rows(name, records):
    fields = ['number', 'unit_id', 'qid', 'entity_name', 'entity_kind', 'role',
              'priority', 'status', 'categories', 'observations', 'fact',
              'file', 'absolute_path', 'source_url', 'source_host', 'sha256',
              'visual_screened', 'identity_verified', 'source_page_verified',
              'url_name_token_match', 'annotation_files']
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for row in records:
            row = dict(row)
            for key in ('categories', 'observations', 'annotation_files'):
                row[key] = ' | '.join(row[key])
            # Prevent external labels/URLs from becoming spreadsheet formulas.
            for key, value in row.items():
                if isinstance(value, str) and value[:1] in ('=', '+', '-', '@'):
                    row[key] = "'" + value
            w.writerow(row)


def priority(cats):
    if {'content_conflict', 'metadata_conflict', 'explicit_reference_conflict'} & set(cats):
        return 'P0_先隔离复核'
    if {'source_review', 'ambiguous_target'} & set(cats):
        return 'P1_核验身份与目标'
    if cats:
        return 'P2_文字与独立性审查'
    return 'P3_未见明显异常但身份未核实'


def build():
    units = read_rows(OUT / 'all_units.jsonl')
    prep = json.loads((OUT / 'preparation.json').read_text())
    assert len(units) == 1225 and prep['role_images'] == 2450
    flags = defaultdict(list)
    pages = []
    annotation_sha = {}
    for path in sorted((OUT / 'annotations').glob('*.json')):
        raw = path.read_bytes()
        annotation_sha[path.name] = hashlib.sha256(raw).hexdigest()
        data = json.loads(raw)
        pages.extend(data['pages'])
        for number, roles, cat, note in data['flags']:
            assert 1 <= number <= len(units) and roles in ('L', 'E', 'LE')
            assert cat in LABELS and note
            for code in roles:
                role = {'L': 'learning', 'E': 'evaluation'}[code]
                entry = dict(category=cat, observation=note, annotation_file=path.name)
                if entry not in flags[(number, role)]:
                    flags[(number, role)].append(entry)
    assert sorted(pages) == list(range(1, math.ceil(len(units) / 20) + 1))
    for page in pages:
        assert (OUT / 'contact_sheets' / f'page_{page:03d}.jpg').is_file()
    records = []
    gallery_units = []
    repair_units = []
    (OUT / 'thumbnails').mkdir(exist_ok=True)
    for unit in units:
        gallery = {k: unit[k] for k in ('number', 'qid', 'entity_name', 'entity_kind', 'fact', 'unit_id')}
        gallery['roles'] = {}
        unit_cats = set()
        for role, image in unit['roles'].items():
            observations = flags[(unit['number'], role)]
            cats = sorted({v['category'] for v in observations})
            unit_cats.update(cats)
            status = 'flagged_for_review' if cats else 'no_obvious_content_conflict_identity_unverified'
            record = dict(number=unit['number'], unit_id=unit['unit_id'], qid=unit['qid'],
                entity_name=unit['entity_name'], entity_kind=unit['entity_kind'], role=role,
                priority=priority(cats), status=status, categories=cats,
                observations=[v['observation'] for v in observations], fact=unit['fact'],
                **{k: image[k] for k in ('file', 'absolute_path', 'source_url', 'source_host', 'sha256')},
                visual_screened=True, screening_page=unit['page'], screening_method='assistant_contact_sheet_review',
                identity_verified=False, source_page_verified=False, human_approved=False,
                url_name_token_match=image['name_tokens_in_source_url'],
                annotation_files=sorted({v['annotation_file'] for v in observations}),
                prediction_or_correctness_used=False)
            records.append(record)
            thumb = OUT / 'thumbnails' / f'{unit["number"]:04d}_{role}.jpg'
            if not thumb.exists():
                with Image.open(image['absolute_path']) as im:
                    im = ImageOps.contain(ImageOps.exif_transpose(im).convert('RGB'), (520, 350))
                    im.save(thumb, quality=86)
            assert thumb.is_file()
            gallery['roles'][role] = dict(
                thumb=os.path.relpath(thumb, OUT),
                image=os.path.relpath(image['absolute_path'], OUT),
                source=image['source_url'], file=image['file'], categories=cats,
                notes=record['observations'], priority=record['priority'],
                url_name_token_match=record['url_name_token_match'])
        gallery['categories'] = sorted(unit_cats)
        gallery['priority'] = priority(unit_cats)
        gallery_units.append(gallery)
        repair_units.append(dict(number=unit['number'], unit_id=unit['unit_id'], qid=unit['qid'],
            entity_name=unit['entity_name'], categories=sorted(unit_cats), priority=priority(unit_cats),
            learning_flags=gallery['roles']['learning']['categories'],
            evaluation_flags=gallery['roles']['evaluation']['categories'],
            recommended_action=('quarantine_pending_repair' if priority(unit_cats).startswith('P0')
                else 'verify_before_reuse'),
            original_data_modified=False, verified_clean=False))
    assert len(records) == len({(r['unit_id'], r['role']) for r in records}) == 2450
    assert all(r['visual_screened'] and not r['identity_verified'] for r in records)
    counts = {}
    for cat, label in LABELS.items():
        selected = [r for r in records if cat in r['categories']]
        counts[cat] = dict(label=label, images=len(selected), units=len({r['unit_id'] for r in selected}),
                          **dict(Counter(r['role'] for r in selected)))
    flagged = [r for r in records if r['categories']]
    p0 = [r for r in records if r['priority'].startswith('P0')]
    summary = dict(scope='formal_experiment_roles_only', units=1225, role_images=2450,
        source_package_entities=1478, source_package_candidates=7390,
        reserve_candidates_visually_reviewed=False, reviewed_contact_pages=sorted(pages),
        all_selected_image_sha_verified=prep['original_sha_verified'],
        original_data_modified=False, source_pages_verified=0, identities_certified=0,
        flagged_images=len(flagged), flagged_units=len({r['unit_id'] for r in flagged}),
        no_flag_images=len(records)-len(flagged), categories=counts,
        priority_images=dict(Counter(r['priority'] for r in records)),
        priority_units=dict(Counter(u['priority'] for u in repair_units)),
        candidate_quarantine_images=len(p0),
        candidate_quarantine_units=len({r['unit_id'] for r in p0}),
        annotation_sha256=annotation_sha,
        interpretation='All roles screened visually. Flags are not all proven identity errors; unflagged roles are not certified clean. No model accuracy was used to select flags.')
    save_rows('全部图片审查.jsonl', records)
    csv_rows('全部图片审查.csv', records)
    csv_rows('全部异常清单.csv', flagged)
    csv_rows('建议隔离复核图片.csv', p0)
    csv_rows('身份与目标待核验.csv', [r for r in records if {'source_review', 'ambiguous_target'} & set(r['categories'])])
    csv_rows('文字提示与近重复.csv', [r for r in records if {'text_identity_cue', 'near_duplicate_review'} & set(r['categories'])])
    for cat in LABELS:
        save_rows(f'{cat}.jsonl', [r for r in records if cat in r['categories']])
    save_rows('逐单元修复清单.jsonl', repair_units)
    save_json('建议隔离单元ID.json', [u['unit_id'] for u in repair_units if u['priority'].startswith('P0')])
    save_json('summary.json', summary)
    write_gallery(gallery_units)
    write_report(summary, records)
    print(json.dumps({k: v for k, v in summary.items() if k != 'annotation_sha256'}, ensure_ascii=False, indent=2))


def write_gallery(units):
    payload = json.dumps(units, ensure_ascii=False).replace('<', '\\u003c')
    labels = json.dumps(LABELS, ensure_ascii=False).replace('<', '\\u003c')
    template = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>实体类正式图片全量筛查</title>
<style>body{font:16px/1.6 system-ui,sans-serif;margin:0;background:#f3f5f7;color:#17202a}header{padding:20px;position:sticky;top:0;background:#fff;z-index:1;border-bottom:1px solid #ddd}h1{font-size:23px;margin:0}p{margin:8px 0}input,select,button{font:inherit;padding:7px;margin:4px;border:1px solid #aaa;border-radius:4px}input{width:min(480px,80vw)}main{max-width:1400px;margin:auto;padding:20px}article{background:#fff;border-radius:8px;margin:18px 0;padding:18px;border-left:6px solid #778}article.p0{border-color:#bd3434}article.p1{border-color:#c98720}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}.image{height:350px;display:flex;align-items:center;justify-content:center;background:#eef1f4}.image img{max-width:100%;max-height:350px;object-fit:contain}.badge{display:inline-block;padding:2px 7px;margin:2px;background:#fff1dc;border-radius:5px;font-size:14px}.muted{color:#596673}.path{word-break:break-all;font-size:13px}h2{font-size:21px;margin:0}h3{font-size:17px}ul{padding-left:22px}footer{text-align:center;padding:15px}@media(max-width:700px){.pair{grid-template-columns:1fr}header{position:static}.image{height:280px}.image img{max-height:280px}}</style>
<header><h1>正式实验图片全量筛查：1,225 单元 / 2,450 张图</h1>
<p>已逐页查看全部正式角色。标记包含疑似问题；未标记也没有认证人物身份。来源链接未经网页核验。原数据、训练和成绩未修改。</p>
<input id="query" placeholder="搜索编号、实体名、QID、事实、备注、路径">
<select id="filter"><option value="flagged">全部已标记</option><option value="p0">建议先隔离复核</option><option value="all">全部正式单元</option><option value="unflagged">未见明显异常，身份未核实</option></select>
<span id="status"></span></header><main id="main"></main><footer><button id="prev">上一页</button><span id="page"></span><button id="next">下一页</button></footer>
<script>const data=__DATA__,labels=__LABELS__;let current=0;const size=24;
const $=s=>document.querySelector(s);const esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
for(const [k,v] of Object.entries(labels)){const o=document.createElement('option');o.value=k;o.textContent=v;$('#filter').appendChild(o)}
function choose(){const q=$('#query').value.toLowerCase().trim(),f=$('#filter').value;return data.filter(u=>(f==='all'||(f==='p0'&&u.priority.startsWith('P0'))||(f==='flagged'&&u.categories.length)||(f==='unflagged'&&!u.categories.length)||u.categories.includes(f))&&(!q||JSON.stringify(u).toLowerCase().includes(q)))}
function role(r,name){return `<section><h3>${name} · ${esc(r.priority)}</h3><div class="image"><a href="${esc(r.image)}" target="_blank"><img loading="lazy" src="${esc(r.thumb)}" alt="${name}图片"></a></div><p>${r.categories.map(c=>`<span class="badge">${esc(labels[c])}</span>`).join('')||'<span class="muted">未见明显内容异常；身份未核实</span>'}</p><ul>${r.notes.map(t=>`<li>${esc(t)}</li>`).join('')}</ul><p class="path">${esc(r.file)}</p><a href="${esc(r.image)}" target="_blank">打开原图</a> · <a href="${esc(r.source)}" target="_blank" rel="noreferrer">记录的来源链接</a><p class="muted path">URL姓名词匹配：${r.url_name_token_match?'有':'无'}（不代表身份认证）</p></section>`}
function render(){const items=choose();current=Math.max(0,Math.min(current,Math.ceil(items.length/size)-1));$('#status').textContent=`共 ${items.length} 个单元`;
$('#main').innerHTML=items.slice(current*size,(current+1)*size).map(u=>`<article id="unit-${u.number}" class="${u.priority.startsWith('P0')?'p0':u.priority.startsWith('P1')?'p1':''}"><h2>#${String(u.number).padStart(4,'0')} ${esc(u.entity_name)} · ${esc(u.qid)}</h2><p>${esc(u.priority)}</p><p><b>目标事实：</b>${esc(u.fact)}</p><p class="muted">类别：${esc(u.entity_kind)}；两图身份均未获来源认证。</p><div class="pair">${role(u.roles.learning,'学习 L')}${role(u.roles.evaluation,'测试 E')}</div></article>`).join('');
$('#page').textContent=` ${items.length?current+1:0} / ${Math.ceil(items.length/size)} `;$('#prev').disabled=current===0;$('#next').disabled=(current+1)*size>=items.length;}
$('#query').oninput=()=>{current=0;render()};$('#filter').onchange=()=>{current=0;render()};$('#prev').onclick=()=>{current--;render();window.scrollTo(0,0)};$('#next').onclick=()=>{current++;render();window.scrollTo(0,0)};render();</script></html>'''
    (OUT / 'gallery.html').write_text(template.replace('__DATA__', payload).replace('__LABELS__', labels))


def write_report(summary, records):
    counts = summary['categories']
    lines = [
        '# 实体类正式图片全量筛查报告', '',
        '日期：2026-10-16。**确实存在实体与配图不符，也存在把作品、文字页或游戏角色当作身份图片的情况。已对正式实验全部 1,225 个单元、2,450 张学习/测试图片完成 62 页逐页视觉筛查。**', '',
        '**这次“全量”限定为正式实验实际使用的图片角色。** 原始包有 1,478 个实体、7,390 张候选图；未选中的候选与储备图没有在本次逐页视觉筛查中全部覆盖，不能用它们直接补位。', '',
        f'共标记 **{summary["flagged_images"]} 张图片、{summary["flagged_units"]} 个单元**的至少一类风险。建议优先隔离复核 **{summary["candidate_quarantine_images"]} 张图片，涉及 {summary["candidate_quarantine_units"]} 个单元**。这不是经人工来源认证的“错图率”；风险类别包含需复核项，并且彼此重叠。', '',
        f'另有 {summary["no_flag_images"]} 张没有在缩略图筛查中发现明显异常，**仍未完成来源和身份认证，不能称作干净数据**。本次认证正确的身份数为 0；这也不代表所有身份都错了。', '',
        '## 交付文件', '',
        '- [可搜索、可筛选的图文对照页面](gallery.html)：每个实体并排呈现学习图与测试图，显示事实、依据、原图与来源。',
        '- [全部异常清单 CSV](全部异常清单.csv)：所有已标记角色，包含图片绝对路径、QID、事实、来源和具体理由。',
        '- [建议隔离复核图片 CSV](建议隔离复核图片.csv)：内容异常、显式标签冲突或元数据冲突角色。',
        '- [身份与目标待核验 CSV](身份与目标待核验.csv)：来源、同名检索、多人图的待核验项。',
        '- [文字提示与近重复 CSV](文字提示与近重复.csv)：提示和独立性风险，不自动等同于错图或事实泄露。',
        '- [全部 2,450 张审查记录 CSV](全部图片审查.csv) / [JSONL](全部图片审查.jsonl)。',
        '- [逐单元修复清单](逐单元修复清单.jsonl) / [建议隔离单元 ID](建议隔离单元ID.json)：只是诊断建议，没有修改原始数据或实验产物。',
        '- [分类统计与覆盖记录](summary.json) / [重点案例放大复核图](重点案例复核.jpg)。', '',
        '## 分类统计', '',
        '| 分类 | 图片数 | 学习图 | 测试图 | 涉及单元数 |',
        '| --- | ---: | ---: | ---: | ---: |',
    ]
    for item in counts.values():
        lines.append(f'| {item["label"]} | {item["images"]} | {item.get("learning",0)} | {item.get("evaluation",0)} | {item["units"]} |')
    lines += ['',
        '“内容明显异常/身份素材不适用”包括动漫、作品代替作者、纯文字页、明显不兼容的器物类型等；不能把该类所有记录统一解释为已确定的另一实体。有些绘画可能与目标相关，但并未证明其能承担本实验的身份识别材料，需按预先统一的材料标准复核。', '',
        '“来源需核验”表示现有材料缺少可信对应或存在疑似同名检索；职业场景、服饰或照片年代只是复核线索，不能据此认证或否定人物身份。“目标不明确”是目标绑定风险，并非多人图一律错误。', '',
        '图片写出姓名，可能让模型直接从文字读出实体；图片写出目标事实，才进一步构成事实提示风险。本次只由助手检查可见文字，没有完成全量 OCR、答案泄漏核对或遮字后的因果实验。', '',
        '## 可以直接理解的具体案例', '',
        '| 单元编号 | 目标 | 实际材料与依据 |',
        '| --- | --- | --- |',
        '| 0030 | 自行车运动员 Gal Glivar | 学习、测试图均是动漫角色。 |',
        '| 0403 | Riace bronzes 两尊古希腊武士铜像 | 广场圆形水池、现代金属结构，目标器物类型不符。 |',
        '| 0551 | 政治人物 Katy Clark | 测试广告直接写 CAITLIN CLARK，并呈篮球语境；这里依据的是图片文字，不是凭脸认人。 |',
        '| 0586 | USS Enterprise (CVN-80) 航母 | 测试图是船身明确写 ENTERPRISE STAR 的货轮。 |',
        '| 0801 | USS Iowa (SSN-797) 潜艇 | 两图均为水面战舰；学习图舰号 61，来源文件名直接含 Iowa-Class-Battleship。 |',
        '| 0955 | USS Kingsville (LCS 36) | 学习图是大型舰艇编队；测试图为写 U.S.S. Kearsarge 的历史战舰明信片。 |',
        '| 1144 | HMS Nottingham (1913)，1916 年沉没 | 配图为现代导弹驱逐舰；测试来源 URL 明确写 1982 年试航，同名舰混用。 |',
        '| 0650 | Q310625 / Marc Bartra | 元数据 name 写 Ronaldo，name_zh、Wikipedia URL、学习事实却均为 Marc Bartra，名称关联存在冲突。 |',
        '| 0231、1111 | TotalEnergies 公司、The Santa Fe New Mexican 报纸 | 被标成 built_landmark，且图像走建筑场景；应先修正实体类型与查询信息。 |', '',
        '![重点案例：左学习图，右测试图](重点案例复核.jpg)', '',
        '## 为什么会进入正式实验', '',
        f'当前构造代码 [{ROOT / "entity_v19_1478/audit/build_package.py"}:173]({ROOT / "entity_v19_1478/audit/build_package.py"}:173) 的图片配对条件只检查：跨实体 SHA 复用、学习/测试 SHA 与来源 URL 不同，以及 dHash 距离大于 5，然后取第一组通过的候选。**这些条件能检查部分重复，不能检查图片是不是目标实体。** 构造记录的 `image_semantics_reviewed: false` 也与此一致。', '',
        '本次发现的路径包括：泛化名称/同名人物检索混淆、作品或影视角色被当作现实人物、同名不同舰船、实体类型错误，以及实体名称字段错配。仅换随机种子、增加训练轮数或调解析器无法修复这些材料问题。', '',
        '## 对当前实验和修复的含义', '',
        'T→V 的测试端、V→T 的训练端、V→V 的两端均可能受错误图片影响。现有四方向分数可以保留为当前有问题材料上的描述性结果，不能据此验证干净的跨模态身份绑定与迁移；也不能单凭本次材料筛查断言分数异常已经完全得到解释或证明预训练泄露。', '',
        '建议先隔离 P0 单元并替换图片或修正实体关联，再复核 P1 的来源与目标、P2 的文字和独立性。未标记 P3 也需要来源认证。每个入选角色应保存可核验的来源页、目标对应依据和明确的目标位置；同名对象要保留 QID、完整消歧名、年代/型号。', '',
        '身份/素材验收与是否包含文字提示应分开记录；图中有姓名不自动算错图。若研究目标是视觉身份绑定，应制定统一的去姓名/去答案文字版本，再独立检查遮字是否损坏目标。', '',
        '验收后按同一批知识单元重建文本/视觉四方向和 direct/statement 两题型。**仅从现有评测结果里删除坏图对应题，无法消除已经用坏图训练造成的影响，最终实验需要重新训练并评测。** 本轮未启动重训，也未按模型答题成绩筛选材料。', '',
        '## 覆盖与限制', '',
        '- 核对冻结来源清单、源实体文件 SHA，以及全部正式图片的 SHA；没有改动原图、原始标签、训练权重、预测或成绩。',
        '- 62/62 接触表完成逐页视觉筛查；每页至多 20 个单元，每单元学习/测试各一张。重点明确案例另作放大复核。',
        '- 审查由助手完成，尚无人类确认；大部分人像的具名身份没有来源认证，不根据面部相似度认人。',
        '- 图片来源 URL 只作为已有元数据；本轮没有逐一访问和核对来源网页。599 张 URL 含多词姓名/别名词，不等于这 599 张正确。',
        '- 缩略图可能漏掉小字、局部目标、细微器物差异；这是一轮覆盖全部正式图片的筛查，不是所有身份和事实都已结案。',
        '- 储备和其余候选图未完成同等视觉覆盖；不能把未标记图片或储备图直接叫作“已清洗集合”。', '',
    ]
    (OUT / '正式图片全量筛查报告.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    build()
