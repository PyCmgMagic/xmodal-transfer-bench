"""Build a Chinese briefing from the completed experiment's audited statistics.

No inference, training, or rescoring is performed. Original artifacts are read only.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'runs/entity_v19_transfer_ep5_20261014'
SCORED = WORK / 'rescoring/native-empty-think-v2'
OUT = WORK / 'reports/briefing_20261016'
MODELS = {'qwen3': 'Qwen3-VL-4B-Instruct', 'glm46': 'GLM-4.6V-Flash'}
TYPES = {'overall': '总体', 'direct': '直接问答题', 'statement': '陈述选择题'}
DIRS = ('TT', 'TV', 'VT', 'VV')
CONTRASTS = {
    'TT_minus_TV': 'T→T − T→V（原始分数）',
    'VV_minus_VT': 'V→V − V→T（原始分数）',
    'gain_TT_minus_TV': '文本学习：ΔT→T − ΔT→V',
    'gain_VV_minus_VT': '视觉学习：ΔV→V − ΔV→T',
    'TT_minus_VT': '文本测试：T→T − V→T',
    'VV_minus_TV': '图片测试：V→V − T→V',
    'matched_minus_cross': '同模态平均 − 跨模态平均',
}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(headers, rows):
    return '\n'.join([
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join('---' for _ in headers) + ' |',
        *('| ' + ' | '.join(map(str, row)) + ' |' for row in rows),
    ])


def export(name, records):
    p = OUT / 'tables' / f'{name}.csv'
    with p.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def statrow(model, category, name, value, **extra):
    return dict(model=model, question_type=category, comparison=name,
                difference_pp=value['difference_pp'],
                ci95_low_pp=value['ci95_pp'][0], ci95_high_pp=value['ci95_pp'][1],
                p_holm=value['p_holm'], units=value['units'], **extra)


def statcells(value):
    low, high = value['ci95_pp']
    return [f"{value['difference_pp']:+.2f}", f'[{low:+.2f}, {high:+.2f}]',
            f"{value['p_holm']:.4f}"]


def figurelink(name, label):
    return f'![{label}]({OUT / "figures" / (name + ".png")})'


def savefig(fig, name):
    fig.savefig(OUT / 'figures' / f'{name}.png', dpi=180, bbox_inches='tight')
    fig.savefig(OUT / 'figures' / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def build():
    for sub in ('figures', 'tables'):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    stats_path = SCORED / 'reports/statistics.json'
    s = read(stats_path)
    completeness = read(SCORED / 'reports/completeness.json')
    protocol = read(SCORED / 'scoring_protocol.json')
    assert s['complete'] and completeness['complete']
    assert s['evaluation_seeds'] == [42]
    assert completeness['logical_predictions'] == completeness['expected_predictions'] == 220576
    assert len(s['per_state_metrics']) == completeness['completed_model_states'] == 22
    assert sum(c['total'] for state in s['per_state_metrics'].values()
               for c in state['cells']) == 220576
    recovered = sum(g.get('recovered_exact_empty_think_letters', 0) for g in protocol['groups'])
    assert recovered == 22812
    labels = [json.loads(line) for line in (WORK / 'data/labels.jsonl').read_text().splitlines()]
    gold = {}
    for test in ('text', 'image'):
        counts = Counter(row['answer'] for row in labels if row['test'] == test)
        assert counts == Counter(dict.fromkeys('ABCD', 2450)), counts
        gold[test] = dict(counts)

    endpoint, gains, contrasts, controls, control_contrasts, rotations, options = [], [], [], [], [], [], []
    matrix_rows, gain_rows, type_rows, base_rows, contrast_rows, control_rows = [], [], [], [], [], []
    option_rows, score_rows, latest_rows = [], [], []
    for model, label in MODELS.items():
        for category, category_label in TYPES.items():
            x = s['endpoint'][model]['all'][category]
            row = [label, category_label]
            for direction in DIRS:
                acc = x['mean_accuracy'][direction]
                endpoint.append(dict(model=model, question_type=category, direction=direction,
                                     accuracy_pct=100 * acc,
                                     baseline_pct=100 * x['baseline_accuracy']['text' if direction[-1] == 'T' else 'image'],
                                     gain_pp=x['gain_stats'][direction]['difference_pp'],
                                     units=x['units'], seed=42, epoch=5, rotations=4))
                row.append(f'{100 * acc:.2f}')
                gains.append(statrow(model, category, direction, x['gain_stats'][direction]))
            type_rows.append(row)
            if category == 'overall':
                matrix_rows.append([label, *row[2:]])
                gain_rows.append([label, *(f"{x['gain_stats'][d]['difference_pp']:+.2f}" for d in DIRS)])
            base_rows.append([label, category_label,
                              f"{100 * x['baseline_accuracy']['text']:.2f}",
                              f"{100 * x['baseline_accuracy']['image']:.2f}"])
            for name, value in x['contrasts'].items():
                contrasts.append(statrow(model, category, name, value))
                if category == 'overall':
                    contrast_rows.append([label, CONTRASTS[name], *statcells(value)])
        for state, values in s['image_controls'][model].items():
            state_label = {'base': '基座', 'TEXT_seed42_epoch5': '文本学习 E5',
                           'VISION_seed42_epoch5': '视觉学习 E5'}[state]
            for category in TYPES:
                x = values[category]
                controls.append(dict(model=model, state=state, question_type=category,
                                     **{key + '_pct': 100 * value for key, value in x['accuracy'].items()},
                                     units=256, rotations=4))
                for name, value in x['contrasts'].items():
                    control_contrasts.append(statrow(model, category, name, value, state=state))
                if category == 'overall':
                    control_rows.append([label, state_label,
                                         *(f"{100 * x['accuracy'][k]:.2f}" for k in ('image', 'no_image', 'wrong_image'))])
        for mode, train_char in (('TEXT', 'T'), ('VISION', 'V')):
            state = f'evaluation/{model}/seed_42/{mode}/epoch_5'
            for test, test_char in (('text', 'T'), ('image', 'V')):
                cells = [c for c in s['per_state_metrics'][state]['cells'] if c['test'] == test]
                distribution = Counter()
                for c in cells:
                    distribution.update(c['predicted_options'])
                n = sum(c['total'] for c in cells)
                invalid = sum(c['invalid'] for c in cells)
                assert n == 9800
                assert sum(distribution.values()) == n
                options.append(dict(model=model, direction=train_char + test_char, total=n,
                                    **{k + '_pct': 100 * distribution[k] / n for k in 'ABCD'},
                                    invalid_pct=100 * invalid / n))
                option_rows.append([label, train_char + '→' + test_char,
                                    *(f'{100 * distribution[k] / n:.2f}' for k in 'ABCD'),
                                    f'{100 * invalid / n:.2f}'])
                for rotation in range(4):
                    subset = [c for c in cells if c['rotation'] == rotation]
                    rotations.append(dict(model=model, direction=train_char + test_char,
                                          rotation=rotation,
                                          accuracy_pct=100 * sum(c['correct'] for c in subset) / sum(c['total'] for c in subset)))
        for category in TYPES:
            x = s['endpoint'][model]['latest398_exploratory'][category]
            latest_rows.append([label, TYPES[category], *(f"{100 * x['mean_accuracy'][d]:.2f}" for d in DIRS)])
        for g in protocol['groups']:
            if g['model'] == model and g['test'] == 'text' and (
                g['state'].startswith('base/') or g['state'].endswith('epoch_5')
            ):
                state = '基座' if g['state'].startswith('base/') else ('T→T' if '/TEXT/' in g['state'] else 'V→T')
                score_rows.append([label, state, f"{100 * g['old_correct'] / g['n']:.2f}",
                                   f"{100 * g['new_correct'] / g['n']:.2f}",
                                   str(g['n'] - g['old_valid']), str(g['n'] - g['new_valid'])])

    for name, records in (('endpoint', endpoint), ('gains', gains), ('contrasts', contrasts),
                          ('image_controls', controls), ('image_control_contrasts', control_contrasts),
                          ('rotations', rotations), ('predicted_options', options)):
        export(name, records)
    curve_records = [dict(model=model, **row) for model in MODELS for row in s['curves_rotation0'][model]]
    export('curves_rotation0', curve_records)
    audit_records = [{key: g.get(key, 0) for key in ('model', 'state', 'test', 'n', 'old_valid', 'new_valid',
                                                   'old_correct', 'new_correct', 'recovered_exact_empty_think_letters')}
                     for g in protocol['groups']]
    export('scoring_audit', audit_records)

    font = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    fontManager.addfont(font)
    plt.rcParams.update({'font.family': FontProperties(fname=font).get_name(), 'font.size': 10,
                         'axes.unicode_minus': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout='constrained')
    for i, (model, label) in enumerate(MODELS.items()):
        for j, (category, category_label) in enumerate(TYPES.items()):
            ax = axes[i, j]
            x = s['endpoint'][model]['all'][category]
            values = np.array([x['mean_accuracy'][d] * 100 for d in DIRS]).reshape(2, 2)
            im = ax.imshow(values, cmap='YlGnBu', vmin=28, vmax=38)
            for n, direction in enumerate(DIRS):
                v = values.flat[n]
                ax.text(n % 2, n // 2,
                        f"{v:.2f}%\nΔ{x['gain_stats'][direction]['difference_pp']:+.2f} pp",
                        ha='center', va='center', color='white' if v > 34.7 else '#12202c', fontsize=12)
            ax.set_xticks([0, 1], ['文字测试 T', '图片测试 V'])
            ax.set_yticks([0, 1], ['文本学习 T', '视觉学习 V'])
            ax.set_title(f'{label}\n{category_label}')
    fig.colorbar(im, ax=axes, shrink=.8, label='Epoch 5 准确率（%）')
    fig.suptitle('四方向终点评测：准确率与相对对应基座的增益\nSeed 42；四次选项轮换平均；V 学习含匿名事实文字', fontsize=14)
    savefig(fig, 'endpoint_matrix')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    colors = ['#2166ac', '#67a9cf', '#b35806', '#f1a340']
    for ax, (model, label) in zip(axes, MODELS.items()):
        for d, color in zip(DIRS, colors):
            rows = sorted((r for r in s['curves_rotation0'][model]
                           if r['probe_id'] == 'overall' and r['direction'] == d), key=lambda r: r['epoch'])
            assert [r['epoch'] for r in rows] == list(range(6))
            ax.plot([r['epoch'] for r in rows], [100 * r['accuracy'] for r in rows],
                    marker='o', label=d[0] + '→' + d[1], color=color,
                    linestyle='--' if d[0] != d[1] else '-')
        ax.set(title=label, xlabel='训练 epoch（0 为基座）', ylabel='准确率（%）', ylim=(27, 38), xticks=range(6))
        ax.grid(alpha=.2)
        ax.legend(ncol=2)
    fig.suptitle('学习曲线：全程使用 rotation 0，同一计分规则\n曲线 E5 与四轮平均主表不是同一统计口径', fontsize=13)
    savefig(fig, 'learning_curves')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    for ax, (model, label) in zip(axes, MODELS.items()):
        for i, state in enumerate(('TEXT_seed42_epoch5', 'VISION_seed42_epoch5')):
            for j, name in enumerate(('image_minus_no_image', 'image_minus_wrong_image')):
                v = s['image_controls'][model][state]['overall']['contrasts'][name]
                y = 3 - (i * 2 + j)
                low, high = v['ci95_pp']
                ax.errorbar(v['difference_pp'], y, xerr=[[v['difference_pp'] - low], [high - v['difference_pp']]],
                            fmt='o', color=colors[i * 2], capsize=4)
                ax.text(6.15, y, f"p={v['p_holm']:.3f}", va='center', fontsize=9)
        ax.axvline(0, color='gray', linestyle='--')
        ax.set(title=label, xlim=(-4, 8), ylim=(-.5, 3.5), xlabel='正确图片带来的准确率差（pp）',
               yticks=[3, 2, 1, 0], yticklabels=['文本学习：正确−无图', '文本学习：正确−错图',
                                             '视觉学习：正确−无图', '视觉学习：正确−错图'])
        ax.grid(axis='x', alpha=.2)
    fig.suptitle('图片依赖对照：训练后未见明确的正确图片优势\n预选 256 单元；误差线为普通配对 95% CI；p 经各自两项 Holm 校正', fontsize=13)
    savefig(fig, 'image_dependence')

    vision_ci_rows = []
    for model, label in MODELS.items():
        for key, text in (('image_minus_no_image', '正确−无图'), ('image_minus_wrong_image', '正确−错图')):
            value = s['image_controls'][model]['VISION_seed42_epoch5']['overall']['contrasts'][key]
            vision_ci_rows.append([label, text, *statcells(value)])
    overall_gain_ci_rows = [[MODELS[r['model']], r['comparison'][0] + '→' + r['comparison'][1],
                             f"{r['difference_pp']:+.2f}", f"[{r['ci95_low_pp']:+.2f}, {r['ci95_high_pp']:+.2f}]",
                             f"{r['p_holm']:.4f}"] for r in gains if r['question_type'] == 'overall']
    material_audit = ROOT / 'diagnostics/entity_v19_material_audit_20261016/材料审核补充报告.md'
    material_note = (f'**后续材料审核补充（2026-10-16）：** 独立于模型成绩抽查 32 个单元，'
                     f'发现动漫、插画和地理对象冲突等进入正式图片角色。图片选择代码只有差异检查，'
                     f'没有实体语义验收；现有数字需作为当前材料上的描述性结果解释。'
                     f'详见[材料审核补充报告]({material_audit})。抽查未认证其余图片身份，'
                     f'异常比例不能直接外推为全量错误率。\n\n' if material_audit.exists() else '')
    full_audit_dir = ROOT / 'diagnostics/entity_v19_full_image_audit_20261016'
    if (full_audit_dir / 'summary.json').exists():
        audit_summary = json.loads((full_audit_dir / 'summary.json').read_text())
        material_note = (
            f'**全量正式图片筛查补充（2026-10-16）：** 已逐页查看正式实验全部 '
            f'{audit_summary["units"]:,} 个单元、{audit_summary["role_images"]:,} 张图片。'
            f'标记 {audit_summary["flagged_images"]:,} 张图片的至少一类风险；'
            f'建议先隔离复核 {audit_summary["candidate_quarantine_units"]} 个单元。'
            f'存在动漫代替现实人物、潜艇配战舰、同名不同舰船及实体字段冲突。'
            f'风险标记不等于全部已确定错图，未标记人像也未认证身份；储备候选图未完成同等视觉覆盖。'
            f'现有结果仅能作为当前材料上的描述性结果，不能验证干净的跨模态身份绑定与迁移。'
            f'详见[正式图片全量筛查报告]({full_audit_dir / "正式图片全量筛查报告.md"})及'
            f'[图文对照页面]({full_audit_dir / "gallery.html"})。原始训练、预测与成绩保留。\n\n')
    text = f'''# 实体类知识跨模态迁移实验汇报

汇报日期：2026-10-16（北京时间）；结果截至 10 月 15 日全部完成。数据集：`entity_v19_1478`。主结果：Seed 42、固定 Epoch 5、四轮选项位置平均。

## 1. 核心结论

{material_note}\
**训练带来小幅答题收益，但目前还不足以证明模型已经形成可靠的“图片身份→所学实体知识”检索能力。** 四个方向的总体增益均为正；两模型、两题型没有呈现一致的同模态优势。正确图、无图、错图的训练后对照未检出明确的正向图片身份收益，是解释迁移矩阵时最需要重视的结果。

Qwen3 的总体准确率为 32.49%–34.51%，相对基座增加 1.37–3.32 个百分点。GLM 为 33.97%–36.73%，增加 2.67–4.87 个百分点。答题表现有所改善，但提升幅度有限，不能只凭终点分数接近就宣称“迁移充分”。

GLM 总体同模态平均比跨模态高 0.95 pp（Holm p=0.0035），但视觉学习后的 V→V 并不明确优于 V→T。Qwen3 总体同模态优势不显著，而且两题型呈现不同方向的结果。

**评分口径须披露：以下内容正确率采用事后统一的格式规范化评分。** GLM 训练后出现的空 `<think></think>` 加单字母答案曾被旧解析器拒绝，导致旧 V→T 几乎为零。修正只处理无歧义格式，未重新推理、未改动原始预测或原始评分；详细审计见第 8 节。

## 2. 实验范围与四方向含义

| 项目 | 本次实际范围 |
| --- | --- |
| 模型 | Qwen3-VL-4B-Instruct；GLM-4.6V-Flash |
| 配对知识单元 | 1,225；每单元各有 direct、statement 一题 |
| 每方向原始题量 | 2,450；终点四轮轮换共 9,800 条逻辑答案 |
| 文本学习 T | 实体名称 + 事实正文 |
| 视觉学习 V | 学习图片 + 匿名事实正文，仍包含文字监督 |
| 文本测试 T | 用实体名称询问知识 |
| 图片测试 V | 用另一张实体图片替代名称，询问知识 |
| 主终点 | 固定第 5 个 epoch，不按测试分数挑 checkpoint |
| 正式评测 seed | 仅 42，按用户缩减范围；43/44 已有训练产物保留但不汇总 |
| 已完成训练 | 原方案 2 模型 × 2 学习模式 × 3 seeds = 12 项，均为 5 epochs |
| 已完成评测 | 22/22 模型状态；220,576/220,576 条逻辑答案 |

T→T 是文本学习后做文本题，T→V 是文本学习后做图片题；V→T、V→V 类推。同一学习模型用于两种测试，不是分别为四方向训练四个模型。学习与测试使用同一批知识单元，目标是测所学知识在不同输入模态下的可访问性；不是测试未学习实体上的泛化。学习图与测试图在所选角色间全局 SHA 不重叠，但这不等于语义和近重复内容都已人工确认。

两题型是不同任务：**direct 为针对属性的直接选择问答，statement 为从完整陈述中选择正确说法**。陈述题可以围绕单事实构造，不应自动称为多事实题。两类各 1,225 题，下文分别报告；总体是等权汇总。

基座与 E5 对每题进行四次循环轮换，使正确答案各到 A/B/C/D 一次。轮换后的逻辑答案不是四份独立知识样本；统计重采样单元仍为 1,225 个知识单元。相同输入可复用确定性输出，220,576 是逻辑评测条数。

## 3. 主结果：先看终点，再看增益

准确率（%）：

{table(['模型', 'T→T', 'T→V', 'V→T', 'V→V'], matrix_rows)}

相对对应输入模态基座的增益（pp，百分点）：

{table(['模型', 'ΔT→T', 'ΔT→V', 'ΔV→T', 'ΔV→V'], gain_rows)}

基座分数（%，与主终点同为四轮平均）：

{table(['模型', '题型', '文本输入', '图片输入'], base_rows)}

增益的配对区间与检验：

{table(['模型', '方向', '增益 pp', '95% CI pp', 'Holm p'], overall_gain_ci_rows)}

总体八项增益在各模型四方向的 Holm 家族内均达到 p<0.05，但这只说明当前题目上的答题表现改变，不能将所有提升都归因于目标事实记忆，更不能单独证明视觉身份绑定。T→T、V→T 减文本基座，T→V、V→V 减图片基座，不能统一减 25%。

{figurelink('endpoint_matrix', '四方向准确率与基座增益')}

## 4. 四方向究竟有什么差异

为避免把输入本身难度差异当成迁移损失，既看原始准确率，也看减去对应基座后的增益。定义 Δ 表示上述增益。同模态平均−跨模态平均为 `(TT + VV − TV − VT) / 2`；此项的两种测试模态基座会相互抵消。

{table(['模型', '比较', '差值 pp', '95% CI pp', 'Holm p'], contrast_rows)}

对 Qwen3，文本学习带给图片题的增益比带给文本题的增益大 1.95 pp（p=0.0156）。视觉学习后 V→T 原始分数比 V→V 高 2.02 pp，但减去文本/图片基座差后，V→V 的增益反而比 V→T 高 0.78 pp，且没有显著差异。因此不能依据原始分数判定“视觉转文字更好”。总体同模态对比 p=0.2350，没有明确优势或劣势。

对 GLM，同为文本测试时，文本学习比视觉学习高 2.19 pp（p=0.0007）；同为图片测试时，两种学习方式只差 0.29 pp，未见明确差异。总体同模态优势为 0.95 pp，但文本学习和视觉学习各自的基座校正增益差都未达到显著，因此不能概括为两个方向都存在稳定迁移损失。

这些检验分别在每模型、每题型的七个比较内做 Holm 校正；不是跨模型差值检验，也没有将“某模型显著、另一模型不显著”等同于模型间差异显著。

## 5. 两类题必须分开解读

{table(['模型', '题型', 'T→T %', 'T→V %', 'V→T %', 'V→V %'], type_rows)}

Qwen3 的 direct 题，视觉学习带给 V→V 的增益比 V→T 大 **3.14 pp**（95% CI [1.47, 4.78]，Holm p=0.0028），有局部视觉同模态增益信号。但 statement 的 V→V 只有 30.20%，低于 V→T 的 33.88%；该题型同模态平均比跨模态低 **1.95 pp**（p=0.0007）。同一模型两题型的描述性趋势明显不同，不能用总体一个数代替。这里没有另做题型×模态交互检验，不能只凭两类 p 值不同宣称交互显著。

GLM 的 direct 四方向为 33.80%–37.45%，statement 为 33.98%–36.02%。同模态平均优势在 direct 为 +0.90 pp（p=0.0985），statement 为 +1.01 pp（p=0.0384）；效应点估计接近，不能解释成“只有 statement 存在效应”。全部题型的增益和七项比较见 CSV。

## 6. 图片对照：当前证据的主要缺口

在推理前固定抽取 256 个单元，每单元两题、四轮选项位置。对照为正确测试图片、移除图片、跨 QID 错图。错图配对没有自身回环；正确图结果复用主评测。以下数值只来自这 256 个单元，不能直接与全量 1,225 单元分数作条件差。

{table(['模型', '状态', '正确图 %', '无图 %', '错图 %'], control_rows)}

重点看视觉学习 E5 的配对差：

{table(['模型', '对照', '差值 pp', '95% CI pp', 'Holm p'], vision_ci_rows)}

**训练后的正确图优势尚未得到统计支持。** 两个模型在 TEXT、VISION E5 的总体对照中都没有检出明确的正向正确图收益。Qwen3 基座反而是正确图比无图低 3.61 pp（p=0.0114），所以不能说“所有图片对照都没有显著差异”。

{figurelink('image_dependence', '训练后正确图片对照的差值与置信区间')}

这提示模型可能借助语言侧的学习、题干或选项线索完成部分题目；也可能受图片可识别性、绑定难度影响。但本轮不能确定哪一项是主因。未显著不等于证明模型完全不看图：子集只有 256 个单元，区间仍容许一定图片收益，图像语义也未穷尽人工核验。

## 7. 学习过程与最新构建子集

{figurelink('learning_curves', '统一 rotation 0 的学习曲线')}

曲线只比较 rotation 0 的 epoch 0–5。GLM 的 T→T 大体持续上升；Qwen3 多方向早期提升后波动，E5 也未呈现所有方向同步提高。终点仍按预设 E5 报告，没有从曲线中挑最高点。不能把曲线末端与四轮平均主表拼接为同一条曲线，也不能仅凭这些曲线认定多训几轮一定有效。

`entity_v19_1478` 是包名，不代表其中所有单元都来自第 19 次构建版本。实际最新版本 `compare-build-review-19-residual-scope-repair` 只有 **398 个单元**，其余来自较早构建版本。最新子集探索性准确率如下：

{table(['模型', '题型', 'T→T %', 'T→V %', 'V→T %', 'V→V %'], latest_rows)}

该子集仍有小幅总体增益，但绝对分数低于全量汇总。它是探索性分层结果，不替换主样本，也不能据此认定构建版本本身造成差异；不同版本包含的实体和问题组成不同。

## 8. 评分格式修正与选项偏好

旧 GLM 评分器接受裸单字母，或可带空 think 头的原生 boxed 答案；却拒绝空 think 头后直接接一个字母。视觉训练后的文字输入大量出现后一种格式。例如：

```text
<think></think>
B
```

这是答案字母明确但不满足原始输出格式的情况。当前事后规则统一接受“可选的空 think 头 + 单个字母/原生 boxed 字母”；仍拒绝非空推理、说明文字、多个答案及截断输出。Qwen3 原规则不变。规则用于所有基座、训练状态、模态、轮换和对照；并非只修 V→T 或只修答对的样本。

文字测试的旧、新结果对照（失效答案始终按错误计入全样本分母）：

{table(['模型', '状态/方向', '原始准确率 %', '格式规范化准确率 %', '原无效数', '现无效数'], score_rows)}

总计恢复 **{recovered:,} 条逻辑输出**的明确答案格式，未改变先前有效答案，未重新推理。原始 22 个状态的预测与评分保留，审计记录含对应 SHA-256。主结果应标为“事后格式规范化内容评分”；原始评分仍可用于报告严格格式遵循情况，不能把它包装成事先冻结的评分协议。

终点预测字母分布（每行 9,800 条，百分比含无效输出分母）：

{table(['模型', '方向', 'A %', 'B %', 'C %', 'D %', '无效 %'], option_rows)}

**标准答案在四轮合并后严格均衡，模型预测并不均衡。** 已实际核对每种测试输入的 gold A/B/C/D 各 2,450 条。比如 Qwen3 T→T 的 A 占 55.38%，D 仅 4.24%。轮换降低了只报告某个固定答案位置造成的偏差，但没有消除模型的位置偏好、选项内容线索或先验知识。纯粹固定选同一字母在这种平衡评测中是 25%，内容相关猜测则不必是 25%；高于 25% 本身不是泄露证据。

## 9. 能支持的结论、限制与后续优先级

本轮支持：当前训练方案能提高部分实体选择题表现；方向差异依赖模型和题型；GLM 有较小的总体同模态优势，Qwen3 没有统一的总体同模态优势。尚不能支持：可靠的实体视觉身份绑定、充分或无损的跨模态迁移、训练随机性上的稳定性，以及“高于随机必然是泄露”。

仅评测 Seed 42。置信区间来自知识单元的 10,000 次配对 bootstrap；轮换先在题内平均，总体再把两题在单元内平均。区间反映当前 seed 下的单元抽样不确定性，不是多 seed 标准差。表中 95% CI 是普通区间，不是多重比较同时区间；显著性解释采用相应家族的 Holm p，不能仅看区间是否越过零。

VISION 仍有匿名文字监督，视觉编码器和对齐模块冻结，只有语言侧 LoRA 可训练；这些是本轮具体方案，结论不能无条件推广到其他视觉学习方式。图片未穷尽人工语义审查；大多数事实缺少首次公开时间，主要依据 Wiki 版本变化，不能据此证明每条知识都晚于模型训练截止时间。训练后准确率仍低，尚需区分事实获取不足、识别失败和测量线索影响。

建议下一步依次进行：

1. **独立审核测量材料。** 抽查学习图/测试图是否确为同一实体、是否足以辨认、错图是否合适，以及题干/选项是否存在绕开实体身份的线索。按预定标准审核，不按模型答对/答错筛题来追求 25%。
2. **拆开检查知识获取与视觉绑定。** 先用独立的事实探测或似然指标判断正文知识是否确实学入，再单独测名称↔图片绑定；避免把所有低分归因于迁移。
3. **冻结下一轮评分和对照。** 将空 think 格式兼容写入事先协议，对修复后的材料进行独立评测；当前结果继续保留事后修正标记。当前按单 seed 完成，不追加其他 seed。

这份汇报仅整理已完成结果，不启动新训练或推理。

## 10. 复核与文件

训练概要：PT 事实正文学习，不套聊天模板、全正文损失；LoRA rank 8 / alpha 32 / dropout 0.05，学习率 1e-4，batch 1、累积 16，max length 8192、max pixels 1,003,520；视觉编码器/对齐模块冻结。推理 greedy、关闭 thinking、max new tokens 32，batch 1。完整参数以 manifest 与 scope override 为准。

本汇报直接读取已完成统计，不重跑 bootstrap 或推理。构建脚本核验了评测总数、状态数、单 seed 范围、格式恢复数、终点分母和实际 gold 分布。

- [完整当前统计 JSON]({stats_path})
- [当前技术统计表]({SCORED / 'reports/report.md'})
- [评测完整性]({SCORED / 'reports/completeness.json'})
- [格式修正审计]({SCORED / 'scoring_protocol.json'})
- [当前/原始评分说明]({WORK / 'CURRENT_RESULTS.md'})
- [原始设计 manifest]({WORK / 'manifest.json'})；[单 seed 实际范围]({WORK / 'evaluation_scope.json'})
- [主结果 CSV]({OUT / 'tables/endpoint.csv'})；[基座增益与区间]({OUT / 'tables/gains.csv'})；[全部配对比较]({OUT / 'tables/contrasts.csv'})
- [图片对照分数]({OUT / 'tables/image_controls.csv'})；[图片对照差值]({OUT / 'tables/image_control_contrasts.csv'})
- [轮换分数]({OUT / 'tables/rotations.csv'})；[预测选项分布]({OUT / 'tables/predicted_options.csv'})；[学习曲线]({OUT / 'tables/curves_rotation0.csv'})
- [评分旧新对照]({OUT / 'tables/scoring_audit.csv'})；[本汇报生成脚本]({Path(__file__).resolve()})

三张图各提供 PNG 和同名 PDF，位于 [figures 目录]({OUT / 'figures'})。
'''
    report = OUT / '实体类跨模态迁移实验汇报.md'
    report.write_text(text, encoding='utf-8')
    validation = dict(complete=True, logical_predictions=220576, model_states=22, seed=42,
                      recovered_exact_empty_think_letters=recovered, gold_by_test=gold,
                      stats_sha256=digest(stats_path), scoring_protocol_sha256=digest(SCORED / 'scoring_protocol.json'),
                      report_sha256=digest(report), endpoint_rows=len(endpoint),
                      contrast_rows=len(contrasts), control_rows=len(controls),
                      bootstrap_reused=True, inference_rerun=False)
    (OUT / 'validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(report=str(report), figures=3, tables=9, validation=validation), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    build()
