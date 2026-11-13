#!/usr/bin/env python3
"""Audit a completed pilot without modifying its inputs or primary results."""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory.resolve()
    primary = json.loads((root / "report.json").read_text())
    assert primary["status"] == "complete"
    samples = {r["id"]: r for r in rows(root / "eval.jsonl")}
    predictions = rows(root / "predictions.jsonl")
    conditions = ("original_1000_rule_sft", "augmented_12_rule_sft")
    audit, stats, indexed = [], {}, {}
    for condition in conditions:
        selected = [r for r in predictions if r["condition"] == condition]
        assert len(selected) == len(samples)
        assert {r["id"] for r in selected} == samples.keys()
        assert len({r["id"] for r in selected}) == len(selected)
        counts = Counter()
        indexed[condition] = {}
        for result in selected:
            sample = samples[result["id"]]
            text = sample["messages"][-1]["content"]
            options = dict(re.findall(r"(?m)^([A-D])\. (.*)$", text))
            assert set(options) == set("ABCD")
            response = result["response"].strip()
            valid = response in "ABCD" and len(response) == 1
            correct = valid and response == sample["gold"]
            assert result["gold"] == sample["gold"]
            assert result["correct"] == correct
            kind = sample["question_type"]
            counts["n"] += 1
            counts["correct"] += correct
            counts["format_errors"] += not valid
            counts[kind + "_n"] += 1
            counts[kind + "_correct"] += correct
            gold_option = options[sample["gold"]]
            predicted_option = options.get(response, "") if valid else ""
            if kind == "diagnosis":
                gold_accept = gold_option.startswith("Accept report.")
                counts["diagnosis_gold_accept"] += gold_accept
                counts["diagnosis_predicted_accept"] += predicted_option.startswith("Accept report.")
                action = "accept" if gold_accept else "replace"
                counts["gold_" + action + "_n"] += 1
                counts["gold_" + action + "_correct"] += correct
            record = {**result, "parsed_answer": response if valid else None,
                      "predicted_option": predicted_option, "gold_option": gold_option,
                      "prompt": text}
            indexed[condition][result["id"]] = record
            audit.append(record)
        assert counts["correct"] == primary["results"][condition]["correct"]
        stats[condition] = dict(counts)
    before, after = (indexed[c] for c in conditions)
    transitions = Counter()
    for key in samples:
        transitions[f"{int(before[key]['correct'])}_to_{int(after[key]['correct'])}"] += 1
    checkpoint = Path(primary["checkpoint"])
    provenance = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [root / "train.jsonl", root / "eval.jsonl", root / "predictions.jsonl",
                     checkpoint / "adapter_model.safetensors", checkpoint / "adapter_config.json"]
    }
    (root / "analysis.json").write_text(json.dumps({"statistics": stats, "paired_transitions": dict(transitions), "sha256": provenance}, ensure_ascii=False, indent=2) + "\n")
    with (root / "question_audit.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit[0]))
        writer.writeheader()
        writer.writerows(audit)
    lines = ["# v7 小规模应用训练结果", "", "同一组12条二值规则的原始24道T2T题；不提供材料，不开启COT。", "",
             "| 模型 | 总计 | direct | diagnosis | 格式错误 |",
             "|---|---|---|---|---|"]
    labels = {conditions[0]: "原1000规则SFT", conditions[1]: "新12规则应用SFT"}
    for condition in conditions:
        s = stats[condition]
        lines.append(f"| {labels[condition]} | {s['correct']}/{s['n']}（{s['correct']/s['n']:.2%}） | {s.get('direct_correct',0)}/{s['direct_n']} | {s.get('diagnosis_correct',0)}/{s['diagnosis_n']} | {s.get('format_errors',0)} |")
    lines += ["", f"逐题配对：原错→新对 {transitions['0_to_1']}；原对→新错 {transitions['1_to_0']}；两次都对 {transitions['1_to_1']}；两次都错 {transitions['0_to_0']}。", "", "## 报告接受偏好", ""]
    for condition in conditions:
        s = stats[condition]
        lines.append(f"- {labels[condition]}：12道报告核验题中接受报告 {s.get('diagnosis_predicted_accept',0)} 次；标准答案接受 {s.get('diagnosis_gold_accept',0)} 次。应纠正报告题答对 {s.get('gold_replace_correct',0)}/{s.get('gold_replace_n',0)}。")
    lines += ["", "## 修改与限制", "", "从原学习规则生成548条训练样本：规则解释、条件与代码回忆、新观察的规则应用、正确/错误报告核验。新观察标签由规则计算，报告核验训练中接受与纠正各128条。测试题及其答案未用于生成训练样本。", "", "仍为Qwen3-VL-4B、LoRA rank8/alpha32/dropout0.05、LR1e-4、batch1/累积16、5个epoch，使用最终checkpoint。原数据集与图片未修改。", "", "本次同时将学习规模从1000条规则降到12条，并增加每条规则的练习；不是同规模消融实验。24道题只覆盖二值规则的文字方向，不能外推到1000条规则、其他规则类型或图片方向。训练练习采用结构化属性表述，仍需独立审核其自然性与覆盖率。", "", "## 逐题变化示例", ""]
    for key in samples:
        if before[key]["response"] == after[key]["response"]:
            continue
        a, b = before[key], after[key]
        lines += [f"### {key}", "", f"原答案：{a['response']}（{'对' if a['correct'] else '错'}）；新答案：{b['response']}（{'对' if b['correct'] else '错'}）；标准答案：{b['gold']}。", "", "```text", b["prompt"], "```", ""]
    (root / "report_zh.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"statistics": stats, "paired_transitions": dict(transitions)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
