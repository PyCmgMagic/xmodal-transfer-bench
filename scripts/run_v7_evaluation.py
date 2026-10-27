#!/usr/bin/env python3
"""Standalone xModal evaluator: preflight, prompts, probes, Swift inference, scoring.

Python >= 3.9; no dependencies beyond the server's existing `swift` installation.
Run --help for options. Gold labels are never passed to Swift. No training occurs.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

VERSION = "xmodal-v7-full-experiment-evaluation-v1"
DIRECTIONS = ("T2T", "T2V", "V2T", "V2V")
SYSTEM = ("You are answering a multiple-choice question. "
          "Return exactly one uppercase letter: A, B, C, or D. "
          "Do not output an explanation, punctuation, markdown, or any other text.")
SUFFIX = ("\n\nAnswer with exactly one uppercase letter (A, B, C, or D). "
          "Do not include an explanation or any other characters.")


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def save_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_rows(path, allow_partial_tail=False):
    """Only an unfinished final line of an interrupted attempt may be ignored."""
    path = Path(path)
    rows, notes = [], []
    data = path.read_bytes()
    lines = data.splitlines()
    for i, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except (ValueError, UnicodeError) as exc:
            if allow_partial_tail and i == len(lines) and not data.endswith(b"\n"):
                notes.append({"file": str(path), "line": i, "reason": "interrupted_partial_tail"})
                break
            raise ValueError(f"Invalid JSON at {path}:{i}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Expected an object at {path}:{i}")
        rows.append(row)
    return rows, notes


def id_map(rows, source):
    result = {}
    for row in rows:
        sid = row.get("id", row.get("sample_id"))
        if not isinstance(sid, str) or not sid or sid in result:
            raise ValueError(f"Missing/duplicate ID in {source}: {sid!r}")
        result[sid] = row
    return result


def checkpoint(path):
    path = Path(path).expanduser().resolve()
    if not (path / "adapter_config.json").is_file() and not (path / "config.json").is_file():
        candidates = [p for p in path.glob("checkpoint-*")
                      if re.fullmatch(r"checkpoint-\d+", p.name)
                      and ((p / "adapter_config.json").is_file() or (p / "config.json").is_file())]
        if not candidates:
            raise ValueError(f"No LoRA or dense checkpoint in {path}; specify --text-adapter / --visual-adapter")
        path = max(candidates, key=lambda p: int(p.name.split("-")[-1]))
    kind = "lora" if (path / "adapter_config.json").is_file() else "dense"
    if kind == "lora":
        weights = sorted([*path.glob("adapter_model*.safetensors"), *path.glob("adapter_model*.bin")])
        tracked = [path / "adapter_config.json", *weights]
    else:
        weights = sorted([*path.glob("model*.safetensors"), *path.glob("pytorch_model*.bin")])
        tracked = [path / "config.json", *weights]
    if not weights or any(p.stat().st_size == 0 for p in weights):
        raise ValueError(f"Missing/empty {kind} weights: {path}")
    return {"path": str(path), "kind": kind, "sha256": {p.name: file_hash(p) for p in tracked}}


def default_adapter(root, mode):
    """Prefer corrected-v3 runs, while retaining compatibility with old runs."""
    names = {
        "T": ("xmodal-qwen3vl4b-corrected-v3-TEXT", "xmodal-qwen3vl4b-t2t"),
        "V": ("xmodal-qwen3vl4b-corrected-v3-VISION", "xmodal-qwen3vl4b-v2v"),
    }[mode]
    for name in names:
        candidate = root / "runs" / name
        if candidate.exists():
            return candidate
    # Let checkpoint() produce the precise missing-checkpoint error.
    return root / "runs" / names[0]


def prepare_row(row, root, direction, answer_format="strict"):
    messages = row.get("messages")
    if not isinstance(messages, list) or any(not isinstance(m, dict) for m in messages):
        raise ValueError(f"Invalid messages: {row.get('id')}")
    roles = [m.get("role") for m in messages]
    if roles not in (["user"], ["system", "user"]):
        raise ValueError(f"Eval must contain only system/user, no assistant answer: {row.get('id')}")
    text = messages[-1].get("content")
    if not isinstance(text, str) or re.findall(r"(?m)^([A-D])\.\s", text) != list("ABCD"):
        raise ValueError(f"Expected ordered A/B/C/D choices: {row.get('id')}")
    images = row.get("images") or []
    if not isinstance(images, list) or len(images) != (1 if direction.endswith("V") else 0):
        raise ValueError(f"{direction} has incorrect image count: {row.get('id')}")
    assets = []
    for name in images:
        if not isinstance(name, str) or "://" in name:
            raise ValueError("Evaluation requires local images")
        p = (root / name).resolve()
        if not p.is_file() or p.stat().st_size == 0:
            raise ValueError(f"Missing/empty image: {p}")
        assets.append(str(p))
    tags = text.count("<image>")
    if assets and tags == 0:
        text = "<image>\n" + text
    elif tags != len(assets):
        raise ValueError(f"Incorrect image placeholders: {row.get('id')}")
    if answer_format == "cot":
        # The frozen dataset already includes a strict-output instruction.
        if text.rstrip().endswith(SUFFIX):
            text = text.rstrip()[:-len(SUFFIX)]
        text = text.rstrip() + "\n\nExplain your reasoning briefly, then finish with exactly FINAL: A, FINAL: B, FINAL: C, or FINAL: D."
        system = "Solve the multiple-choice question. Give concise reasoning and finish with a terminal FINAL: letter."
    else:
        if not text.rstrip().endswith(SUFFIX):
            text = text.rstrip() + SUFFIX
        system = SYSTEM
    # Whitelist fields so audit/gold metadata cannot become inference input.
    result = {"id": row["id"], "qid": row.get("qid", row["id"].split("-")[0]),
              "messages": [{"role": "system", "content": system}, {"role": "user", "content": text}]}
    if assets:
        result["images"] = assets
    return result


def _messages_for_key(row):
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Prediction has no messages for input verification")
    messages = list(messages)
    if any(not isinstance(m, dict) for m in messages):
        raise ValueError("Prediction has malformed messages for input verification")
    if messages and messages[-1].get("role") == "assistant":
        messages.pop()
    return [{"role": m.get("role"), "content": m.get("content")} for m in messages]


def _image_key(value):
    """Normalize ms-swift image metadata without requiring one output schema."""
    if isinstance(value, str):
        return os.path.normcase(os.path.normpath(value))
    if isinstance(value, dict):
        # Different ms-swift versions may return a path under one of these
        # keys, or return a serialized image object with no path at all.
        for name in ("path", "image", "url", "name", "file", "value"):
            candidate = value.get(name)
            if isinstance(candidate, str):
                return os.path.normcase(os.path.normpath(candidate))
        return "<image-object>"
    return f"<{type(value).__name__}>"


def prompt_text_key(row):
    return digest({"messages": _messages_for_key(row)})


def prompt_key(row):
    messages = _messages_for_key(row)
    assets = row.get("images") or []
    if isinstance(assets, str):
        assets = [assets]
    if not isinstance(assets, list):
        assets = [assets]
    return digest({"messages": messages, "images": [_image_key(p) for p in assets]})


def align(rows, inputs, source):
    """Match by ID and verify prompt; fallback only to unique prompt+image, NEVER row order."""
    expected = id_map(inputs, source)
    by_prompt = defaultdict(list)
    by_text = defaultdict(list)
    for sid, row in expected.items():
        by_prompt[prompt_key(row)].append(sid)
        by_text[prompt_text_key(row)].append(sid)
    result = {}
    for raw in rows:
        sid = raw.get("id", raw.get("sample_id"))
        key = prompt_key(raw)
        if sid is None:
            matches = by_prompt.get(key, [])
            if len(matches) == 1:
                sid = matches[0]
            else:
                # Some Swift releases serialize images as objects or omit
                # image paths from the returned row. The prompt remains a
                # safe fallback because the evaluator rejects duplicate
                # prompt keys in each direction during preparation.
                text_matches = by_text.get(prompt_text_key(raw), [])
                if len(text_matches) != 1:
                    raise ValueError(f"Cannot uniquely restore ID from prompt+image in {source}; matches={len(matches)}, text_matches={len(text_matches)}")
                sid = text_matches[0]
        if sid not in expected or prompt_text_key(raw) != prompt_text_key(expected[sid]):
            raise ValueError(f"ID/prompt/image mismatch in {source}: {sid!r}")
        if sid in result:
            raise ValueError(f"Duplicate prediction in {source}: {sid}")
        if not isinstance(raw.get("response"), str):
            raise ValueError(f"Prediction has no string response in {source}: {sid}")
        result[sid] = {**raw, "id": sid, "qid": expected[sid]["qid"]}
    return result


def letter(response, answer_format="strict"):
    if not isinstance(response, str):
        return None
    text = response.strip()
    if text in ("A", "B", "C", "D"):
        return text
    if answer_format == "glm-box":
        # Accept only the observed GLM protocol, never letters in prose or
        # nonempty reasoning. Raw responses remain unchanged for auditing.
        match = re.fullmatch(
            r"(?:<think>\s*</think>\s*)?"
            r"<\|begin_of_box\|>\s*([ABCD])\s*<\|end_of_box\|>", text)
        if match:
            return match.group(1)
    if answer_format == "cot":
        # Never extract letters from reasoning, examples, or conflicting
        # conclusions. Accept only an explicit final answer at the end.
        if "<think>" in text and "</think>" not in text:
            return None
        final = text.rsplit("</think>", 1)[-1].strip()
        if final in ("A", "B", "C", "D"):
            return final
        match = re.search(r"\bFINAL\s*[:：]\s*([ABCD])\s*$", final, re.I)
        markers = re.findall(r"\bFINAL\s*[:：]\s*([ABCD])\b", final, re.I)
        if match and len({x.upper() for x in markers}) == 1:
            return match.group(1).upper()
        boxes = re.findall(r"<\|begin_of_box\|>\s*([ABCD])\s*<\|end_of_box\|>", final)
        match = re.search(r"<\|begin_of_box\|>\s*([ABCD])\s*<\|end_of_box\|>\s*$", final)
        if match and len(set(boxes)) == 1 and all(x.upper() == match.group(1) for x in markers):
            return match.group(1)
    return None


def score(inputs, predictions, labels, audit, direction, answer_format="strict"):
    records = []
    for row in inputs:
        sid = row["id"]
        response = predictions[sid]["response"] if sid in predictions else None
        selected = letter(response, answer_format)
        gold = labels[sid]["answer"]
        if sid not in predictions:
            status = "missing_prediction"
        elif not response.strip():
            status = "empty_prediction"
        elif selected is None:
            status = "format_violation"
        else:
            status = "correct" if selected == gold else "wrong_answer"
        records.append({"id": sid, "qid": row["qid"], "gold": gold, "predicted": selected,
                        "response": response, "status": status, "correct": status == "correct",
                        "probe_id": audit.get(sid, {}).get("probe_id", sid.rsplit("-", 1)[-1]),
                        "entity_kind": audit.get(sid, {}).get("entity_kind", "unknown"),
                        "evaluation_group": labels[sid]["evaluation_group"],
                        "bias_watchlist": labels[sid]["bias_watchlist"]})
        records[-1]["question_type"] = labels[sid]["question_type"]

    def metrics(items):
        counts = Counter(r["status"] for r in items)
        n = len(items)
        present = n - counts["missing_prediction"]
        valid = counts["correct"] + counts["wrong_answer"]
        return {"total": n, "matched": present, "valid": valid, "correct": counts["correct"],
                "wrong_answer": counts["wrong_answer"], "missing_prediction": counts["missing_prediction"],
                "empty_prediction": counts["empty_prediction"], "format_violation": counts["format_violation"],
                "row_coverage": present / n if n else 0, "valid_coverage": valid / n if n else 0,
                "accuracy": counts["correct"] / n if n else 0,
                "accuracy_on_valid_only": counts["correct"] / valid if valid else None}

    report = {"direction": direction, "answer_format": answer_format,
              "normalized_answers": sum(r["predicted"] is not None and
                                        isinstance(r["response"], str) and
                                        r["response"].strip() != r["predicted"] for r in records),
              **metrics(records)}
    report["result_application"] = metrics([r for r in records if r["evaluation_group"] == "result_application"])
    report["rule_report_audit"] = metrics([r for r in records if r["evaluation_group"] == "rule_report_audit"])
    report["bias_watchlist"] = metrics([r for r in records if r["bias_watchlist"]])
    report["overall_interpretation"] = "Mixed scope: use result_application for applied-result performance and rule_report_audit separately."
    for key in ("probe_id", "entity_kind", "evaluation_group", "question_type"):
        groups = defaultdict(list)
        for r in records:
            groups[r[key]].append(r)
        report["by_" + key] = {k: metrics(v) for k, v in sorted(groups.items())}
    entities = defaultdict(list)
    for r in records:
        entities[r["qid"]].append(r)
    report["entity_exact_accuracy"] = sum(all(r["correct"] for r in rs) for rs in entities.values()) / len(entities)
    report["entities"] = len(entities)
    # Wrong choices are normal model outcomes. Missing/invalid output remains an observed failure.
    report["status"] = "complete" if report["valid"] == report["total"] else "complete_with_output_errors"
    if report["matched"] != report["total"]:
        report["status"] = "incomplete"
    return report, records


def swift_command(args, executable, adapter, dataset, output):
    model = args.model
    cmd = [*executable, "infer", "--model", model]
    if adapter:
        # A dense checkpoint is a complete model and must be loaded as --model;
        # only LoRA checkpoints are passed through --adapters.
        if (Path(adapter) / "adapter_config.json").is_file():
            cmd.extend(["--adapters", adapter])
        else:
            cmd[cmd.index("--model") + 1] = adapter
    cmd.extend([
            "--load_args", "false", "--use_chat_template", "true",
            "--infer_backend", "transformers", "--torch_dtype", "bfloat16",
            "--val_dataset", str(dataset), "--val_dataset_shuffle", "false", "--dataset_shuffle", "false",
            # Keep audit IDs in the JSONL for safe result alignment, but do
            # not forward them as model.generate kwargs. Qwen3-VL rejects
            # metadata such as id/qid/dataset at generation time.
            "--remove_unused_columns", "true", "--strict", "true",
            "--max_batch_size", str(args.batch_size), "--write_batch_size", str(args.write_batch_size),
            "--max_new_tokens", str(args.max_new_tokens), "--temperature", "0",
            "--seed", str(args.seed), "--data_seed", str(args.seed), "--max_pixels", str(args.max_pixels),
            "--enable_thinking", "true" if args.answer_format == "cot" else "false", "--stream", "false", "--result_path", str(output)])
    return cmd


def child_environment(root, gpu, max_pixels):
    env = os.environ.copy()
    for name in list(env):
        if (name in {"NPROC_PER_NODE", "NNODES", "NODE_RANK", "MASTER_ADDR", "MASTER_PORT",
                     "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK",
                     "ROLE_WORLD_SIZE", "CUDA_VISIBLE_DEVICES"} or name.startswith("TORCHELASTIC_")):
            env.pop(name, None)
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), ROOT_IMAGE_DIR=str(root), MAX_PIXELS=str(max_pixels),
               PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    return env


def stop_child(proc):
    if proc.poll() is not None:
        return
    if os.name == "posix":
        os.killpg(proc.pid, signal.SIGINT)
    else:
        proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
        proc.wait()


def run_command(cmd, log, env, cwd):
    print("$ " + shlex.join(cmd), flush=True)
    kwargs = {"start_new_session": True} if os.name == "posix" else {"creationflags": subprocess.CREATE_NO_WINDOW}
    with Path(log).open("w", encoding="utf-8") as out:
        out.write("$ " + shlex.join(cmd) + "\n")
        out.flush()
        # Tee Swift's merged stdout/stderr to the terminal and a durable log.
        proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", bufsize=1, **kwargs)
        try:
            for line in proc.stdout:
                out.write(line)
                out.flush()
                print(line, end="", flush=True)
            return proc.wait()
        except BaseException:
            stop_child(proc)
            raise
        finally:
            proc.stdout.close()


@contextlib.contextmanager
def run_lock(path):
    """OS lock is released on process death; the persistent file is harmless."""
    with Path(path).open("a+b") as f:
        f.seek(0)
        if f.read(1) == b"":
            f.write(b"0")
            f.flush()
        f.seek(0)
        try:
            if os.name == "posix":
                import fcntl
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise ValueError(f"Another evaluation is using {path.parent}") from exc
        try:
            yield
        finally:
            f.seek(0)
            if os.name == "posix":
                fcntl.flock(f, fcntl.LOCK_UN)
            else:
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


def collect_attempts(folder, state, inputs):
    predictions, notes = {}, []
    expected = id_map(inputs, folder)
    for attempt in state.get("attempts", []):
        raw = folder / attempt["raw"]
        if not raw.is_file():
            if attempt["status"] == "complete":
                raise ValueError(f"Completed result is missing: {raw}")
            continue
        rows, warnings = read_rows(raw, allow_partial_tail=attempt["status"] != "complete")
        allowed = [expected[sid] for sid in attempt["ids"]]
        aligned = align(rows, allowed, raw)
        if predictions.keys() & aligned.keys():
            raise ValueError(f"Duplicate results across attempts: {raw}")
        predictions.update(aligned)
        notes.extend(warnings)
    return predictions, notes


def infer_attempt(args, executable, adapter, inputs, folder, state, env, root):
    number = len(state["attempts"]) + 1
    stem = f"attempt_{number:03d}"
    dataset, raw = folder / (stem + ".input.jsonl"), folder / (stem + ".raw.jsonl")
    if raw.exists():
        raise ValueError(f"Refusing to append to untracked result: {raw}")
    # Keep sample IDs in the evaluator's state, but do not put audit metadata
    # into Swift's model input. Transformers' generate() rejects id/qid and
    # some ms-swift versions also expose dataset as a model kwarg. The prompt
    # and image are sufficient for inference; align() restores IDs afterward.
    swift_inputs = [{k: row[k] for k in ("messages", "images") if k in row} for row in inputs]
    save_rows(dataset, swift_inputs)
    cmd = swift_command(args, executable, adapter, dataset, raw)
    attempt = {"input": dataset.name, "raw": raw.name, "log": stem + ".log", "command": cmd,
               "ids": [r["id"] for r in inputs], "started_at": now(), "status": "running"}
    state["attempts"].append(attempt)
    save_json(folder / "state.json", state)
    try:
        code = run_command(cmd, folder / attempt["log"], env, root)
        attempt.update(returncode=code, status="complete" if code == 0 else "failed")
    except BaseException:
        attempt["status"] = "interrupted"
        raise
    finally:
        attempt["ended_at"] = now()
        save_json(folder / "state.json", state)
    return code


def perform_direction(args, executable, root, output, direction, inputs, adapter, labels, audit):
    folder = output / direction
    folder.mkdir(exist_ok=True)
    path = folder / "state.json"
    state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"attempts": []}
    env = child_environment(root, args.gpu if direction.startswith("T") else args.visual_gpu, args.max_pixels)
    adapter_path = adapter["path"] if adapter else None
    save_rows(folder / "eval.input.jsonl", inputs)
    predictions, notes = collect_attempts(folder, state, inputs)
    error = None
    try:
        if not predictions:
            # Probe is a fixed prefix, not Swift's random val_dataset_sample. It is reused in the final result.
            probe = inputs[:args.probe_size]
            code = infer_attempt(args, executable, adapter_path, probe, folder, state, env, root)
            predictions, notes = collect_attempts(folder, state, inputs)
            if code != 0:
                raise RuntimeError(f"Swift probe exited with code {code}")
        probe_ids = {r["id"] for r in inputs[:args.probe_size]}
        if not probe_ids.issubset(predictions):
            pending_probe = [r for r in inputs[:args.probe_size] if r["id"] not in predictions]
            code = infer_attempt(args, executable, adapter_path, pending_probe, folder, state, env, root)
            predictions, notes = collect_attempts(folder, state, inputs)
            if code != 0 or not probe_ids.issubset(predictions):
                raise RuntimeError("Probe inference is incomplete")
        invalid = [sid for sid in probe_ids
                   if letter(predictions[sid]["response"], args.answer_format) is None]
        if invalid:
            print(f"[{direction}] probe contains {len(invalid)} invalid responses; retaining them and continuing the full dataset", flush=True)
        pending = [r for r in inputs if r["id"] not in predictions]
        if pending:
            print(f"[{direction}] saved={len(predictions)}, remaining={len(pending)}", flush=True)
            code = infer_attempt(args, executable, adapter_path, pending, folder, state, env, root)
            predictions, notes = collect_attempts(folder, state, inputs)
            if code != 0:
                raise RuntimeError(f"Swift exited with code {code}; partial results retained")
    except (RuntimeError, ValueError, OSError) as exc:
        error = str(exc)
        # Collect complete rows flushed before a nonzero exit; never replace invalid responses.
        predictions, notes = collect_attempts(folder, state, inputs)
    except KeyboardInterrupt:
        predictions, notes = collect_attempts(folder, state, inputs)
        save_rows(folder / "predictions.jsonl", [predictions[r["id"]] for r in inputs if r["id"] in predictions])
        raise
    report, records = score(inputs, predictions, labels, audit, direction, args.answer_format)
    report.update(adapter=adapter_path, notes=notes, updated_at=now())
    if error:
        report.update(status="blocked", error=error)
    save_rows(folder / "predictions.jsonl", [predictions[r["id"]] for r in inputs if r["id"] in predictions])
    save_rows(folder / "results.jsonl", records)
    save_rows(folder / "errors.jsonl", [r for r in records if not r["correct"]])
    save_json(folder / "report.json", report)
    print(f"[{direction}] {report['status']}: rows={report['matched']}/{report['total']} "
          f"valid={report['valid']}/{report['total']} accuracy={report['accuracy']:.2%}", flush=True)
    return report


def summary(output, reports, directions):
    rows = [{"direction": d, **reports.get(d, {"status": "not_started"})} for d in directions]
    save_json(output / "summary.json", {"version": VERSION, "updated_at": now(), "directions": rows})
    keys = ["direction", "status", "total", "matched", "valid", "correct", "accuracy", "valid_coverage",
            "empty_prediction", "format_violation", "missing_prediction", "entity_exact_accuracy"]
    with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# xModal evaluation", "", "| Direction | Status | Rows | Valid | Accuracy (all) |",
             "|---|---|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['direction']} | {r['status']} | {r.get('matched', 0)}/{r.get('total', '-')} | "
                     f"{r.get('valid', 0)} | {r.get('accuracy', 0):.2%} |")
    lines.extend(["", "Empty, missing and invalid responses count as incorrect in Accuracy (all).",
                  "Full reports contain direct/diagnosis, entity type, and entity-level breakdowns."])
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("/mnt/yunpan2/tanjiyang/xmodal/entity"))
    p.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--text-adapter", type=Path, help="Text checkpoint or run directory (default: corrected-v3 TEXT, then legacy t2t)")
    p.add_argument("--visual-adapter", type=Path, help="Visual checkpoint or run directory (default: corrected-v3 VISION, then legacy v2v)")
    p.add_argument("--base-model-only", action="store_true",
                   help="Evaluate the unadapted base model; do not load either LoRA adapter")
    p.add_argument("--answer-format", choices=("strict", "glm-box", "cot"), default="strict",
                   help="strict: single A/B/C/D; glm-box: also accept a GLM box with optional empty think block; preserve raw output")
    p.add_argument("--directions", nargs="+", choices=DIRECTIONS, default=list(DIRECTIONS))
    p.add_argument("--gpu", default="0", help="One GPU for T2T/T2V; directions run sequentially")
    p.add_argument("--visual-gpu", default=None, help="One GPU for V2T/V2V; defaults to --gpu")
    p.add_argument("--parallel", action="store_true",
                   help="Run one text direction and one visual direction concurrently on separate GPUs")
    p.add_argument("--output-dir", type=Path, help="New run directory; default: runs/evaluation/<timestamp>")
    p.add_argument("--resume", action="store_true", help="Resume same --output-dir, reusing saved rows, even wrong/invalid ones")
    p.add_argument("--dry-run", action="store_true", help="Validate data/checkpoints and save plan; do not run Swift")
    p.add_argument("--probe-size", type=int, default=3, help="Initial fixed-prefix format probe, included in final predictions")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--write-batch-size", type=int, default=32, help="Flush every N predictions for resumability")
    p.add_argument("--max-new-tokens", type=int, default=4)
    p.add_argument("--max-pixels", type=int, default=1003520)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--swift-bin", default="swift", help="Swift executable path or name")
    return p


def resolve_swift(name):
    value = shutil.which(name)
    if not value:
        raise ValueError(f"Cannot find {name}; activate the xmodal environment first")
    return [value]


def execute(args):
    for key in ("probe_size", "batch_size", "write_batch_size", "max_new_tokens", "max_pixels"):
        if getattr(args, key) < 1:
            raise ValueError(f"--{key.replace('_', '-')} must be positive")
    if len(set(args.directions)) != len(args.directions):
        raise ValueError("Duplicate directions")
    args.visual_gpu = args.visual_gpu or args.gpu
    if "," in args.gpu or "," in args.visual_gpu:
        raise ValueError("Each worker uses one GPU; do not provide comma-separated GPU IDs")
    if args.resume and not args.output_dir:
        raise ValueError("--resume requires the original --output-dir")
    root = args.data_root.expanduser().resolve()
    output = (args.output_dir or root / "runs" / "evaluation" / time.strftime("%Y%m%d_%H%M%S")).resolve()
    print(f"Output: {output}", flush=True)
    executable = [args.swift_bin] if args.dry_run else resolve_swift(args.swift_bin)
    labels_path = root / "audit" / "eval_labels.jsonl"
    labels = id_map(read_rows(labels_path)[0], labels_path)
    if not labels or any(r.get("answer") not in ("A", "B", "C", "D") for r in labels.values()):
        raise ValueError("Gold labels are empty or contain an invalid answer")
    audit_path = root / "audit" / "sample_audit.jsonl"
    audit = id_map(read_rows(audit_path)[0], audit_path) if audit_path.is_file() else {}
    adapters = {}
    if not args.base_model_only:
        for mode, arg in [("T", args.text_adapter), ("V", args.visual_adapter)]:
            if any(d.startswith(mode) for d in args.directions):
                adapters[mode] = checkpoint(arg or default_adapter(root, mode))
                print(f"{mode} learning checkpoint: {adapters[mode]['path']}", flush=True)
    elif args.text_adapter or args.visual_adapter:
        raise ValueError("--base-model-only cannot be combined with --text-adapter or --visual-adapter")
    datasets, source_hashes, assets = {}, {}, {}
    for d in args.directions:
        source = root / d / "eval.jsonl"
        rows = read_rows(source)[0]
        ids = id_map(rows, source)
        if set(ids) != set(labels):
            raise ValueError(f"{d} and gold ID sets differ: eval={len(ids)}, gold={len(labels)}")
        datasets[d] = [prepare_row(row, root, d, args.answer_format) for row in rows]
        source_hashes[d] = file_hash(source)
        for row in datasets[d]:
            for name in row.get("images", []):
                if name not in assets:
                    st = Path(name).stat()
                    assets[name] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
        print(f"{d}: {len(rows)} samples checked", flush=True)
    versions = {}
    for name in ("ms-swift", "transformers", "peft", "torch"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    config = {"version": VERSION, "data_root": str(root), "model": args.model,
              "base_model_only": args.base_model_only,
              "directions": args.directions, "adapters": adapters, "sources": source_hashes,
              "prepared_hashes": {d: digest(rows) for d, rows in datasets.items()},
              "labels_sha256": file_hash(labels_path), "audit_sha256": file_hash(audit_path) if audit else None,
              "image_inventory_sha256": digest(assets),
              "generation": {k: getattr(args, k) for k in ("max_new_tokens", "max_pixels", "seed", "batch_size", "probe_size")}}
    # Preserve signatures of existing strict runs; changing the answer protocol
    # must use a new output directory rather than silently changing old scores.
    if args.answer_format != "strict":
        config["answer_format"] = args.answer_format
    signature = digest(config)
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output / ".run.lock"):
        manifest = output / "manifest.json"
        if args.resume:
            if not manifest.is_file() or json.loads(manifest.read_text(encoding="utf-8"))["signature"] != signature:
                raise ValueError("Cannot resume: dataset, checkpoint or settings changed; use a new output directory")
        else:
            if any(p.name != ".run.lock" for p in output.iterdir()):
                raise ValueError(f"Output is not empty: {output}; use --resume or a new directory")
            save_json(manifest, {"signature": signature, "config": config, "created_at": now(), "versions": versions})
        plan = {}
        for d in args.directions:
            selected_adapter = adapters[d[0]]["path"] if d[0] in adapters else None
            plan[d] = swift_command(args, executable, selected_adapter,
                                    output / d / "eval.input.jsonl", output / d / "example.raw.jsonl")
        save_json(output / "plan.json", plan)
        if args.dry_run:
            print("Preflight passed. Use the same options plus --resume to start inference.")
            return 0
        reports = {}
        # Reload reports so an interruption during a later direction retains earlier summaries.
        for d in args.directions:
            report_path = output / d / "report.json"
            if report_path.is_file():
                reports[d] = json.loads(report_path.read_text(encoding="utf-8"))

        def run_one(direction):
            try:
                return direction, perform_direction(args, executable, root, output, direction,
                                                    datasets[direction], adapters.get(direction[0]), labels, audit)
            except Exception as exc:
                report = {"direction": direction, "status": "blocked", "error": str(exc)}
                save_json(output / direction / "report.json", report)
                print(f"[{direction}] ERROR: {exc}", file=sys.stderr)
                return direction, report

        def run_gpu_group(group):
            # A GPU owns one model process at a time. The two GPU groups can run together.
            group_reports = {}
            for direction in group:
                direction, report = run_one(direction)
                group_reports[direction] = report
            return group_reports

        try:
            if args.parallel and len(args.directions) > 1:
                # Partition by physical GPU. This keeps T2T/T2V serialized on the text GPU
                # and V2T/V2V serialized on the visual GPU, while both groups overlap.
                groups = defaultdict(list)
                for d in args.directions:
                    groups[args.gpu if d.startswith("T") else args.visual_gpu].append(d)
                with ThreadPoolExecutor(max_workers=len(groups), thread_name_prefix="xmodal-gpu") as pool:
                    futures = [pool.submit(run_gpu_group, group) for group in groups.values()]
                    for future in as_completed(futures):
                        reports.update(future.result())
                        summary(output, reports, args.directions)
            else:
                for d in args.directions:
                    direction, report = run_one(d)
                    reports[direction] = report
                    summary(output, reports, args.directions)
        except KeyboardInterrupt:
            summary(output, reports, args.directions)
            print(f"\nStopped. Saved raw results remain in {output}; rerun with --resume.")
            return 130
    print(f"Summary: {output / 'summary.md'}", flush=True)
    return 0 if all(r["status"] in ("complete", "complete_with_output_errors") for r in reports.values()) else 1


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return execute(args)
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nStopped before inference.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
