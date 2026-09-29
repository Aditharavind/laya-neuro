#!/usr/bin/env python3
"""Evaluate a fine-tuned laya-neuro checkpoint: accuracy + calibration (ECE, Brier).

This is the metric pair the deep-research flagged as essentially absent from
the EEG workload-classification literature (a systematic review found zero
mentions of ECE/Brier/reliability diagrams in that subfield) -- so this
script, not raw accuracy, is the project's actual point of differentiation.

Run once per split protocol, honest (cross_subject) first:
    python eval/evaluate.py --checkpoint checkpoints/laya_neuro_cross_subject \
        --data-dir data/cross_subject --out reports/eval_cross_subject.json
    python eval/evaluate.py --checkpoint checkpoints/laya_neuro_within_subject \
        --data-dir data/within_subject --out reports/eval_within_subject.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eeg.baselines import expected_calibration_error, brier_score  # noqa: E402


def load_jsonl(path: str) -> list:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate(agent, jsonl_path: str, label: str, batch_size: int = 64) -> dict:
    rows = load_jsonl(jsonl_path)
    if not rows:
        return {"n": 0, "accuracy": 0.0, "ece": 0.0, "brier": 0.0}
    questions = json.loads(rows[0]["questions"])

    states = [json.loads(r["state"]) for r in rows]
    labels = []
    for r in rows:
        gold = json.loads(r["gold"])["high_workload"]["probabilities"]
        labels.append(1 if gold["true"] > gold["false"] else 0)

    results = agent.predict_batch(states, questions, batch_size=batch_size)
    probs = []
    for result in results:
        answer = result["answers"]["high_workload"]
        p_true = float(answer.get("noul", answer.get("probabilities", {}).get("true", 0.5)))
        probs.append([1 - p_true, p_true])

    probs = np.array(probs)
    labels = np.array(labels)
    preds = np.argmax(probs, axis=1)
    acc = float(np.mean(preds == labels))
    ece = expected_calibration_error(probs, labels)
    brier = brier_score(probs, labels)

    report = {"n": len(labels), "accuracy": acc, "ece": ece, "brier": brier}
    print(f"[{label}] n={len(labels)} accuracy={acc:.4f} ECE={ece:.4f} Brier={brier:.4f}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", required=True, help="directory with test.jsonl (e.g. data/cross_subject)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import laya
    import torch
    agent = laya.Agent(args.checkpoint, device="cuda" if torch.cuda.is_available() else "cpu")

    report = {"checkpoint": args.checkpoint, "data_dir": args.data_dir}
    report["test"] = evaluate(agent, os.path.join(args.data_dir, "test.jsonl"), os.path.basename(args.data_dir))

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
