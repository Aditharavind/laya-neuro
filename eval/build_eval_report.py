#!/usr/bin/env python3
"""Build a single eval_report.json covering baselines + Laya, both protocols.

Retrains the three classical baselines (cheap, seconds) and runs the given
Laya checkpoint on its test set, computing accuracy/ECE/Brier plus binned
reliability-diagram data for every model, so eval/make_results_card.py can
render everything from one file.

Usage:
    python eval/build_eval_report.py --data data --laya-checkpoint checkpoints/laya_neuro_demo \
        --out eval_report.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eeg.baselines import FEATURE_KEYS, expected_calibration_error, brier_score  # noqa: E402

N_BINS = 10


def reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = N_BINS) -> list:
    confidences = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    correct = (preds == labels).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        n = int(mask.sum())
        out.append({
            "bin_lo": float(lo), "bin_hi": float(hi), "n": n,
            "mean_confidence": float(confidences[mask].mean()) if n else None,
            "accuracy": float(correct[mask].mean()) if n else None,
        })
    return out


def score(probs: np.ndarray, labels: np.ndarray) -> dict:
    preds = np.argmax(probs, axis=1)
    return {
        "n": len(labels),
        "accuracy": float(np.mean(preds == labels)),
        "ece": expected_calibration_error(probs, labels),
        "brier": brier_score(probs, labels),
        "reliability": reliability_bins(probs, labels),
    }


def run_baselines(df: pd.DataFrame, data_dir: str, protocol: str) -> dict:
    train_idx = np.load(os.path.join(data_dir, protocol, "train_idx.npy"))
    val_idx = np.load(os.path.join(data_dir, protocol, "val_idx.npy"))
    test_idx = np.load(os.path.join(data_dir, protocol, "test_idx.npy"))
    fit_idx = np.concatenate([train_idx, val_idx])

    X = df[FEATURE_KEYS].values
    y = df["label"].values
    scaler = StandardScaler().fit(X[fit_idx])
    X_fit, X_test = scaler.transform(X[fit_idx]), scaler.transform(X[test_idx])
    y_fit, y_test = y[fit_idx], y[test_idx]

    models = {
        "LDA": LinearDiscriminantAnalysis(),
        "SVM": SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    }
    out = {}
    for name, model in models.items():
        model.fit(X_fit, y_fit)
        probs = model.predict_proba(X_test)
        out[name] = score(probs, y_test)
        print(f"  [{protocol}] {name}: acc={out[name]['accuracy']:.4f} ece={out[name]['ece']:.4f}")
    return out


def run_laya(checkpoint: str, jsonl_path: str) -> dict:
    import laya
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = laya.Agent(checkpoint, device=device)

    rows = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    questions = json.loads(rows[0]["questions"])
    states = [json.loads(r["state"]) for r in rows]
    labels = np.array([1 if json.loads(r["gold"])["high_workload"]["probabilities"]["true"] > 0.5 else 0 for r in rows])

    results = agent.predict_batch(states, questions, batch_size=64)
    probs = []
    for result in results:
        answer = result["answers"]["high_workload"]
        p_true = float(answer.get("noul", answer.get("probabilities", {}).get("true", 0.5)))
        probs.append([1 - p_true, p_true])
    probs = np.array(probs)

    out = score(probs, labels)
    print(f"  [Laya cross_subject]: acc={out['accuracy']:.4f} ece={out['ece']:.4f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    ap.add_argument("--laya-checkpoint", default=os.path.join(os.path.dirname(__file__), "..", "checkpoints", "laya_neuro_demo"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "eval_report.json"))
    args = ap.parse_args()

    df = pd.read_parquet(os.path.join(args.data, "stew_features.parquet"))

    report = {}
    print("running baselines...")
    for protocol in ["cross_subject", "within_subject"]:
        report[protocol] = run_baselines(df, args.data, protocol)

    print("running Laya (cross_subject only)...")
    report["cross_subject"]["Laya"] = run_laya(
        args.laya_checkpoint, os.path.join(args.data, "cross_subject", "test.jsonl"))

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
