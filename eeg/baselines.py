#!/usr/bin/env python3
"""Classical ML baselines on the same engineered EEG features, same splits.

Trains SVM (RBF), Random Forest, and LDA -- the three models the
deep-research flagged as the standard comparison points in EEG
cognitive-workload literature -- on `data/stew_features.parquet`, using the
exact same train/val/test index arrays `eeg/build_dataset.py` used to build
the Laya JSONL splits (`*_idx.npy`), so the numbers are directly comparable.

Reports accuracy, ECE and Brier score per protocol per model, so Laya's own
eval script can be checked against a real baseline rather than an assumed one.

Usage:
    python eeg/baselines.py --data data
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

FEATURE_KEYS = ["delta_mean", "theta_mean", "alpha_mean", "beta_mean", "gamma_mean",
                "frontal_theta", "posterior_alpha", "frontal_theta_posterior_alpha_ratio",
                "engagement_index", "theta_alpha_ratio"]


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    confidences = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    correct = (preds == labels).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / len(labels)) * abs(bin_acc - bin_conf)
    return float(ece)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def run_protocol(df: pd.DataFrame, data_dir: str, protocol: str):
    train_idx = np.load(os.path.join(data_dir, protocol, "train_idx.npy"))
    val_idx = np.load(os.path.join(data_dir, protocol, "val_idx.npy"))
    test_idx = np.load(os.path.join(data_dir, protocol, "test_idx.npy"))
    # baselines don't need a held-out calibration split the way Laya's temperature
    # fitting does; fold val into train for a fair sample-count comparison
    fit_idx = np.concatenate([train_idx, val_idx])

    X = df[FEATURE_KEYS].values
    y = df["label"].values
    scaler = StandardScaler().fit(X[fit_idx])
    X_fit, X_test = scaler.transform(X[fit_idx]), scaler.transform(X[test_idx])
    y_fit, y_test = y[fit_idx], y[test_idx]

    models = {
        "LDA": LinearDiscriminantAnalysis(),
        "SVM (RBF)": SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1),
    }

    print(f"\n=== {protocol} protocol | fit n={len(fit_idx)} | test n={len(test_idx)} ===")
    results = {}
    for name, model in models.items():
        model.fit(X_fit, y_fit)
        probs = model.predict_proba(X_test)
        preds = np.argmax(probs, axis=1)
        acc = float(np.mean(preds == y_test))
        ece = expected_calibration_error(probs, y_test)
        brier = brier_score(probs, y_test)
        results[name] = {"accuracy": acc, "ece": ece, "brier": brier}
        print(f"  {name:15s} accuracy={acc:.4f}  ECE={ece:.4f}  Brier={brier:.4f}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    args = ap.parse_args()

    df = pd.read_parquet(os.path.join(args.data, "stew_features.parquet"))
    all_results = {}
    for protocol in ["cross_subject", "within_subject"]:
        all_results[protocol] = run_protocol(df, args.data, protocol)

    print("\n=== summary ===")
    print(f"{'protocol':15s} {'model':15s} {'acc':>8s} {'ECE':>8s} {'brier':>8s}")
    for protocol, models in all_results.items():
        for name, r in models.items():
            print(f"{protocol:15s} {name:15s} {r['accuracy']:8.4f} {r['ece']:8.4f} {r['brier']:8.4f}")


if __name__ == "__main__":
    main()
