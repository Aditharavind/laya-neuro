#!/usr/bin/env python3
"""Extract features from all STEW epochs and build Laya typed-decision datasets.

Two evaluation protocols, both honestly labeled (this project's whole point
is not to conflate them, per the deep-research finding that naive random
splits on EEG data are leakage-inflated because EEG carries a strong
subject-identity signal):

  - "within-subject" (naive): a random 70/15/15 split of epochs that does
    NOT respect subject boundaries -- the same subject's other epochs can
    appear in both train and test. This is the inflated, easier setting;
    reported for comparison with literature numbers that use this protocol,
    not presented as the honest number.
  - "cross-subject" (grouped): uses STEW's own pre-made, verified
    subject-disjoint fold 0 (12 held-out subjects, zero overlap with the
    other 36 training subjects) as the test set. This is the number that
    actually matters for "does this generalize to a new person."

Usage:
    python eeg/build_dataset.py --raw-dir data/raw --out data
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eeg.features import extract_features  # noqa: E402

QUESTION = {
    "high_workload": {
        "type": "noul",
        "instructions": "Given this subject's EEG-derived features for this 2-second window, "
                        "is this subject currently experiencing high cognitive workload?",
    }
}
CONFIDENT = 0.95
FEATURE_KEYS = ["delta_mean", "theta_mean", "alpha_mean", "beta_mean", "gamma_mean",
                "frontal_theta", "posterior_alpha", "frontal_theta_posterior_alpha_ratio",
                "engagement_index", "theta_alpha_ratio"]


def make_row(feats: dict, label: int) -> dict:
    state = {k: round(feats[k], 5) for k in FEATURE_KEYS}
    p_true = CONFIDENT if label == 1 else 1 - CONFIDENT
    return {
        "state": json.dumps(state),
        "questions": json.dumps(QUESTION),
        "gold": json.dumps({"high_workload": {"probabilities": {"true": p_true, "false": 1 - p_true}}}),
    }


def write_jsonl(rows: list, path: str):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    ap.add_argument("--seed", type=int, default=20260929)
    args = ap.parse_args()

    print("loading raw STEW arrays...")
    X = np.load(os.path.join(args.raw_dir, "STEW_X.npy"))
    y = pd.read_csv(os.path.join(args.raw_dir, "STEW_y.csv"), header=None)[0].values
    subj = pd.read_csv(os.path.join(args.raw_dir, "STEW_subject_id.csv"), header=None)[0].values
    test_idx_cross = np.loadtxt(os.path.join(args.raw_dir, "test_indices_fold_0.txt"), dtype=int)

    print(f"extracting features for {len(X)} epochs...")
    feats = []
    for i in range(len(X)):
        feats.append(extract_features(X[i]))
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{len(X)}")

    df = pd.DataFrame(feats)
    df["label"] = y
    df["subject_id"] = subj
    df.to_parquet(os.path.join(args.out, "stew_features.parquet"))
    print(f"saved feature table -> {os.path.join(args.out, 'stew_features.parquet')}")

    rng = random.Random(args.seed)
    n = len(df)

    # --- cross-subject split: STEW's own pre-verified subject-disjoint fold 0 ---
    cross_test_mask = np.zeros(n, dtype=bool)
    cross_test_mask[test_idx_cross] = True
    cross_train_all_idx = np.where(~cross_test_mask)[0]
    cross_test_idx = np.where(cross_test_mask)[0]
    # hold out a subject-disjoint slice of the training subjects for calibration (val)
    train_subjects = sorted(set(subj[cross_train_all_idx]))
    rng.shuffle(train_subjects)
    n_val_subj = max(1, len(train_subjects) // 6)
    val_subjects = set(train_subjects[:n_val_subj])
    cross_val_idx = [i for i in cross_train_all_idx if subj[i] in val_subjects]
    cross_train_idx = [i for i in cross_train_all_idx if subj[i] not in val_subjects]

    os.makedirs(os.path.join(args.out, "cross_subject"), exist_ok=True)
    for name, idx in [("train", cross_train_idx), ("val", cross_val_idx), ("test", cross_test_idx)]:
        rows = [make_row(feats[i], int(y[i])) for i in idx]
        write_jsonl(rows, os.path.join(args.out, "cross_subject", f"{name}.jsonl"))
        np.save(os.path.join(args.out, "cross_subject", f"{name}_idx.npy"), np.array(idx))

    # --- within-subject (naive) split: pure random, ignores subject boundaries ---
    all_idx = list(range(n))
    rng.shuffle(all_idx)
    n_test = int(0.15 * n)
    n_val = int(0.15 * n)
    naive_test_idx = all_idx[:n_test]
    naive_val_idx = all_idx[n_test:n_test + n_val]
    naive_train_idx = all_idx[n_test + n_val:]

    os.makedirs(os.path.join(args.out, "within_subject"), exist_ok=True)
    for name, idx in [("train", naive_train_idx), ("val", naive_val_idx), ("test", naive_test_idx)]:
        rows = [make_row(feats[i], int(y[i])) for i in idx]
        write_jsonl(rows, os.path.join(args.out, "within_subject", f"{name}.jsonl"))
        np.save(os.path.join(args.out, "within_subject", f"{name}_idx.npy"), np.array(idx))

    print("\ndone. cross_subject/ = honest (subject-disjoint) protocol; "
          "within_subject/ = naive (leakage-prone) protocol, kept for literature comparison only.")


if __name__ == "__main__":
    main()
