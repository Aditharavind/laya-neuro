#!/usr/bin/env python3
"""Interactive inference demo for the fine-tuned laya-neuro EEG model.

Unlike a text-claim demo, you can't type raw EEG by hand -- so this pulls a
real 2-second, 14-channel epoch from STEW's held-out cross-subject test set
(subjects the model never trained on), extracts the same engineered
features used at training time, and shows the model's prediction next to
the actual self-reported workload label for that epoch.

Usage:
    python app.py                                    # loads the local cross-subject checkpoint
    python app.py --checkpoint Aditharavind/laya-neuro   # loads from HF
    python app.py --n 10                              # show 10 random held-out epochs, non-interactively
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from eeg.features import extract_features

QUESTION = {
    "high_workload": {
        "type": "noul",
        "instructions": "Given this subject's EEG-derived features for this 2-second window, "
                        "is this subject currently experiencing high cognitive workload?",
    }
}
FEATURE_KEYS = ["delta_mean", "theta_mean", "alpha_mean", "beta_mean", "gamma_mean",
                "frontal_theta", "posterior_alpha", "frontal_theta_posterior_alpha_ratio",
                "engagement_index", "theta_alpha_ratio"]

DEFAULT_CHECKPOINT = os.path.join(os.path.dirname(__file__), "checkpoints", "laya_neuro_cross_subject")
DEFAULT_RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
DEFAULT_TEST_IDX = os.path.join(os.path.dirname(__file__), "data", "cross_subject", "test_idx.npy")


def ask(agent, epoch: np.ndarray, true_label: int, subject_id: int) -> None:
    feats = extract_features(epoch)
    state = {k: round(feats[k], 5) for k in FEATURE_KEYS}

    result = agent.predict(state, QUESTION)
    answer = result["answers"]["high_workload"]
    p_true = float(answer.get("noul", answer.get("probabilities", {}).get("true", 0.5)))
    verdict = "HIGH workload" if p_true >= 0.5 else "LOW workload"
    confidence = p_true if p_true >= 0.5 else 1 - p_true
    truth = "HIGH workload" if true_label == 1 else "LOW workload"
    correct = "correct" if (p_true >= 0.5) == (true_label == 1) else "WRONG"

    bar_len = 30
    filled = int(round(p_true * bar_len))
    bar = "#" * filled + "-" * (bar_len - filled)
    print(f"  held-out subject #{subject_id} (never seen in training)")
    print(f"  features: {json.dumps(state)}")
    print(f"  model says: {verdict} (confidence {confidence:.1%})  P(high)=[{bar}] {p_true:.1%}")
    print(f"  actual self-reported label: {truth}  -> {correct}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                     help="local checkpoint dir or a Hugging Face repo id "
                          "(default: ./checkpoints/laya_neuro_cross_subject)")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: auto-detect)")
    ap.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    ap.add_argument("--n", type=int, default=None, help="show N random held-out epochs non-interactively and exit")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    import laya
    import torch

    import pandas as pd
    X = np.load(os.path.join(args.raw_dir, "STEW_X.npy"))
    y = pd.read_csv(os.path.join(args.raw_dir, "STEW_y.csv"), header=None)[0].values
    subj = pd.read_csv(os.path.join(args.raw_dir, "STEW_subject_id.csv"), header=None)[0].values
    test_idx = np.load(DEFAULT_TEST_IDX)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading {args.checkpoint} on {device} ...")
    agent = laya.Agent(args.checkpoint, device=device)
    print(f"loaded. {len(test_idx)} held-out (cross-subject) test epochs available.\n")

    rng = np.random.default_rng(args.seed)

    def show_one():
        i = int(rng.choice(test_idx))
        ask(agent, X[i], int(y[i]), int(subj[i]))

    if args.n is not None:
        for _ in range(args.n):
            show_one()
        return

    print("Press Enter to sample a random held-out EEG epoch, or 'q' to quit.\n")
    while True:
        try:
            cmd = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if cmd in ("q", "quit", "exit"):
            break
        show_one()


if __name__ == "__main__":
    main()
