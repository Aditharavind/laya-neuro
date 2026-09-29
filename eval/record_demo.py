#!/usr/bin/env python3
"""Render a short demo video: input EEG -> extracted features -> model output.

Three panels, all showing real data for real held-out subjects (never seen
in training): the raw 14-channel EEG waveform for the 2-second epoch
(left), the engineered features actually fed to Laya as the "model input"
(top right), and the model's live prediction vs. the subject's real
self-reported workload label (bottom right, "model output"). A fixed random
seed picks the sampled epochs -- this is not a cherry-picked "best of" reel,
so it may show wrong predictions too, matching the real ~66% test accuracy
this checkpoint scores overall (shown as a caption, not hidden).

Usage:
    python eval/record_demo.py --checkpoint checkpoints/laya_neuro_demo --out demo.mp4
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eeg.features import extract_features, CHANNELS  # noqa: E402

FEATURE_KEYS = ["delta_mean", "theta_mean", "alpha_mean", "beta_mean", "gamma_mean",
                "frontal_theta", "posterior_alpha", "frontal_theta_posterior_alpha_ratio",
                "engagement_index", "theta_alpha_ratio"]
FEATURE_LABELS = ["delta", "theta", "alpha", "beta", "gamma", "frontal\ntheta",
                   "posterior\nalpha", "ft/pa\nratio", "engagement\nindex", "theta/alpha\nratio"]
QUESTION = {
    "high_workload": {
        "type": "noul",
        "instructions": "Given this subject's EEG-derived features for this 2-second window, "
                        "is this subject currently experiencing high cognitive workload?",
    }
}

# validated categorical palette (dataviz skill), reused from swarm-laya's demo
BLUE = "#2a78d6"      # LOW workload
ORANGE = "#eb6834"    # HIGH workload
GREEN = "#1baf7a"     # correct
RED = "#e34948"       # wrong
INK = "#1a1a17"
MUTED = "#5a5952"
BG = "#fafaf9"

REVEAL_FRAMES = 20     # frames revealing the waveform left-to-right
BAR_FILL_FRAMES = 15   # frames animating the output probability bar
HOLD_FRAMES = 20        # frames holding the final result
FPS = 24


def render_frame(epoch, feats_state, p_true, true_label, subject_id, reveal_frac, bar_frac, final, overall_acc):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1], height_ratios=[1, 1.1],
                           left=0.06, right=0.97, top=0.90, bottom=0.08, hspace=0.55, wspace=0.28)

    fig.suptitle("laya-neuro: EEG → features → model decision (live, held-out subject)",
                 fontsize=15, color=INK, fontweight="bold", x=0.06, ha="left")

    # --- left: raw EEG waveform, revealed left-to-right ---
    ax_wave = fig.add_subplot(gs[:, 0])
    n_samples = epoch.shape[1]
    reveal_n = max(2, int(n_samples * reveal_frac))
    t = np.arange(n_samples) / 128.0
    offset_step = 6.0
    for ch_i in range(epoch.shape[0]):
        sig = epoch[ch_i, :reveal_n]
        sig_norm = (sig - epoch[ch_i].mean()) / (epoch[ch_i].std() + 1e-6)
        ax_wave.plot(t[:reveal_n], sig_norm + ch_i * offset_step, color=BLUE, linewidth=0.8, alpha=0.85)
        ax_wave.text(-0.08, ch_i * offset_step, CHANNELS[ch_i], fontsize=7, color=MUTED,
                     ha="right", va="center")
    ax_wave.set_xlim(0, 2.0)
    ax_wave.set_ylim(-offset_step, epoch.shape[0] * offset_step)
    ax_wave.set_xlabel("seconds", fontsize=9, color=MUTED)
    ax_wave.set_title(f"Raw EEG input — subject #{subject_id} (held-out, never in training)",
                       fontsize=10, color=INK, loc="left")
    ax_wave.set_yticks([])
    for spine in ["top", "right", "left"]:
        ax_wave.spines[spine].set_visible(False)
    ax_wave.set_facecolor(BG)

    # --- top right: engineered features (the actual model input) ---
    ax_feat = fig.add_subplot(gs[0, 1])
    vals = [feats_state[k] for k in FEATURE_KEYS]
    bars = ax_feat.bar(range(len(vals)), vals, color=MUTED, width=0.6)
    ax_feat.set_xticks(range(len(vals)))
    ax_feat.set_xticklabels([lbl.replace("\n", " ") for lbl in FEATURE_LABELS],
                             fontsize=6.5, color=MUTED, rotation=35, ha="right")
    ax_feat.set_title("Model input — engineered features (JSON state)", fontsize=10, color=INK, loc="left")
    ax_feat.tick_params(axis="y", labelsize=7, colors=MUTED)
    for spine in ["top", "right"]:
        ax_feat.spines[spine].set_visible(False)
    ax_feat.set_facecolor(BG)

    # --- bottom right: model output ---
    ax_out = fig.add_subplot(gs[1, 1])
    ax_out.set_xlim(0, 1)
    ax_out.set_ylim(0, 1)
    ax_out.axis("off")
    ax_out.set_title("Model output", fontsize=10, color=INK, loc="left")

    shown_p = p_true * bar_frac
    bar_color = ORANGE if shown_p >= 0.5 else BLUE
    ax_out.barh([0.75], [shown_p], height=0.18, color=bar_color, left=0)
    ax_out.barh([0.75], [1.0], height=0.18, color="none", edgecolor=MUTED, linewidth=1)
    ax_out.text(0.0, 0.95, "P(high workload)", fontsize=9, color=MUTED)
    ax_out.text(1.0, 0.75, f"{shown_p:.0%}", fontsize=11, color=INK, ha="right", va="center")

    if final:
        verdict = "HIGH workload" if p_true >= 0.5 else "LOW workload"
        conf = p_true if p_true >= 0.5 else 1 - p_true
        truth = "HIGH workload" if true_label == 1 else "LOW workload"
        correct = (p_true >= 0.5) == (true_label == 1)
        result_color = GREEN if correct else RED
        result_text = "✓ matches self-report" if correct else "✗ does not match self-report"
        ax_out.text(0.0, 0.45, f"model says: {verdict}", fontsize=13, color=INK, fontweight="bold")
        ax_out.text(0.0, 0.32, f"confidence: {conf:.0%}", fontsize=10, color=MUTED)
        ax_out.text(0.0, 0.15, f"actual self-reported label: {truth}", fontsize=10, color=MUTED)
        ax_out.text(0.0, 0.02, result_text, fontsize=10, color=result_color, fontweight="bold")

    fig.text(0.5, 0.015,
              f"Overall honest test accuracy on 7,128 held-out epochs from subjects never seen in training: {overall_acc:.1%}"
              " — individual examples shown here are a random, unfiltered sample, not cherry-picked.",
              fontsize=8, color=MUTED, ha="center")

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.path.join(os.path.dirname(__file__), "..", "checkpoints", "laya_neuro_demo"))
    ap.add_argument("--raw-dir", default=os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
    ap.add_argument("--test-idx", default=os.path.join(os.path.dirname(__file__), "..", "data", "cross_subject", "test_idx.npy"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "demo.mp4"))
    ap.add_argument("--n-examples", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--overall-acc", type=float, default=0.663, help="the real, full-test-set accuracy to caption")
    args = ap.parse_args()

    import laya
    import torch
    import imageio.v2 as imageio

    X = np.load(os.path.join(args.raw_dir, "STEW_X.npy"))
    y = pd.read_csv(os.path.join(args.raw_dir, "STEW_y.csv"), header=None)[0].values
    subj = pd.read_csv(os.path.join(args.raw_dir, "STEW_subject_id.csv"), header=None)[0].values
    test_idx = np.load(args.test_idx)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {args.checkpoint} on {device} ...")
    agent = laya.Agent(args.checkpoint, device=device)

    rng = np.random.default_rng(args.seed)
    picks = rng.choice(test_idx, size=args.n_examples, replace=False)

    writer = imageio.get_writer(args.out, fps=FPS, codec="libx264", quality=8, macro_block_size=1)

    for k, i in enumerate(picks):
        i = int(i)
        epoch = X[i]
        true_label = int(y[i])
        subject_id = int(subj[i])
        feats = extract_features(epoch)
        state = {kk: round(feats[kk], 5) for kk in FEATURE_KEYS}
        result = agent.predict(state, QUESTION)
        answer = result["answers"]["high_workload"]
        p_true = float(answer.get("noul", answer.get("probabilities", {}).get("true", 0.5)))

        print(f"[{k + 1}/{len(picks)}] subject={subject_id} true={'HIGH' if true_label else 'LOW'} "
              f"pred_p_high={p_true:.3f}")

        for f in range(REVEAL_FRAMES):
            frac = (f + 1) / REVEAL_FRAMES
            writer.append_data(render_frame(epoch, state, p_true, true_label, subject_id, frac, 0.0, False, args.overall_acc))
        for f in range(BAR_FILL_FRAMES):
            frac = (f + 1) / BAR_FILL_FRAMES
            writer.append_data(render_frame(epoch, state, p_true, true_label, subject_id, 1.0, frac, f == BAR_FILL_FRAMES - 1, args.overall_acc))
        for _ in range(HOLD_FRAMES):
            writer.append_data(render_frame(epoch, state, p_true, true_label, subject_id, 1.0, 1.0, True, args.overall_acc))

    writer.close()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
