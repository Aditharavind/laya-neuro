#!/usr/bin/env python3
"""Render a single shareable results image from eval_report.json."""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
RED = "#e34948"
VIOLET = "#4a3aa7"
INK = "#0b0b0b"
MUTED = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#e3e2dd"

MODEL_ORDER = ["LDA", "SVM", "RandomForest", "Laya"]
MODEL_LABELS = ["LDA", "SVM (RBF)", "Random\nForest", "Laya\n(early ckpt)"]
MODEL_COLORS = [MUTED, AQUA, VIOLET, ORANGE]


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=os.path.join(os.path.dirname(__file__), "..", "eval_report.json"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results_card.png"))
    args = ap.parse_args()

    with open(args.report) as f:
        r = json.load(f)

    cs = r["cross_subject"]
    ws = r["within_subject"]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        "text.color": INK, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED,
    })

    fig = plt.figure(figsize=(10.8, 15.5), dpi=150, facecolor=SURFACE)
    gs = fig.add_gridspec(5, 2, height_ratios=[0.9, 2.0, 2.0, 1.8, 1.6],
                           hspace=0.85, wspace=0.32, left=0.09, right=0.94, top=0.945, bottom=0.035)

    # --- Title ---
    fig.text(0.09, 0.975, "laya-neuro", fontsize=30, fontweight="bold", color=INK, ha="left")
    fig.text(0.09, 0.957, "Calibrated, subject-disjoint EEG cognitive workload classification",
              fontsize=12.5, color=MUTED, ha="left")

    # --- Stat tiles row ---
    laya_cs = cs["Laya"]
    best_baseline_acc = max(cs[m]["accuracy"] for m in ["LDA", "SVM", "RandomForest"])
    tiles = [
        ("Laya accuracy\n(honest, cross-subject)", f"{laya_cs['accuracy']*100:.1f}%", ORANGE),
        ("Best baseline\n(same honest protocol)", f"{best_baseline_acc*100:.1f}%", MUTED),
        ("Laya ECE\n(calibration error)", f"{laya_cs['ece']*100:.1f}%", VIOLET),
        ("Held-out subjects\n(never in training)", "12 / 48", BLUE),
    ]
    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    for i, (label, value, color) in enumerate(tiles):
        x = 0.0 + i * 0.26
        ax.text(x, 0.65, value, fontsize=24, fontweight="bold", color=color, ha="left", va="center",
                 transform=ax.transAxes)
        ax.text(x, 0.12, label, fontsize=9.5, color=MUTED, ha="left", va="center", transform=ax.transAxes)

    # --- Accuracy: honest vs naive protocol, all models ---
    ax1 = fig.add_subplot(gs[1, :])
    x = np.arange(len(MODEL_ORDER))
    w = 0.35
    cs_vals = [cs[m]["accuracy"] * 100 if m in cs else None for m in MODEL_ORDER]
    ws_vals = [ws[m]["accuracy"] * 100 if m in ws else None for m in MODEL_ORDER]
    cs_vals_plot = [v if v is not None else 0 for v in cs_vals]
    ws_vals_plot = [v if v is not None else 0 for v in ws_vals]
    ax1.bar(x - w / 2, cs_vals_plot, width=w, color=BLUE, label="Cross-subject (honest)", zorder=3)
    bars2 = ax1.bar(x + w / 2, ws_vals_plot, width=w, color="#b9c9dd", label="Within-subject (naive, leakage-prone)", zorder=3)
    for i, v in enumerate(ws_vals):
        if v is None:
            bars2[i].set_visible(False)
            ax1.text(x[i] + w / 2, 3, "not yet\ntrained", fontsize=7.5, color=MUTED, ha="center", va="bottom")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(MODEL_LABELS, fontsize=9.5)
    ax1.set_ylabel("Accuracy (%)", fontsize=10)
    ax1.set_ylim(0, 85)
    ax1.axhline(50, color=RED, linewidth=1, linestyle="--", zorder=2)
    ax1.text(len(MODEL_ORDER) - 0.5, 51.5, "chance", fontsize=8, color=RED, ha="right")
    style_axes(ax1)
    ax1.legend(loc="upper left", frameon=False, fontsize=9, ncol=1)
    ax1.set_title("Accuracy: honest (subject-disjoint) vs. naive (leakage-prone) split",
                  fontsize=13, fontweight="bold", color=INK, pad=10, loc="left")

    # --- Calibration: ECE and Brier, cross-subject only ---
    ax2 = fig.add_subplot(gs[2, 0])
    ece_vals = [cs[m]["ece"] * 100 for m in MODEL_ORDER]
    bars = ax2.bar(x, ece_vals, width=0.55, color=MODEL_COLORS, zorder=3)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(MODEL_LABELS, fontsize=8.5)
    ax2.set_ylabel("Expected Calibration Error (%)", fontsize=9.5)
    ax2.set_ylim(0, max(ece_vals) * 1.4)
    style_axes(ax2)
    ax2.set_title("Calibration error (lower = better)\ncross-subject protocol", fontsize=11, fontweight="bold",
                  color=INK, pad=10, loc="left")

    # --- Reliability diagram: Laya vs best baseline, cross-subject ---
    ax3 = fig.add_subplot(gs[2, 1])
    ax3.plot([0, 1], [0, 1], color=GRID, linewidth=1.5, linestyle="--", zorder=1, label="Perfect calibration")
    best_name = min(["LDA", "SVM", "RandomForest"], key=lambda m: cs[m]["ece"])
    for name, color in [(best_name, MUTED), ("Laya", ORANGE)]:
        bins = cs[name]["reliability"]
        conf = [b["mean_confidence"] for b in bins if b["n"] > 0]
        acc = [b["accuracy"] for b in bins if b["n"] > 0]
        label = "Laya (early ckpt)" if name == "Laya" else f"{name} (best baseline)"
        ax3.plot(conf, acc, marker="o", markersize=4, color=color, linewidth=1.8, label=label, zorder=3)
    ax3.set_xlim(0.45, 1.0)
    ax3.set_ylim(0.3, 1.0)
    ax3.set_xlabel("Mean predicted confidence", fontsize=9.5)
    ax3.set_ylabel("Observed accuracy", fontsize=9.5)
    style_axes(ax3)
    ax3.legend(loc="upper left", frameon=False, fontsize=8)
    ax3.set_title("Reliability diagram\ncross-subject protocol", fontsize=11, fontweight="bold", color=INK,
                  pad=10, loc="left")

    # --- Brier score ---
    ax4 = fig.add_subplot(gs[3, 0])
    brier_vals = [cs[m]["brier"] for m in MODEL_ORDER]
    ax4.bar(x, brier_vals, width=0.55, color=MODEL_COLORS, zorder=3)
    ax4.set_xticks(list(x))
    ax4.set_xticklabels(MODEL_LABELS, fontsize=8.5)
    ax4.set_ylabel("Brier score (lower = better)", fontsize=9.5)
    ax4.set_ylim(0, max(brier_vals) * 1.3)
    style_axes(ax4)
    ax4.set_title("Brier score\ncross-subject protocol", fontsize=11, fontweight="bold", color=INK, pad=10, loc="left")

    # --- Setup summary ---
    ax5 = fig.add_subplot(gs[3, 1])
    ax5.axis("off")
    ax5.set_title("Setup", fontsize=11.5, fontweight="bold", color=INK, pad=10, loc="left")
    lines = [
        ("Base model", "Laya (ModernBERT-large, 421M)"),
        ("Dataset", "STEW: 48 subjects, 14-ch\nEmotiv EPOC EEG, 128 Hz"),
        ("Features", "10 engineered (Welch PSD band\npower + literature ratios)"),
        ("Laya training", "2 epochs, 3,000/17,820 rows\n(early checkpoint, more in progress)"),
    ]
    y = 0.90
    for label, val in lines:
        ax5.text(0.0, y, label, fontsize=9, fontweight="bold", color=INK, ha="left", va="top",
                  transform=ax5.transAxes)
        ax5.text(0.42, y, val, fontsize=8.3, color=MUTED, ha="left", va="top", transform=ax5.transAxes)
        y -= 0.26

    # --- Note ---
    ax6 = fig.add_subplot(gs[4, :])
    ax6.axis("off")
    note = ("Cross-subject test = STEW's own pre-verified fold of 12 subjects with zero epochs anywhere in\n"
            "training (the honest generalization number). Within-subject = naive random split ignoring subject\n"
            "boundaries (the easier, leakage-prone protocol common in this literature). SVM and Random Forest\n"
            "both gain accuracy under the naive split; LDA barely does -- consistent with EEG's subject-identity\n"
            "signal being exploitable by higher-capacity models. Laya here is an early checkpoint (2 epochs,\n"
            "17% of available training data); a full run is in progress.")
    ax6.text(0.0, 0.95, note, fontsize=8.5, color=MUTED, ha="left", va="top", transform=ax6.transAxes, linespacing=1.6)

    # --- Footer ---
    fig.text(0.09, 0.012,
              "Open-source · fine-tunes github.com/NandhaKishorM/laya · data: STEW (CC BY 4.0)",
              fontsize=9, color=MUTED, ha="left")

    fig.savefig(args.out, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
