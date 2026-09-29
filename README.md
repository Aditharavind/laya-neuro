# laya-neuro (NeuroLaya)

*An unofficial, independent fine-tune of [Laya](https://github.com/NandhaKishorM/laya)
by Nandhakishor M — not affiliated with or endorsed by the original project.*

> **Honesty check on the field's own biggest trap.** EEG carries a strong
> subject-identity signal, so a naive random train/test split on EEG data
> lets a classifier partly learn "which subject is this" and use it as a
> shortcut to recall that subject's workload label — inflating accuracy well
> above what it would score on a genuinely new person. This repo reports
> **both** protocols side by side, always labeled, so the honest number is
> never hidden behind the easier one. See Results.

Fine-tunes [Laya](https://github.com/NandhaKishorM/laya) (by Nandhakishor M /
Convai Innovations) — a non-autoregressive "System 1" decision model that
answers typed questions (`choice` / `score` / `noul`) over a state in a
single forward pass — as a lightweight, **calibrated** binary classifier for
EEG-based cognitive workload, using the public [STEW dataset](https://huggingface.co/datasets/monster-monash/STEW)
(48 subjects, 14-channel Emotiv EPOC EEG, resting-baseline vs. SIMKAP-multitasking).

This project exists because a literature review found something specific and
fixable: the EEG cognitive-workload classification field **almost never
reports calibration** (Expected Calibration Error, Brier score, reliability
diagrams) — evaluation is accuracy-only, and a large fraction of published
studies use leakage-prone random splits that inflate that accuracy in the
first place. Neither problem needs a new architecture to fix — both need
standard, already-documented tools (subject-disjoint splits, ECE/Brier/
temperature scaling) that this subfield simply hasn't been applying. See
[the deep-research summary](#grounding) for the specific sources behind
every claim above.

**Pretrained on Hugging Face:**
[dataset](https://huggingface.co/datasets/Aditharavind/laya-neuro-decisions) ·
[cross-subject model](https://huggingface.co/Aditharavind/laya-neuro) (headline, honest) ·
[within-subject model](https://huggingface.co/Aditharavind/laya-neuro-within-subject) (literature-comparison only)

## Pipeline

```
eeg/features.py        Welch-PSD band-power + engineered ratio feature extraction (raw epoch -> ~10 features)
eeg/build_dataset.py    STEW epochs -> Laya-format JSONL, both split protocols
eeg/baselines.py        classical ML (SVM/RF/LDA) on the same features, same splits -- direct comparison
training/               preprocess.py (tokenize) + train.py (single-GPU RLCD fine-tuning, same recipe as Laya's own notebook)
eval/evaluate.py        accuracy + ECE + Brier score for a fine-tuned checkpoint
app.py                  interactive demo: sample a real held-out EEG epoch, see the model's live prediction vs. ground truth
```

```bash
python eeg/build_dataset.py --raw-dir data/raw --out data
python eeg/baselines.py --data data
python training/preprocess.py --data-dir data/cross_subject --out-dir data/preprocessed_cross_subject
python training/train.py --data-dir data/preprocessed_cross_subject --output-dir checkpoints/laya_neuro_cross_subject --epochs 6
python eval/evaluate.py --checkpoint checkpoints/laya_neuro_cross_subject --data-dir data/cross_subject
python app.py
```

## Data

[STEW](https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset)
(Lim, Sourina & Wang), pulled from its non-gated, CC BY 4.0 Hugging Face
mirror [`monster-monash/STEW`](https://huggingface.co/datasets/monster-monash/STEW):
48 subjects, 14-channel Emotiv EPOC EEG at 128 Hz, pre-segmented into
28,512 two-second epochs, each labeled with a binarized self-reported
workload rating (high/low). This mirror only exposes the binarized label,
not the original 1-9 NASA-TLX-style rating, which is why this project uses
a `noul` (binary) Laya question rather than a `score` question.

**Two evaluation protocols, both always reported together:**

| Protocol | What it is | Use |
|---|---|---|
| `cross_subject/` | STEW's own pre-verified, subject-disjoint fold (test = 12 held-out subjects, zero epochs from those subjects anywhere in train/val) | **The honest headline number** — does this generalize to a new person |
| `within_subject/` | Pure random 70/15/15 split, ignoring subject boundaries | Kept only to compare against literature numbers that use this (easier, leakage-prone) protocol — never presented as the generalization claim |

## Features

Rather than feeding Laya raw 14×256 EEG samples, each epoch is reduced to
~10 named, literature-grounded features via Welch's method PSD
(`eeg/features.py`): relative power in 5 canonical bands (delta/theta/alpha/
beta/gamma), frontal theta, posterior alpha, their ratio, the engagement
index (beta/(alpha+theta)), and theta/alpha ratio. This was sanity-checked
against a large, subject-diverse sample (2000 low- and 2000 high-workload
epochs spanning all 48 subjects) and confirmed to move in the
literature-expected direction: theta rises, alpha falls, and the
frontal-theta/posterior-alpha ratio rises substantially (1.19 -> 1.61) with
workload — consistent with the [Chikhi, Matton & Blanchet (2022)](https://onlinelibrary.wiley.com/doi/10.1111/psyp.14009)
meta-analysis (theta g=0.68, alpha g=-0.25 across 24 studies).

## Results

*(filled in after training + evaluation complete — see `reports/`)*

## Grounding

This project's dataset choice, feature pipeline, baseline family, and
calibration methodology all come from a dedicated research pass (see
`reports/` — gitignored, internal working notes) rather than assumption.
Key sources: STEW dataset card and IEEE DataPort listing; Chikhi, Matton &
Blanchet (2022) EEG-workload meta-analysis; a systematic review of EEG
workload-classification methodology (model prevalence, leakage rate, and
the absence of calibration reporting in this subfield); Guo et al. (2017)
on temperature scaling.

## License / attribution

Fine-tunes [Laya](https://github.com/NandhaKishorM/laya) (Nandhakishor M /
Convai Innovations, Apache 2.0). Training code adapted from Laya's own
fine-tuning notebook (see header comments in `training/train.py`). Dataset:
[STEW](https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset)
(Lim, Sourina & Wang), via the [`monster-monash/STEW`](https://huggingface.co/datasets/monster-monash/STEW)
Hugging Face mirror, CC BY 4.0.
