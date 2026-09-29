---
license: apache-2.0
base_model: convaiinnovations/laya
base_model_relation: finetune
tags:
- eeg
- cognitive-workload
- neurotechnology
- laya
- rlcd
- calibration
datasets:
- Aditharavind/laya-neuro-decisions
pipeline_tag: text-classification
---

# laya-neuro (early checkpoint)

*An unofficial, independent fine-tune of [Laya](https://github.com/NandhaKishorM/laya)
by Nandhakishor M / Convai Innovations — not affiliated with or endorsed by
the original project.*

> **Early checkpoint, not the final result.** Trained on 3,000 of the
> 17,820 available training examples for 2 epochs (a fast initial run).
> A full run on all training data is in progress. See
> [Aditharavind/laya-neuro](https://github.com/Aditharavind/laya-neuro) for
> the full pipeline and updated results.

A [Laya](https://github.com/NandhaKishorM/laya) checkpoint (ModernBERT-large
encoder, 421M params) fine-tuned via RLCD to classify cognitive workload
(high/low) from EEG-derived features, using the public
[STEW dataset](https://huggingface.co/datasets/monster-monash/STEW)
(48 subjects, 14-channel Emotiv EPOC EEG, resting-baseline vs.
SIMKAP-multitasking).

**Evaluated on the honest, subject-disjoint protocol**: the test set is 12
subjects held out entirely from training (STEW's own pre-verified
subject-disjoint fold), so this number reflects generalization to a new
person, not a same-subject shortcut.

## Training

- Base: `convaiinnovations/laya`
- Data: [Aditharavind/laya-neuro-decisions](https://huggingface.co/datasets/Aditharavind/laya-neuro-decisions), `cross_subject/` split, 3,000-example random subset of the 17,820-example train set (early run)
- Recipe: RLCD (GRPO-style policy gradient over proper scoring rules + soft cross-entropy), 2 epochs, single 6GB consumer GPU, 8-bit AdamW (bitsandbytes)

## Results (this checkpoint)

| Metric | Cross-subject test (n=7,128, 12 held-out subjects) |
|---|---|
| Accuracy | 66.3% |
| ECE | 8.3% |
| Brier score | 0.449 |

For comparison, classical ML baselines trained on identical features and
the identical subject-disjoint split: LDA 67.3%, SVM (RBF) 67.1%, Random
Forest 65.7%. This early Laya checkpoint is already in the same range as
those baselines despite seeing under 17% of the available training data.

## Input format

```json
{
  "state": {"delta_mean": 0.4, "theta_mean": 0.14, "alpha_mean": 0.08, "beta_mean": 0.11,
            "gamma_mean": 0.05, "frontal_theta": 0.14, "posterior_alpha": 0.11,
            "frontal_theta_posterior_alpha_ratio": 1.3, "engagement_index": 0.5,
            "theta_alpha_ratio": 1.7},
  "questions": {"high_workload": {"type": "noul",
                "instructions": "Given this subject's EEG-derived features for this 2-second window, is this subject currently experiencing high cognitive workload?"}}
}
```

Features are relative EEG band power (Welch PSD, 5 canonical bands) plus
engineered ratios, extracted per 2-second epoch — see `eeg/features.py` in
the GitHub repo for the exact extraction code.

## License / attribution

Fine-tunes [Laya](https://github.com/NandhaKishorM/laya) (Nandhakishor M /
Convai Innovations, Apache 2.0). Dataset:
[STEW](https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset)
(Lim, Sourina & Wang), via the [`monster-monash/STEW`](https://huggingface.co/datasets/monster-monash/STEW)
Hugging Face mirror, CC BY 4.0.
