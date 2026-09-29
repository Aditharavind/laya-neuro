---
license: cc-by-4.0
task_categories:
- text-classification
tags:
- eeg
- cognitive-workload
- neurotechnology
- laya
size_categories:
- 10K<n<100K
---

# laya-neuro Decisions

*Part of laya-neuro, an unofficial, independent fine-tune of
[Laya](https://github.com/NandhaKishorM/laya) by Nandhakishor M — not
affiliated with or endorsed by the original project. This dataset is a
derivative of [STEW](https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset)
(Lim, Sourina & Wang), via its Hugging Face mirror
[`monster-monash/STEW`](https://huggingface.co/datasets/monster-monash/STEW)
(CC BY 4.0) — engineered features and Laya-format rows derived from that
source's raw EEG, not original data collection.*

EEG-derived cognitive-workload decision states for
[Laya](https://github.com/NandhaKishorM/laya)'s typed-decision fine-tuning
recipe. Each row is one 2-second EEG epoch: a `state` (10 engineered
features — relative band power per canonical band plus literature-grounded
ratios), a fixed `questions` schema (one `noul`/binary question,
`high_workload`), and `gold` — a confident soft probability distribution
matching STEW's binarized self-reported workload label.

## How it was generated

STEW's HF mirror provides 28,512 pre-segmented, 14-channel (Emotiv EPOC),
128 Hz, 2-second epochs with a binarized high/low workload label and a
subject ID. `eeg/features.py` (in the source repo) reduces each epoch to
~10 named features via Welch's method PSD: relative power in 5 canonical
bands (delta/theta/alpha/beta/gamma), frontal theta, posterior alpha, their
ratio, the engagement index (beta/(alpha+theta)), and theta/alpha ratio.
`eeg/build_dataset.py` turns those into Laya-format rows and produces two
split protocols.

## Splits — two protocols, always reported separately

| split | protocol | rows | notes |
|---|---|---|---|
| `cross_subject/{train,val,test}` | subject-disjoint | 17,820 / 3,564 / 7,128 | **the honest protocol** — `test` is STEW's own pre-verified fold of 12 subjects with zero epochs anywhere in train/val |
| `within_subject/{train,val,test}` | naive random | 19,960 / 4,276 / 4,276 | ignores subject boundaries — kept only for comparison with literature numbers that use this easier, leakage-prone protocol |

`stew_features.parquet` also included: the full 28,512-row engineered
feature table (with `label` and `subject_id` columns) used to train the
classical ML baselines (SVM/RF/LDA) reported alongside Laya.

## Schema

```json
{
  "state": "{\"delta_mean\": 0.618, \"theta_mean\": 0.141, \"alpha_mean\": 0.083, \"beta_mean\": 0.113, \"gamma_mean\": 0.048, \"frontal_theta\": 0.145, \"posterior_alpha\": 0.109, \"frontal_theta_posterior_alpha_ratio\": 1.326, \"engagement_index\": 0.504, \"theta_alpha_ratio\": 1.706}",
  "questions": "{\"high_workload\": {\"type\": \"noul\", \"instructions\": \"Given this subject's EEG-derived features for this 2-second window, is this subject currently experiencing high cognitive workload?\"}}",
  "gold": "{\"high_workload\": {\"probabilities\": {\"true\": 0.95, \"false\": 0.05}}}"
}
```

## Use

```python
from datasets import load_dataset
ds = load_dataset("Aditharavind/laya-neuro-decisions", data_files="cross_subject/train.jsonl")
```

Or point `training/preprocess.py` in the source repo at these files directly.

## License

CC BY 4.0, matching the source [STEW](https://huggingface.co/datasets/monster-monash/STEW)
mirror's license. Attribution: Lim, W.L., Sourina, O., Wang, L.P. (2018),
"STEW: Simultaneous Task EEG Workload Dataset," IEEE Transactions on
Neural Systems and Rehabilitation Engineering.
