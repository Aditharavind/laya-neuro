"""EEG -> compact engineered feature set, for a lightweight typed-decision classifier.

Pipeline (per the project's deep-research: Welch PSD is the standard,
implementable core of EEG band-power feature extraction; MNE/YASA are
convenience wrappers around the same scipy.signal.welch primitive, not used
here to keep this dependency-light):

    raw epoch (14 channels x 256 samples, 2s @ 128Hz)
    -> Welch PSD per channel (1s / 128-sample window, 50% overlap)
    -> integrate PSD over 5 canonical bands per channel
    -> relative band power (band / total 0.5-45Hz power) per channel
    -> a small set of literature-grounded engineered ratios:
       engagement index = beta / (alpha + theta)  (Pope, Bogart & Bartolome)
       frontal theta vs. posterior alpha contrast  (theta rises frontally,
       alpha falls posteriorly with workload -- the most literature-grounded
       single contrast for workload, per the project's meta-analysis review)

This produces roughly 10-15 named features -- a "minimal" tier per the
research (vs. a 70-feature 5-bands x 14-channels tier), appropriate for a
state a typed-decision model reads as text, not a wide numeric vector.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch

FS = 128  # Hz, STEW's native sampling rate

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

# Standard 14-channel Emotiv EPOC montage order, as used to collect STEW.
CHANNELS = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
FRONTAL_IDX = [CHANNELS.index(c) for c in ["AF3", "F7", "F3", "FC5", "F4", "F8", "AF4"]]
POSTERIOR_IDX = [CHANNELS.index(c) for c in ["P7", "O1", "O2", "P8"]]


def band_power_per_channel(epoch: np.ndarray, fs: int = FS) -> dict:
    """epoch: (n_channels, n_samples). Returns {band: array of shape (n_channels,)} in relative power."""
    nperseg = min(128, epoch.shape[-1])
    freqs, psd = welch(epoch, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, axis=-1)
    total_mask = (freqs >= 0.5) & (freqs <= 45.0)
    total_power = np.trapz(psd[:, total_mask], freqs[total_mask], axis=-1)
    total_power = np.maximum(total_power, 1e-12)

    out = {}
    for band, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs <= hi)
        band_power = np.trapz(psd[:, mask], freqs[mask], axis=-1)
        out[band] = band_power / total_power  # relative power, per channel
    return out


def extract_features(epoch: np.ndarray, fs: int = FS) -> dict:
    """epoch: (14, 256) raw EEG. Returns a flat dict of named engineered features."""
    bp = band_power_per_channel(epoch, fs=fs)

    feats = {}
    for band in BANDS:
        feats[f"{band}_mean"] = float(np.mean(bp[band]))

    frontal_theta = float(np.mean(bp["theta"][FRONTAL_IDX]))
    posterior_alpha = float(np.mean(bp["alpha"][POSTERIOR_IDX]))
    beta_mean = feats["beta_mean"]
    alpha_mean = feats["alpha_mean"]
    theta_mean = feats["theta_mean"]

    feats["frontal_theta"] = frontal_theta
    feats["posterior_alpha"] = posterior_alpha
    feats["frontal_theta_posterior_alpha_ratio"] = frontal_theta / max(posterior_alpha, 1e-6)
    feats["engagement_index"] = beta_mean / max(alpha_mean + theta_mean, 1e-6)
    feats["theta_alpha_ratio"] = theta_mean / max(alpha_mean, 1e-6)
    return feats
