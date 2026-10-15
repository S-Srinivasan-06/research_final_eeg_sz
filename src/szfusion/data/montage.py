"""Channel definitions and the two constants of the token pipeline. Every value here is explained.

ELECTRODES_19: the 19 referential 10-20 electrodes of the LinOSS branch, in the order of the HyBrain authors' code
  (github.com/hhyy0401/seizure, the DCRNN repository order). TUSZ labels them "EEG <name>-REF" (averaged reference)
  or "EEG <name>-LE" (linked ears); the token pipeline takes the electrode name before the "-" and does not
  re-reference. A recording is skipped if any of the 19 is missing (in TUSZ v2.0.6 that is always FZ/PZ).
SWAP_PERM_19: the hemisphere swap used as training augmentation (p = 0.5): left/right exchange of the 7 pairs
  FP1/FP2, F3/F4, C3/C4, O1/O2, F7/F8, T3/T4, T5/T6. P3/P4 are NOT swapped and the midline (FZ, CZ, PZ) stays: this
  is exactly the authors' get_swap_pairs_idx(), which we kept so the augmentation matches the published baseline.
TOKEN_MEAN, TOKEN_STD: one global z-score for all log|FFT| values, the scalar mean and standard deviation stored in
  the HyBrain authors' marker files means/stds_seq2seq_fft_12s_szdetect_single.pkl (TUSZ train). Gain augmentation
  s ~ U(0.8, 1.2) becomes the additive shift log(s) / TOKEN_STD in this domain, because log|FFT(s x)| = log s + log|FFT(x)|.
TCP22: the 22 bipolar channels of the TUSZ TCP montage, on which TUSZ gives per-channel seizure labels (.csv).
CHAINS16: the 16 longitudinal bipolar chains of the CatBoost branch (Zabihi et al. 2026).
"""
import numpy as np

ELECTRODES_19 = ["FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
                 "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ"]
SWAP_PAIRS_19 = [("FP1", "FP2"), ("F3", "F4"), ("C3", "C4"), ("O1", "O2"), ("F7", "F8"), ("T3", "T4"), ("T5", "T6")]


def swap_perm(names=ELECTRODES_19, pairs=SWAP_PAIRS_19):
    """Index permutation that exchanges each pair (applied to the channel axis)."""
    perm = np.arange(len(names))
    for a, b in pairs:
        i, j = names.index(a), names.index(b)
        perm[i], perm[j] = j, i
    return perm


SWAP_PERM_19 = swap_perm()
TOKEN_MEAN = 3.9594627590488702
TOKEN_STD = 1.5625200363244962

TCP22 = ["FP1-F7", "F7-T3", "T3-T5", "T5-O1", "FP2-F8", "F8-T4", "T4-T6", "T6-O2", "A1-T3", "T3-C3", "C3-CZ",
         "CZ-C4", "C4-T4", "T4-A2", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2"]
CHAINS16 = ["FP1-F7", "F7-T3", "T3-T5", "T5-O1", "FP2-F8", "F8-T4", "T4-T6", "T6-O2",
            "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2"]


def channel_electrodes(channels=TCP22, electrodes=ELECTRODES_19):
    """For each bipolar channel, the indices of its electrodes among `electrodes` (A1/A2 are not among them)."""
    return [[electrodes.index(e) for e in c.split("-") if e in electrodes] for c in channels]
