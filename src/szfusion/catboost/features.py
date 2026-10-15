"""Zabihi et al. 2026 (Sci Rep) features: PARTIAL port of [GH fxpipeline.py] (repo read via WebFetch summary, 2026-10-12).

The paper's fxpipeline.py has 941 base columns; formulas live in an unread Supplement, so this module implements
the NAMED families with repo parameters where known; NOT yet ported [DEFAULT-omitted]: SWI-3Hz spike-wave,
sharpness asymmetry, hemispheric/AP asymmetry indices, alpha burst fraction, posterior coherence & PLV, TKEO,
median/95% edge freq, _summarize_ts temporal stats, 5-s subwindow band powers. Count < 941 -> see README.
  load_bipolar18(rec, dcfg) -> x [16,T] volts (200 Hz, notch 60 + HP1 0.5 + LP4 35, zero-phase), names
  window_labels_zabihi(t0, win, intervals) -> y  (overlap >= min(9 s, 0.2 x seizure duration))
  extract_window_features(X [N,18,W], fs, fcfg) -> F float32 [N,K], feature_names(fcfg) -> K names
"""
from __future__ import annotations

import math

import numpy as np

from szfusion.data.tusz import REF19, load_eeg

PAIRS18 = [("FP1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"), ("FP2", "F8"), ("F8", "T4"), ("T4", "T6"),
           ("T6", "O2"), ("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"), ("FP2", "F4"), ("F4", "C4"),
           ("C4", "P4"), ("P4", "O2")]          # [GH preprocess_TUHEEGSeizure.py:47-52] 16 pairs (spec's 18 was wrong)
PAIRS16 = PAIRS18
NAMES18 = [f"{a}-{b}" for a, b in PAIRS16]
# region -> channel indices = LEFT_UP / LEFT_POST / RIGHT_UP / RIGHT_POST  [GH fxpipeline.py region constants]
REGIONS = {"la": [0, 1, 8, 9], "lp": [2, 3, 10, 11], "ra": [4, 5, 12, 13], "rp": [6, 7, 14, 15]}
# chain groups for PLV variants (spec names lp/lu/rp/ru; mapping is [DEFAULT]): l/r temporal + l/r parasagittal
CHAINS = {"all": list(range(16)), "lu": [0, 1, 2, 3], "lp": [8, 9, 10, 11], "ru": [4, 5, 6, 7], "rp": [12, 13, 14, 15]}
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}   # [GH fxpipeline.py BANDS]
PLV_BANDS = {"delta": (2, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}  # [GH _band_global_plv_features]


# ----------------------------------------------------------------------------- loading / labels
def _filters(x, fs, dcfg):
    from scipy.signal import butter, iirnotch, filtfilt, sosfiltfilt
    if dcfg.get("notch_hz"):
        b, a = iirnotch(dcfg["notch_hz"], Q=dcfg.get("notch_q", 30.0), fs=fs)
        x = filtfilt(b, a, x, axis=-1)
    if dcfg.get("hp_hz"):
        x = sosfiltfilt(butter(dcfg.get("hp_order", 1), dcfg["hp_hz"], "highpass", fs=fs, output="sos"), x, axis=-1)
    if dcfg.get("lp_hz"):
        x = sosfiltfilt(butter(dcfg.get("lp_order", 4), dcfg["lp_hz"], "lowpass", fs=fs, output="sos"), x, axis=-1)
    return x


def load_bipolar18(rec, dcfg):
    """common.load_eeg('ref19', no filtering) -> 16 longitudinal bipolar pairs -> filters -> VOLTS
    [GH preprocess_TUHEEGSeizure.py:47-52 pairs, :389-405 filters, :197-200 uV->V]. Resampler: scipy
    resample_poly (repo: MNE raw.resample, :369) [ADAPT]. Missing pairs zero-filled (repo tolerates <=2) [ADAPT]."""
    fs = dcfg["fs"]
    ref, _ = load_eeg(rec, "ref19", fs=fs, bandpass=None, notch=None)
    ix = {e: i for i, e in enumerate(REF19)}
    x = np.stack([ref[ix[a]].astype(np.float64) - ref[ix[b]] for a, b in PAIRS16]) * dcfg.get("scale", 1e-6)
    return np.ascontiguousarray(_filters(x, fs, dcfg), dtype=np.float32), list(NAMES18)


def window_labels_zabihi(t0, win_sec, intervals, abs_sec=9.0, frac=0.20):
    """Positive iff overlap with some merged seizure >= min(abs_sec, frac x that seizure's duration)."""
    t0 = np.asarray(t0, float); t1 = t0 + win_sec
    y = np.zeros(len(t0), bool)
    for a, b in intervals:
        ov = np.clip(np.minimum(t1, b) - np.maximum(t0, a), 0, None)
        y |= (ov > 0) & (ov >= min(abs_sec, frac * (b - a)))
    return y.astype(np.int8)


# ----------------------------------------------------------------------------- feature families
def _bandpower(f, P, lo, hi):
    m = (f >= lo) & (f < hi)
    return P[..., m].sum(-1) * (f[1] - f[0])


def _base_channel_features(X, fs, fcfg):
    """X [N,C,W] -> dict name -> [N,C]."""
    from scipy.signal import welch, butter, sosfiltfilt, hilbert
    eps = 1e-12
    N, C, W = X.shape
    out = {}
    d1 = np.diff(X, axis=-1); d2 = np.diff(d1, axis=-1)
    v0, v1, v2 = X.var(-1), d1.var(-1), d2.var(-1)
    # time domain [M]
    out["rms"] = np.sqrt((X ** 2).mean(-1))
    out["line_length"] = np.abs(d1).mean(-1)
    out["zcr"] = (X[..., 1:] * X[..., :-1] < 0).sum(-1) * fs / (W - 1)   # [GH fxpipeline.py zero-cross]
    mob = np.sqrt(v1 / (v0 + eps))
    out["hjorth_activity"] = v0; out["hjorth_mobility"] = mob
    out["hjorth_complexity"] = np.sqrt(v2 / (v1 + eps)) / (mob + eps)
    # Welch spectral [M, GH]
    nper = max(256, min(W, int(round(fcfg.get("welch_sec", 4) * fs))))   # [GH fxpipeline.py Welch]
    f, P = welch(X, fs=fs, nperseg=nper, noverlap=nper // 2, axis=-1)
    lo_t, hi_t = fcfg.get("total_range", [0.5, 30])
    bp = {b: _bandpower(f, P, lo, hi) for b, (lo, hi) in BANDS.items()}
    tot = _bandpower(f, P, lo_t, hi_t) + eps
    for b in BANDS:
        out[f"welch_rel_{b}"] = bp[b] / tot
    out["ratio_theta_over_delta"] = bp["theta"] / (bp["delta"] + eps)
    out["ratio_beta_over_theta"] = bp["beta"] / (bp["theta"] + eps)
    out["ratio_(alpha+beta)_over_(delta+theta)"] = (bp["alpha"] + bp["beta"]) / (bp["delta"] + bp["theta"] + eps)
    m = (f >= lo_t) & (f < hi_t); ft, Pt = f[m], P[..., m]
    pn = Pt / (Pt.sum(-1, keepdims=True) + eps)
    out["spectral_entropy"] = -(np.where(pn > 0, pn * np.log2(pn + eps), 0)).sum(-1) / math.log2(len(ft))
    lf, lp = np.log10(ft[ft >= 1]), np.log10(Pt[..., ft >= 1] + eps)  # 1/f slope: LS fit log P vs log f
    lfc = lf - lf.mean()
    out["aperiodic_slope"] = ((lp - lp.mean(-1, keepdims=True)) * lfc).sum(-1) / (lfc ** 2).sum()
    am = (f >= 8) & (f < 13)
    out["alpha_peak_hz"] = f[am][np.argmax(P[..., am], -1)]
    # dominant frequency + its dynamics over 5-s subwindows [GH]
    dom = ft[np.argmax(Pt, -1)]
    out["domfreq_hz"] = dom
    half = Pt >= 0.5 * Pt.max(-1, keepdims=True)
    bw = (half.sum(-1)) * (ft[1] - ft[0])
    out["domfreq_bw_hz"] = bw; out["domfreq_Q"] = dom / (bw + eps)
    sub = int(fcfg.get("subwin_sec", 5) * fs); ns = W // sub
    if ns >= 2:
        Xs = X[..., : ns * sub].reshape(N, C, ns, sub)
        fs_, Ps = welch(Xs, fs=fs, nperseg=sub, axis=-1)
        ms = (fs_ >= lo_t) & (fs_ < hi_t)
        ds = fs_[ms][np.argmax(Ps[..., ms], -1)]  # [N,C,ns]
        out["domfreq_std_over_time_hz"] = ds.std(-1)
        out["domfreq_cv_over_time"] = ds.std(-1) / (ds.mean(-1) + eps)
        tt = (np.arange(ns) - (ns - 1) / 2) * (sub / fs / 60.0)
        out["domfreq_slope_hz_per_min"] = (ds * tt).sum(-1) / (tt ** 2).sum()
        # within-window dynamics [M]: SD of line length across subwindows
        out["subwin_ll_cv"] = (lambda L: L.std(-1) / (L.mean(-1) + eps))(np.abs(np.diff(Xs, axis=-1)).mean(-1))
    # spectral shape per band [GH]
    for b, (lo, hi) in BANDS.items():
        mb = (f >= lo) & (f < hi); Pb = P[..., mb] + eps
        out[f"shape_crest_{b}"] = Pb.max(-1) / Pb.mean(-1)
        out[f"shape_flatness_{b}"] = np.exp(np.log(Pb).mean(-1)) / Pb.mean(-1)
    # theta rhythmicity: max normalised autocorrelation at lags 1/8..1/4 s [DEFAULT]
    th = sosfiltfilt(butter(4, [4, 8], "bandpass", fs=fs, output="sos"), X, axis=-1)
    thc = th - th.mean(-1, keepdims=True)
    lags = range(int(fs / 8), int(fs / 4) + 1)
    den = (thc ** 2).sum(-1) + eps
    out["theta_rhythmicity_autocorr"] = np.max([(thc[..., :-L] * thc[..., L:]).sum(-1) / den for L in lags], 0)
    # wavelet relative energies (db4, 5 levels) [DEFAULT]
    if True:                      # PyWavelets is required: without it 11 columns would silently disappear
        import pywt
        coeffs = pywt.wavedec(X, fcfg.get("wavelet", "db4"), level=fcfg.get("wavelet_level", 5), axis=-1)
        E = np.stack([(c ** 2).sum(-1) for c in coeffs], -1)  # [N,C,L+1] A5,D5..D1
        Er = E / (E.sum(-1, keepdims=True) + eps)
        # at 200 Hz: D5 3.1-6.25 (theta-ish), D4 6.25-12.5 (alpha), D3 12.5-25 (beta), A5 0-3.1 (delta)
        for j, b in enumerate(["delta", "theta", "alpha", "beta"]):
            out[f"wav_rel_{b}"] = Er[..., j]
        for j in range(Er.shape[-1]):
            out[f"wav_energy_L{j}"] = np.log(E[..., j] + eps)
        out["wav_entropy"] = -(Er * np.log2(Er + eps)).sum(-1)
    # envelope [GH]
    for lo, hi in ((4, 30), (8, 30)):
        xb = sosfiltfilt(butter(4, [lo, hi], "bandpass", fs=fs, output="sos"), X, axis=-1)
        env = np.abs(hilbert(xb, axis=-1))
        out[f"env_mean_{lo}_{hi}"] = env.mean(-1)
        out[f"env_cv_{lo}_{hi}"] = env.std(-1) / (env.mean(-1) + eps)
    return out


def _plv_features(X, fs):
    """PLV in canonical bands: mean over pairs in each chain group + leading eigenvalue (normalised)."""
    from scipy.signal import butter, sosfiltfilt, hilbert
    out = {}
    for b, (lo, hi) in PLV_BANDS.items():
        ph = np.angle(hilbert(sosfiltfilt(butter(4, [lo, hi], "bandpass", fs=fs, output="sos"), X, axis=-1), axis=-1))
        z = np.exp(1j * ph)                                  # [N,C,W]
        plv = np.abs(np.einsum("ncw,ndw->ncd", z, z.conj()) / X.shape[-1])  # [N,C,C]
        for g, idx in CHAINS.items():
            sub = plv[:, idx][:, :, idx]; n = len(idx)
            out[f"plv_mean_{g}_{b}"] = (sub.sum((1, 2)) - n) / (n * (n - 1))
        out[f"plv_leading_eig_{b}"] = np.linalg.eigvalsh(plv)[:, -1] / X.shape[1]
    return out


def feature_names(fcfg=None, channel_names=None) -> list:
    """Names via a dry run on a tiny random input (cheap, guarantees order matches extract)."""
    fs = (fcfg or {}).get("_fs", 200); W = int((fcfg or {}).get("_win_sec", 60) * fs)
    _, names = _extract(np.random.default_rng(0).standard_normal((1, 16, W)), fs, fcfg or {})
    return names


def _extract(X, fs, fcfg):
    X = np.asarray(X, dtype=np.float64)
    ch = _base_channel_features(X, fs, fcfg)
    cols, names = [], []
    for name, v in ch.items():  # region aggregation: median, p95, p95-median focality [GH fxpipeline.py]
        for r, idx in REGIONS.items():
            med = np.median(v[:, idx], 1); p95 = np.percentile(v[:, idx], 95, axis=1)
            cols += [med, p95, p95 - med]; names += [f"{name}_{r}", f"{name}_{r}_p95", f"{name}_{r}_focal"]
    if fcfg.get("plv", True):
        for name, v in _plv_features(X, fs).items():
            cols.append(v); names.append(name)
    F = np.nan_to_num(np.stack(cols, 1), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return F, names


def extract_window_features(X, fs, fcfg=None):
    return _extract(X, fs, fcfg or {})[0]
