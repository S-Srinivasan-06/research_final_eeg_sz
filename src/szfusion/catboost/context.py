"""Temporal context, ported from [GH infer.py:151-298 add_temporal_context_features] (k=7, per file_id).

For every base feature f (11 new columns -> 941 x 12 = 11,292 in the paper):
  f_ctx_mean, f_ctx_min, f_ctx_max, f_ctx_std (ddof=0), f_ctx_range, f_ctx_z = (f - mean)/(std + 1e-6)
  f_d1_rate = (f_i - f_{i-1}) / (t_i - t_{i-1});  f_d1_mean_pm7, f_d1_std_pm7 (rolling +-k over d1_rate)
  f_d2_rate = (f_{i+1} - 2 f_i + f_{i-1}) / median_stride^2
  f_long_slope_pm7 = (f_{i+k} - f_{i-k}) / (t_{i+k} - t_{i-k})
Centred windows truncated at file edges (min_periods=1); non-positive denominators / out-of-range -> NaN
(CatBoost handles NaN natively). Non-causal (uses future windows) -> offline only [M].
add_context(df, feat_cols, ccfg) -> (df_out, all_feature_cols); df is re-sorted by (rec_id, t0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STATS = ["ctx_mean", "ctx_min", "ctx_max", "ctx_std", "ctx_range", "ctx_z", "d1_rate", "d1_mean_pm{k}",
         "d1_std_pm{k}", "d2_rate", "long_slope_pm{k}"]


def _file_context(A, t, k):
    """A [n,d] float64, t [n] -> [n, 11*d] in STATS order."""
    n, d = A.shape
    w = 2 * k + 1
    roll = lambda M: pd.DataFrame(M).rolling(w, center=True, min_periods=1)
    r = roll(A)
    mean, mn, mx = r.mean().to_numpy(), r.min().to_numpy(), r.max().to_numpy()
    std = r.std(ddof=0).to_numpy()
    rng = mx - mn
    z = (A - mean) / (std + 1e-6)
    dt = np.r_[np.nan, np.diff(t)]
    dt = np.where(dt > 0, dt, np.nan)
    d1 = np.vstack([np.full((1, d), np.nan), np.diff(A, axis=0)]) / dt[:, None]
    r1 = roll(d1)
    d1m, d1s = r1.mean().to_numpy(), r1.std(ddof=0).to_numpy()
    diffs = np.diff(t); diffs = diffs[diffs > 0]
    ms = float(np.median(diffs)) if len(diffs) else np.nan
    d2 = np.full((n, d), np.nan)
    if n >= 3 and ms == ms and ms > 0:
        d2[1:-1] = (A[2:] - 2 * A[1:-1] + A[:-2]) / ms ** 2
    ls = np.full((n, d), np.nan)
    if n > 2 * k:
        den = t[2 * k:] - t[:-2 * k]
        ls[k:n - k] = (A[2 * k:] - A[:-2 * k]) / np.where(den > 0, den, np.nan)[:, None]
    return np.concatenate([mean, mn, mx, std, rng, z, d1, d1m, d1s, d2, ls], axis=1)


def add_context(df: pd.DataFrame, feat_cols, ccfg) -> tuple:
    ccfg = ccfg or {}
    k = int(ccfg.get("k", 0))
    if ccfg.get("mode", "zabihi") == "none" or k <= 0 or len(df) == 0:
        return df, list(feat_cols)
    df = df.sort_values(["rec_id", "t0"], kind="stable").reset_index(drop=True)
    src = list(feat_cols)
    A_all = df[src].to_numpy(np.float64)
    t_all = df["t0"].to_numpy(np.float64)
    rid = df["rec_id"].to_numpy()
    bounds = np.flatnonzero(np.r_[True, rid[1:] != rid[:-1], True])
    stats = [s.format(k=k) for s in STATS]
    names = [f"{c}_{s}" for s in stats for c in src]
    out = np.empty((len(df), len(names)), np.float32)
    for a, b in zip(bounds[:-1], bounds[1:]):
        out[a:b] = _file_context(A_all[a:b], t_all[a:b], k)
    extra = pd.DataFrame(out, columns=names, index=df.index)
    return pd.concat([df, extra], axis=1), src + names
