"""How and why the CatBoost and LinOSS branches make different detections and different mistakes (TUSZ eval, 1 s).

  python -m szfusion.eval.complementarity [--fusion_1s <work>/tusz/fusion_1s] [--tokens <work>/tokens/tusz]
         [--out results/tusz/complementarity.json] [--figs docs/figures] [--n_boot 2000]

Inputs (nothing is selected or tuned here; every operating point is the dev-frozen one in selection.json):
  eval_tracks.npz, selection.json of the 1 s fusion stage (fusion/run_1s.py). Default folder: the official run
  (<work>/tusz/fusion_1s) when its eval_tracks.npz exists, else the numerically identical S1-gate copy
  (eval/by_type.py default_fusion_dir). The row mapping to the token store is evidence.load (asserted).
  The eval token store (tokens.npy, y_ch.npy, index.json): z-scored log|FFT| of each referential electrode,
  1 Hz bins 0-99 Hz, no filtering (tokens.py). This is exactly what the LinOSS branch reads.
Branches: CatBoost = `catboost`, LinOSS = `linoss_mean` (5-seed mean, the stacker input), fusion = `fused_mean`.
Operating points:
  ev, ev2  event points 1 and 2 (selection.json configs[m].ev / .ev2: smoothing k, thr, gap, min_dur; dev FA/24h
           budgets 16.56 / 33.12). Each model is read at its own frozen point.
  f1_thr   per-second dev-F1 threshold on the raw (unsmoothed) per-second probability (configs[m].f1_thr).

1. Event Venn. A true event (run of y = 1) is detected by a model at a point if any of its predicted seconds
   overlaps it (by_type.detection, identical to protocol_1s.event_scores). Groups: both / CatBoost only / LinOSS
   only / neither; for each, how many the fusion detects at the same point, and the events' type (by_type rule),
   duration, labelled channels (evidence.py rule: TCP channels of the recording's valid channels with a .csv label in
   at least one second of the event) and signed latency of each model.
2. False alarms. A false-alarm (FA) event is a predicted event overlapping no annotated second. A CatBoost FA event is
   "shared" if it overlaps a LinOSS FA event of the same point (and vice versa); the fusion "keeps" an FA event if a
   fused predicted event of the same point overlaps it. Second level, at the f1_thr of each branch: FA second = y = 0
   and p > f1_thr; groups CatBoost only / LinOSS only / both; fusion keeps = fused p > fused f1_thr.
   Spectral content of FA seconds from the tokens: for each second, the mean of the tokens over the bins of a band
   (BANDS) and over the 19 electrodes, converted to dB of amplitude (DB_PER_TOKEN) and expressed relative to the mean
   of the same recording's background (y = 0) seconds ("relative dB"); this removes recording-level offsets (montage,
   amplifier). Electrode spread: number of the 19 electrodes whose band value exceeds that electrode's background
   mean in the recording by more than SPREAD_Z background SDs. Spectral statistics are computed on "remote" FA
   seconds, at least REMOTE_S seconds from any annotated seizure second of the recording, so that seconds smeared
   over from a seizure boundary do not count as artefact false alarms; the counts near seizures are reported apart.
   Effect sizes: difference of group means (LinOSS-only minus CatBoost-only), 95 % percentile bootstrap CI over
   recordings (n_boot resamples, seed BOOT_SEED). Event-level FA spectra: each FA event's mean over its seconds.
3. Temporal behaviour. Event-aligned mean per-second probability (raw tracks) at onset and offset, +-WIN_ALIGN s,
   over events of at least MIN_DUR_ALIGN s; seconds outside the recording's scored prefix or inside another
   annotated event are excluded. Half-maximum time of each mean curve: baseline = mean of the curve far from the
   event (onset: [-120, -90] s; offset: [+90, +120] s), plateau = mean inside (onset: [+20, +40] s; offset:
   [-40, -20] s); time of the first (onset) / last (offset) second at which the curve is >= the midpoint; 10-90 %
   width likewise. Per event (all events): onset time-to-threshold = start of the first run of p > f1_thr that
   overlaps the event, minus the onset (truncated at -WIN_ALIGN s); offset = end of the last such run minus the
   annotated end (truncated at +WIN_ALIGN s). Paired LinOSS - CatBoost differences over events reached by both,
   bootstrap CI over recordings.
4. Disagreement map: 2-D density of (z_C, z_L) logits for ictal and background seconds; quadrants at the two
   f1_thr; fused positive rate per quadrant; the stacker's (lr_ctx) contour at the fused f1_thr drawn with its two
   context inputs S15(z_L), S61(z_L) set (a) equal to z_L (a steady LinOSS track) and (b) to their eval medians over
   background seconds.
5. Correlation of the branch logits (Pearson and Spearman) by seizure type (ictal seconds of the type's events) and per
   patient (distribution only; no ID is stored, RULES.md 4).

Every number written comes from this code. Hypotheses are labelled as such in the JSON.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import mannwhitneyu, spearmanr  # noqa: E402

from szfusion import paths  # noqa: E402
from szfusion.data.montage import TCP22, TOKEN_STD  # noqa: E402
from szfusion.eval import by_type as BT  # noqa: E402
from szfusion.eval.mechanism import provenance  # noqa: E402
from szfusion.fusion import protocol_1s as P  # noqa: E402
from szfusion.fusion.stacker import Stacker, logit  # noqa: E402

MODELS = {"catboost": "CatBoost", "linoss_mean": "LinOSS", "fused_mean": "fusion"}
POINTS = ["ev", "ev2"]
# 1 Hz token bins (bin f = f Hz). Bands are half-open [lo, hi) in Hz; delta starts at 1 (bin 0 is DC).
BANDS = {"delta_0.5-4": list(range(1, 4)), "theta_4-8": list(range(4, 8)), "alpha_8-13": list(range(8, 13)),
         "beta_13-30": list(range(13, 30)), "gamma_30-45": list(range(30, 45)),
         "high_45-99_excl_59-61": [b for b in range(45, 100) if b not in (59, 60, 61)], "line_60": [60]}
HF_LF = (("gamma_30-45", "high_45-99_excl_59-61"), ("delta_0.5-4", "theta_4-8"))   # spectral-shape contrast:
#   mean of the high-frequency bands minus mean of the low-frequency bands (relative dB); insensitive to a broadband
#   level change, so it isolates "more high- than low-frequency power"
DB_PER_TOKEN = TOKEN_STD * 20 / math.log(10)   # one token unit = TOKEN_STD natural-log amplitude units = 13.57 dB
REMOTE_S = 120          # s; an FA second counts as "remote" if no annotated seizure second lies within +-120 s
NEAR_S = 60             # s; FA seconds within 60 s of a seizure second are reported as "near seizure"
SPREAD_Z = 1.0          # electrode spread threshold: band value > background mean + 1 background SD (same recording)
WIN_ALIGN = 120         # s, half-width of the event-aligned window
MIN_DUR_ALIGN = 60      # s, events used for the mean onset/offset curves
BOOT_SEED = 0
N_BOOT_EVENTS = 1000    # resamples of events (within a seizure type) for the CI of the by-type logit correlation
COL = {"catboost": "#eb6834", "linoss_mean": "#2a78d6", "fused_mean": "#1baf7a", "neither": "#8a8985"}
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False})


# ----------------------------------------------------------------------------- helpers
def boot_weights(n_rec, n_boot, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_rec, np.full(n_rec, 1 / n_rec), size=n_boot).astype(np.float64)


def rec_sums(vals, rec_of, n_rec):
    """vals (m,) or (m, F), rec_of (m,) -> per-recording sums (n_rec, F) and counts (n_rec,)."""
    v = vals if vals.ndim == 2 else vals[:, None]
    S = np.zeros((n_rec, v.shape[1])); np.add.at(S, rec_of, v)
    C = np.bincount(rec_of, minlength=n_rec).astype(np.float64)
    return S, C


def boot_diff(a, ra, b, rb, n_rec, W, names=None):
    """Mean(a) - mean(b) with a percentile 95 % CI over recordings (W = bootstrap multinomial weights)."""
    Sa, Ca = rec_sums(a, ra, n_rec); Sb, Cb = rec_sums(b, rb, n_rec)
    pt = Sa.sum(0) / max(Ca.sum(), 1) - Sb.sum(0) / max(Cb.sum(), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        bs = (W @ Sa) / (W @ Ca)[:, None] - (W @ Sb) / (W @ Cb)[:, None]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5], axis=0)
    out = [{"diff": float(p), "ci95": [float(l), float(h)]} for p, l, h in zip(pt, lo, hi)]
    return dict(zip(names, out)) if names else out[0]


def boot_mean(a, ra, n_rec, W):
    S, C = rec_sums(a, ra, n_rec)
    with np.errstate(invalid="ignore", divide="ignore"):
        bs = (W @ S) / (W @ C)[:, None]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5], axis=0)
    return [{"mean": float(m), "ci95": [float(l), float(h)]} for m, l, h in zip(S.sum(0) / max(C.sum(), 1), lo, hi)]


def q(x):
    x = np.asarray([v for v in x if v is not None], float)
    if len(x) == 0:
        return None
    return {"n": int(len(x)), "median": float(np.median(x)), "q25": float(np.quantile(x, .25)),
            "q75": float(np.quantile(x, .75)), "mean": float(x.mean())}


def runs_mask(st, en, n):
    d = np.zeros(n + 1, np.int64); np.add.at(d, st, 1); np.add.at(d, en, -1)
    return np.cumsum(d)[:-1] > 0


def pred(D, m, pt):
    ev = D["sel"][m][pt]
    ps = P.smooth(D["T"][m].astype(np.float64), D["rec"], ev["k"])
    st, en = P.events(ps, D["rec"], ev["thr"], ev["gap"], ev["min_dur"])
    cy = np.r_[0, np.cumsum(D["y"])]
    fa = (cy[en] - cy[st]) == 0
    return st, en, fa, runs_mask(st, en, len(ps))


# ----------------------------------------------------------------------------- per-event table
def event_table(D):
    tab = BT.true_event_table(D)
    ts, te = D["te"]
    for t, s, e in zip(tab, ts, te):
        r = int(D["rec"][s]); i0 = D["blocks"][r][0]; info = D["index"][r]
        a, b = int(s - i0), int(e - i0)
        valid = [TCP22.index(c) for c in info["valid_channels"]]
        yc = np.asarray(D["y_ch"][info["offset"] + a:info["offset"] + b])[:, valid]
        t["n_channels"] = int(yc.max(0).sum()); t["n_valid"] = len(valid)
        t["channel_share"] = float(yc.max(0).mean())
        t["rec"] = r
    return tab


# ----------------------------------------------------------------------------- 1. event Venn
def event_venn(D, tab):
    out = {}
    for pt in POINTS:
        det, lat = {}, {}
        for m in MODELS:
            d_, l_, _ = BT.detection(D, m, pt)
            det[m], lat[m] = d_, l_
        c, l, f = det["catboost"], det["linoss_mean"], det["fused_mean"]
        groups = {"both": c & l, "catboost_only": c & ~l, "linoss_only": ~c & l, "neither": ~c & ~l}
        res = {"operating_points": {m: D["sel"][m][pt] for m in MODELS}, "n_events": len(tab),
               "detected": {m: int(det[m].sum()) for m in MODELS}}
        for g, msk in groups.items():
            idx = np.flatnonzero(msk)
            ev = [tab[i] for i in idx]
            res[g] = {
                "n_events": int(len(idx)), "fusion_detects": int(f[idx].sum()),
                "fusion_detects_share": float(f[idx].mean()) if len(idx) else None,
                "type_counts": dict(Counter(t["type"] for t in ev).most_common()),
                "group_counts": dict(Counter(t["group"] for t in ev).most_common()),
                "duration_s": q([t["dur"] for t in ev]),
                "n_channels_labelled": q([t["n_channels"] for t in ev]),
                "channel_share": q([t["channel_share"] for t in ev]),
                "events_on_all_valid_channels": int(sum(t["channel_share"] == 1.0 for t in ev)),
                "latency_s": {m: q([lat[m][i] for i in idx]) for m in MODELS}}
        a_, b_ = np.flatnonzero(groups["catboost_only"]), np.flatnonzero(groups["linoss_only"])
        res["catboost_only_vs_linoss_only_mannwhitney_p"] = {
            k: float(mannwhitneyu([tab[i][k] for i in a_], [tab[i][k] for i in b_]).pvalue) if len(a_) and len(b_) else None
            for k in ("dur", "n_channels", "channel_share")}
        res["fusion_detects_outside_union"] = int((f & ~c & ~l).sum())
        res["fusion_misses_inside_union"] = int((~f & (c | l)).sum())
        out[pt] = res
    return out


# ----------------------------------------------------------------------------- token features
def token_features(D, X):
    """Per scored second: relative band dB (electrode mean) (n, B), relative spectrum dB (n, 100), electrode spread
    per band (n, B). Background statistics per recording from its y = 0 scored seconds."""
    n = len(D["y"]); B = len(BANDS)
    band_rel = np.zeros((n, B), np.float32); spec_rel = np.zeros((n, 100), np.float32)
    spread = np.zeros((n, B), np.int8); band_abs = np.zeros((n, B), np.float32)
    bidx = list(BANDS.values())
    for r, (i0, i1) in D["blocks"].items():
        o = D["index"][r]["offset"]
        x = np.asarray(X[o:o + (i1 - i0)], np.float32) * DB_PER_TOKEN          # (m, 19, 100) dB of amplitude
        be = np.stack([x[:, :, b].mean(-1) for b in bidx], -1)                 # (m, 19, B)
        bg = D["y"][i0:i1] == 0
        if bg.sum() < 2:
            bg = np.ones_like(bg)
        mu, sd = be[bg].mean(0), be[bg].std(0) + 1e-6                           # (19, B)
        bm = be.mean(1)                                                          # (m, B)
        band_abs[i0:i1] = bm
        band_rel[i0:i1] = bm - bm[bg].mean(0)
        sp = x.mean(1)
        spec_rel[i0:i1] = sp - sp[bg].mean(0)
        spread[i0:i1] = ((be - mu) / sd > SPREAD_Z).sum(1)
    names = list(BANDS)
    hf = band_rel[:, [names.index(b) for b in HF_LF[0]]].mean(1); lf = band_rel[:, [names.index(b) for b in HF_LF[1]]].mean(1)
    band_rel = np.c_[band_rel, hf - lf].astype(np.float32)                      # last column: hf_minus_lf
    return band_rel, spec_rel, spread, band_abs


def dist_to_seizure(D):
    """Per scored second: distance (s) to the nearest annotated seizure second of the same recording (inf if none)."""
    n = len(D["y"]); out = np.full(n, np.inf)
    for r, (i0, i1) in D["blocks"].items():
        yy = np.flatnonzero(D["y"][i0:i1])
        if len(yy) == 0:
            continue
        t = np.arange(i1 - i0)
        k = np.searchsorted(yy, t)
        left = np.where(k > 0, t - yy[np.maximum(k - 1, 0)], np.inf)
        right = np.where(k < len(yy), yy[np.minimum(k, len(yy) - 1)] - t, np.inf)
        d = np.minimum(left, right)
        out[i0:i1] = d
    return out


# ----------------------------------------------------------------------------- 2. false alarms
def fa_analysis(D, F, dist, W):
    band_rel, spec_rel, spread, _ = F
    bs_ = list(BANDS); bn = bs_ + ["hf_minus_lf"]; n_rec = len(D["index"]); rec = D["rec"].astype(np.int64); y = D["y"]
    bg = y == 0
    out = {"bands_hz_bins": {k: [v[0], v[-1]] for k, v in BANDS.items()},
           "hf_minus_lf": f"mean of {HF_LF[0]} minus mean of {HF_LF[1]} (relative dB)",
           "unit": f"dB of amplitude relative to the recording's background mean (1 token unit = {DB_PER_TOKEN:.3f} dB)"}
    # reference rows: ictal and all background seconds
    ict = np.flatnonzero(y == 1); bgi = np.flatnonzero(bg)
    out["reference_ictal_rel_db"] = dict(zip(bn, boot_mean(band_rel[ict], rec[ict], n_rec, W)))
    out["reference_ictal_spread"] = dict(zip(bs_, boot_mean(spread[ict].astype(float), rec[ict], n_rec, W)))
    out["reference_background_spread"] = dict(zip(bs_, [float(v) for v in spread[bgi].mean(0)]))

    # ---- second level at f1_thr
    p = {m: D["T"][m].astype(np.float64) for m in MODELS}
    on = {m: p[m] > D["sel"][m]["f1_thr"] for m in MODELS}
    fc, fl = bg & on["catboost"], bg & on["linoss_mean"]
    grp = {"catboost_only": fc & ~fl, "linoss_only": fl & ~fc, "both": fc & fl}
    sec = {"thresholds": {m: D["sel"][m]["f1_thr"] for m in MODELS},
           "n_fa_seconds": {"catboost": int(fc.sum()), "linoss_mean": int(fl.sum()), "fused_mean": int((bg & on["fused_mean"]).sum())}}
    rows = {}
    for g, msk in grp.items():
        i = np.flatnonzero(msk); rr = np.flatnonzero(msk & (dist >= REMOTE_S)); rows[g] = rr
        sec[g] = {"n_seconds": int(len(i)), "n_recordings": int(len(np.unique(rec[i]))),
                  "fusion_keeps": int(on["fused_mean"][i].sum()),
                  "fusion_keeps_share": float(on["fused_mean"][i].mean()) if len(i) else None,
                  "near_seizure_le_60s": int((dist[i] <= NEAR_S).sum()),
                  "near_seizure_share": float((dist[i] <= NEAR_S).mean()) if len(i) else None,
                  "remote_ge_120s": int(len(rr)), "remote_recordings": int(len(np.unique(rec[rr]))),
                  "remote_rel_db": dict(zip(bn, boot_mean(band_rel[rr], rec[rr], n_rec, W))) if len(rr) else None,
                  "remote_spread": dict(zip(bs_, boot_mean(spread[rr].astype(float), rec[rr], n_rec, W))) if len(rr) else None}
    a, b = rows["linoss_only"], rows["catboost_only"]
    sec["effect_linoss_only_minus_catboost_only_remote"] = {
        "rel_db": boot_diff(band_rel[a], rec[a], band_rel[b], rec[b], n_rec, W, bn),
        "spread": boot_diff(spread[a].astype(float), rec[a], spread[b].astype(float), rec[b], n_rec, W, bs_)}
    # each group vs ictal (are FA seconds seizure-like?)
    for g in grp:
        r_ = rows[g]
        if len(r_):
            sec[g]["remote_minus_ictal_rel_db"] = boot_diff(band_rel[r_], rec[r_], band_rel[ict], rec[ict], n_rec, W, bn)
    out["seconds_at_f1_thr"] = sec
    spectra = {g: spec_rel[rows[g]].mean(0) if len(rows[g]) else None for g in grp}
    spectra["ictal"] = spec_rel[ict].mean(0)

    # ---- event level at ev / ev2
    ev_out = {}
    for pt in POINTS:
        Pm = {m: pred(D, m, pt) for m in MODELS}
        famask = {m: runs_mask(Pm[m][0][Pm[m][2]], Pm[m][1][Pm[m][2]], len(y)) for m in MODELS}
        res = {}
        for m, other in (("catboost", "linoss_mean"), ("linoss_mean", "catboost")):
            st, en, fa, _ = Pm[m]
            cmo = np.r_[0, np.cumsum(famask[other])]; cmf = np.r_[0, np.cumsum(Pm["fused_mean"][3])]
            cmp_ = np.r_[0, np.cumsum(Pm[other][3])]
            s_, e_ = st[fa], en[fa]
            shared = (cmo[e_] - cmo[s_]) > 0
            kept = (cmf[e_] - cmf[s_]) > 0
            ov_any = (cmp_[e_] - cmp_[s_]) > 0
            dmin = np.array([dist[a_:b_].min() for a_, b_ in zip(s_, e_)]) if len(s_) else np.array([])
            feats = np.array([band_rel[a_:b_].mean(0) for a_, b_ in zip(s_, e_)]) if len(s_) else np.zeros((0, len(bn)))
            spr = np.array([spread[a_:b_].mean(0) for a_, b_ in zip(s_, e_)]) if len(s_) else np.zeros((0, len(bs_)))
            r_ = rec[s_]
            d = {"n_fa_events": int(len(s_)), "fa_seconds": int((e_ - s_).sum()),
                 "n_recordings": int(len(np.unique(r_))),
                 "shared_with_other_fa": int(shared.sum()), "overlapping_any_other_predicted_event": int(ov_any.sum()),
                 "fusion_keeps": int(kept.sum()), "fusion_keeps_shared": int((kept & shared).sum()),
                 "fusion_keeps_unshared": int((kept & ~shared).sum()),
                 "duration_s": q(e_ - s_), "unshared_duration_s": q((e_ - s_)[~shared]),
                 "min_distance_to_seizure_s": {"n_in_recordings_with_seizure": int(np.isfinite(dmin).sum()),
                                               "n_within_120s": int((dmin <= REMOTE_S).sum())}}
            us = ~shared
            if us.sum():
                d["unshared_rel_db"] = dict(zip(bn, boot_mean(feats[us], r_[us], n_rec, W)))
                d["unshared_spread"] = dict(zip(bs_, boot_mean(spr[us], r_[us], n_rec, W)))
            res[m] = d; res[m + "__arrays"] = (feats[us], spr[us], r_[us])
        a_f, a_s, a_r = res.pop("linoss_mean__arrays"); b_f, b_s, b_r = res.pop("catboost__arrays")
        if len(a_f) and len(b_f):
            res["effect_linoss_only_minus_catboost_only_events"] = {
                "rel_db": boot_diff(a_f, a_r, b_f, b_r, n_rec, W, bn),
                "spread": boot_diff(a_s, a_r, b_s, b_r, n_rec, W, bs_)}
        # union clusters: connected components of CatBoost FA mask OR LinOSS FA mask
        u = famask["catboost"] | famask["linoss_mean"]
        us_, ue_ = P._runs(u, D["rec"])
        cc = np.r_[0, np.cumsum(famask["catboost"])]; cl = np.r_[0, np.cumsum(famask["linoss_mean"])]
        hc, hl = (cc[ue_] - cc[us_]) > 0, (cl[ue_] - cl[us_]) > 0
        res["union_clusters"] = {"catboost_only": int((hc & ~hl).sum()), "linoss_only": int((~hc & hl).sum()),
                                 "both": int((hc & hl).sum())}
        sc = {"catboost_only": famask["catboost"] & ~famask["linoss_mean"],
              "linoss_only": famask["linoss_mean"] & ~famask["catboost"],
              "both": famask["catboost"] & famask["linoss_mean"]}
        res["fa_seconds_venn"] = {k: {"n_seconds": int(v.sum()), "fusion_predicted": int((v & Pm["fused_mean"][3]).sum())}
                                  for k, v in sc.items()}
        res["fusion_fa_events"] = int(Pm["fused_mean"][2].sum())
        ev_out[pt] = res
    out["events"] = ev_out
    return out, spectra


# ----------------------------------------------------------------------------- 3. temporal behaviour
def temporal(D, tab, W):
    y = D["y"]; rec = D["rec"].astype(np.int64); n_rec = len(D["index"]); lags = np.arange(-WIN_ALIGN, WIN_ALIGN + 1)
    ts, te = D["te"]
    in_ev = np.full(len(y), -1, np.int64)
    for k, (s, e) in enumerate(zip(ts, te)):
        in_ev[s:e] = k
    curves, per_ev = {}, {}
    sel = [k for k, t in enumerate(tab) if t["dur"] >= MIN_DUR_ALIGN]
    for m in MODELS:
        p = D["T"][m].astype(np.float64)
        for kind in ("onset", "offset"):
            M = np.full((len(sel), len(lags)), np.nan)
            for j, k in enumerate(sel):
                s, e = ts[k], te[k]; r = int(rec[s]); i0, i1 = D["blocks"][r]
                ref = s if kind == "onset" else e - 1
                t = ref + lags; ok = (t >= i0) & (t < i1)
                tt = t[ok]; good = (in_ev[tt] == k) | (in_ev[tt] == -1)
                vals = np.full(len(lags), np.nan); vv = p[tt].copy(); vv[~good] = np.nan; vals[ok] = vv
                M[j] = vals
            curves[(m, kind)] = M
        # per-event time to threshold
        thr = D["sel"][m]["f1_thr"]; on = p > thr
        st, en = P._runs(on, D["rec"])
        on_t, off_t = [], []
        for s, e in zip(ts, te):
            ov = np.flatnonzero((st < e) & (en > s))
            if len(ov) == 0:
                on_t.append(None); off_t.append(None); continue
            on_t.append(int(max(st[ov[0]] - s, -WIN_ALIGN))); off_t.append(int(min(en[ov[-1]] - e, WIN_ALIGN)))
        per_ev[m] = (on_t, off_t)

    def halfmax(c, kind):
        if kind == "onset":
            base = np.nanmean(c[(lags >= -120) & (lags <= -90)]); plat = np.nanmean(c[(lags >= 20) & (lags <= 40)])
            def first(level):
                return int(lags[np.flatnonzero(c >= level)[0]]) if np.any(c >= level) else None
            lv = lambda f: base + f * (plat - base)
            t50, t10, t90 = first(lv(.5)), first(lv(.1)), first(lv(.9))
        else:
            base = np.nanmean(c[(lags >= 90) & (lags <= 120)]); plat = np.nanmean(c[(lags >= -40) & (lags <= -20)])
            def last(level):
                return int(lags[np.flatnonzero(c >= level)[-1]]) if np.any(c >= level) else None
            lv = lambda f: base + f * (plat - base)
            t50, t10, t90 = last(lv(.5)), last(lv(.1)), last(lv(.9))
        w = None if t10 is None or t90 is None else abs(t90 - t10)
        return {"baseline_p": float(base), "plateau_p": float(plat), "t_half_s": t50, "t10_s": t10, "t90_s": t90,
                "width_10_90_s": w}

    mean_curves, out = {}, {"n_events_curves": len(sel), "min_duration_s": MIN_DUR_ALIGN, "half_max": {},
                             "time_to_threshold": {}, "paired_linoss_minus_catboost": {}}
    for (m, kind), M in curves.items():
        c = np.nanmean(M, 0); mean_curves[(m, kind)] = c
        out["half_max"].setdefault(m, {})[kind] = halfmax(c, kind)
        out["half_max"][m][kind]["p_at_minus60_plus60"] = [float(c[lags == -60][0]), float(c[lags == 60][0])]
    for m in MODELS:
        on_t, off_t = per_ev[m]
        out["time_to_threshold"][m] = {"threshold": D["sel"][m]["f1_thr"], "events_reached": int(sum(v is not None for v in on_t)),
                                       "onset_s": q(on_t), "offset_s": q(off_t),
                                       "onset_le_minus30_s": int(sum(v is not None and v <= -30 for v in on_t)),
                                       "offset_ge_plus30_s": int(sum(v is not None and v >= 30 for v in off_t))}
    ev_rec = rec[ts]
    for kind, j in (("onset", 0), ("offset", 1)):
        a, b = per_ev["linoss_mean"][j], per_ev["catboost"][j]
        idx = [i for i in range(len(a)) if a[i] is not None and b[i] is not None]
        dv = np.array([a[i] - b[i] for i in idx], float)
        S, C = rec_sums(dv, ev_rec[idx], n_rec)
        bs = (W @ S)[:, 0] / (W @ C)
        out["paired_linoss_minus_catboost"][kind] = {
            "n_events": len(idx), "mean_diff_s": float(dv.mean()), "median_diff_s": float(np.median(dv)),
            "mean_ci95": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))],
            "share_linoss_earlier": float((dv < 0).mean()) if kind == "onset" else None,
            "share_linoss_ends_earlier": float((dv < 0).mean()) if kind == "offset" else None}
    return out, mean_curves, lags, per_ev


# ----------------------------------------------------------------------------- 4. disagreement map
def disagreement(D):
    y = D["y"]; zc = logit(D["T"]["catboost"].astype(np.float64)); zl = logit(D["T"]["linoss_mean"].astype(np.float64))
    th = {m: D["sel"][m]["f1_thr"] for m in MODELS}
    c = D["T"]["catboost"] > th["catboost"]; l = D["T"]["linoss_mean"] > th["linoss_mean"]
    f = D["T"]["fused_mean"] > th["fused_mean"]
    st = Stacker.from_dict(D["sel"]["fused_mean"]["stacker"])
    chk = st.predict(zc, zl, D["rec"])
    max_dev = float(np.max(np.abs(chk - D["T"]["fused_mean"].astype(np.float64))))
    out = {"stacker": D["sel"]["fused_mean"]["stacker"]["cand"], "stacker_recomputed_max_abs_diff": max_dev}
    for name, msk in (("ictal", y == 1), ("background", y == 0)):
        n = int(msk.sum()); d = {"n_seconds": n}
        for qn, qm in (("cb_pos_lin_pos", c & l), ("cb_pos_lin_neg", c & ~l), ("cb_neg_lin_pos", ~c & l),
                       ("cb_neg_lin_neg", ~c & ~l)):
            mm = msk & qm
            d[qn] = {"n_seconds": int(mm.sum()), "share": float(mm.sum() / n),
                     "fused_positive": int((mm & f).sum()),
                     "fused_positive_share": float((mm & f).sum() / mm.sum()) if mm.sum() else None}
        out[name] = d
    s15, s61 = P.smooth(zl, D["rec"], 15), P.smooth(zl, D["rec"], 61)
    bg = y == 0
    out["context_medians_background"] = {"S15_zL": float(np.median(s15[bg])), "S61_zL": float(np.median(s61[bg]))}
    out["context_medians_ictal"] = {"S15_zL": float(np.median(s15[y == 1])), "S61_zL": float(np.median(s61[y == 1]))}
    return out, zc, zl, st


def contour_grid(st, ctx, g):
    """Fused p on a (z_C, z_L) grid; ctx = None -> context features equal z_L, else (S15, S61) constants."""
    ZC, ZL = np.meshgrid(g, g)
    zc, zl = ZC.ravel(), ZL.ravel()
    s15 = zl if ctx is None else np.full_like(zl, ctx[0]); s61 = zl if ctx is None else np.full_like(zl, ctx[1])
    X = (np.c_[zc, zl, zc * zl, s15, s61] - np.array(st.p["mean"])) / np.array(st.p["scale"])
    return ZC, ZL, (1 / (1 + np.exp(-(X @ np.array(st.p["coef"]) + st.p["intercept"])))).reshape(ZC.shape)


# ----------------------------------------------------------------------------- 5. correlation
def correlations(D, tab):
    y = D["y"]; zc = logit(D["T"]["catboost"].astype(np.float64)); zl = logit(D["T"]["linoss_mean"].astype(np.float64))
    ts, te = D["te"]
    rows_by = {}
    for t, s, e in zip(tab, ts, te):
        rows_by.setdefault(t["type"], []).append(np.arange(s, e))
        rows_by.setdefault("group/" + t["group"], []).append(np.arange(s, e))

    def rho(i):
        if len(i) < 10:
            return {"n_seconds": int(len(i))}
        return {"n_seconds": int(len(i)), "pearson": float(np.corrcoef(zc[i], zl[i])[0, 1]),
                "spearman": float(spearmanr(zc[i], zl[i]).statistic)}
    out = {"all_ictal": rho(np.flatnonzero(y == 1)), "all_background": rho(np.flatnonzero(y == 0)), "by_type": {}}
    rng = np.random.default_rng(BOOT_SEED)
    for k, v in sorted(rows_by.items()):
        i = np.concatenate(v); out["by_type"][k] = {"n_events": len(v), **rho(i)}
        if len(v) >= 5 and len(i) >= 10:
            bs = []
            for _ in range(N_BOOT_EVENTS):
                j = np.concatenate([v[t] for t in rng.integers(0, len(v), len(v))])
                bs.append(np.corrcoef(zc[j], zl[j])[0, 1])
            out["by_type"][k]["pearson_ci95_events"] = [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]
    pat = D["patient"]
    per = {"background": [], "ictal": []}
    for p_ in np.unique(pat):
        m = pat == p_
        b = np.flatnonzero(m & (y == 0)); i = np.flatnonzero(m & (y == 1))
        if len(b) >= 600:
            per["background"].append(np.corrcoef(zc[b], zl[b])[0, 1])
        if len(i) >= 60:
            per["ictal"].append(np.corrcoef(zc[i], zl[i])[0, 1])
    out["per_patient_pearson"] = {
        "rule": "background: patients with >= 600 background seconds; ictal: patients with >= 60 ictal seconds; "
                "distribution only, no patient IDs",
        **{k: {**q(v), "min": float(np.min(v)), "max": float(np.max(v)), "n_negative": int(np.sum(np.array(v) < 0))}
           for k, v in per.items()}}
    return out


# ----------------------------------------------------------------------------- figures
def fig_venn(ven, fa, path):
    fig, axs = plt.subplots(2, 2, figsize=(12, 7))
    gname = {"both": "both", "catboost_only": "CatBoost only", "linoss_only": "LinOSS only", "neither": "neither"}
    gcol = {"both": "#4a3aa7", "catboost_only": COL["catboost"], "linoss_only": COL["linoss_mean"], "neither": COL["neither"]}
    for j, pt in enumerate(POINTS):
        ax = axs[0, j]; v = ven[pt]
        ks = list(gname)
        for i, k in enumerate(ks):
            n, fd = v[k]["n_events"], v[k]["fusion_detects"]
            ax.barh(i, n, color=gcol[k], alpha=.35, height=.6)
            ax.barh(i, fd, color=gcol[k], height=.6)
            ax.text(n + 3, i, f"{n} events, fusion detects {fd}", va="center", fontsize=8.5)
        ax.set_yticks(range(len(ks)), [gname[k] for k in ks]); ax.invert_yaxis()
        ax.set_xlim(0, max(v[k]["n_events"] for k in ks) * 1.75)
        ax.set_xlabel("annotated events (eval)")
        ax.set_title(f"Annotated events by branch detection, event point {j + 1}\n(dark = also detected by the fusion)",
                     fontsize=9.5, loc="left")
        ax = axs[1, j]; e = fa["events"][pt]
        rows = [("CatBoost FA events", e["catboost"]), ("LinOSS FA events", e["linoss_mean"])]
        for i, (lab, d) in enumerate(rows):
            n, sh, kp = d["n_fa_events"], d["shared_with_other_fa"], d["fusion_keeps"]
            c = COL["catboost"] if i == 0 else COL["linoss_mean"]
            ax.barh(i, n, color=c, alpha=.35, height=.6)
            ax.barh(i, sh, color="#4a3aa7", height=.3)
            ax.barh(i + .3, kp, color=c, height=.15)
            ax.text(n + .5, i, f"{n} FA events:\n{sh} shared, fusion keeps {kp}", va="center", fontsize=8.5)
        ax.set_yticks(range(2), [r[0] for r in rows]); ax.invert_yaxis()
        ax.set_xlim(0, max(r[1]["n_fa_events"] for r in rows) * 1.7)
        ax.set_xlabel("false-alarm events (eval)")
        u = e["union_clusters"]
        ax.set_title(f"False-alarm events, event point {j + 1}; violet = shared with the other\n"
                     f"branch, thin bar = kept by the fusion (fusion: {e['fusion_fa_events']} FA events)",
                     fontsize=9.5, loc="left")
        ax.text(.98, .04, f"union clusters: CatBoost only {u['catboost_only']},\nLinOSS only {u['linoss_only']}, "
                f"both {u['both']}", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#444")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_spectra(spectra, fa, path):
    fig, axs = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axs[0]; f = np.arange(100)
    sty = {"catboost_only": (COL["catboost"], "CatBoost-only FA s"), "linoss_only": (COL["linoss_mean"], "LinOSS-only FA s"),
           "both": ("#4a3aa7", "shared FA s"), "ictal": ("#222222", "ictal s")}
    sec = fa["seconds_at_f1_thr"]
    for k, (c, lab) in sty.items():
        if spectra.get(k) is None:
            continue
        n = sec[k]["remote_ge_120s"] if k != "ictal" else None
        ax.plot(f[1:], spectra[k][1:], color=c, lw=1.6 if k != "ictal" else 1.2,
                ls="-" if k != "ictal" else "--", label=lab + (f" (n={n})" if n is not None else ""))
    ax.axhline(0, color="#999", lw=.8)
    ax.axvspan(35, 99, color="#eee", zorder=-1)
    ax.text(37, 6.2, "above the CatBoost 35 Hz low-pass", fontsize=8, color="#555")
    ax.set_xlabel("frequency (Hz, 1 Hz token bins)"); ax.set_ylabel("dB vs recording background mean")
    ax.set_title("Electrode-mean spectrum of remote false-alarm seconds (>= 120 s from\n"
                 "any seizure), each branch at its dev-F1 threshold; ictal seconds for reference", fontsize=9.5, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.13), ncol=2)
    ax = axs[1]
    sets = [("remote FA seconds, dev-F1 thresholds", sec["effect_linoss_only_minus_catboost_only_remote"]["rel_db"], "#333333")]
    for j, pt in enumerate(POINTS):
        e = fa["events"][pt].get("effect_linoss_only_minus_catboost_only_events")
        if e:
            sets.append((f"unshared FA events, event point {j + 1}", e["rel_db"], ["#8a8985", "#4a3aa7"][j]))
    names = list(sets[0][1]); yy = np.arange(len(names))
    for k_, (lab, eff, c) in enumerate(sets):
        off = (k_ - (len(sets) - 1) / 2) * .22
        d = np.array([eff[k]["diff"] for k in names]); lo = np.array([eff[k]["ci95"][0] for k in names])
        hi = np.array([eff[k]["ci95"][1] for k in names])
        ax.errorbar(d, yy + off, xerr=[d - lo, hi - d], fmt="o", color=c, ms=4, capsize=2, label=lab)
    ax.axvline(0, color="#999", lw=.8)
    ax.set_yticks(yy, [k.replace("_", " ") for k in names])
    ax.set_ylim(len(names) + .9, -.6)
    ax.legend(frameon=False, fontsize=7.5, loc="lower center", ncol=1)
    ax.set_xlabel("LinOSS-only minus CatBoost-only (dB, 95 % CI over recordings)")
    ax.set_title("Band power difference of false alarms\n(> 0: more power in LinOSS-only false alarms)", fontsize=9.5,
                 loc="left")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_onset(mean_curves, lags, per_ev, tmp, path):
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1, 1, .8]})
    for ax, kind in zip(axs[:2], ("onset", "offset")):
        for m, lab in MODELS.items():
            ax.plot(lags, mean_curves[(m, kind)], color=COL[m], lw=1.8, label=lab)
            h = tmp["half_max"][m][kind]["t_half_s"]
            if h is not None:
                ax.axvline(h, color=COL[m], lw=.8, ls=":")
        ax.axvline(0, color="#333", lw=.8)
        ax.set_xlabel(f"seconds from annotated {kind}")
        ax.set_ylabel("mean per-second probability")
        ax.set_title(f"Aligned at seizure {kind} ({tmp['n_events_curves']} events >= {MIN_DUR_ALIGN} s);\n"
                     "dotted = half-maximum time of each mean curve", fontsize=9.5, loc="left")
        ax.legend(frameon=False, fontsize=8, loc="upper right" if kind == "onset" else "upper left",
                  bbox_to_anchor=(1, .9) if kind == "onset" else (0, .9))
    ax = axs[2]
    data, labs, cols = [], [], []
    for kind, j in (("onset", 0), ("offset", 1)):
        for m in ("catboost", "linoss_mean", "fused_mean"):
            v = [x for x in per_ev[m][j] if x is not None]
            data.append(v); labs.append(f"{MODELS[m]}\n{kind}\nn={len(v)}"); cols.append(COL[m])
    bp = ax.boxplot(data, patch_artist=True, widths=.6, showfliers=False, medianprops=dict(color="#111"))
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_alpha(.5)
    ax.set_xticks(range(1, len(labs) + 1), labs, fontsize=7.5)
    ax.axhline(0, color="#333", lw=.8)
    ax.set_ylabel("s (first crossing - onset | last crossing - end)")
    ax.set_title("Per event: raw track > its dev-F1 threshold\n(truncated at +-120 s)", fontsize=9.5, loc="left")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_density(D, zc, zl, st, dis, path):
    th = {m: D["sel"][m]["f1_thr"] for m in MODELS}
    g = np.linspace(-11.6, 11.6, 117)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.8), sharex=True, sharey=True, constrained_layout=True)
    ZC, ZL, Pst = contour_grid(st, None, g)
    cm = dis["context_medians_background"]
    _, _, Pmed = contour_grid(st, (cm["S15_zL"], cm["S61_zL"]), g)
    for ax, (name, msk) in zip(axs, (("ictal", D["y"] == 1), ("background", D["y"] == 0))):
        H, xe, ye = np.histogram2d(zc[msk], zl[msk], bins=[g, g])
        H = H / H.sum()
        im = ax.pcolormesh(xe, ye, np.log10(np.where(H > 0, H, np.nan)).T, cmap="Greys", vmin=-6.5, vmax=-1,
                           shading="flat")
        ax.contour(ZC, ZL, Pst, levels=[th["fused_mean"]], colors=[COL["fused_mean"]], linewidths=2)
        ax.contour(ZC, ZL, Pmed, levels=[th["fused_mean"]], colors=[COL["fused_mean"]], linewidths=1.5,
                   linestyles="--")
        ax.axvline(logit(np.array([th["catboost"]]))[0], color=COL["catboost"], lw=1.2)
        ax.axhline(logit(np.array([th["linoss_mean"]]))[0], color=COL["linoss_mean"], lw=1.2)
        d = dis[name]
        for qn, (x, y_) in {"cb_pos_lin_pos": (.97, .97), "cb_pos_lin_neg": (.97, .03), "cb_neg_lin_pos": (.03, .97),
                            "cb_neg_lin_neg": (.03, .03)}.items():
            dd = d[qn]
            fp = "-" if dd["fused_positive_share"] is None else f"{dd['fused_positive_share'] * 100:.0f} %"
            txt = f"{dd['share'] * 100:.1f} % of s\nfused + {fp}"
            ax.text(x, y_, txt, transform=ax.transAxes, ha="right" if x > .5 else "left",
                    va="top" if y_ > .5 else "bottom", fontsize=8,
                    bbox=dict(boxstyle="round,pad=.25", fc="white", ec="#ccc", alpha=.9))
        ax.set_title(f"{name} seconds (n = {d['n_seconds']:,})", fontsize=10, loc="left")
        ax.set_xlabel("CatBoost logit z_C")
        ax.set_xlim(-6, 5); ax.set_ylim(-10, 7)
    axs[0].set_ylabel("LinOSS logit z_L (5-seed mean)")
    cb = fig.colorbar(im, ax=axs, shrink=.85, pad=.01); cb.set_label("log10 share of seconds per cell")
    fig.supxlabel("Orange / blue lines: dev-F1 thresholds of CatBoost / LinOSS. Green: lr_ctx stacker at the fused dev-F1 "
                  "threshold (solid: context S15, S61 = z_L;\ndashed: context at the background medians). Corner boxes: "
                  "share of seconds per quadrant and the share of them the fusion marks positive.", fontsize=8)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


# ----------------------------------------------------------------------------- main
def main(argv=None):
    work = paths.get("work_dir")
    ap = argparse.ArgumentParser()
    ap.add_argument("--fusion_1s", default=None)
    ap.add_argument("--tokens", default=str(work / "tokens" / "tusz"))
    ap.add_argument("--out", default="results/tusz/complementarity.json")
    ap.add_argument("--figs", default="docs/figures")
    ap.add_argument("--n_boot", type=int, default=2000)
    a = ap.parse_args(argv)
    fdir = Path(a.fusion_1s) if a.fusion_1s else BT.default_fusion_dir()
    src = BT.source_label(fdir)
    D = BT.load(fdir, a.tokens)
    n_rec = len(D["index"])
    # bootstrap over the scored recordings only (weights indexed by store position; unscored positions get 0)
    scored = np.array(sorted(D["blocks"]))
    W = np.zeros((a.n_boot, n_rec)); W[:, scored] = boot_weights(len(scored), a.n_boot)
    tab = event_table(D)
    print("event Venn"); ven = event_venn(D, tab)
    X = np.load(Path(a.tokens) / "eval" / "tokens.npy", mmap_mode="r")
    print("token features"); F = token_features(D, X)
    dist = dist_to_seizure(D)
    print("false alarms"); fa, spectra = fa_analysis(D, F, dist, W)
    print("temporal"); tmp, mean_curves, lags, per_ev = temporal(D, tab, W)
    print("disagreement"); dis, zc, zl, st = disagreement(D)
    print("correlation"); cor = correlations(D, tab)
    figs = Path(a.figs); figs.mkdir(parents=True, exist_ok=True)
    fig_venn(ven, fa, figs / "complementarity_venn.png")
    fig_spectra(spectra, fa, figs / "complementarity_fa_spectra.png")
    fig_onset(mean_curves, lags, per_ev, tmp, figs / "complementarity_onset.png")
    fig_density(D, zc, zl, st, dis, figs / "complementarity_density.png")
    out = {"provenance": {**provenance(BT.command("complementarity", src, a.out, f" --figs {a.figs} --n_boot {a.n_boot}")),
                          "source": src, "n_boot": a.n_boot, "boot_seed": BOOT_SEED,
                          "bootstrap_unit": "recordings (scored eval recordings, resampled with replacement)"},
           "definitions": {"branches": MODELS, "doc": __doc__.split("\n\n", 1)[1],
                           "constants": {"REMOTE_S": REMOTE_S, "NEAR_S": NEAR_S, "SPREAD_Z": SPREAD_Z,
                                         "WIN_ALIGN": WIN_ALIGN, "MIN_DUR_ALIGN": MIN_DUR_ALIGN,
                                         "DB_PER_TOKEN": DB_PER_TOKEN}},
           "data": {"scored_seconds": int(len(D["y"])), "ictal_seconds": int(D["y"].sum()),
                    "scored_recordings": len(D["blocks"]), "true_events": len(tab)},
           "hypothesis_H1": {"statement": "HYPOTHESIS (tested, not assumed): LinOSS reads unfiltered 0-99 Hz referential "
                                          "spectra, CatBoost reads 0.5-35 Hz filtered bipolar features, so LinOSS-only "
                                          "false alarms carry more high-frequency (>= 30 Hz) power and CatBoost-only "
                                          "false alarms more low-frequency (< 8 Hz) power.",
                             "test": "fa.seconds_at_f1_thr.effect_linoss_only_minus_catboost_only_remote and "
                                     "fa.events.<point>.effect_linoss_only_minus_catboost_only_events"},
           "event_venn": ven, "false_alarms": fa, "temporal": tmp, "disagreement": dis, "correlation": cor}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print("written", a.out)


if __name__ == "__main__":
    main()
