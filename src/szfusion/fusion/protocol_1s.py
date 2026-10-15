"""The common evaluation protocol for CatBoost, OscSeq (LinOSS) and their fusion: per-second tracks.

Unit: one second [s, s+1) of a recording. Recordings: those in BOTH the token store (the authors' channel rule: dev
1,666 of 2,203, eval all 880) and CatBoost's window table (recordings shorter than 60 s have no CatBoost window).
Label: y_sec of the token store = 1 if the second overlaps a seizure interval of the recording's .csv_bi (the
same term-level annotation CatBoost was trained from; CatBoost's own 60 s window label rule is NOT used here).
CatBoost per-second probability: mean of the ensemble scores of all 60 s windows (stride 15 s) covering that
second (1-4 windows). Window t0 + 60 s <= signal length, so CatBoost covers a prefix [0, c) of each recording; the
tail (< 15 s) is uncovered. OscSeq per-second probability: its own track (every second). Scored seconds = seconds
covered by both, identical for every model (a prefix of each recording, so events stay contiguous).

Metrics on scored seconds:
  AUROC, AUPRC (threshold-free, raw probabilities);
  F1 / recall / precision per second at the threshold maximising dev per-second F1 (raw probabilities);
  events: centred moving average over k s -> threshold -> runs -> merge gaps <= g s -> drop events < d s;
  a true event (run of y = 1) is detected if any predicted second overlaps it (any-overlap, no tolerance, stricter
  than SzCORE); a predicted event overlapping no true event is one false alarm; FA/24h = FA / scored hours x 24.
  (k, threshold, g, d) are chosen on dev only, per model, by one rule: maximise dev event sensitivity subject to
  dev FA/24h <= 16.6 (= the CatBoost paper's 0.69 FA/h dev constraint); ties -> fewer false alarms. A secondary
  operating point uses the same rule with a 33.1 FA/24h budget. Grid: k in {1,5,15,31,61}, g in {0,10,30,60},
  d in {1,10}, 40 thresholds at dev quantiles 0.90-0.9995 of the smoothed track.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score



FA_BUDGET_24H = 0.69 * 24            # primary operating point (paper's dev constraint, 16.6 FA/24h)
FA_BUDGET2_24H = 2 * 0.69 * 24       # secondary operating point (33.1 FA/24h), same rule
GRID = {"k": [1, 5, 15, 31, 61], "gap": [0, 10, 30, 60], "min_dur": [1, 10]}
N_THR = 40


def catboost_track(scores_npz, index, total_sec):
    """Per-second CatBoost probability aligned to the token store rows (NaN = not covered)."""
    z = np.load(scores_npz)
    acc = np.zeros(total_sec); cnt = np.zeros(total_sec)
    order = np.argsort(z["rec_id"], kind="stable")
    rid, t0, sc = z["rec_id"][order], z["t0"][order], z["score"][order].astype(np.float64)
    b = np.flatnonzero(np.r_[True, rid[1:] != rid[:-1], True])
    pos = {rid[i]: (i, j) for i, j in zip(b[:-1], b[1:])}
    n_rec = 0
    for e in index:
        if e["rec_id"] not in pos:
            continue
        i, j = pos[e["rec_id"]]; n_rec += 1
        o, n = e["offset"], e["n_sec"]
        for t, s in zip(t0[i:j], sc[i:j]):
            a = int(round(t)); c = min(a + 60, n)
            acc[o + a:o + c] += s; cnt[o + a:o + c] += 1
    out = np.full(total_sec, np.nan); ok = cnt > 0
    out[ok] = acc[ok] / cnt[ok]
    return out, n_rec


def load_split(split, tokens_root, catboost_scores, linoss_dirs):
    """-> dict with y, rec (recording number per row), patient per row, scored mask, tracks {'catboost', names...}."""
    d = Path(tokens_root) / split
    index = [e for e in json.loads((d / "index.json").read_text()) if e["status"] == "done"]
    y = np.load(d / "y_sec.npy").astype(np.int8)
    total = len(y)
    rec = np.full(total, -1, np.int32); pat = np.empty(total, object)
    for ri, e in enumerate(index):
        rec[e["offset"]:e["offset"] + e["n_sec"]] = ri; pat[e["offset"]:e["offset"] + e["n_sec"]] = e["patient"]
    cb, n_cb_rec = catboost_track(catboost_scores, index, total)
    tracks = {"catboost": cb}
    for name, path in linoss_dirs.items():
        tracks[name] = np.load(path / f"prob_{split}.npy").astype(np.float64)
    scored = rec >= 0
    for t in tracks.values():
        scored &= ~np.isnan(t)
    idx = np.flatnonzero(scored)
    info = {"split": split, "store_recordings": len(index), "recordings_with_catboost_windows": n_cb_rec,
            "scored_recordings": int(len(np.unique(rec[idx]))), "scored_hours": round(len(idx) / 3600, 2),
            "store_hours": round(total / 3600, 2), "seizure_seconds_scored": int(y[idx].sum()),
            "seizure_seconds_store": int(y.sum())}
    D = {"y": y[idx], "rec": rec[idx], "patient": pat[idx], "info": info,
         "tracks": {k: v[idx] for k, v in tracks.items()}}
    D["info"]["true_events_scored"] = len(true_events(D)[0])
    return D


# ----------------------------------------------------------------------------- events (vectorised)
def _runs(mask, rec):
    """Start/end (exclusive) of runs of True that do not cross recordings."""
    m = mask.astype(np.int8)
    brk = np.r_[True, rec[1:] != rec[:-1]]
    st = np.flatnonzero(m & (brk | np.r_[True, m[:-1] == 0]))
    en_mask = m & np.r_[rec[1:] != rec[:-1], True] | (m & np.r_[m[1:] == 0, True])
    en = np.flatnonzero(en_mask) + 1
    return st, en


def true_events(D):
    return _runs(D["y"] == 1, D["rec"])


def smooth(p, rec, k):
    """Centred moving average over k seconds within each recording (edges: mean over available seconds)."""
    if k <= 1:
        return p
    h = k // 2
    out = np.empty_like(p)
    b = np.flatnonzero(np.r_[True, rec[1:] != rec[:-1], True])
    ker = np.ones(k)
    for a, c in zip(b[:-1], b[1:]):
        x = p[a:c]
        num = np.convolve(x, ker, "full")[h:h + len(x)]
        den = np.convolve(np.ones(len(x)), ker, "full")[h:h + len(x)]
        out[a:c] = num / den
    return out


def events(ps, rec, thr, gap, min_dur):
    st, en = _runs(ps > thr, rec)
    if len(st) == 0:
        return st, en
    new = np.r_[True, (st[1:] - en[:-1] > gap) | (rec[st[1:]] != rec[en[:-1] - 1])]
    g = np.cumsum(new) - 1
    s2 = st[new]; e2 = np.zeros(len(s2), np.int64); np.maximum.at(e2, g, en)
    keep = (e2 - s2) >= max(min_dur, 1)
    return s2[keep], e2[keep]


def event_scores(D, ps, thr, gap, min_dur, te=None):
    ts, tend = te if te is not None else true_events(D)
    ps_, pe_ = events(ps, D["rec"], thr, gap, min_dur)
    cy = np.r_[0, np.cumsum(D["y"])]
    fa = int(((cy[pe_] - cy[ps_]) == 0).sum())
    diff = np.zeros(len(ps) + 1, np.int64); np.add.at(diff, ps_, 1); np.add.at(diff, pe_, -1)
    pm = np.cumsum(diff)[:-1] > 0
    cm = np.r_[0, np.cumsum(pm)]
    det = int(((cm[tend] - cm[ts]) > 0).sum())
    hours = len(ps) / 3600
    tp = int((pm & (D["y"] == 1)).sum()); fp = int((pm & (D["y"] == 0)).sum()); fn = int((~pm & (D["y"] == 1)).sum())
    return {"event_sens": det / max(len(ts), 1), "fa_per_24h": fa / hours * 24, "n_true": int(len(ts)),
            "n_detected": det, "n_pred": int(len(ps_)), "n_fa": fa,
            "event_prec": (len(ps_) - fa) / max(len(ps_), 1),
            "time_recall": tp / max(tp + fn, 1), "time_ppv": tp / max(tp + fp, 1)}


def tune_events(D, p, budget=FA_BUDGET_24H, rows=None):
    """Dev grid -> (best params, best dev scores, all rows). Rule: max event sens s.t. FA/24h <= budget,
    ties -> fewer false alarms. Pass rows from an earlier call to re-select for another budget without recomputing."""
    if rows is None:
        te = true_events(D)
        rows = []
        for k, gap, md in itertools.product(GRID["k"], GRID["gap"], GRID["min_dur"]):
            ps = smooth(p, D["rec"], k)
            for thr in np.unique(np.quantile(ps, np.linspace(0.90, 0.9995, N_THR))):
                r = event_scores(D, ps, float(thr), gap, md, te)
                rows.append({"k": k, "thr": float(thr), "gap": gap, "min_dur": md, **r})
    best = max(rows, key=lambda r: (r["fa_per_24h"] <= budget, r["event_sens"], -r["fa_per_24h"]))
    return {k: best[k] for k in ("k", "thr", "gap", "min_dur")}, best, rows


def best_f1_threshold(y, p):
    qs = np.unique(np.quantile(p, np.linspace(0.5, 0.9995, 400)))
    order = np.argsort(-p); ps, ys = p[order], y[order]
    ctp = np.cumsum(ys); P = ys.sum()
    best = (-1, 0.5)
    for q in qs:
        n = np.searchsorted(-ps, -q, side="left")        # number of seconds with p > q
        tp = ctp[n - 1] if n > 0 else 0
        f1 = 2 * tp / max(n + P, 1)
        if f1 > best[0]:
            best = (f1, float(q))
    return best[1], best[0]


def sec_scores(y, p, thr):
    pr = p > thr
    tp = int((pr & (y == 1)).sum()); fp = int((pr & (y == 0)).sum()); fn = int((~pr & (y == 1)).sum())
    return {"auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1), "recall": tp / max(tp + fn, 1), "precision": tp / max(tp + fp, 1),
            "f1_thr": thr}


def full_scores(D, p, sel):
    """All common-protocol metrics for one probability track with dev-frozen selections sel = {f1_thr, ev}."""
    r = sec_scores(D["y"], p, sel["f1_thr"])
    for tag in ("ev", "ev2"):
        if tag in sel:
            ev = sel[tag]
            r.update({f"{tag}_{k}": v for k, v in event_scores(D, smooth(p, D["rec"], ev["k"]), ev["thr"], ev["gap"],
                                                              ev["min_dur"]).items()})
    return r
