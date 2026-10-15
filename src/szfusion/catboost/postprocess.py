"""Event post-processing + in-house scorer, ported from [GH infer.py:361-564 postprocess_and_event_report].

Per file (sorted by start): stride = median positive start diff (fallback 15 s) [:361-368];
k = round(SMOOTH_SEC/stride), forced odd, centred moving average np.convolve(mode='same') [:370-381, :521-522];
mask = smoothed >= thr [:524]; dilation radius = ceil(DILATE_SEC/stride), kernel 2r+1 [:384-396, :525-526];
intervals [start, end] of positive windows, merged if gap <= MERGE_GAP_SEC [:398-422];
drop events < MIN_PRED_EVENT_SEC [:530-534]; a GT event is detected if overlap >= MIN_EVENT_OVERLAP_SEC
[:556-559]; a predicted event overlapping no GT event is a false alarm [:561-564]. FA reported per hour.
Repo defaults (event_params.json): SMOOTH 30, DILATE 20, MERGE_GAP 40, MIN_PRED 10, MIN_OVERLAP 10, thr 0.44.
`common.metrics.event_metrics` (pure any-overlap, FA/24h) is ALSO reported on the same mask for comparability.
"""
from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd

from szfusion.eval.metrics import event_metrics

DEFAULTS = dict(threshold=0.44, smooth_sec=30.0, dilate_sec=20.0, merge_gap_sec=40.0, min_pred_event_sec=10.0,
                min_event_overlap_sec=10.0)


def _stride(t0):
    d = np.diff(np.sort(t0)); d = d[d > 0]
    return float(np.median(d)) if len(d) else 15.0


def _movavg_same(p, k):
    k = max(1, int(k))
    if k % 2 == 0:
        k += 1
    if k <= 1:
        return p.astype(float)
    # 'full' + centred slice == mode="same", but keeps len(p) when the file is shorter than k
    # (np.convolve(..., "same") returns max(len(p), k) values)
    full = np.convolve(p, np.ones(k) / k, mode="full")
    s = (k - 1) // 2
    return full[s:s + len(p)]


def _dilate(mask, r):
    if r <= 0:
        return mask
    L = min(2 * r + 1, len(mask))
    if L % 2 == 0:
        L -= 1
    if L <= 1:
        return mask
    return np.convolve(mask.astype(float), np.ones(L), mode="same") > 0


def _mask_to_intervals(t0, mask, win_sec, gap):
    iv = []
    for a, on in zip(t0, mask):
        if not on:
            continue
        b = a + win_sec
        if iv and a - iv[-1][1] <= gap:
            iv[-1][1] = max(iv[-1][1], b)
        else:
            iv.append([a, b])
    return [(a, b) for a, b in iv]


def file_pipeline(t0, p, params, win_sec):
    """-> (smoothed probs, final window mask, predicted intervals) for one file."""
    o = np.argsort(t0); t0, p = np.asarray(t0)[o], np.asarray(p, float)[o]
    st = _stride(t0)
    sm = _movavg_same(p, round(params["smooth_sec"] / st))
    mask = _dilate(sm >= params["threshold"], math.ceil(params["dilate_sec"] / st))
    iv = [(a, b) for a, b in _mask_to_intervals(t0, mask, win_sec, params["merge_gap_sec"])
          if b - a >= params["min_pred_event_sec"]]
    return o, sm, mask, iv


def run(meta, scores, params, win_sec):
    """meta: DataFrame(rec_id, t0, ...) aligned with scores -> (smoothed [N], mask [N], {rec_id: intervals})."""
    meta = meta.reset_index(drop=True)
    sm_all = np.zeros(len(meta)); mask_all = np.zeros(len(meta), bool); events = {}
    for rid, g in meta.groupby("rec_id", sort=False):
        idx = g.index.to_numpy()
        o, sm, mask, iv = file_pipeline(g["t0"].to_numpy(float), np.asarray(scores)[idx], params, win_sec)
        sm_all[idx[o]] = sm; mask_all[idx[o]] = mask; events[rid] = iv
    return sm_all, mask_all, events


def _ov(a, b, c, d):
    return max(0.0, min(b, d) - max(a, c))


def score_events(meta, events, intervals, min_overlap):
    """In-house scorer [GH infer.py:556-564] -> event sensitivity, FA/h, PPV, time-based recall/PPV."""
    durs = meta.groupby("rec_id", sort=False)["rec_duration"].first()
    n_true = n_det = n_pred = n_fp = 0
    t_true = t_pred = t_hit = 0.0
    for rid in durs.index:
        tru, pe = list(intervals.get(rid, [])), events.get(rid, [])
        n_true += len(tru); n_pred += len(pe)
        n_det += sum(any(_ov(a, b, c, d) >= min_overlap for c, d in pe) for a, b in tru)
        n_fp += sum(not any(_ov(a, b, c, d) > 0 for c, d in tru) for a, b in pe)
        t_true += sum(b - a for a, b in tru); t_pred += sum(b - a for a, b in pe)
        t_hit += sum(_ov(a, b, c, d) for a, b in tru for c, d in pe)
    hours = float(durs.sum()) / 3600
    return dict(event_sens=n_det / n_true if n_true else float("nan"),
                fa_per_h=n_fp / hours if hours else float("nan"),
                event_ppv=(n_pred - n_fp) / n_pred if n_pred else float("nan"),
                time_recall=t_hit / t_true if t_true else float("nan"),
                time_ppv=t_hit / t_pred if t_pred else float("nan"),
                n_true=n_true, n_detected=n_det, n_pred=n_pred, n_false_alarms=n_fp, hours=hours)


def apply(meta, scores, params, win_sec, stride_sec, intervals):
    """Frozen params -> {'inhouse': ..., 'any_overlap_common': ...}."""
    meta = meta.reset_index(drop=True)
    sm, mask, ev = run(meta, scores, params, win_sec)
    res = {"inhouse": score_events(meta, ev, intervals, params["min_event_overlap_sec"])}
    # comparable any-overlap scorer on the same final mask (merge gap applied identically, no min_dur beyond)
    res["any_overlap_common"] = event_metrics(meta, mask.astype(float), 0.5, win_sec, stride_sec, intervals,
                                              1, params["min_pred_event_sec"], params["merge_gap_sec"])
    return res, sm, mask


def tune(meta, scores, y, win_sec, stride_sec, intervals, pcfg):
    """Dev grid [M]: s.t. window spec >= spec_min and FA/h <= fa_h_max, maximise event sens; tie-break
    time recall, then event PPV. Returns (best params, dev metrics, grid rows)."""
    meta = meta.reset_index(drop=True)
    g = pcfg["grid"]; y = np.asarray(y)
    best, rows = None, []
    for th, sm_s, dl, gap, mo, md in itertools.product(g["threshold"], g["smooth_sec"], g["dilate_sec"],
                                                       g["merge_gap_sec"], g["min_event_overlap_sec"],
                                                       g["min_pred_event_sec"]):
        p = dict(threshold=float(th), smooth_sec=float(sm_s), dilate_sec=float(dl), merge_gap_sec=float(gap),
                 min_event_overlap_sec=float(mo), min_pred_event_sec=float(md))
        _, mask, ev = run(meta, scores, p, win_sec)
        m = score_events(meta, ev, intervals, mo)
        neg = y == 0
        spec = float((~mask[neg]).mean()) if neg.any() else float("nan")
        ok = spec >= pcfg.get("spec_min", 0.93) and m["fa_per_h"] <= pcfg.get("fa_h_max", 0.69)
        key = (ok, np.nan_to_num(m["event_sens"], nan=-1), np.nan_to_num(m["time_recall"], nan=-1),
               np.nan_to_num(m["event_ppv"], nan=-1))
        rows.append({**p, **m, "window_spec": spec, "feasible": ok})
        if best is None or key > best[0]:
            best = (key, p, {**m, "window_spec": spec, "feasible": ok})
    return best[1], best[2], rows
