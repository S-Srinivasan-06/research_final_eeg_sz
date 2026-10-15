"""The 60 s window test on TUSZ: the windows, labels, post-processing and event scorer of Zabihi et al. (2026).

Windows: 60 s long, every 15 s, exactly the CatBoost branch's windows (catboost/extract.py), restricted to the
recordings in the token store (so every model sees the same windows).
Label: 1 if the window overlaps a seizure by at least min(9 s, 20 % of that seizure's duration).
Event pipeline (catboost/postprocess.py, a port of the authors' infer.py):
  1. Moving-average smoothing of the window scores.
  2. A threshold, then dilation of the positive windows.
  3. Merging of gaps, then removal of events shorter than 10 s.
  4. A true seizure counts as detected if a predicted event overlaps it by at least 10 s.
  5. A predicted event that overlaps no seizure is one false alarm.
Dev tuning rule (the paper's): maximise event sensitivity subject to window specificity >= 0.93 and false alarms
<= 0.69 per hour. The grid is smoothing {15, 30, 45} s × dilation {0, 20, 30} s × merge gap {20, 40, 60} s ×
15 thresholds at dev score quantiles 0.80-0.995.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from szfusion.catboost import postprocess
from szfusion.eval.metrics import window_metrics

WIN, STRIDE = 60.0, 15.0


def logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def windows(split, tokens_root, catboost_scores, feat_dir):
    """-> start row of each window in the token store, label, patient, CatBoost window logit, meta, true intervals."""
    import pandas as pd
    z = np.load(catboost_scores)
    index = {e["rec_id"]: e for e in json.loads((Path(tokens_root) / split / "index.json").read_text())
             if e["status"] == "done"}
    keep = np.array([r in index for r in z["rec_id"]])
    meta = pd.DataFrame({k: z[k][keep] for k in ("rec_id", "patient", "t0", "rec_duration", "y")})
    start = np.array([index[r]["offset"] + min(int(round(t)), index[r]["n_sec"] - 60)
                      for r, t in zip(meta["rec_id"], meta["t0"])], np.int64)
    with open(Path(feat_dir) / split / "intervals.json") as fh:
        iv = {k: [tuple(v) for v in vs] for k, vs in json.load(fh).items()}
    return (start, meta["y"].to_numpy().astype(np.int64), meta["patient"].to_numpy(),
            logit(z["score"][keep].astype(np.float64)), meta, iv)


def tune(meta, score, iv, extra_thr=()):
    """Dev grid -> (chosen parameters, all rows), by the paper's rule."""
    y = meta["y"].to_numpy(); neg = y == 0
    thrs = sorted(set(np.quantile(score, np.linspace(0.80, 0.995, 15)).tolist()) | set(extra_thr))
    best, rows = None, []
    for th, sm, dl, gap in itertools.product(thrs, [15.0, 30.0, 45.0], [0.0, 20.0, 30.0], [20.0, 40.0, 60.0]):
        p = dict(threshold=float(th), smooth_sec=sm, dilate_sec=dl, merge_gap_sec=gap, min_pred_event_sec=10.0,
                 min_event_overlap_sec=10.0)
        _, mask, ev = postprocess.run(meta, score, p, WIN)
        r = postprocess.score_events(meta, ev, iv, 10.0)
        spec = float((~mask[neg]).mean())
        ok = spec >= 0.93 and r["fa_per_h"] <= 0.69
        rows.append({**p, **r, "window_spec": spec, "feasible": ok})
        key = (ok, r["event_sens"], r["time_recall"], r["event_ppv"])
        if best is None or key > best[0]:
            best = (key, p)
    return best[1], rows


def score(meta, s, params, iv):
    """Frozen parameters -> window AUROC/AUPRC/F1 and the paper's event metrics."""
    ev, _, mask = postprocess.apply(meta, s, params, WIN, STRIDE, iv)
    wm = window_metrics(meta["y"].to_numpy(), s, params["threshold"])
    return {"auroc": wm["auroc"], "auprc": wm["auprc"], "f1_at_thr": wm["f1"], **ev["inhouse"],
            "fa_per_24h_any_overlap": ev["any_overlap_common"]["fa_per_24h"]}
