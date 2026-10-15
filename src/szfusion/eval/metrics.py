"""Window-level and SzCORE-style any-overlap event-level metrics."""
from __future__ import annotations

import numpy as np


def _nan_if_single_class(fn, y, s):
    try:
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(fn(y, s))
    except ValueError:
        return float("nan")


def window_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    from sklearn.metrics import roc_auc_score, average_precision_score, cohen_kappa_score
    y = np.asarray(y_true).astype(int).ravel()
    s = np.asarray(y_score, dtype=np.float64).ravel()
    p = (s >= threshold).astype(int)
    tp = int(((p == 1) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    fp = int(((p == 1) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    kappa = float(cohen_kappa_score(y, p)) if len(np.unique(np.r_[y, p])) > 1 else float("nan")
    return dict(auroc=_nan_if_single_class(roc_auc_score, y, s),
                auprc=_nan_if_single_class(average_precision_score, y, s),
                acc=float((p == y).mean()) if len(y) else float("nan"),
                sens=sens, spec=spec, prec=prec, f1=f1, kappa=kappa, threshold=float(threshold),
                n=int(len(y)), n_pos=int(y.sum()), tp=tp, fp=fp, tn=tn, fn=fn)


def best_threshold(y_true, y_score, rule: str = "f1") -> float:
    """Choose a threshold (ON DEV ONLY). rule: 'f1' | 'youden' | 'sens>=0.95' (any 'sens>=x')."""
    from sklearn.metrics import precision_recall_curve, roc_curve
    y = np.asarray(y_true).astype(int).ravel()
    s = np.asarray(y_score, dtype=np.float64).ravel()
    if len(np.unique(y)) < 2:
        return 0.5
    if rule == "f1":
        pr, rc, th = precision_recall_curve(y, s)
        f1 = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-12, None)
        return float(th[int(np.nanargmax(f1))])
    fpr, tpr, th = roc_curve(y, s)
    th = np.clip(th, -np.inf, np.max(s))  # sklearn puts +inf/max+1 first
    if rule == "youden":
        return float(th[int(np.argmax(tpr - fpr))])
    if rule.startswith("sens>="):
        target = float(rule.split(">=")[1])
        ok = np.where(tpr >= target)[0]
        return float(th[ok[0]]) if len(ok) else float(np.min(s))
    raise ValueError(rule)


def _intervals_from_mask(t0, pos, win_sec, merge_gap, min_dur):
    ev = []
    for a, on in zip(t0, pos):
        if not on:
            continue
        b = a + win_sec
        if ev and a <= ev[-1][1] + merge_gap:
            ev[-1][1] = max(ev[-1][1], b)
        else:
            ev.append([a, b])
    return [(a, b) for a, b in ev if (b - a) >= min_dur]


def _overlaps(a, b, intervals):
    return any(min(b, d) > max(a, c) for c, d in intervals)


def predicted_events(meta, y_score, threshold, win_sec, smooth_k=1, min_dur=0.0, merge_gap=0.0):
    """-> {rec_id: [(start, stop), ...]} from thresholded (optionally moving-average smoothed) scores.
    Each positive window contributes [t0, t0+win_sec]; overlapping/adjacent spans (gap <= merge_gap)
    merge; events shorter than min_dur are dropped."""
    import pandas as pd
    df = pd.DataFrame({"rec_id": np.asarray(meta["rec_id"]), "t0": np.asarray(meta["t0"], float),
                       "s": np.asarray(y_score, float).ravel()})
    out = {}
    for rid, g in df.groupby("rec_id", sort=False):
        g = g.sort_values("t0")
        s = g["s"].to_numpy()
        if smooth_k and smooth_k > 1:
            s = np.convolve(s, np.ones(smooth_k) / smooth_k, mode="same")
        out[rid] = _intervals_from_mask(g["t0"].to_numpy(), s >= threshold, win_sec, merge_gap, min_dur)
    return out


def event_metrics(meta, y_score, threshold, win_sec, stride_sec, true_intervals_by_rec,
                  smooth_k: int = 1, min_dur: float = 0.0, merge_gap: float = 0.0) -> dict:
    """SzCORE-style any-overlap scoring over the recordings present in `meta`
    (needs columns rec_id, t0, rec_duration). A true event is detected if any predicted event
    overlaps it; a predicted event overlapping no true event is a false alarm.
    FA/24h = FP / scored_hours * 24, scored_hours = sum of rec_duration of scored recordings.
    NOTE: no pre/post tolerance and no long-event splitting (pure any-overlap)."""
    import pandas as pd
    meta = meta if isinstance(meta, pd.DataFrame) else pd.DataFrame(meta)
    pred = predicted_events(meta, y_score, threshold, win_sec, smooth_k, min_dur, merge_gap)
    durs = meta.groupby("rec_id", sort=False)["rec_duration"].first()
    n_true = n_det = n_pred = n_fp = 0
    for rid in durs.index:
        tru = list(true_intervals_by_rec.get(rid, []))
        pe = pred.get(rid, [])
        n_true += len(tru)
        n_det += sum(_overlaps(a, b, pe) for a, b in tru)
        n_pred += len(pe)
        n_fp += sum(not _overlaps(a, b, tru) for a, b in pe)
    hours = float(durs.sum()) / 3600.0
    sens = n_det / n_true if n_true else float("nan")
    prec = (n_pred - n_fp) / n_pred if n_pred else float("nan")
    f1 = 2 * sens * prec / (sens + prec) if (n_true and n_pred and (sens + prec) > 0) else float("nan")
    return dict(event_sens=sens, event_prec=prec, event_f1=f1,
                fa_per_24h=(n_fp / hours * 24.0) if hours > 0 else float("nan"),
                n_true_events=int(n_true), n_detected=int(n_det), n_pred_events=int(n_pred),
                n_false_alarms=int(n_fp), scored_hours=hours, threshold=float(threshold),
                smooth_k=int(smooth_k), min_dur=float(min_dur), merge_gap=float(merge_gap),
                win_sec=float(win_sec), stride_sec=float(stride_sec))
