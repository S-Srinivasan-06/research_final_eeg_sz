"""Why the 12 s and 60 s tests score lower than the 1 s test: measured causes, on the frozen TUSZ eval outputs.

  python -m szfusion.eval.resolution [--out results/tusz/resolution.json]

Inputs: the same folders as eval/ablation.py (official run when complete, otherwise the numerically identical
S1-gate copy; the source is written per resolution), the eval and dev token stores (index.json, y_sec.npy), the
CatBoost committee scores (<work>/tusz/catboost/scores_{dev,eval}.npz), the 5 LinOSS runs (prob_dev.npy), the
CatBoost feature folder (intervals.json, 60 s windows) and the HyBrain 12 s marker lists (12 s clips).

Nothing here changes a reported result. Every threshold applied to eval is a dev-frozen one; the only new threshold
(section b, F1 of the pooled 1 s track) is chosen on dev. Everything else on eval is DESCRIPTIVE (labelled so in the
JSON): it decomposes the frozen scores, it does not select anything.

Units and labels (the three tests):
  1 s   every scored second; y = 1 if the second overlaps a .csv_bi seizure interval (any overlap).
  12 s  the HyBrain clips [12k, 12k + 12) of their marker lists; y = the authors' label (any seizure second).
  60 s  the CatBoost windows (60 s, stride 15 s); y = 1 if the overlap with a seizure is >= min(9 s, 20 % of it).
Ictal content of a unit = number of its seconds with y_sec = 1 in the token store (the 1 s label), so the three
resolutions are compared on one ground truth.

(a) label definition: positive units split into fully ictal (all seconds ictal) and boundary (some seconds ictal);
    negative units into pure background (no ictal second) and sub-threshold (60 s only: ictal seconds below the
    overlap rule). At 1 s the analogue of a boundary unit is an edge second (first or last second of an annotated
    event, the only seconds whose any-overlap label can cover a partly ictal second). Per group: count, recall (or
    false-positive rate) at the frozen threshold, median probability, and AUROC against pure background.
(b) head vs unit/label: the fused per-second track of the 1 s test (fused_mean) is pooled over each 12 s clip / 60 s
    window (log-mean-exp of logits, as clips12.lme, and the mean probability) and scored with the unit labels. Then
        unit/label effect = pooled-track metric − per-second metric of the same track,
        head effect       = trained 12 s / 60 s fusion head (look_mean_cb) − pooled-track metric,
    both on the units whose seconds are all scored at 1 s (the 1 s test covers a prefix of each recording). F1 of the
    pooled track uses a threshold chosen on dev (best dev F1 of the same pooling of the dev fused track; the dev
    track is the refit stacker of selection.json applied to dev, i.e. in-sample for the stacker).
(c) counts and prevalence per resolution.
(d) other causes: seizures shorter than the unit; recall by the duration of the longest seizure inside the unit;
    share of positive units that come from long seizures; peri-ictal background (pure-background units within NEAR_S
    of a seizure, groups neg_pure_near / neg_pure_far in section a); 60 s window overlap (each second is in up to 4
    windows).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from szfusion import paths
from szfusion.eval.ablation import config_sha256, sources
from szfusion.eval.mechanism import provenance
from szfusion.fusion import protocol_1s as P
from szfusion.fusion.stacker import Stacker, logit as slogit
from szfusion.units import clips12, windows60

MODELS = {"fusion": ("fused_mean", "look_mean_cb"), "catboost": ("catboost", "catboost"),
          "linoss": ("linoss_mean", "look_mean")}          # name -> (1 s track, 12 s / 60 s score key)
DUR_BINS = [(0, 12), (12, 60), (60, 180), (180, np.inf)]  # s, seizure-duration bins (half-open)
NEAR_S = 60                    # s; a pure-background unit is "near" a seizure if one of its seconds is <= 60 s from an
                               # annotated seizure second of the same recording (peri-ictal background)
sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731


def rd(p):
    return json.loads(Path(p).read_text())


def auc(y, s):
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None


def ap(y, s):
    y = np.asarray(y)
    return float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else None


def runs(y):
    d = np.diff(np.r_[0, np.asarray(y, np.int8), 0])
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


# ----------------------------------------------------------------------------- store-level arrays
def store_arrays(tokens, split):
    td = Path(tokens) / split
    index = [e for e in json.loads((td / "index.json").read_text()) if e["status"] == "done"]
    y = np.asarray(np.load(td / "y_sec.npy", mmap_mode="r")).astype(np.int8)
    evdur = np.zeros(len(y), np.int64)        # per store second: length (s) of the annotated seizure it belongs to
    edge = np.zeros(len(y), bool)             # first or last second of a seizure
    dist = np.full(len(y), np.iinfo(np.int64).max // 4, np.int64)   # s to the nearest seizure second (same recording)
    n_ev, durs = 0, []
    for e in index:
        o, n = e["offset"], e["n_sec"]
        ic = np.flatnonzero(y[o:o + n])
        if len(ic):
            t = np.arange(n); j = np.clip(np.searchsorted(ic, t), 0, len(ic) - 1)
            dist[o:o + n] = np.minimum(np.abs(ic[j] - t), np.abs(ic[np.maximum(j - 1, 0)] - t))
        for a, b in zip(*runs(y[o:o + n])):
            evdur[o + a:o + b] = b - a; edge[o + a] = edge[o + b - 1] = True
            durs.append(int(b - a)); n_ev += 1
    return index, y, evdur, edge, np.array(durs), dist


def to_store(rec, vals, index, total):
    """Scored-second track -> store-length array (NaN where not scored). Scored seconds are a prefix per recording."""
    out = np.full(total, np.nan)
    b = np.flatnonzero(np.r_[True, rec[1:] != rec[:-1], True])
    rows = np.empty(len(rec), np.int64)
    for i, j in zip(b[:-1], b[1:]):
        e = index[int(rec[i])]
        assert j - i <= e["n_sec"]
        rows[i:j] = e["offset"] + np.arange(j - i)
    out[rows] = vals
    return out, rows


# ----------------------------------------------------------------------------- (a) groups
def unit_groups(c, y, U, dmin):
    """c = ictal seconds per unit (1 s label), y = unit label, dmin = min distance (s) of the unit to a seizure."""
    return {"neg_pure_near": (y == 0) & (c == 0) & (dmin <= NEAR_S), "neg_pure_far": (y == 0) & (c == 0) & (dmin > NEAR_S),
            "pos_full": (y == 1) & (c == U), "pos_boundary": (y == 1) & (c > 0) & (c < U),
            "pos_no_ictal_second": (y == 1) & (c == 0), "neg_pure": (y == 0) & (c == 0),
            "neg_subthreshold": (y == 0) & (c > 0)}


def group_stats(groups, y, p, thr, ge):
    above = p >= thr if ge else p > thr
    pure = groups["neg_pure"]
    out = {}
    for g, m in groups.items():
        n = int(m.sum())
        r = {"n": n}
        if n:
            r["frac_above_thr"] = float(above[m].mean())       # recall for positives, FP rate for negatives
            r["median_prob"] = float(np.median(p[m]))
            if g.startswith("pos") or g in ("neg_subthreshold", "neg_pure_near"):
                yy = np.r_[np.ones(n), np.zeros(int(pure.sum()))]
                r["auroc_vs_pure_background"] = auc(yy, np.r_[p[m], p[pure]])
        out[g] = r
    clean = groups["pos_full"] | pure
    out["clean_units_only"] = {"n": int(clean.sum()), "auroc": auc(y[clean], p[clean]), "auprc": ap(y[clean], p[clean]),
                               "note": "fully ictal positives vs pure background only (boundary and sub-threshold "
                                       "units removed); DESCRIPTIVE"}
    no_sub = ~groups["neg_subthreshold"]
    out["without_subthreshold_negatives"] = {"n": int(no_sub.sum()), "auroc": auc(y[no_sub], p[no_sub]),
                                             "auprc": ap(y[no_sub], p[no_sub])}
    return out


# ----------------------------------------------------------------------------- (b) pooling
def pool(track, start, U):
    W = track[start[:, None] + np.arange(U)]
    lme = clips12.lme(_logit_nan(W))
    ok = ~np.isnan(W)
    mean = np.where(ok.any(1), np.where(ok, W, 0).sum(1) / np.maximum(ok.sum(1), 1), np.nan)
    full = ~np.isnan(W).any(1)
    return lme, mean, full


def _logit_nan(W):
    out = np.full(W.shape, np.nan)
    ok = ~np.isnan(W)
    out[ok] = slogit(W[ok])
    return out


def fill(z):
    return np.where(np.isnan(z), np.nanmin(z), z)


def thr_metrics(y, p, thr, ge=False):
    pr = p >= thr if ge else p > thr
    tp = int((pr & (y == 1)).sum()); fp = int((pr & (y == 0)).sum())
    fn = int((~pr & (y == 1)).sum()); tn = int((~pr & (y == 0)).sum())
    return {"f1": 2 * tp / max(2 * tp + fp + fn, 1), "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "specificity": tn / max(tn + fp, 1), "thr": float(thr)}


def dev_fused_track(tokens, catboost_dir, runs_dir, sel_1s):
    names = sel_1s["linoss_runs"]
    dirs = {f"linoss_s{i}": Path(runs_dir) / n for i, n in enumerate(names)}
    D = P.load_split("dev", tokens, Path(catboost_dir) / "scores_dev.npz", dirs)
    lin = np.mean([D["tracks"][n] for n in dirs], 0)
    st = Stacker.from_dict(sel_1s["configs"]["fused_mean"]["stacker"])
    fused = st.predict(slogit(D["tracks"]["catboost"]), slogit(lin), D["rec"])
    return D, {"fused_mean": fused, "catboost": D["tracks"]["catboost"], "linoss_mean": lin}


# ----------------------------------------------------------------------------- main analysis
def analyse(a):
    S = sources()
    work = paths.get("work_dir")
    f1s, u12, u60 = S["1s"][0], S["12s"][0], S["60s"][0]
    sel1 = rd(f1s / "selection.json"); sel12 = rd(u12 / "selection.json"); sel60 = rd(u60 / "selection.json")
    T = np.load(f1s / "eval_tracks.npz")
    Z12, Z60 = np.load(u12 / "eval_scores.npz"), np.load(u60 / "eval_scores.npz")
    index, ys, evdur, edge, durs, dist = store_arrays(a.tokens, "eval")
    rec = T["rec"].astype(np.int64)
    tr = {}
    for n in ("fused_mean", "catboost", "linoss_mean"):
        tr[n], rows = to_store(rec, T[n].astype(np.float64), index, len(ys))
    assert np.array_equal(ys[rows], T["y"].astype(np.int8)), "1 s track labels differ from the store"
    y1 = T["y"].astype(np.int8)
    R = {"sources": {k: v[1] for k, v in S.items()}}

    # ---------------- (c) counts and prevalence
    st12, y12 = Z12["start"].astype(np.int64), Z12["y"].astype(np.int64)
    st60, y60 = Z60["start"].astype(np.int64), Z60["y"].astype(np.int64)
    c12 = ys[st12[:, None] + np.arange(12)].sum(1); c60 = ys[st60[:, None] + np.arange(60)].sum(1)
    R["c_counts"] = {
        "1s": {"units": int(len(y1)), "positives": int(y1.sum()), "prevalence": float(y1.mean()),
               "unit": "scored second", "label": "any overlap with a .csv_bi seizure (y_sec)"},
        "12s": {"units": int(len(y12)), "positives": int(y12.sum()), "prevalence": float(y12.mean()),
                "unit": "HyBrain 12 s clip", "label": "HyBrain list label (any seizure second)",
                "label_vs_1s_label_agreement": float(((c12 > 0).astype(int) == y12).mean())},
        "60s": {"units": int(len(y60)), "positives": int(y60.sum()), "prevalence": float(y60.mean()),
                "unit": "60 s window, stride 15 s", "label": "overlap >= min(9 s, 20 % of the seizure)",
                "windows_per_second_max": 4,
                "label_vs_any_ictal_second_agreement": float(((c60 > 0).astype(int) == y60).mean())},
        "eval_store_seizures": int(len(durs)),
        "seizure_duration_s_quantiles_0_25_50_75_100": [float(q) for q in np.quantile(durs, [0, .25, .5, .75, 1])]}

    # ---------------- (a) label definition / boundary units
    A = {"descriptive": True}
    # 1 s: edge seconds
    e1 = edge[rows]
    g1 = {"pos_full": (y1 == 1) & ~e1, "pos_boundary": (y1 == 1) & e1, "pos_no_ictal_second": np.zeros(len(y1), bool),
          "neg_pure": y1 == 0, "neg_subthreshold": np.zeros(len(y1), bool),
          "neg_pure_near": (y1 == 0) & (dist[rows] <= NEAR_S), "neg_pure_far": (y1 == 0) & (dist[rows] > NEAR_S)}
    A["1s"] = {"definition": "pos_full = interior ictal seconds; pos_boundary = first/last second of an annotated "
                             "seizure (the only seconds an any-overlap label can mark while partly ictal)",
               "positive_boundary_share": float(g1["pos_boundary"].sum() / max(y1.sum(), 1)), "models": {}}
    for m, (k1, _) in MODELS.items():
        A["1s"]["models"][m] = group_stats(g1, y1, T[k1].astype(np.float64), sel1["configs"][k1]["f1_thr"], False)
    for U, Z, st, y, c, ge, thrs in ((12, Z12, st12, y12, c12, False, {k: sel12["thr"][k] for k in sel12["thr"]}),
                                     (60, Z60, st60, y60, c60, True,
                                      {k: v["threshold"] for k, v in sel60["post"].items()})):
        G = unit_groups(c, y, U, dist[st[:, None] + np.arange(U)].min(1))
        key = f"{U}s"
        A[key] = {"positive_boundary_share": float(G["pos_boundary"].sum() / max(y.sum(), 1)),
                  "positive_full_share": float(G["pos_full"].sum() / max(y.sum(), 1)),
                  "negatives_with_ictal_seconds_share": float(G["neg_subthreshold"].sum() / max((y == 0).sum(), 1)),
                  "ictal_fraction_of_positive_units_quantiles_0_25_50_75_100":
                      [float(q) for q in np.quantile(c[y == 1] / U, [0, .25, .5, .75, 1])],
                  "group_counts": {g: int(m.sum()) for g, m in G.items()}, "models": {}}
        for m, (_, ku) in MODELS.items():
            z = Z[ku]; p = sig(fill(z)) if ku == "catboost" else sig(z)
            A[key]["models"][m] = group_stats(G, y, p, thrs[ku], ge)
    R["a_label_definition"] = A

    # ---------------- (b) pooled 1 s track vs trained heads
    B = {"descriptive_on_eval": True, "threshold_for_pooled_F1": "best dev F1 of the same pooling on the dev fused "
                                                                  "track (in-sample stacker refit); chosen on dev"}
    D, dev_tr = dev_fused_track(a.tokens, a.catboost, a.runs, sel1)
    dindex = [e for e in json.loads((Path(a.tokens) / "dev" / "index.json").read_text()) if e["status"] == "done"]
    dys = np.asarray(np.load(Path(a.tokens) / "dev" / "y_sec.npy", mmap_mode="r")).astype(np.int8)
    dstore = {}
    for n, v in dev_tr.items():
        dstore[n], drows = to_store(D["rec"], v, dindex, len(dys))
    assert np.array_equal(dys[drows], D["y"]), "dev track labels differ from the store"
    dst12, _, dy12, _ = clips12.clips(a.markers, a.tokens, "dev")
    dst60, dy60, *_ = windows60.windows("dev", a.tokens, Path(a.catboost) / "scores_dev.npz", a.feat_dir)
    B["per_second_reference"] = {m: {"auroc": auc(y1, T[k1]), "auprc": ap(y1, T[k1])} for m, (k1, _) in MODELS.items()}
    for U, Z, st, y, dst, dy, head_thr, ge in (
            (12, Z12, st12, y12, dst12, dy12, sel12["thr"], False),
            (60, Z60, st60, y60, dst60, dy60, {k: v["threshold"] for k, v in sel60["post"].items()}, True)):
        key = f"{U}s"; out = {}
        for m, (k1, ku) in MODELS.items():
            lme, mean, full = pool(tr[k1], st, U)
            dlme, dmean, _ = pool(dstore[k1], dst, U)
            zh = Z[ku]; ph = sig(fill(zh)) if ku == "catboost" else sig(zh)
            r = {"units_fully_scored_at_1s": int(full.sum()), "units": int(len(y)),
                 "positives_fully_scored": int(y[full].sum())}
            # per-second metric of the same track restricted to the seconds of the fully scored units (reference)
            secs = np.unique((st[full][:, None] + np.arange(U)).ravel())
            r["per_second_on_unit_seconds"] = {"auroc": auc(ys[secs], tr[k1][secs]), "auprc": ap(ys[secs], tr[k1][secs]),
                                               "seconds": int(len(secs)), "positives": int(ys[secs].sum())}
            for name, s_eval, s_dev in (("pooled_lme", sig(fill(lme)), sig(fill(dlme))),
                                        ("pooled_mean", fill(mean), fill(dmean))):
                thr, _ = P.best_f1_threshold(dy, s_dev)
                r[name] = {"all_units": {"auroc": auc(y, s_eval), "auprc": ap(y, s_eval),
                                         **thr_metrics(y, s_eval, thr)},
                           "fully_scored_units": {"auroc": auc(y[full], s_eval[full]), "auprc": ap(y[full], s_eval[full]),
                                                  **thr_metrics(y[full], s_eval[full], thr)}}
            r["trained_head"] = {"key": ku,
                                 "all_units": {"auroc": auc(y, ph), "auprc": ap(y, ph), **thr_metrics(y, ph, head_thr[ku], ge)},
                                 "fully_scored_units": {"auroc": auc(y[full], ph[full]), "auprc": ap(y[full], ph[full]),
                                                        **thr_metrics(y[full], ph[full], head_thr[ku], ge)}}
            ps = r["per_second_on_unit_seconds"]; pl = r["pooled_lme"]["fully_scored_units"]
            hd = r["trained_head"]["fully_scored_units"]
            r["decomposition_fully_scored_units"] = {
                "unit_label_effect_auroc": pl["auroc"] - ps["auroc"], "head_effect_auroc": hd["auroc"] - pl["auroc"],
                "unit_label_effect_auprc": pl["auprc"] - ps["auprc"], "head_effect_auprc": hd["auprc"] - pl["auprc"],
                "pooling": "log-mean-exp"}
            out[m] = r
        B[key] = out
    R["b_head_vs_unit"] = B

    # ---------------- (d) seizure duration vs unit length
    Dd = {"descriptive": True,
          "seizures_shorter_than_12s": int((durs < 12).sum()), "seizures_shorter_than_60s": int((durs < 60).sum()),
          "n_seizures": int(len(durs)),
          "share_shorter_than_12s": float((durs < 12).mean()), "share_shorter_than_60s": float((durs < 60).mean())}
    ed1 = evdur[rows]
    for key, U, Z, st, y, ge, thrs, k_unit in (
            ("1s", 1, None, None, y1, False, None, None),
            ("12s", 12, Z12, st12, y12, False, sel12["thr"], None),
            ("60s", 60, Z60, st60, y60, True, {k: v["threshold"] for k, v in sel60["post"].items()}, None)):
        if U == 1:
            dmax = ed1
        else:
            dmax = evdur[st[:, None] + np.arange(U)].max(1)
        pos = y == 1
        rr = {"positive_units_by_longest_seizure_s": {}, "recall_at_frozen_thr_by_longest_seizure_s": {}}
        for lo, hi in DUR_BINS:
            b = f"{lo}-{'inf' if hi == np.inf else int(hi)}"
            m = pos & (dmax >= lo) & (dmax < hi)
            rr["positive_units_by_longest_seizure_s"][b] = {"n": int(m.sum()), "share": float(m.sum() / max(pos.sum(), 1))}
            rec_m = {}
            for mm, (k1, ku) in MODELS.items():
                if U == 1:
                    p = T[k1].astype(np.float64); th = sel1["configs"][k1]["f1_thr"]
                else:
                    z = Z[ku]; p = sig(fill(z)) if ku == "catboost" else sig(z); th = thrs[ku]
                above = p >= th if ge else p > th
                rec_m[mm] = float(above[m].mean()) if m.any() else None
            rr["recall_at_frozen_thr_by_longest_seizure_s"][b] = rec_m
        rr["positive_units_with_no_annotated_second"] = int((pos & (dmax == 0)).sum())
        Dd[key] = rr
    R["d_other_causes"] = Dd
    R["summary"] = summary(R)
    return R


def summary(R):
    c, A, B, D = R["c_counts"], R["a_label_definition"], R["b_head_vs_unit"], R["d_other_causes"]
    s = []
    s.append(f"(c) Units and prevalence: 1 s {c['1s']['units']:,} seconds, {c['1s']['prevalence']:.4f} positive; "
             f"12 s {c['12s']['units']:,} clips, {c['12s']['prevalence']:.4f}; 60 s {c['60s']['units']:,} windows, "
             f"{c['60s']['prevalence']:.4f}.")
    s.append(f"(a) Boundary units: {A['1s']['positive_boundary_share']:.3f} of positive seconds are edge seconds, "
             f"against {A['12s']['positive_boundary_share']:.3f} of positive 12 s clips and "
             f"{A['60s']['positive_boundary_share']:.3f} of positive 60 s windows that are only partly ictal; "
             f"{A['60s']['negatives_with_ictal_seconds_share']:.4f} of negative 60 s windows contain ictal seconds "
             f"(sub-threshold overlap).")
    for k in ("12s", "60s"):
        fm = A[k]["models"]["fusion"]
        s.append(f"(a) Fusion at {k}, frozen threshold: recall {fm['pos_full']['frac_above_thr']:.3f} on fully ictal "
                 f"units vs {fm['pos_boundary']['frac_above_thr']:.3f} on boundary units; AUROC vs pure background "
                 f"{fm['pos_full']['auroc_vs_pure_background']:.3f} vs {fm['pos_boundary']['auroc_vs_pure_background']:.3f}"
                 f"; clean units only (descriptive) AUROC {fm['clean_units_only']['auroc']:.3f}.")
    f1m = A["1s"]["models"]["fusion"]
    s.append(f"(a) Fusion at 1 s: recall {f1m['pos_full']['frac_above_thr']:.3f} on interior seconds vs "
             f"{f1m['pos_boundary']['frac_above_thr']:.3f} on edge seconds.")
    for k in ("12s", "60s"):
        f = B[k]["fusion"]; d = f["decomposition_fully_scored_units"]
        s.append(f"(b) {k}, units fully scored at 1 s ({f['units_fully_scored_at_1s']:,}/{f['units']:,}): per-second "
                 f"AUROC of the fused track on their seconds {f['per_second_on_unit_seconds']['auroc']:.4f}; pooled "
                 f"(log-mean-exp) {f['pooled_lme']['fully_scored_units']['auroc']:.4f}; trained head "
                 f"{f['trained_head']['fully_scored_units']['auroc']:.4f} -> unit/label effect "
                 f"{d['unit_label_effect_auroc']:+.4f}, head effect {d['head_effect_auroc']:+.4f} (AUROC); AUPRC "
                 f"{d['unit_label_effect_auprc']:+.4f} / {d['head_effect_auprc']:+.4f}. Pooled-track F1 at a dev "
                 f"threshold {f['pooled_lme']['all_units']['f1']:.3f} vs trained head "
                 f"{f['trained_head']['all_units']['f1']:.3f} (all units).")
    s.append("(d) Peri-ictal background (pure-background units within 60 s of a seizure), fusion false-positive rate at "
             "the frozen threshold, near vs far: " + "; ".join(
                 f"{k} {A[k]['models']['fusion']['neg_pure_near']['frac_above_thr']:.4f} "
                 f"(n = {A[k]['models']['fusion']['neg_pure_near']['n']:,}) vs "
                 f"{A[k]['models']['fusion']['neg_pure_far']['frac_above_thr']:.4f}" for k in ("1s", "12s", "60s")) + ".")
    s.append(f"(d) {D['seizures_shorter_than_12s']} of {D['n_seizures']} eval seizures are shorter than 12 s and "
             f"{D['seizures_shorter_than_60s']} shorter than 60 s.")
    for k in ("1s", "12s", "60s"):
        r = D[k]["recall_at_frozen_thr_by_longest_seizure_s"]
        s.append(f"(d) {k} fusion recall by seizure duration: " +
                 ", ".join(f"{b} s {v['fusion']:.3f}" if v["fusion"] is not None else f"{b} s –" for b, v in r.items())
                 + "; share of positive units: " +
                 ", ".join(f"{b} {v['share']:.3f}" for b, v in D[k]["positive_units_by_longest_seizure_s"].items()) + ".")
    return s


def main(argv=None):
    work = paths.get("work_dir")
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--tokens", default=str(work / "tokens" / "tusz"))
    ap_.add_argument("--catboost", default=str(work / "tusz" / "catboost"))
    ap_.add_argument("--runs", default=str(work / "kaggle_out" / "szfusion-linoss-tusz" / "runs"))
    ap_.add_argument("--feat_dir", default=str(paths.get("work_dir").parent / "cb_feats"),
                     help="CatBoost feature folder with {dev,eval}/intervals.json (default: the development features "
                          "next to work_dir, used under the 2026-10-15 10:40 amendment)")
    ap_.add_argument("--markers", default=str(work / "markers" / "tusz"))
    ap_.add_argument("--out", default="results/tusz/resolution.json")
    a = ap_.parse_args(argv)
    R = analyse(a)
    out = {"provenance": {**provenance(f"python -m szfusion.eval.resolution --out {a.out}"),
                          "config_sha256": config_sha256(), "source": R["sources"],
                          "inputs": "<work>/{tusz|s1_gate}/{fusion_1s,u12,u60}, <work>/tokens/tusz, "
                                    "<work>/tusz/catboost, LinOSS runs s0-s4 (prob_dev.npy), cb_feats, markers/tusz"},
           **R}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print("\n".join(R["summary"]))


if __name__ == "__main__":
    main()
