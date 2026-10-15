"""Patient-level bootstrap confidence intervals for the frozen test results (TUSZ eval 1 s; CHB-MIT test).

  python -m szfusion.eval.bootstrap tusz   [--fusion_1s F] [--tokens T] [--out results/tusz/bootstrap.json]
  python -m szfusion.eval.bootstrap chbmit [--meta M] [--linoss L0..L4] [--catboost C] [--scores S]
                                           [--out results/chbmit/bootstrap.json]

Unit of resampling: the patient. Seizures, false alarms and per-second errors cluster within a patient, so seconds,
clips or events are not independent; resampling patients with replacement keeps that dependence inside each
replicate. N_BOOT replicates (seed SEED); each draws as many patients as the test set has, with replacement; a patient
drawn m times enters every metric with weight m (all its rows, events, false alarms and hours). The 95 % interval is
the 2.5-97.5 percentile interval of the replicates. The paired difference fused - branch is computed on the same
replicate (same patients) for both models; reported with its interval and the fraction of replicates where it is > 0.
Nothing is re-selected: every threshold and event rule is the dev-frozen one; the per-event detection and per-patient
false-alarm counts are those of the full test set (events never cross recordings, so they never cross patients).

TUSZ (eval_tracks.npz + selection.json of the 1 s fusion stage; default fusion folder as in by_type.py):
  fused_mean, catboost, linoss_mean: AUROC, AUPRC (raw probabilities), F1 at the frozen dev-F1 threshold
  (configs.<m>.f1_thr), event sensitivity and FA/24h at event points 1 (ev) and 2 (ev2), as fusion/protocol_1s.
CHB-MIT (HyBrain protocol, test; patient = chb case, clips_test.npz `patient`): per-clip / per-second predictions are
rebuilt with eval/chbmit_scores.load and the frozen selection.json (stackers and thresholds); its test stage is not
re-run. 12 s clips and 1 s point-wise: fused_mean, linoss_mean, catboost: AUROC, AUPRC, F1 at the frozen threshold.

Weighted AUROC / AUPRC: identical to sklearn roc_auc_score / average_precision_score with sample_weight (ties grouped;
checked against sklearn on the point estimate and on one replicate, see `checks` in the output).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from szfusion import paths
from szfusion.eval import by_type as BT
from szfusion.eval.mechanism import provenance
from szfusion.fusion import protocol_1s as P

N_BOOT = 2000              # bootstrap replicates
SEED = 0                   # numpy default_rng seed
CI = (2.5, 97.5)           # percentiles of the 95 % interval
MODELS = ["fused_mean", "catboost", "linoss_mean"]
UNIT_NOTE = ("Resampling unit = patient (with replacement): events, false alarms and per-second errors cluster by "
             "patient, so resampling seconds or events would understate the variance.")


# ----------------------------------------------------------------------------- weighted threshold-free metrics
class Ranked:
    """One score track, sorted once; weighted AUROC/AUPRC for any per-patient weight vector."""

    def __init__(self, p, y, pid):
        o = np.argsort(-p, kind="mergesort")
        ps = p[o]
        self.y, self.pid = y[o].astype(np.float64), pid[o]
        self.starts = np.flatnonzero(np.r_[True, ps[1:] != ps[:-1]])   # groups of tied scores

    def metrics(self, wp):
        w = wp[self.pid]
        pw = w * self.y
        Pg = np.add.reduceat(pw, self.starts); Ng = np.add.reduceat(w - pw, self.starts)
        cP, cN = np.cumsum(Pg), np.cumsum(Ng)
        Pt, Nt = cP[-1], cN[-1]
        if Pt <= 0 or Nt <= 0:
            return np.nan, np.nan
        auroc = float(np.sum(Ng * (cP - Pg / 2)) / (Pt * Nt))
        prec = cP / np.maximum(cP + cN, 1e-300)
        auprc = float(np.sum(Pg * prec) / Pt)
        return auroc, auprc


def counts_by_patient(pid, n_pat, *cols):
    return [np.bincount(pid, weights=c.astype(np.float64), minlength=n_pat) for c in cols]


def ci(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    lo, hi = np.percentile(v, CI)
    return [float(lo), float(hi)]


def summarise(point, reps, models, metrics, ref="fused_mean"):
    out = {}
    for m in models:
        out[m] = {k: {"point": point[m][k], "ci95": ci(reps[m][k]), "n_valid": int(np.sum(~np.isnan(reps[m][k])))}
                  for k in metrics}
    diffs = {}
    for b in models:
        if b == ref:
            continue
        diffs[f"{ref}-{b}"] = {}
        for k in metrics:
            d = np.asarray(reps[ref][k]) - np.asarray(reps[b][k]); d = d[~np.isnan(d)]
            diffs[f"{ref}-{b}"][k] = {"point": point[ref][k] - point[b][k], "ci95": ci(d),
                                      "frac_gt_0": float(np.mean(d > 0))}
    return out, diffs


def boot_loop(n_pat, fn):
    rng = np.random.default_rng(SEED)
    reps = None
    for _ in range(N_BOOT):
        w = np.bincount(rng.integers(0, n_pat, n_pat), minlength=n_pat).astype(np.float64)
        r = fn(w)
        if reps is None:
            reps = {m: {k: [] for k in r[m]} for m in r}
        for m in r:
            for k, v in r[m].items():
                reps[m][k].append(v)
    return reps


# ----------------------------------------------------------------------------- TUSZ
def tusz(fusion_dir, tokens_root):
    D = BT.load(fusion_dir, tokens_root)
    _, pid = np.unique(D["patient"], return_inverse=True)
    n_pat = int(pid.max()) + 1
    y = D["y"].astype(np.int64)
    ts, te = D["te"]
    ev_pid = pid[ts]
    sec = np.bincount(pid, minlength=n_pat).astype(np.float64)
    ntrue = np.bincount(ev_pid, minlength=n_pat).astype(np.float64)
    pre = {}
    for m in MODELS:
        p = D["T"][m].astype(np.float64)
        sel = D["sel"][m]
        pr = p > sel["f1_thr"]
        tp, fp, fn = counts_by_patient(pid, n_pat, pr & (y == 1), pr & (y == 0), ~pr & (y == 1))
        evs = {}
        for pt in ("ev", "ev2"):
            st, en, _ = BT.predicted_mask(D, m, pt)
            cy = np.r_[0, np.cumsum(y)]
            fa = (cy[en] - cy[st]) == 0
            det, _, _ = BT.detection(D, m, pt)
            evs[pt] = (np.bincount(ev_pid, weights=det.astype(float), minlength=n_pat),
                       np.bincount(pid[st[fa]], minlength=n_pat).astype(np.float64))
        pre[m] = {"rank": Ranked(p, y, pid), "tp": tp, "fp": fp, "fn": fn, "ev": evs}

    def fn_(w):
        r = {}
        for m, q in pre.items():
            au, ap = q["rank"].metrics(w)
            tp, fp, fn = w @ q["tp"], w @ q["fp"], w @ q["fn"]
            d = {"auroc": au, "auprc": ap, "f1": 2 * tp / max(2 * tp + fp + fn, 1)}
            for pt, (det, fa) in q["ev"].items():
                d[f"{pt}_event_sens"] = (w @ det) / (w @ ntrue) if w @ ntrue > 0 else np.nan
                d[f"{pt}_fa_per_24h"] = (w @ fa) / (w @ sec) * 86400
            r[m] = d
        return r

    point = fn_(np.ones(n_pat))
    # checks against the scorer / sklearn
    checks = {}
    for m in MODELS:
        p = D["T"][m].astype(np.float64)
        ref = P.full_scores(D, p, D["sel"][m])
        checks[m] = {"auroc_sklearn": ref["auroc"], "auprc_sklearn": ref["auprc"], "f1_scorer": ref["f1"],
                     "ev_event_sens_scorer": ref["ev_event_sens"], "ev_fa_per_24h_scorer": ref["ev_fa_per_24h"],
                     "ev2_event_sens_scorer": ref["ev2_event_sens"], "ev2_fa_per_24h_scorer": ref["ev2_fa_per_24h"]}
        for k in ("auroc", "auprc", "f1"):
            assert abs(point[m][k] - ref[k]) < 1e-9, (m, k, point[m][k], ref[k])
        for k in ("ev_event_sens", "ev_fa_per_24h", "ev2_event_sens", "ev2_fa_per_24h"):
            assert abs(point[m][k] - ref[k]) < 1e-9, (m, k, point[m][k], ref[k])
    w1 = np.bincount(np.random.default_rng(123).integers(0, n_pat, n_pat), minlength=n_pat).astype(float)
    p = D["T"]["fused_mean"].astype(np.float64)
    sk = (roc_auc_score(y, p, sample_weight=w1[pid]), average_precision_score(y, p, sample_weight=w1[pid]))
    mine = pre["fused_mean"]["rank"].metrics(w1)
    assert abs(sk[0] - mine[0]) < 1e-9 and abs(sk[1] - mine[1]) < 1e-9, (sk, mine)
    checks["weighted_replicate_fused_mean"] = {"sklearn": list(map(float, sk)), "ours": list(mine)}
    reps = boot_loop(n_pat, fn_)
    metrics = list(point["fused_mean"])
    res, diffs = summarise(point, reps, MODELS, metrics)
    info = {"n_patients": n_pat, "n_patients_with_events": int((ntrue > 0).sum()), "n_true_events": int(len(ts)),
            "scored_seconds": int(len(y)), "scored_hours": float(len(y) / 3600)}
    return {"data": "TUSZ v2.0.6 eval, 1 s scored seconds (eval_tracks.npz)", "info": info, "results": res,
            "paired_differences": diffs, "checks": checks}


# ----------------------------------------------------------------------------- CHB-MIT
def chbmit(a):
    from szfusion.eval.chbmit_scores import load, sig
    from szfusion.fusion.stacker import WindowStacker
    from szfusion.units.clips12 import lme
    sel = json.loads((Path(a.scores) / "selection.json").read_text())
    published = json.loads((Path(a.scores) / "test_results.json").read_text())
    published = published.get("results", published)
    yc, ys, pat, zl, zc = load("test", SimpleNamespace(meta=a.meta, linoss=a.linoss, catboost=a.catboost))
    _, pid_c = np.unique(pat, return_inverse=True)
    n_pat = int(pid_c.max()) + 1
    out, checks = {}, {}
    for task in ("clip12", "point1s"):
        y = (yc if task == "clip12" else ys.ravel()).astype(np.int64)
        pid = pid_c if task == "clip12" else np.repeat(pid_c, 12)
        zc_t = zc if task == "clip12" else np.repeat(zc, 12)
        zl_t = lme(zl["mean"]) if task == "clip12" else zl["mean"].ravel()
        tracks = {"fused_mean": WindowStacker.from_dict(sel["stack"][f"{task}/fused_mean"]).predict(zc_t, zl_t),
                  "linoss_mean": sig(zl_t), "catboost": sig(zc_t)}
        pre = {}
        for m, p in tracks.items():
            pr = p > sel["thr"][f"{task}/{m}"]
            tp, fp, fn = counts_by_patient(pid, n_pat, pr & (y == 1), pr & (y == 0), ~pr & (y == 1))
            pre[m] = {"rank": Ranked(p, y, pid), "tp": tp, "fp": fp, "fn": fn}

        def fn_(w, pre=pre):
            r = {}
            for m, q in pre.items():
                au, ap = q["rank"].metrics(w)
                tp, fp, fn = w @ q["tp"], w @ q["fp"], w @ q["fn"]
                r[m] = {"auroc": au, "auprc": ap, "f1": 2 * tp / max(2 * tp + fp + fn, 1)}
            return r

        point = fn_(np.ones(n_pat))
        for m in tracks:
            ref = published[f"{task}/{m}"]
            checks[f"{task}/{m}"] = {k: ref[k] for k in ("auroc", "auprc", "f1")}
            for k in ("auroc", "auprc", "f1"):
                assert abs(point[m][k] - ref[k]) < 1e-9, (task, m, k, point[m][k], ref[k])
        reps = boot_loop(n_pat, fn_)
        res, diffs = summarise(point, reps, ["fused_mean", "linoss_mean", "catboost"], ["auroc", "auprc", "f1"])
        out[task] = {"n_rows": int(len(y)), "n_pos": int(y.sum()),
                     "n_patients_with_pos": int((np.bincount(pid, weights=y, minlength=n_pat) > 0).sum()),
                     "results": res, "paired_differences": diffs}
    return {"data": "CHB-MIT, HyBrain protocol test split (chb01-chb22, within-patient 70/15/15 clip lists)",
            "n_patients": n_pat,
            "note": "Patient = chb case. The split is within patient, so every test patient also contributed "
                    "training clips; the interval covers patient-to-patient variation of the test clips only.",
            "checks_point_estimate_equals_test_results_json": checks, **out}


def main(argv=None):
    work = paths.get("work_dir")
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["tusz", "chbmit"])
    ap.add_argument("--fusion_1s", default=None)
    ap.add_argument("--tokens", default=str(work / "tokens" / "tusz"))
    ap.add_argument("--meta", default=str(work / "kaggle_out" / "szfusion-chbmit-linoss" / "store_meta"))
    ap.add_argument("--linoss", nargs="+",
                    default=[str(work / "kaggle_out" / "szfusion-chbmit-linoss" / "linoss" / f"s{i}") for i in range(5)])
    ap.add_argument("--catboost", default=str(work / "kaggle_out" / "szfusion-chbmit-catboost" / "catboost_scores"))
    ap.add_argument("--scores", default=str(work / "chbmit" / "scores"), help="chbmit_scores output (selection.json)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    base = {"n_boot": N_BOOT, "seed": SEED, "ci": "95 % percentile interval (2.5, 97.5)", "unit_note": UNIT_NOTE}
    if a.dataset == "tusz":
        a.out = a.out or "results/tusz/bootstrap.json"
        fdir = Path(a.fusion_1s) if a.fusion_1s else BT.default_fusion_dir()
        src = BT.source_label(fdir)
        prov = {**provenance(BT.command("bootstrap tusz", src, a.out)), "source": src}
        res = tusz(fdir, a.tokens)
    else:
        a.out = a.out or "results/chbmit/bootstrap.json"
        prov = {**provenance(f"python -m szfusion.eval.bootstrap chbmit --meta <store_meta> --linoss s0..s4 "
                             f"--catboost <catboost_scores> --scores <work>/chbmit/scores --out {a.out}"),
                "source": "C3 test stage (frozen selection.json)"}
        res = chbmit(a)
    out = {"provenance": prov, **base, **res}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    blocks = [("", res)] if a.dataset == "tusz" else [(t, res[t]) for t in ("clip12", "point1s")]
    for tag, b in blocks:
        for m, r in b["results"].items():
            print(tag, m, "  ".join(f"{k} {v['point']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]" for k, v in r.items()))
        for d, r in b["paired_differences"].items():
            print(tag, d, "  ".join(f"{k} {v['point']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] P>0 {v['frac_gt_0']:.3f}"
                                    for k, v in r.items()))


if __name__ == "__main__":
    main()
