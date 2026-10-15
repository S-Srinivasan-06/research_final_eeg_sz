"""ROC, precision-recall and event FROC curves of the CatBoost × LinOSS fusion and its two branches (TUSZ eval).

  python -m szfusion.viz.curves [--out docs/figures]

Writes
  roc_pr_1s.{png,svg}       1 s: ROC and precision-recall of fused_mean, catboost, linoss_mean on the scored seconds
                            (eval_tracks.npz); the dev-frozen per-second operating point (best dev F1 threshold,
                            selection.json configs.<model>.f1_thr, rule p > thr) is marked with a circle.
  roc_pr_12s_60s.{png,svg}  12 s clips and 60 s windows: the same for look_mean_cb (fusion), catboost and look_mean
                            (LinOSS branch) from eval_scores.npz; operating points: 12 s thr (best dev F1, p > thr),
                            60 s post.threshold of the dev post-processing rule applied to the raw window probability
                            (p >= thr, as the stored F1).
  froc_1s.png               1 s event sensitivity against false alarms per 24 h. For each model the smoothing k, merge
                            gap and minimum duration are frozen at its dev event point 1 (selection.json ev) and only
                            the threshold is swept over quantiles of the smoothed eval track plus the frozen
                            threshold itself (a curve, not a selection), drawn in threshold order; scorer = fusion/protocol_1s.event_scores (any overlap). Circle = event point 1,
                            square = event point 2 (its own dev-frozen k / gap / min_dur), both read from
                            eval_results.json.
Inputs: the folders of eval/ablation.py sources() (official when complete, else S1 gate; the source is printed in
each figure's footer). No threshold is chosen on eval.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve  # noqa: E402

from szfusion.eval.ablation import sources  # noqa: E402
from szfusion.fusion import protocol_1s as P  # noqa: E402

COL = {"fusion": "#0F6E56", "catboost": "#D4537E", "linoss": "#378ADD"}     # as viz/froc.py
NAME = {"fusion": "CatBoost × LinOSS fusion", "catboost": "CatBoost branch (w/o LinOSS)",
        "linoss": "LinOSS branch (w/o CatBoost)"}
K1 = {"fusion": "fused_mean", "catboost": "catboost", "linoss": "linoss_mean"}
KU = {"fusion": "look_mean_cb", "catboost": "catboost", "linoss": "look_mean"}
N_PLOT = 4000                 # max points drawn per curve (curves are thinned for drawing only, metrics use all points)
FROC_Q = np.r_[np.linspace(0.80, 0.99, 60), np.linspace(0.9902, 0.99995, 80)]   # threshold quantiles of the track
sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False})


def thin(x, y):
    if len(x) <= N_PLOT:
        return x, y
    i = np.unique(np.r_[np.linspace(0, len(x) - 1, N_PLOT).astype(int), len(x) - 1])
    return x[i], y[i]


def op_point(y, p, thr, ge):
    pr = p >= thr if ge else p > thr
    tp = (pr & (y == 1)).sum(); fp = (pr & (y == 0)).sum()
    return fp / (y == 0).sum(), tp / (y == 1).sum(), tp / max(tp + fp, 1)     # FPR, TPR (= recall), precision


def roc_pr(ax_roc, ax_pr, y, scores, thrs, ge, title):
    for m, p in scores.items():
        fpr, tpr, _ = roc_curve(y, p)
        pre, rec, _ = precision_recall_curve(y, p)
        a, b = roc_auc_score(y, p), average_precision_score(y, p)
        ax_roc.plot(*thin(fpr, tpr), color=COL[m], lw=1.4, label=f"{NAME[m]}  (AUROC {a:.3f})")
        ax_pr.plot(*thin(rec[::-1], pre[::-1]), color=COL[m], lw=1.4, label=f"{NAME[m]}  (AUPRC {b:.3f})")
        f, t, pr = op_point(y, p, thrs[m], ge)
        ax_roc.plot(f, t, "o", ms=6, mfc="white", mec=COL[m], mew=1.6, zorder=5)
        ax_pr.plot(t, pr, "o", ms=6, mfc="white", mec=COL[m], mew=1.6, zorder=5)
    ax_roc.plot([0, 1], [0, 1], ":", color="0.6", lw=0.8)
    prev = y.mean()
    ax_pr.axhline(prev, ls=":", color="0.6", lw=0.8)
    ax_pr.text(0.01, prev - 0.012, f"prevalence {prev:.3f}", ha="left", va="top", fontsize=7.5, color="0.4")
    ax_roc.set(xlabel="false-positive rate (1 − specificity)", ylabel="true-positive rate (sensitivity)",
               xlim=(0, 1), ylim=(0, 1.01), title=f"{title}: ROC")
    ax_pr.set(xlabel="recall (sensitivity)", ylabel="precision", xlim=(0, 1), ylim=(0, 1.01),
              title=f"{title}: precision–recall")
    for ax in (ax_roc, ax_pr):
        ax.legend(loc="lower right" if ax is ax_roc else "lower left", bbox_to_anchor=None if ax is ax_roc else (0.0, 0.10),
                  fontsize=7.5, frameon=ax is not ax_roc, facecolor="white", edgecolor="none", framealpha=0.9)
        ax.grid(alpha=0.25, lw=0.5)
    # inset: ROC zoom on the low-FPR region, where the operating points lie
    ins = ax_roc.inset_axes([0.42, 0.36, 0.40, 0.34])
    ins.set_facecolor("white"); ins.patch.set_alpha(1.0); ins.set_zorder(4)
    for m, p in scores.items():
        fpr, tpr, _ = roc_curve(y, p)
        k = fpr <= 0.1
        ins.plot(*thin(fpr[k], tpr[k]), color=COL[m], lw=1.1)
        f, t, _ = op_point(y, p, thrs[m], ge)
        ins.plot(f, t, "o", ms=4.5, mfc="white", mec=COL[m], mew=1.3)
    ins.set_xlim(0, 0.1); ins.set_ylim(0, 1); ins.tick_params(labelsize=6.5)
    ins.set_title("FPR ≤ 0.10", fontsize=7); ins.grid(alpha=0.25, lw=0.5)


def footer(fig, txt):
    fig.text(0.01, 0.005, txt, fontsize=7, color="0.35", ha="left", va="bottom")


def fig_1s(S, out):
    d, src = S["1s"]
    T = np.load(d / "eval_tracks.npz"); sel = json.loads((d / "selection.json").read_text())["configs"]
    y = T["y"].astype(np.int8)
    scores = {m: T[K1[m]].astype(np.float64) for m in K1}
    thrs = {m: sel[K1[m]]["f1_thr"] for m in K1}
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.6))
    roc_pr(ax[0], ax[1], y, scores, thrs, False, f"1 s (n = {len(y):,} s, {int(y.sum()):,} ictal)")
    footer(fig, f"TUSZ v2.0.6 eval, scored seconds; circles = dev-frozen best-dev-F1 thresholds. Source: {src}.")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for ext in ("png", "svg"):
        fig.savefig(out / f"roc_pr_1s.{ext}", dpi=200)
    plt.close(fig)


def fig_units(S, out):
    fig, ax = plt.subplots(2, 2, figsize=(10.5, 9.0))
    srcs = []
    for row, (res, U) in enumerate((("12s", 12), ("60s", 60))):
        d, src = S[res]; srcs.append(f"{U} s: {src}")
        Z = np.load(d / "eval_scores.npz"); sel = json.loads((d / "selection.json").read_text())
        y = Z["y"].astype(int)
        scores = {}
        for m, k in KU.items():
            z = Z[k]
            scores[m] = sig(np.where(np.isnan(z), np.nanmin(z), z)) if k == "catboost" else sig(z)
        thrs = {m: (sel["thr"][KU[m]] if U == 12 else sel["post"][KU[m]]["threshold"]) for m in KU}
        unit = "clips" if U == 12 else "windows"
        roc_pr(ax[row, 0], ax[row, 1], y, scores, thrs, U == 60,
               f"{U} s (n = {len(y):,} {unit}, {int(y.sum()):,} positive)")
    footer(fig, "TUSZ v2.0.6 eval. Circles: 12 s best-dev-F1 threshold; 60 s window threshold of the dev post-processing "
                f"rule (p ≥ thr). Sources: {', '.join(srcs)}.")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    for ext in ("png", "svg"):
        fig.savefig(out / f"roc_pr_12s_60s.{ext}", dpi=200)
    plt.close(fig)


def froc_1s(S, out):
    d, src = S["1s"]
    T = np.load(d / "eval_tracks.npz"); sel = json.loads((d / "selection.json").read_text())["configs"]
    R = json.loads((d / "eval_results.json").read_text())["results"]
    D = {"y": T["y"].astype(np.int8), "rec": T["rec"].astype(np.int64)}
    te = P.true_events(D)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for m, k in K1.items():
        ev = sel[k]["ev"]
        ps = P.smooth(T[k].astype(np.float64), D["rec"], ev["k"])
        pts = []
        for thr in np.unique(np.r_[np.quantile(ps, FROC_Q), ev["thr"]])[::-1]:      # high -> low threshold
            r = P.event_scores(D, ps, float(thr), ev["gap"], ev["min_dur"], te)
            pts.append((r["fa_per_24h"], r["event_sens"]))
        x, yv = np.array(pts).T                       # drawn in threshold order (merging can make it non-monotone)
        ax.plot(x, yv, color=COL[m], lw=1.5,
                label=f"{NAME[m]}  (k {ev['k']} s, gap {ev['gap']} s, min {ev['min_dur']} s)")
        r = R[k]
        ax.plot(r["ev_fa_per_24h"], r["ev_event_sens"], "o", ms=7, mfc="white", mec=COL[m], mew=1.8, zorder=5)
        ax.plot(r["ev2_fa_per_24h"], r["ev2_event_sens"], "s", ms=6.5, mfc="white", mec=COL[m], mew=1.8, zorder=5)
    ax.plot([], [], "o", mfc="white", mec="0.3", label="event point 1 (dev FA/24h ≤ 16.56), frozen")
    ax.plot([], [], "s", mfc="white", mec="0.3", label="event point 2 (dev FA/24h ≤ 33.12), frozen")
    ax.set(xscale="log", xlim=(0.5, 200), ylim=(0, 1), xlabel="false alarms per 24 h (log scale)",
           ylabel="event sensitivity (any overlap)",
           title=f"1 s event FROC, TUSZ eval ({len(te[0])} seizures, {len(D['y']) / 3600:.1f} h)")
    ax.grid(alpha=0.25, lw=0.5, which="both")
    ax.legend(loc="upper left", fontsize=7.5, frameon=False)
    footer(fig, "Smoothing / gap / minimum duration frozen at each model's dev point 1; threshold swept on eval "
                f"(a curve, not a selection). Source: {src}.")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "froc_1s.png", dpi=200)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    S = sources()
    print({k: v[1] for k, v in S.items()})
    fig_1s(S, out); fig_units(S, out); froc_1s(S, out)


if __name__ == "__main__":
    main()
