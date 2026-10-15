"""Detection graphs: whole-recording timelines of the CatBoost × LinOSS fusion and its two branches (TUSZ eval, 1 s).

  python -m szfusion.viz.timeline [--out docs/figures/timeline_cases]

Case rule (fixed before looking at any timeline; no recording or patient ID is printed, rendered or written):
  candidates  eval recordings with at least one annotated seizure detected by the fusion at event point 1
              (selection.json configs.fused_mean.ev, any overlap, fusion/protocol_1s.py);
  value       number of annotated seizure seconds of the recording (scored seconds);
  cases       the candidates whose value is closest to the 25th, 50th and 75th percentile of the candidate values
              (ties -> smaller value, then store order; a recording is not used twice: the next closest is taken).
Panels per case, over the whole scored recording (x in minutes):
  grey spans   annotated seizure seconds (y), on every row;
  row 1        CatBoost branch per-second probability (the mean of the 60 s windows covering the second);
  row 2        LinOSS branch per-second probability (5-seed mean);
  row 3        fused per-second probability (thin) and its centred moving average over k s (thick), with the event
               threshold (dashed); k and the threshold are the dev-frozen event point 1;
  row 4        fused event mask: green = predicted event overlapping an annotated seizure, red = false alarm (a
               predicted event overlapping no annotated second, also marked with a red triangle above row 3).
Inputs: the 1 s folder of eval/ablation.py sources() (official when complete, else the S1 gate).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from szfusion.eval.ablation import sources  # noqa: E402
from szfusion.fusion import protocol_1s as P  # noqa: E402

PCTS = (25, 50, 75)             # percentiles of seizure seconds per recording that define the three cases
COL = {"catboost": "#D4537E", "linoss_mean": "#378ADD", "fused_mean": "#0F6E56"}
plt.rcParams.update({"font.size": 8.5, "font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False})


def spans(mask):
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def pick(cands, vals, pcts):
    order = np.lexsort((cands, vals))         # by value, then store order
    used, out = set(), []
    for q in pcts:
        target = np.percentile(vals, q)
        best = min((i for i in order if cands[i] not in used), key=lambda i: (abs(vals[i] - target), vals[i], cands[i]))
        used.add(cands[best]); out.append((q, float(target), int(cands[best]), int(vals[best])))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures/timeline_cases")
    a = ap.parse_args(argv)
    d, src = sources()["1s"]
    T = np.load(d / "eval_tracks.npz"); sel = json.loads((d / "selection.json").read_text())["configs"]
    rec, y = T["rec"].astype(np.int64), T["y"].astype(np.int8)
    ev = sel["fused_mean"]["ev"]
    pf = T["fused_mean"].astype(np.float64)
    ps = P.smooth(pf, rec, ev["k"])
    st, en = P.events(ps, rec, ev["thr"], ev["gap"], ev["min_dur"])
    ts, te = P.true_events({"y": y, "rec": rec})
    cy = np.r_[0, np.cumsum(y)]
    is_fa = (cy[en] - cy[st]) == 0
    det = np.array([bool(((st < e) & (en > s)).any()) for s, e in zip(ts, te)])
    b = np.flatnonzero(np.r_[True, rec[1:] != rec[:-1], True]); blocks = list(zip(b[:-1], b[1:]))
    cands, vals = [], []
    for i, j in blocks:
        if det[(ts >= i) & (ts < j)].any():
            cands.append(i); vals.append(int(y[i:j].sum()))
    cands, vals = np.array(cands), np.array(vals)
    cases = pick(cands, vals, PCTS)
    print(f"source {src}; {len(cands)} candidate recordings; seizure seconds quantiles "
          f"{[float(np.percentile(vals, q)) for q in PCTS]}")
    fig = plt.figure(figsize=(11, 3.7 * len(cases) + 0.8))
    outer = fig.add_gridspec(len(cases), 1, hspace=0.55)
    for ci, (q, target, i0, v) in enumerate(cases):
        j0 = next(j for i, j in blocks if i == i0)
        sl = slice(i0, j0); n = j0 - i0; t = np.arange(n) / 60
        ann = spans(y[sl] == 1)
        ev_in = [(s - i0, e - i0, fa) for s, e, fa in zip(st, en, is_fa) if i0 <= s < j0]
        n_sz = int(((ts >= i0) & (ts < j0)).sum()); n_det = int(det[(ts >= i0) & (ts < j0)].sum())
        n_fa = sum(fa for *_, fa in ev_in)
        g = outer[ci].subgridspec(4, 1, height_ratios=[1, 1, 1.25, 0.32], hspace=0.32)
        axs = [fig.add_subplot(g[k]) for k in range(4)]
        for k, (key, lab) in enumerate((("catboost", "CatBoost\nbranch"), ("linoss_mean", "LinOSS\nbranch"),
                                        ("fused_mean", "fusion"))):
            ax = axs[k]
            for s, e in ann:
                ax.axvspan(s / 60, e / 60, color="0.82", lw=0, zorder=0)
            p = T[key][sl].astype(np.float64)
            if key == "fused_mean":
                ax.plot(t, p, color=COL[key], lw=0.4, alpha=0.45, rasterized=True)
                ax.plot(t, ps[sl], color=COL[key], lw=1.3, rasterized=True)
                ax.axhline(ev["thr"], ls="--", color="0.25", lw=0.8)
                for s, e, fa in ev_in:
                    if fa:
                        ax.plot((s + e) / 120, 1.06, "v", color="#D63C3C", ms=6, clip_on=False)
            else:
                ax.plot(t, p, color=COL[key], lw=0.6, rasterized=True)
            ax.set_ylim(0, 1); ax.set_xlim(0, n / 60); ax.set_yticks([0, 0.5, 1]); ax.tick_params(axis="y", labelsize=7)
            ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=8)
            ax.tick_params(labelbottom=False)
        ax = axs[3]
        for s, e in ann:
            ax.axvspan(s / 60, e / 60, ymin=0.55, ymax=1, color="0.6", lw=0)
        for s, e, fa in ev_in:
            ax.axvspan(s / 60, e / 60, ymin=0, ymax=0.45, color="#D63C3C" if fa else "#3FA34D", lw=0)
        ax.set_xlim(0, n / 60); ax.set_yticks([]); ax.spines["left"].set_visible(False)
        ax.set_ylabel("events", rotation=0, ha="right", va="center", fontsize=8)
        ax.set_xlabel("time in recording (min)")
        axs[0].set_title(f"Case {ci + 1}: recording closest to the {q}th percentile of seizure seconds "
                         f"(target {target:.0f} s)
this recording: {v} s in {n_sz} seizure{'s' if n_sz != 1 else ''}, "
                         f"{n / 60:.0f} min scored; fusion detects {n_det}/{n_sz}, false alarms {n_fa}",
                         loc="left", fontsize=9, pad=10)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = [Patch(color="0.82", label="annotated seizure"), Line2D([], [], color=COL["catboost"], label="CatBoost p(t)"),
         Line2D([], [], color=COL["linoss_mean"], label="LinOSS p(t)"),
         Line2D([], [], color=COL["fused_mean"], lw=1.3, label=f"fused p(t), {ev['k']} s mean"),
         Line2D([], [], ls="--", color="0.25", label=f"event threshold {ev['thr']:.3f}"),
         Patch(color="#3FA34D", label="fused event (true detection)"), Patch(color="#D63C3C", label="false alarm")]
    fig.legend(handles=h, loc="lower center", ncol=7, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 0.022))
    fig.text(0.5, 0.006, f"TUSZ v2.0.6 eval, event point 1 (k {ev['k']} s, gap {ev['gap']} s, min {ev['min_dur']} s, "
             f"frozen on dev). Candidates: {len(cands)} recordings with >= 1 detected seizure. Source: {src}.",
             ha="center", fontsize=7, color="0.35")
    fig.subplots_adjust(left=0.09, right=0.99, top=0.955, bottom=0.085)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=180)
    for ci, (q, target, i0, v) in enumerate(cases):
        print(f"case {ci + 1}: p{q} target {target:.1f} s -> {v} s")


if __name__ == "__main__":
    main()
