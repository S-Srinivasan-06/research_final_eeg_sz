"""Event-level FROC on the 60 s test: event sensitivity against false alarms per hour (as Zabihi et al. 2026, Fig. 3).

  python -m szfusion.viz.froc --unit60 O60 --tokens T --catboost C --feat_dir F --out docs/figures/froc_60s [--label L]

Inputs are the frozen outputs of units/overlooker.py (selection.json, eval_scores.npz). For each model (CatBoost
branch, LinOSS branch, CatBoost × LinOSS fusion):
  * the post-processing (smoothing, dilation, merge gap, 10 s minimum event, >= 10 s detection overlap) is frozen at
    that model's dev choice (windows60.tune), and only the threshold is swept over the eval scores. This draws a
    curve, like a ROC curve; points on it are not selections.
  * the dev-chosen operating point (the threshold chosen on dev as well) is marked with a circle.
  * scoring = the paper's event scorer (catboost/postprocess.score_events, micro-averaged over recordings).
The star is the published operating point of the CatBoost ensemble (Zabihi et al. 2026: 0.749 at 0.682 FA/h,
TUSZ v2.0.3). Its full curve is not reproduced, because only that point is reported numerically.
Writes <out>.svg, <out>.png and <out>.json (every point).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from szfusion.catboost import postprocess
from szfusion.units import windows60

PAPER = (0.682, 0.749)
MODELS = [("look_mean_cb", "CatBoost × LinOSS fusion", "#0F6E56"), ("catboost", "CatBoost branch", "#D4537E"),
          ("look_mean", "LinOSS branch", "#378ADD")]
sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731


def curve(meta, iv, p, prm, n=90):
    pts = []
    for th in np.unique(np.quantile(p, np.r_[np.linspace(0.5, 0.99, n - 20), np.linspace(0.991, 0.9999, 20)])):
        q = dict(prm, threshold=float(th))
        _, _, ev = postprocess.run(meta, p, q, windows60.WIN)
        r = postprocess.score_events(meta, ev, iv, q["min_event_overlap_sec"])
        pts.append((r["fa_per_h"], r["event_sens"], float(th)))
    return sorted(pts)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit60", required=True); ap.add_argument("--tokens", required=True)
    ap.add_argument("--catboost", required=True); ap.add_argument("--feat_dir", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--label", default="")
    a = ap.parse_args()
    sel = json.loads((Path(a.unit60) / "selection.json").read_text())
    Z = np.load(Path(a.unit60) / "eval_scores.npz")
    _, _, _, _, meta, iv = windows60.windows("eval", a.tokens, Path(a.catboost) / "scores_eval.npz", a.feat_dir)
    meta = meta.reset_index(drop=True)
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    out = {"label": a.label, "paper_point": PAPER, "models": {}}
    for key, name, col in MODELS:
        z = Z[key]; p = sig(np.where(np.isnan(z), np.nanmin(z), z))
        prm = sel["post"][key]
        pts = curve(meta, iv, p, prm)
        _, _, ev = postprocess.run(meta, p, prm, windows60.WIN)
        op = postprocess.score_events(meta, ev, iv, prm["min_event_overlap_sec"])
        fa, se, _ = zip(*pts)
        ax.plot(fa, se, "-", color=col, lw=1.6, marker=".", ms=3, label=f"{name} (threshold sweep)")
        ax.plot(op["fa_per_h"], op["event_sens"], "o", color=col, ms=8, mec="white", mew=1.2)
        at = max([s for f, s, _ in pts if f <= PAPER[0]], default=float("nan"))
        need = min([f for f, s, _ in pts if s >= PAPER[1]], default=float("nan"))
        out["models"][key] = {"name": name, "post_processing": prm, "curve": pts,
                              "dev_operating_point": {"fa_per_h": op["fa_per_h"], "event_sens": op["event_sens"]},
                              "sens_at_paper_fa": at, "fa_needed_for_paper_sens": need}
    ax.plot([], [], "o", color="#444441", ms=7, mec="white", label="operating point chosen on dev")
    ax.plot(*PAPER, "*", color="#BA7517", ms=16, mec="black", mew=0.6, zorder=5,
            label="CatBoost ensemble, published (Zabihi et al. 2026; TUSZ v2.0.3)")
    ax.annotate("thr=0.44\nFA/h=0.68\nSe=0.75", PAPER, xytext=(PAPER[0] + 0.35, PAPER[1] + 0.12), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlim(0, 4.0); ax.set_ylim(0, 1.0); ax.grid(alpha=0.25)
    ax.set_xlabel("False alarms / hour (event level, micro)"); ax.set_ylabel("Event sensitivity (micro)")
    ax.set_title(f"60 s test, TUSZ v2.0.6 eval {a.label}".strip(), loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    o = Path(a.out); o.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(o.with_suffix(".svg")); fig.savefig(o.with_suffix(".png"), dpi=200)
    o.with_suffix(".json").write_text(json.dumps(out, indent=1, default=float))
    for k, m in out["models"].items():
        print(f"{m['name']:28s} dev point {m['dev_operating_point']['event_sens']:.3f} @ {m['dev_operating_point']['fa_per_h']:.3f} FA/h"
              f" | sens at <=0.682 FA/h {m['sens_at_paper_fa']:.3f} | FA/h for >=0.749: {m['fa_needed_for_paper_sens']:.2f}")


if __name__ == "__main__":
    main()
