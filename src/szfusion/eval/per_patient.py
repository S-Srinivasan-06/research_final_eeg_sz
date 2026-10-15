"""TUSZ eval per-patient event sensitivity and false alarms at event point 1: the fusion against each branch.

  python -m szfusion.eval.per_patient [--fusion_1s F] [--tokens T] [--out results/tusz/per_patient.json]
                                      [--fig docs/figures/per_patient.png]

Input and event rules as in eval/by_type.py (frozen 1 s fusion outputs; event point 1 = selection.json
configs.<model>.ev; any-overlap detection and false alarms as fusion/protocol_1s.event_scores).
Per patient (all scored recordings of the patient pooled):
  n_events     true events (runs of seizure seconds)
  event_sens   detected events / n_events (patients with n_events >= 1 only)
  n_fa         false alarms (predicted events overlapping no seizure second)
No patient ID is written or drawn (TUSZ data-use agreement; RULES.md 4): patients are listed by rank only. Rank order:
patients with events by fused_mean sensitivity (descending), then by n_events (descending), then by the CatBoost and
LinOSS sensitivities (descending); then the patients without events by fused_mean false alarms (descending). The ranks
are not stable across runs and carry no identity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from szfusion import paths  # noqa: E402
from szfusion.eval import by_type as BT  # noqa: E402
from szfusion.eval.mechanism import provenance  # noqa: E402

MODELS = ["fused_mean", "catboost", "linoss_mean"]
LABEL = {"fused_mean": "fusion", "catboost": "CatBoost branch", "linoss_mean": "LinOSS branch (5-seed mean)"}
COLOR = {"fused_mean": "#2a78d6", "catboost": "#eb6834", "linoss_mean": "#1baf7a"}   # dataviz reference slots 1-3
POINT = "ev"                                                                        # event point 1


def per_patient(D):
    _, pid = np.unique(D["patient"], return_inverse=True)
    n_pat = int(pid.max()) + 1
    ts, _ = D["te"]
    y = D["y"].astype(np.int64); cy = np.r_[0, np.cumsum(y)]
    n_ev = np.bincount(pid[ts], minlength=n_pat)
    sens, fa = {}, {}
    for m in MODELS:
        st, en, _ = BT.predicted_mask(D, m, POINT)
        det, _, _ = BT.detection(D, m, POINT)
        nd = np.bincount(pid[ts], weights=det.astype(float), minlength=n_pat)
        sens[m] = np.where(n_ev > 0, nd / np.maximum(n_ev, 1), np.nan)
        fa[m] = np.bincount(pid[st[(cy[en] - cy[st]) == 0]], minlength=n_pat)
    return n_pat, n_ev, sens, fa


def order(n_ev, sens, fa):
    w = np.flatnonzero(n_ev > 0); wo = np.flatnonzero(n_ev == 0)
    w = sorted(w, key=lambda i: (-sens["fused_mean"][i], -n_ev[i], -sens["catboost"][i], -sens["linoss_mean"][i]))
    wo = sorted(wo, key=lambda i: (-fa["fused_mean"][i], -fa["catboost"][i], -fa["linoss_mean"][i]))
    return list(w), list(wo)


def figure(path, n_ev, sens, w):
    x = np.arange(1, len(w) + 1)
    fig, (ax, axn) = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True, gridspec_kw={"height_ratios": [2.4, 1]})
    off = {"fused_mean": 0.0, "catboost": -0.22, "linoss_mean": 0.22}
    for m in ("catboost", "linoss_mean", "fused_mean"):
        v = sens[m][w]
        ax.scatter(x + off[m], v, s=46 if m == "fused_mean" else 30, color=COLOR[m], label=LABEL[m],
                   edgecolor="white", linewidth=1.0, zorder=3 if m == "fused_mean" else 2,
                   marker="o" if m == "fused_mean" else ("s" if m == "catboost" else "^"))
    ax.set_ylim(-0.05, 1.05); ax.set_ylabel("event sensitivity\n(per patient)")
    ax.grid(axis="y", color="#e4e3df", lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False); axn.spines[s].set_visible(False)
    h, lab = ax.get_legend_handles_labels()
    o = [lab.index(LABEL[m]) for m in MODELS]
    ax.legend([h[i] for i in o], [lab[i] for i in o], loc="upper right", frameon=False, fontsize=8.5, ncol=3,
              bbox_to_anchor=(1.0, 1.13))
    ax.set_title("TUSZ eval, event point 1 (dev-frozen): per-patient event sensitivity, patients with >= 1 seizure",
                 loc="left", fontsize=10)
    axn.bar(x, n_ev[w], width=0.7, color="#9a9890")
    axn.set_ylabel("events"); axn.set_xlabel("patient rank (by fusion sensitivity; no identity)")
    axn.grid(axis="y", color="#e4e3df", lw=0.6); axn.set_axisbelow(True)
    for xi, n in zip(x, n_ev[w]):
        axn.text(xi, n, str(int(n)), ha="center", va="bottom", fontsize=7, color="#52514e")
    axn.set_ylim(0, n_ev[w].max() * 1.18)
    axn.set_xticks(x); axn.tick_params(axis="x", labelsize=7.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    work = paths.get("work_dir")
    ap = argparse.ArgumentParser()
    ap.add_argument("--fusion_1s", default=None)
    ap.add_argument("--tokens", default=str(work / "tokens" / "tusz"))
    ap.add_argument("--out", default="results/tusz/per_patient.json")
    ap.add_argument("--fig", default="docs/figures/per_patient.png")
    a = ap.parse_args(argv)
    fdir = Path(a.fusion_1s) if a.fusion_1s else BT.default_fusion_dir()
    src = BT.source_label(fdir)
    D = BT.load(fdir, a.tokens)
    n_pat, n_ev, sens, fa = per_patient(D)
    w, wo = order(n_ev, sens, fa)
    rows = [{"rank": r + 1, "n_events": int(n_ev[i]),
             **{f"event_sens_{m}": (None if np.isnan(sens[m][i]) else float(sens[m][i])) for m in MODELS},
             **{f"n_fa_{m}": int(fa[m][i]) for m in MODELS}} for r, i in enumerate(w + wo)]
    summ = {}
    for m in MODELS:
        s = sens[m][w]
        summ[m] = {"median_patient_sens": float(np.median(s)), "mean_patient_sens": float(np.mean(s)),
                   "quartiles_patient_sens": [float(q) for q in np.percentile(s, [25, 50, 75])],
                   "n_patients_sens_0": int((s == 0).sum()), "n_patients_sens_1": int((s == 1).sum()),
                   "pooled_event_sens": float(np.nansum(sens[m][w] * n_ev[w]) / n_ev[w].sum()),
                   "total_fa": int(fa[m].sum()), "n_patients_with_fa": int((fa[m] > 0).sum()),
                   "max_fa_one_patient": int(fa[m].max()),
                   "fa_in_patients_without_events": int(fa[m][wo].sum())}
    paired = {}
    for b in ("catboost", "linoss_mean"):
        d = sens["fused_mean"][w] - sens[b][w]; f = fa["fused_mean"] - fa[b]
        paired[f"fused_mean-{b}"] = {"sens_patients_higher": int((d > 0).sum()), "sens_patients_equal": int((d == 0).sum()),
                                     "sens_patients_lower": int((d < 0).sum()),
                                     "fa_patients_fewer": int((f < 0).sum()), "fa_patients_equal": int((f == 0).sum()),
                                     "fa_patients_more": int((f > 0).sum())}
    out = {"provenance": {**provenance(BT.command("per_patient", src, a.out, f" --fig {a.fig}")), "source": src},
           "definitions": {"data": "TUSZ v2.0.6 eval, scored seconds of the 1 s fusion stage",
                           "event_point": "event point 1 (configs.<model>.ev), frozen on dev",
                           "event_sens": "detected / true events of the patient (any overlap, protocol_1s)",
                           "n_fa": "predicted events of the patient overlapping no seizure second",
                           "ranks": "no patient IDs (TUSZ-restricted); rank = order described in per_patient.py; "
                                    "patients without events follow, ranked by fused_mean false alarms"},
           "n_patients": n_pat, "n_patients_with_events": len(w), "n_events": int(n_ev.sum()),
           "events_per_patient_quartiles": [float(q) for q in np.percentile(n_ev[w], [0, 25, 50, 75, 100])],
           "summary": summ, "paired_patients": paired, "patients_by_rank": rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    Path(a.fig).parent.mkdir(parents=True, exist_ok=True)
    figure(a.fig, n_ev, sens, w)
    print(json.dumps({k: out[k] for k in ("n_patients", "n_patients_with_events", "n_events",
                                          "events_per_patient_quartiles", "summary", "paired_patients")}, indent=1))


if __name__ == "__main__":
    main()
