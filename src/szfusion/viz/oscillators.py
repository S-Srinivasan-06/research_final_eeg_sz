"""Figure: what the trained LinOSS layers compute.

  python -m szfusion.viz.oscillators --runs <work>/kaggle_out/szfusion-linoss-tusz/runs --out docs/figures/oscillators

(a) Impulse responses h_p[k] (linoss/layer.py:linoss_kernel) of five learned oscillators of seed 0, layer
    seq_blocks.0.fwd (the first whole-head block, forward direction). The five are chosen by a fixed rule: the
    oscillators at the 2nd, 25th, 50th, 75th and 98th percentiles of that layer's learned natural periods. Each curve
    is divided by its own maximum |h| (shape, not gain, is shown).
(b) Learned natural periods T = 1 / f (f = LinOSSLayer.frequencies_hz at the 1 Hz token rate, IM discretisation:
    f = atan(dt sqrt(A)) / 2 pi) of every oscillator of every LinOSS layer of the 5 seeds (per seed: per-electrode block
    2 x 32, whole-head blocks 2 x 2 x 96, i.e. 448), as histograms on log-spaced bins, against the initial range
    5-240 s of the final recipe (path.md 4.2; results.json cfg.osc_periods). Values beyond the bin range are drawn in
    the edge bins; an oscillator whose A reached 0 (relu) has f = 0 and an infinite period. The share of learned periods outside
    the initial range is printed and stated in the legend.
Inputs: best.pt and results.json of each seed folder s0..s4. Output: <out>.png (dpi 200) and <out>.svg.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from szfusion.linoss.layer import LinOSSLayer, linoss_kernel  # noqa: E402
from szfusion.linoss.model import build  # noqa: E402

KERNEL_LAGS = 240          # lags shown in (a), seconds (= the 240 s training segment)
PICK_Q = (0.02, 0.25, 0.5, 0.75, 0.98)   # period quantiles of the oscillators drawn in (a)
BINS = np.logspace(np.log10(2), np.log10(1000), 50)   # period bins (s) of (b)

plt.rcParams.update({"font.size": 8.5, "font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "svg.fonttype": "none"})


def load(run_dir: Path):
    res = json.loads((run_dir / "results.json").read_text())
    m = build(res["cfg"])
    m.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)["model"])
    return m.eval(), res


def periods(model):
    out = {}
    for name, mod in model.named_modules():
        if isinstance(mod, LinOSSLayer):
            with torch.no_grad():
                out[name] = (1 / mod.frequencies_hz(1.0)).numpy()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="folder with the seed folders s0..s4")
    ap.add_argument("--seeds", nargs="+", default=["s0", "s1", "s2", "s3", "s4"])
    ap.add_argument("--out", default="docs/figures/oscillators")
    a = ap.parse_args(argv)
    runs = Path(a.runs)
    m0, res0 = load(runs / a.seeds[0])
    lo, hi = res0["cfg"]["osc_periods"]
    lay = m0.seq_blocks[0].fwd
    with torch.no_grad():
        A, dt = lay.oscillators()
        h = linoss_kernel(A, dt, KERNEL_LAGS, lay.disc).numpy()
        per = (1 / lay.frequencies_hz(1.0)).numpy()
    order = np.argsort(per)
    pick = [int(order[int(round(q * (len(per) - 1)))]) for q in PICK_Q]

    groups = {"per-electrode block (el_blocks)": [], "whole-head blocks (seq_blocks)": []}
    for s in a.seeds:
        m, r = load(runs / s)
        assert r["cfg"]["osc_periods"] == [lo, hi], r["cfg"]
        for name, p in periods(m).items():
            groups["per-electrode block (el_blocks)" if name.startswith("el_") else "whole-head blocks (seq_blocks)"].append(p)
    allp = np.concatenate([np.concatenate(v) for v in groups.values()])
    n_inf = int((~np.isfinite(allp)).sum())       # A = relu(A_raw) = 0: no restoring force, an integrator (f = 0)
    out_lo, out_hi = float((allp < lo).mean()), float((allp > hi).mean())     # inf counts as > hi

    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={"width_ratios": [1.45, 1]})
    cols = ["#08306B", "#2171B5", "#4292C6", "#6BAED6", "#9ECAE1"]
    for c, p, q in zip(cols, pick, PICK_Q):
        ax[0].plot(np.arange(KERNEL_LAGS), h[p] / np.abs(h[p]).max(), color=c, lw=1.1,
                   label=f"P{int(q * 100):02d}: period {per[p]:.0f} s")
    ax[0].axhline(0, color="#888780", lw=0.5)
    ax[0].set_xlim(0, KERNEL_LAGS - 1)
    ax[0].set_xlabel("lag k (s)"); ax[0].set_ylabel("h_p[k] / max |h_p|")
    ax[0].set_title("(a) impulse responses, seed 0, seq_blocks.0.fwd", loc="left", fontsize=8.5)
    ax[0].legend(frameon=False, fontsize=6.8, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3)
    for (k, v), c in zip(groups.items(), ["#6A51A3", "#2171B5"]):
        ax[1].hist(np.clip(np.concatenate(v), BINS[0], BINS[-1]), bins=BINS, histtype="step", lw=1.2, color=c,
                   density=True, label=f"{k}, n = {sum(len(x) for x in v)}")
    ax[1].axvspan(lo, hi, color="#DEEBF7", zorder=0, lw=0, label=f"initial range {lo:g}-{hi:g} s")
    ax[1].set_xscale("log"); ax[1].set_xlabel("learned natural period (s)"); ax[1].set_ylabel("density")
    ax[1].set_title(f"(b) all oscillators, {len(a.seeds)} seeds", loc="left", fontsize=8.5)
    ax[1].legend(frameon=False, fontsize=6.6, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1)
    txt = f"outside the initial range:\n< {lo:g} s: {100 * out_lo:.2f} %\n> {hi:g} s: {100 * out_hi:.2f} %"
    if n_inf:
        txt += f"\n(incl. {n_inf} with A = 0,\ninfinite period)"
    ax[1].text(0.98, 0.97, txt, transform=ax[1].transAxes, ha="right", va="top", fontsize=6.6)
    fig.tight_layout()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    q = np.quantile(allp[np.isfinite(allp)], [0, 0.1, 0.5, 0.9, 1])
    print(json.dumps({"n_oscillators": int(len(allp)), "period_s_q0_10_50_90_100": [round(float(x), 2) for x in q],
                      "n_infinite_period": n_inf, "share_below_init": round(out_lo, 4), "share_above_init": round(out_hi, 4),
                      "picked_periods_s": [round(float(per[p]), 1) for p in pick]}))


if __name__ == "__main__":
    main()
