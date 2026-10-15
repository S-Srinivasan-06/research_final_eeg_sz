"""Fusion-only 60 s FROC against the published CatBoost ensemble curve (Zabihi et al. 2026, Fig. 3).

  python -m szfusion.viz.froc_vs_paper --froc docs/figures/froc_60s_devrun.json --out docs/figures/froc_60s_fusion_vs_paper

The fusion curve and its dev operating point come from viz/froc.py (post-processing frozen at the dev choice,
threshold swept on eval). The paper's curve is docs/figures/zabihi2026_fig3_digitized.json, digitized from the
published figure (method and ±0.02 accuracy stated in that file). It is drawn dotted, as a visual reference only.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("--froc", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--label", default="(development run, reproduced exactly by this code)")
a = ap.parse_args()
F = json.loads(Path(a.froc).read_text())["models"]["look_mean_cb"]
Z = json.loads((Path(__file__).resolve().parents[3] / "docs/figures/zabihi2026_fig3_digitized.json").read_text())
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(7.2, 4.6))
zx, zy = zip(*Z["points_fa_per_h_sens"])
ax.plot(zx, zy, ":", color="#D4537E", lw=2.0, marker="o", ms=3,
        label="CatBoost ensemble, published curve (Zabihi et al. 2026, Fig. 3; TUSZ v2.0.3; digitized)")
ax.plot(0.682, 0.749, "*", color="#D4537E", ms=15, mec="black", mew=0.6, zorder=5,
        label="published operating point: Se 0.749 at 0.682 FA/h (thr 0.44)")
fx, fy, _ = zip(*F["curve"])
ax.plot(fx, fy, "-", color="#0F6E56", lw=2.0, marker=".", ms=3, label="CatBoost × LinOSS fusion (TUSZ v2.0.6 eval)")
op = F["dev_operating_point"]
ax.plot(op["fa_per_h"], op["event_sens"], "o", color="#0F6E56", ms=9, mec="white", mew=1.4, zorder=6,
        label=f"fusion operating point chosen on dev: Se {op['event_sens']:.3f} at {op['fa_per_h']:.3f} FA/h")
ax.set_xlim(0, 4.0); ax.set_ylim(0, 1.0); ax.grid(alpha=0.25)
ax.set_xlabel("False alarms / hour (event level, micro)"); ax.set_ylabel("Event sensitivity (micro)")
ax.set_title(f"60 s test, event-level FROC {a.label}", loc="left", fontsize=9)
ax.legend(frameon=False, fontsize=7.5, loc="lower right")
fig.tight_layout(); o = Path(a.out)
fig.savefig(o.with_suffix(".svg")); fig.savefig(o.with_suffix(".png"), dpi=200)
