"""Why the fusion helps: how much the CatBoost and LinOSS branches share their errors, on the frozen 1 s eval tracks.

  python -m szfusion.eval.mechanism --fusion_1s <work>/tusz/fusion_1s --out results/tusz/fusion_mechanism.json

Input: eval_tracks.npz (per scored second: rec, y, catboost, linoss_mean, fused_mean, ...) and selection.json of the
1 s fusion stage (fusion/run_1s.py). Nothing is selected here; every threshold is either frozen on dev or defined by
a fixed false-alarm budget, and the numbers are descriptive statistics of the eval tracks.
Branches: CatBoost = `catboost`; LinOSS = `linoss_mean` (the 5-seed mean, the input of `fused_mean`); fusion =
`fused_mean`.

Quantities:
  rho_background, rho_ictal  Pearson correlation of the two branch logits z = log(p / (1 - p)) (p clipped to
      [1e-5, 1 - 1e-5], stacker.logit) over the background seconds (y = 0) and over the ictal seconds (y = 1).
  shared false alarms        at each branch's dev-F1 threshold (selection.json configs[*].f1_thr, frozen on dev), a
      false-alarm second is a background second with p > threshold. share_cb_fa_also_linoss = |FA_C ∩ FA_L| / |FA_C|,
      and the reverse.
  ictal caught at a 1 % background budget   for each track, the threshold is the 99th percentile of its own
      probabilities over the eval background seconds (so 1 % of background seconds lie above it), and the value is the
      fraction of ictal seconds above that threshold: the ROC curve read at FPR = 0.01. The threshold uses eval labels,
      so this is a description of the eval ROC curves, not an operating point.
  separation gain sqrt(2 / (1 + rho))   if two scores each separate the classes by d' (equal mean gap and variance)
      and their within-class correlation is rho, their average separates them by d' * sqrt(2 / (1 + rho)): the mean gap
      is unchanged and the variance shrinks to (1 + rho) / 2. Evaluated at rho_background and at rho_ictal. It is the
      ceiling of what an equal-weight average of two equally good branches can gain, not a prediction for the stacker.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from szfusion.fusion.stacker import logit

BUDGET_FPR = 0.01          # background budget for the "ictal caught" figure: 1 % of the eval background seconds


def provenance(cmd: str) -> dict:
    """git commit (with a '+dirty' flag), date, command. No path is stored (RULES.md 4)."""
    repo = Path(__file__).resolve().parents[3]
    try:
        c = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001
        c = "unknown"
    return {"git_commit": c, "date": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "command": cmd}


def ictal_at_budget(y, p, fpr=BUDGET_FPR):
    thr = float(np.quantile(p[y == 0], 1 - fpr))
    return {"thr": thr, "ictal_frac": float((p[y == 1] > thr).mean()),
            "background_frac_above": float((p[y == 0] > thr).mean())}


def analyse(fusion_dir) -> dict:
    d = Path(fusion_dir)
    T = np.load(d / "eval_tracks.npz")
    sel = json.loads((d / "selection.json").read_text())["configs"]
    y = T["y"].astype(np.int8)
    pc, pl, pf = (T[k].astype(np.float64) for k in ("catboost", "linoss_mean", "fused_mean"))
    zc, zl = logit(pc), logit(pl)
    bg, ic = y == 0, y == 1
    rho_bg = float(np.corrcoef(zc[bg], zl[bg])[0, 1])
    rho_ic = float(np.corrcoef(zc[ic], zl[ic])[0, 1])
    tc, tl = sel["catboost"]["f1_thr"], sel["linoss_mean"]["f1_thr"]
    fa_c, fa_l = bg & (pc > tc), bg & (pl > tl)
    both = int((fa_c & fa_l).sum())
    return {
        "inputs": {"tracks": "eval_tracks.npz (1 s fusion stage, eval)", "catboost": "catboost",
                   "linoss": "linoss_mean (5-seed mean)", "fusion": "fused_mean",
                   "scored_seconds": int(len(y)), "ictal_seconds": int(ic.sum()), "background_seconds": int(bg.sum())},
        "logit_correlation": {"rho_background": rho_bg, "rho_ictal": rho_ic},
        "false_alarm_seconds_at_dev_f1_threshold": {
            "thr_catboost": tc, "thr_linoss": tl,
            "n_fa_catboost": int(fa_c.sum()), "n_fa_linoss": int(fa_l.sum()), "n_fa_both": both,
            "share_cb_fa_also_linoss": both / max(int(fa_c.sum()), 1),
            "share_linoss_fa_also_cb": both / max(int(fa_l.sum()), 1)},
        f"ictal_caught_at_{int(BUDGET_FPR * 100)}pct_background": {
            "catboost": ictal_at_budget(y, pc), "linoss_mean": ictal_at_budget(y, pl),
            "fused_mean": ictal_at_budget(y, pf)},
        "separation_gain_sqrt_2_over_1_plus_rho": {
            "at_rho_background": float(np.sqrt(2 / (1 + rho_bg))), "at_rho_ictal": float(np.sqrt(2 / (1 + rho_ic)))},
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fusion_1s", required=True, help="1 s fusion folder with eval_tracks.npz and selection.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", default="official", help="label stored in the provenance (official / s1_gate)")
    a = ap.parse_args(argv)
    res = analyse(a.fusion_1s)
    cmd = "python -m szfusion.eval.mechanism --fusion_1s <work>/" + \
          ("tusz/fusion_1s" if a.source == "official" else "s1_gate/fusion_1s") + f" --out {a.out} --source {a.source}"
    out = {"provenance": {**provenance(cmd), "source": a.source}, **res}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    json.dump(res, sys.stdout, indent=1); print()


if __name__ == "__main__":
    main()
