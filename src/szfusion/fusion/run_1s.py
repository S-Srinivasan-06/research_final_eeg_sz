"""1 s fusion of the CatBoost and LinOSS per-second tracks: dev selection, then one eval scoring.

  python -m szfusion.fusion.run_1s dev  --tokens T --catboost C --linoss R0 R1 R2 R3 R4 --out O
  python -m szfusion.fusion.run_1s eval --tokens T --catboost C --linoss R0 ... R4 --out O    (refuses to re-score)

T: token store root; C: folder with scores_dev.npz / scores_eval.npz of the CatBoost committee; R*: LinOSS runs (each
with prob_dev.npy / prob_eval.npy); O: output folder.

Leakage: CatBoost was trained on train only, and dev is unseen by it. The LinOSS seeds used dev only to choose their
best epoch (one number per seed), so their dev scores are slightly optimistic. That can only push the stacker to
over-weight LinOSS. The stacker is fitted on dev.

Dev stage (path.md 4.4), for every LinOSS configuration (each seed and the 5-seed mean `linoss_mean`):
  1. Fusion selection: 5-fold patient-grouped CV over the candidates (stacker.py), scored by out-of-fold per-second
     AUPRC. Ties within 0.002 go to the simpler candidate, in the order late, lr, lr_ctx.
  2. The chosen stacker is refitted on all of dev and written to selection.json as coefficients.
  3. On the out-of-fold dev predictions: the per-second F1 threshold (the best dev F1), and event post-processing at
     two false-alarm budgets (protocol_1s.tune_events).
Eval stage: applies the frozen selection.json to eval and writes eval_results.json once.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

from szfusion.fusion import protocol_1s as P
from szfusion.fusion.stacker import CANDS, Stacker, logit
from szfusion.util.monitor import Log


def oof(cand, w, zc, zl, y, rec, folds):
    out = np.zeros(len(y))
    for tr, te in folds:
        out[te] = Stacker(cand, w).fit(zc[tr], zl[tr], y[tr], rec[tr]).predict(zc[te], zl[te], rec[te])
    return out


def select_fusion(D, lin_name, log):
    y, rec = D["y"], D["rec"]
    zc, zl = logit(D["tracks"]["catboost"]), logit(D["tracks"][lin_name])
    folds = list(GroupKFold(5).split(zc, y, D["patient"]))
    table, best = [], None
    for cand in CANDS:
        for w in ([round(0.1 * i, 1) for i in range(11)] if cand == "late" else [None]):
            p = oof(cand, w, zc, zl, y, rec, folds)
            ap = float(average_precision_score(y, p))
            table.append({"cand": cand, "w": w, "oof_auprc": ap})
            if best is None or ap > best[0] + (0.002 if cand != "late" or best[1] != "late" else 0):
                best = (ap, cand, w, p)
    log.table(f"fusion CV on dev ({lin_name}): out-of-fold per-second AUPRC", ["candidate", "w", "oof AUPRC"],
              [[r["cand"], "-" if r["w"] is None else r["w"], round(r["oof_auprc"], 4)] for r in table])
    return best[1], best[2], best[3], table


def add_mean(D, names):
    if len(names) > 1:
        D["tracks"]["linoss_mean"] = np.mean([D["tracks"][n] for n in names], 0)
        return names + ["linoss_mean"]
    return names


def run(stage, tokens, catboost, linoss, out, force=False):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    log = Log(out, name=f"fuse_{stage}")
    dirs = {f"linoss_s{i}": Path(r) for i, r in enumerate(linoss)}
    D = P.load_split(stage, tokens, Path(catboost) / f"scores_{stage}.npz", dirs)
    log.event(f"{stage}_set", **D["info"])
    lin = add_mean(D, list(dirs))
    if stage == "dev":
        sel, rows = {"linoss_runs": [str(Path(r).name) for r in linoss], "dev_info": D["info"], "configs": {}}, {}
        cfgs = {"catboost": D["tracks"]["catboost"], **{n: D["tracks"][n] for n in lin}}
        for n in lin:
            cand, w, p_oof, cv = select_fusion(D, n, log)
            fused = "fused" + n[len("linoss"):]
            st = Stacker(cand, w).fit(logit(D["tracks"]["catboost"]), logit(D["tracks"][n]), D["y"], D["rec"])
            sel["configs"][fused] = {"linoss": n, "stacker": st.to_dict(), "cv": cv}
            cfgs[fused] = p_oof
        for name, p in cfgs.items():
            t = time.time()
            f1_thr, _ = P.best_f1_threshold(D["y"], p)
            ev, _, grid = P.tune_events(D, p, P.FA_BUDGET_24H)
            ev2, _, _ = P.tune_events(D, p, P.FA_BUDGET2_24H, grid)
            sel["configs"].setdefault(name, {}).update({"f1_thr": f1_thr, "ev": ev, "ev2": ev2})
            rows[name] = P.full_scores(D, p, sel["configs"][name])
            log.event("dev_selection", config=name, f1_thr=round(f1_thr, 4), ev=ev, ev2=ev2, sec=round(time.time() - t, 1))
        (out / "selection.json").write_text(json.dumps(sel, indent=2, default=float))
        (out / "dev_results.json").write_text(json.dumps({"info": D["info"], "results": rows}, indent=2, default=float))
    else:
        f_out = out / "eval_results.json"
        if f_out.exists() and not force:
            log("eval_results.json exists: eval was already scored once. Not re-scoring."); log.close(); return
        sel = json.loads((out / "selection.json").read_text())
        zc = logit(D["tracks"]["catboost"])
        cfgs = {"catboost": D["tracks"]["catboost"], **{n: D["tracks"][n] for n in lin}}
        for n in lin:
            fused = "fused" + n[len("linoss"):]
            st = Stacker.from_dict(sel["configs"][fused]["stacker"])
            cfgs[fused] = st.predict(zc, logit(D["tracks"][n]), D["rec"])
        rows = {name: P.full_scores(D, p, sel["configs"][name]) for name, p in cfgs.items()}
        np.savez_compressed(out / "eval_tracks.npz", rec=D["rec"], y=D["y"],      # per-second; for SzCORE + figures
                            **{k: v.astype(np.float32) for k, v in cfgs.items()})
        f_out.write_text(json.dumps({"info": D["info"], "results": rows}, indent=2, default=float))
    log.table(f"{stage.upper()} per-second results", ["config", "AUROC", "AUPRC", "F1", "ev sens", "FA/24h"],
              [[k, round(r["auroc"], 4), round(r["auprc"], 4), round(r["f1"], 4), round(r["ev_event_sens"], 3),
                round(r["ev_fa_per_24h"], 2)] for k, r in rows.items()])
    log.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["dev", "eval"])
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--catboost", required=True)
    ap.add_argument("--linoss", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true", help="only to regenerate the same eval file (never to re-select)")
    a = ap.parse_args()
    run(a.stage, a.tokens, a.catboost, a.linoss, a.out, a.force)


if __name__ == "__main__":
    main()
