"""CHB-MIT scoring under the HyBrain protocol (path.md stage C3): 12 s clip detection and 1 s point-wise detection.

  python -m szfusion.eval.chbmit_scores dev  --meta M --linoss L0 ... L4 --catboost C --out O
  python -m szfusion.eval.chbmit_scores test --meta M --linoss L0 ... L4 --catboost C --out O   (refuses to re-score)

M: store_meta of the training kernel (clips_{dev,test}.npz, y_sec.npy). L*: LinOSS seed folders (prob_dev.npy,
prob_test.npy: per-second probabilities, NaN outside the clips). C: CatBoost folder (scores_dev.npz,
scores_test.npz: one row per clip, keyed by rec_id and t0 = 12k).
Scores:
  LinOSS clip   z_L = log-mean-exp of the 12 per-second logits (clips12.lme); 5-seed ensemble = mean of seed logits
  CatBoost clip z_C = logit of the committee score of the clip
  fused clip    WindowStacker on [z_C, z_L, z_C·z_L], fitted on dev clips
  per second    LinOSS: its per-second logit. CatBoost: the logit of its clip, for every second of the clip.
                Fused: WindowStacker on [z_C, z_L(t), z_C·z_L(t)], fitted on dev seconds.
Selection (dev only): the stacker coefficients, and the F1 threshold (the best dev F1, protocol_1s.best_f1_threshold;
HyBrain: "the threshold that maximises dev F1"). The dev rows are 5-fold patient-grouped out-of-fold predictions.
Metrics on test: AUROC, AUPRC, F1, precision, recall (clips: the authors' labels; seconds: summary.txt labels).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from szfusion.fusion.protocol_1s import best_f1_threshold
from szfusion.fusion.stacker import WindowStacker
from szfusion.units.clips12 import lme, logit

sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731


def metrics(y, p, thr):
    pr = p > thr
    tp = int((pr & (y == 1)).sum()); fp = int((pr & (y == 0)).sum()); fn = int((~pr & (y == 1)).sum())
    return {"auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1), "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "thr": float(thr), "n": int(len(y)), "pos": int(y.sum())}


def load(split, a):
    C = np.load(Path(a.meta) / f"clips_{split}.npz")
    Y = np.load(Path(a.meta) / "y_sec.npy")
    index = json.loads((Path(a.meta) / "index.json").read_text())
    start, yc, pat = C["start"], C["y"], C["patient"]
    rows = start[:, None] + np.arange(12)
    zl = {}
    for i, r in enumerate(a.linoss):
        p = np.load(Path(r) / f"prob_{split}.npy").astype(np.float64)[rows]
        assert not np.isnan(p).any(), f"{r}: NaN inside listed {split} clips"
        zl[f"s{i}"] = logit(p)                                                    # (n_clips, 12)
    zl["mean"] = np.mean([zl[f"s{i}"] for i in range(len(a.linoss))], 0)
    cb = np.load(Path(a.catboost) / f"scores_{split}.npz")                       # one row per clip
    score = {(r, int(round(t0))): s for r, t0, s in zip(cb["rec_id"], cb["t0"], cb["score"])}
    offs = np.array([e["offset"] for e in index]); ids = [e["rec_id"] for e in index]   # index is sorted by offset
    ri = np.searchsorted(offs, start, side="right") - 1
    zc = logit(np.array([score[(ids[i], int(st - offs[i]))] for i, st in zip(ri, start)], np.float64))
    return yc, Y[rows], pat, zl, zc


def oof(zc, zl, y, groups):
    o = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(y, y, groups):
        o[te] = WindowStacker().fit(zc[tr], zl[tr], y[tr]).predict(zc[te], zl[te])
    return o


def run(stage, a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    f_sel, f_res = out / "selection.json", out / f"{stage}_results.json"
    if stage == "test" and f_res.exists():
        print("test already scored once; not re-scoring"); return
    yc, ys, pat, zl, zc = load("dev" if stage == "dev" else "test", a)
    sel = json.loads(f_sel.read_text()) if stage == "test" else {"thr": {}, "stack": {}}
    res = {}
    for task in ("clip12", "point1s"):
        y = yc if task == "clip12" else ys.ravel()
        g = pat if task == "clip12" else np.repeat(pat, 12)
        zc_t = zc if task == "clip12" else np.repeat(zc, 12)
        rows = {"catboost": sig(zc_t)}
        for k, z in zl.items():
            zl_t = lme(z) if task == "clip12" else z.ravel()
            rows[f"linoss_{k}"] = sig(zl_t)
            name = f"fused_{k}"
            if stage == "dev":
                rows[name] = oof(zc_t, zl_t, y, g)
                sel["stack"][f"{task}/{name}"] = WindowStacker().fit(zc_t, zl_t, y).to_dict()
            else:
                rows[name] = WindowStacker.from_dict(sel["stack"][f"{task}/{name}"]).predict(zc_t, zl_t)
        for name, p in rows.items():
            if stage == "dev":
                sel["thr"][f"{task}/{name}"] = best_f1_threshold(y, p)[0]
            res[f"{task}/{name}"] = metrics(y, p, sel["thr"][f"{task}/{name}"])
        for grp in ("linoss", "fused"):                                              # mean ± sd over seeds
            ks = [f"{task}/{grp}_s{i}" for i in range(len(a.linoss))]
            res[f"{task}/{grp}_seeds"] = {m: [float(np.mean([res[k][m] for k in ks])), float(np.std([res[k][m] for k in ks], ddof=1))]
                                          for m in ("auroc", "auprc", "f1")}
    if stage == "dev":
        f_sel.write_text(json.dumps(sel, indent=2))
    f_res.write_text(json.dumps(res, indent=2))
    for k, r in res.items():
        if "seeds" in k:
            print(f"{k:30s} " + "  ".join(f"{m} {v[0]:.3f}±{v[1]:.3f}" for m, v in r.items()))
        elif k.endswith(("catboost", "_mean")):
            print(f"{k:30s} AUROC {r['auroc']:.3f}  AUPRC {r['auprc']:.3f}  F1 {r['f1']:.3f}  (n={r['n']}, pos={r['pos']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["dev", "test"])
    ap.add_argument("--meta", required=True)
    ap.add_argument("--linoss", nargs="+", required=True)
    ap.add_argument("--catboost", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a.stage, a)


if __name__ == "__main__":
    main()
