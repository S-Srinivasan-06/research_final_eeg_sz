"""Score dev and eval with a trained CatBoost committee (CPU, deterministic given the saved models).

  python -m szfusion.catboost.score --feat_dir <work>/cb_feats/tusz --models <work>/tusz/catboost --out <work>/tusz/catboost

The inference path is the same as in catboost/train.py:
  1. features.parquet is read.
  2. add_context adds ±7 windows, per recording, non-causal (context.py).
  3. Each committee scores its windows: base (20 models, 588 base columns), full (20 models, 7,056 columns) and
     sens (10 models, 7,056 columns).
  4. The models' probabilities are averaged within each committee.
  5. The score is the equal mean of the three committee means.
Recordings are processed in chunks. The context is computed per recording, so chunking does not change the result.
Output: scores_{dev,eval}.npz with score, p_base, p_full, p_sens, rec_id, patient, t0, rec_duration and y, one row
per 60 s window. These contain recording IDs and labels, so they are never committed.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from szfusion.catboost.context import add_context
from szfusion.catboost.extract import load_config
from szfusion.util.monitor import Log


def load_models(model_dir, log):
    from catboost import CatBoostClassifier
    com = {"base": [], "full": [], "sens": []}
    for p in sorted(Path(model_dir).glob("cb_*.cbm")):
        m = CatBoostClassifier(); m.load_model(str(p)); com[p.stem.split("_")[1]].append(m)
    log.event("models", **{k: len(v) for k, v in com.items()})
    assert [len(com[k]) for k in ("base", "full", "sens")] == [20, 20, 10], "expected 20 base, 20 full, 10 sens models"
    return com


def score_split(feat_dir, split, com, base_cols, full_cols, ccfg, threads, log, chunk_recs=150):
    import pandas as pd
    t = time.time()
    df = pd.read_parquet(Path(feat_dir) / split / "features.parquet")
    df = df.sort_values(["rec_id", "t0"], kind="stable").reset_index(drop=True)
    recs = df["rec_id"].unique()
    log.event("features", split=split, rows=len(df), recordings=len(recs), pos=int(df["y"].sum()))
    parts = {k: [] for k in com}; metas = []
    for ci in range(0, len(recs), chunk_recs):
        sub = df[df["rec_id"].isin(set(recs[ci:ci + chunk_recs]))]
        sub, cols = add_context(sub, base_cols, ccfg)
        assert cols == full_cols, "context column names differ from the trained models' feature_cols_full.txt"
        Xb = sub[base_cols].to_numpy(np.float32); Xf = sub[full_cols].to_numpy(np.float32)
        for k, ms in com.items():
            X = Xb if k == "base" else Xf
            parts[k].append(np.mean([m.predict_proba(X, thread_count=threads)[:, 1] for m in ms], 0))
        metas.append(sub[["rec_id", "patient", "t0", "rec_duration", "y"]].copy())
        log(f"  [{split}] recordings {min(ci + chunk_recs, len(recs))}/{len(recs)}  {time.time() - t:.0f}s")
    meta = pd.concat(metas, ignore_index=True)
    p = {k: np.concatenate(v) for k, v in parts.items()}
    return meta, np.mean(np.vstack([p["base"], p["full"], p["sens"]]), 0), p


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_dir", required=True)
    ap.add_argument("--models", required=True, help="folder with cb_*.cbm and feature_cols_{base,full}.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["dev", "eval"])
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    log = Log(out, name="score"); log.event("config", **vars(a))
    cfg = load_config()
    base_cols = (Path(a.models) / "feature_cols_base.txt").read_text().split("\n")
    full_cols = (Path(a.models) / "feature_cols_full.txt").read_text().split("\n")
    com = None
    for split in a.splits:
        f = out / f"scores_{split}.npz"
        if f.exists():
            log(f"{f.name} exists, skipping (delete it to recompute)"); continue
        com = com or load_models(a.models, log)
        t = time.time()
        meta, score, p = score_split(a.feat_dir, split, com, base_cols, full_cols, cfg["context"], a.threads, log)
        np.savez(f, score=score.astype(np.float32), **{f"p_{k}": v.astype(np.float32) for k, v in p.items()},
                 rec_id=meta["rec_id"].to_numpy().astype(str), patient=meta["patient"].to_numpy().astype(str),
                 t0=meta["t0"].to_numpy(), rec_duration=meta["rec_duration"].to_numpy(), y=meta["y"].to_numpy())
        log.event("scored", split=split, rows=len(score), sec=round(time.time() - t, 1))
    log.close()


if __name__ == "__main__":
    main()
