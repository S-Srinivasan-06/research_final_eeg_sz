"""Per-recording feature extraction (multiprocessing, resumable) -> <out>/<split>/recs/<rec_id>.npz,
then merged to <out>/<split>/features.parquet (+ intervals.json, summary.json).

EVERY window of EVERY recording is kept (no subsampling) for all splits; train negatives are subsampled
later in train_ensemble.py, AFTER temporal context is computed, so context never sees gaps.

  python -m szfusion.catboost.extract --split dev --out <work>/cb_feats/tusz --workers 14 [--max_recordings 2]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from szfusion import paths
from szfusion.catboost.features import (NAMES18, extract_window_features, feature_names, load_bipolar18,
                                        window_labels_zabihi)
from szfusion.data.tusz import ALL_MONTAGES, list_recordings, window_recording

CONFIG = paths.REPO / "configs" / "tusz_final.yaml"


def load_config(path=None) -> dict:
    with open(path or CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["catboost"]
    cfg["features"]["_fs"] = cfg["data"]["fs"]; cfg["features"]["_win_sec"] = cfg["data"]["win_sec"]
    return cfg


def _process(args):
    rec, cfg, rec_dir = args
    out = Path(rec_dir) / f"{rec.rec_id}.npz"
    if out.exists():
        return rec.rec_id, "cached", 0
    d = cfg["data"]
    try:
        x, names = load_bipolar18(rec, d)
    except Exception as e:  # unreadable EDF -> logged, skipped
        return rec.rec_id, f"error: {e!r}", 0
    X, _, t0 = window_recording(x, d["fs"], rec.seizure_intervals, d["win_sec"], d["stride_sec"], "any")
    y = window_labels_zabihi(t0, d["win_sec"], rec.seizure_intervals, d["label_abs_sec"], d["label_frac"])
    chunk = int(cfg["extract"].get("chunk_windows", 256))
    F = [extract_window_features(X[i:i + chunk], d["fs"], cfg["features"]) for i in range(0, len(y), chunk)]
    nk = len(feature_names(cfg["features"], names))
    F = np.concatenate(F) if F else np.zeros((0, nk), np.float32)
    dur = rec.duration if not math.isnan(rec.duration) else x.shape[1] / d["fs"]
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, F=F, y=y.astype(np.int8), t0=t0.astype(np.float64), rec_duration=np.float64(dur),
             patient=rec.patient, session=rec.session, montage_dir=rec.montage_dir)
    os.replace(tmp, out)  # atomic -> a killed run never leaves a half-written file
    return rec.rec_id, "ok", len(y)


def select_recordings(cfg, split, max_recordings=None, seed=0):
    d = cfg["data"]
    recs = list_recordings(paths.get("tusz_root") / "edf", split, d.get("montage_dirs") or ALL_MONTAGES)
    if max_recordings:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(recs))
        order = sorted(order, key=lambda i: not recs[i].has_seizure)  # seizure recordings first (smoke/debug)
        recs = [recs[i] for i in sorted(order[:max_recordings])]
    return recs


def merge(cfg, split, out_dir, recs):
    """Concatenate per-recording npz -> features.parquet (float32 features + meta columns)."""
    import pandas as pd
    rec_dir = Path(out_dir) / split / "recs"
    names = feature_names(cfg["features"], _channel_names(cfg))
    frames = []
    for rec in recs:
        p = rec_dir / f"{rec.rec_id}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        if len(z["y"]) == 0:
            continue
        df = pd.DataFrame(z["F"], columns=names)
        df.insert(0, "y", z["y"]); df.insert(0, "t0", z["t0"])
        df.insert(0, "rec_duration", float(z["rec_duration"]))
        df.insert(0, "montage_dir", str(z["montage_dir"])); df.insert(0, "session", str(z["session"]))
        df.insert(0, "patient", str(z["patient"])); df.insert(0, "rec_id", rec.rec_id)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["rec_id"] + names)
    out = Path(out_dir) / split
    try:
        df.to_parquet(out / "features.parquet", index=False)
        path = out / "features.parquet"
    except Exception:  # pyarrow missing -> pickle fallback
        path = out / "features.pkl"; df.to_pickle(path)
    kept = set(df["rec_id"].unique()) if len(df) else set()
    with open(out / "intervals.json", "w") as f:
        json.dump({r.rec_id: r.seizure_intervals for r in recs if r.rec_id in kept}, f)
    return path, df


def _channel_names(cfg):
    return list(NAMES18)


def load_split(out_dir, split):
    import pandas as pd
    d = Path(out_dir) / split
    df = pd.read_parquet(d / "features.parquet") if (d / "features.parquet").exists() else \
        pd.read_pickle(d / "features.pkl")
    with open(d / "intervals.json") as f:
        iv = {k: [tuple(v) for v in vs] for k, vs in json.load(f).items()}
    return df, iv


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "dev", "eval"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--max_recordings", type=int, default=None)
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--no_merge", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.data_root:
        cfg["data"]["root"] = a.data_root
    recs = select_recordings(cfg, a.split, a.max_recordings, cfg.get("seed", 0))
    rec_dir = Path(a.out) / a.split / "recs"; rec_dir.mkdir(parents=True, exist_ok=True)
    t = time.time()
    jobs = [(r, cfg, str(rec_dir)) for r in recs]
    errors, n_win = [], 0
    if a.workers > 1:
        from multiprocessing import Pool
        with Pool(a.workers) as pool:
            it = pool.imap_unordered(_process, jobs, chunksize=1)
            for k, (rid, status, n) in enumerate(it):
                n_win += n
                if status.startswith("error"):
                    errors.append({"rec_id": rid, "error": status})
                if k % 50 == 0:
                    print(f"[{a.split}] {k + 1}/{len(jobs)} {time.time() - t:.0f}s", flush=True)
    else:
        for k, j in enumerate(jobs):
            rid, status, n = _process(j)
            n_win += n
            if status.startswith("error"):
                errors.append({"rec_id": rid, "error": status})
    summary = {"split": a.split, "n_recordings": len(recs), "n_new_windows": n_win, "errors": errors,
               "sec": round(time.time() - t, 1), "config": cfg}
    if not a.no_merge:
        path, df = merge(cfg, a.split, a.out, recs)
        summary.update(features_file=str(path), n_windows=int(len(df)),
                       n_pos=int(df["y"].sum()) if len(df) else 0,
                       n_features=len(feature_names(cfg["features"], _channel_names(cfg))))
    with open(Path(a.out) / a.split / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, default=str)[:800])
    return summary


if __name__ == "__main__":
    main()
