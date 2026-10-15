"""CHB-MIT store and clip tables under the HyBrain split (path.md stage C2, input side).

  python -m szfusion.chbmit_store --rec <dir(s) with rec/*.npz> --markers <HyBrain marker dir> --out <out>

Inputs: the per-recording files of tokens_chbmit.py, and the authors' 12 s split lists
{train,dev,test}Set_seq2seq_12s_{sz,nosz}.txt (github.com/hhyy0401/seizure @ 694bdd4d, src/data/file_markers_chb).
Each list line is "<recording>.edf_<k>.h5,<label>": clip k = seconds [12k, 12k + 12), and label 1 if the clip overlaps
a seizure. The split is within patient: 70 / 15 / 15 % of each patient's clips, shuffled with seed 123.
Outputs in <out>:
  tokens/all/tokens.npy  (total_sec, 22, 100) float16  z = (log|FFT| - mu) / sigma. mu and sigma are the mean and
                                                       standard deviation of every token value in the TRAIN clips.
  tokens/all/y_sec.npy   per-second labels;  tokens/all/index.json  one entry per recording (rec_id, patient,
                         offset, n_sec, status 'done')
  clips_{train,dev,test}.npz   start (store row of the clip's first second), y (authors' label), patient
  cbfeats/{train,dev,test}/features.parquet  one row per clip: rec_id, patient, t0 = 12k, rec_duration, y, 588
                         CatBoost features;  intervals.json  seizure intervals (s) per recording, for the scorer
  stats.json             mu, sigma, clip counts, checks (every listed clip maps; labels agree with y_sec)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SPLITS = {"train": "trainSet", "dev": "devSet", "test": "testSet"}


def read_markers(marker_dir, split):
    out = []
    for kind in ("sz", "nosz"):
        for line in (Path(marker_dir) / f"{SPLITS[split]}_seq2seq_12s_{kind}.txt").read_text().split():
            key, lab = line.split(",")
            rec, k = key.split(".edf_")
            out.append((rec, int(k.split(".h5")[0]), int(lab)))
    return out


def runs(y):
    d = np.diff(np.r_[0, y.astype(np.int8), 0])
    return [(float(a), float(b)) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]


def main(argv=None):
    import pandas as pd
    from numpy.lib.format import open_memmap
    from szfusion.catboost.extract import load_config
    from szfusion.catboost.features import NAMES18, feature_names
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", nargs="+", required=True)
    ap.add_argument("--markers", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out); (out / "tokens" / "all").mkdir(parents=True, exist_ok=True)
    files = {}
    for d in a.rec:
        for f in Path(d).rglob("rec/*.npz"):
            files[f.stem] = f
    recs = sorted(files)
    index, off = [], 0
    for r in recs:
        z = np.load(files[r])
        index.append({"rec_id": r, "patient": str(z["patient"]), "offset": off, "n_sec": int(z["n_sec"]),
                      "status": "done"})
        off += int(z["n_sec"])
    by = {e["rec_id"]: e for e in index}
    mk = {s: read_markers(a.markers, s) for s in SPLITS}
    stats = {"recordings": len(index), "seconds": off}
    for s, m in mk.items():
        miss = [x for x in m if x[0] not in by or 12 * (x[1] + 1) > by[x[0]]["n_sec"]]
        stats[f"{s}_clips"] = len(m); stats[f"{s}_unmapped"] = len(miss)
        if miss:
            raise ValueError(f"{len(miss)} {s} clips do not map to the recordings: {miss[:3]}")
    # mu, sigma over the train clips (float64 accumulation)
    tr_by_rec = {}
    for r, k, _ in mk["train"]:
        tr_by_rec.setdefault(r, []).append(k)
    n = s1 = s2 = 0.0
    for r, ks in tr_by_rec.items():
        tok = np.load(files[r])["tok"].astype(np.float64)
        sel = np.concatenate([tok[12 * k:12 * k + 12] for k in ks])
        n += sel.size; s1 += sel.sum(); s2 += (sel ** 2).sum()
    mu = s1 / n; sigma = float(np.sqrt(s2 / n - mu ** 2))
    stats.update(mu=mu, sigma=sigma)
    X = open_memmap(out / "tokens" / "all" / "tokens.npy", mode="w+", dtype=np.float16, shape=(off, 22, 100))
    Y = np.zeros(off, np.int8)
    for e in index:
        z = np.load(files[e["rec_id"]])
        o, n_s = e["offset"], e["n_sec"]
        X[o:o + n_s] = ((z["tok"].astype(np.float32) - mu) / sigma).astype(np.float16)
        Y[o:o + n_s] = z["y"]
    X.flush(); del X
    np.save(out / "tokens" / "all" / "y_sec.npy", Y)
    (out / "tokens" / "all" / "index.json").write_text(json.dumps(index))
    cfg = load_config()
    names = feature_names(dict(cfg["features"], _fs=200, _win_sec=12), list(NAMES18))
    for s, m in mk.items():
        start = np.array([by[r]["offset"] + 12 * k for r, k, _ in m], np.int64)
        y = np.array([lab for _, _, lab in m], np.int64)
        yy = np.array([Y[st:st + 12].max() for st in start])
        stats[f"{s}_label_agreement"] = float((yy == y).mean())
        np.savez(out / f"clips_{s}.npz", start=start, y=y, patient=np.array([by[r]["patient"] for r, _, _ in m]))
        rows = {}
        for r, k, lab in m:
            rows.setdefault(r, []).append((k, lab))
        parts = []
        for r, kl in rows.items():
            F = np.load(files[r])["feat"]
            ks = np.array([k for k, _ in kl])
            df = pd.DataFrame(F[ks], columns=names)
            df.insert(0, "y", [lab for _, lab in kl]); df.insert(0, "t0", 12.0 * ks)
            df.insert(0, "rec_duration", float(by[r]["n_sec"])); df.insert(0, "patient", by[r]["patient"])
            df.insert(0, "rec_id", r)
            parts.append(df)
        d = out / "cbfeats" / s; d.mkdir(parents=True, exist_ok=True)
        pd.concat(parts, ignore_index=True).to_parquet(d / "features.parquet", index=False)
        iv = {r: runs(Y[by[r]["offset"]:by[r]["offset"] + by[r]["n_sec"]]) for r in rows}
        (d / "intervals.json").write_text(json.dumps(iv))
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
