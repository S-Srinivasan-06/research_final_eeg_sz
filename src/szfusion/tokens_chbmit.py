"""CHB-MIT per-recording inputs for both branches (path.md stage C1): tokens for LinOSS, 12 s clip features for
CatBoost, per-second labels.

  python -m szfusion.tokens_chbmit --raw <dir with chbNN/> --out <out> --patients chb01 chb02 ...   (resumable)

For every EDF of a patient (data/chbmit.py has the channel and annotation rules), one file <out>/rec/<rec>.npz:
  tok   (n_sec, 22, 100) float16  log|FFT| per 1 s step, as tokens.py on TUSZ but NOT yet z-scored:
        FFT resampling (scipy.signal.resample) 256 Hz -> 200 Hz to round(200 · n / fs) samples, as the
        authors' preprocess_chb.py; 200-sample steps; |scipy.fftpack.fft(n = 200)| bins 0-99; zeros -> 1e-8; ln.
        The global z-score is fitted on the train clips later (chbmit_store.py).
  y     (n_sec,) int8               1 if the second overlaps a summary.txt seizure: [floor(start), ceil(end))
  feat  (n_clips, 588) float32      CatBoost features of each 12 s clip k = seconds [12k, 12k + 12)
        (catboost/features.py with the TUSZ settings of configs/tusz_final.yaml: the 16 longitudinal chains, µV -> V,
        resample_poly 256 -> 200 Hz, 60 Hz notch Q30, 0.5 Hz high-pass order 1, 35 Hz low-pass order 4, zero phase).
        CHB-MIT names T7/P7/T8/P8 for the TUSZ T3/T5/T4/T6.
  n_sec, fs, patient
Recordings that miss one of the 22 channels are listed in <out>/skipped.json, with the reason.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from szfusion.data import chbmit

FS, S, CLIP = 200, 200, 12
# CatBoost chain order (catboost/features.py PAIRS16, regions la/lp/ra/rp index into this order), in CHB-MIT names
CHAINS16_CHB = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
                "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2"]


def process(job):
    """One EDF -> npz. Never raises: returns (rec, status)."""
    edf, out_dir, intervals, patient, cfg = job
    rec = Path(edf).stem
    out = Path(out_dir) / "rec" / f"{rec}.npz"
    if out.exists():
        return rec, "cached"
    try:
        import pyedflib
        from scipy.fftpack import fft
        from scipy.signal import resample, resample_poly
        from szfusion.catboost.features import _filters, extract_window_features
        with pyedflib.EdfReader(str(edf)) as f:
            labels = f.getSignalLabels()
            try:
                idx = chbmit.channel_indices(labels)
            except KeyError as e:
                return rec, f"skipped: missing channel {e}"
            fs = f.getSampleFrequency(idx[0])
            sig = np.stack([f.readSignal(i) for i in idx]).astype(np.float64)        # (22, n) µV
        n = sig.shape[1]
        # tokens (the authors' resampling and features)
        x = resample(sig, num=int(round(FS * n / fs)), axis=1) if fs != FS else sig
        n_sec = x.shape[1] // S
        a = fft(x[:, :n_sec * S].reshape(22, n_sec, S).transpose(1, 0, 2), n=S, axis=-1)[..., :S // 2]
        amp = np.abs(a); amp[amp == 0.0] = 1e-8
        tok = np.log(amp).astype(np.float16)
        y = np.zeros(n_sec, np.int8)
        for s0, s1 in intervals:
            y[max(int(np.floor(s0)), 0):min(int(np.ceil(s1)), n_sec)] = 1
        # CatBoost clip features (TUSZ CatBoost signal path)
        d = cfg["data"]
        ch = [chbmit.CHB22.index(c) for c in CHAINS16_CHB]
        bip = resample_poly(sig[ch], FS, int(fs), axis=1) * d.get("scale", 1e-6) if fs != FS else sig[ch] * 1e-6
        bip = _filters(bip, FS, d).astype(np.float32)
        n_clip = n_sec // CLIP
        W = CLIP * FS
        X = np.stack([bip[:, k * W:(k + 1) * W] for k in range(n_clip)]) if n_clip else np.zeros((0, 16, W), np.float32)
        fc = dict(cfg["features"], _fs=FS, _win_sec=CLIP)
        feat = np.concatenate([extract_window_features(X[i:i + 64], FS, fc) for i in range(0, n_clip, 64)]) \
            if n_clip else np.zeros((0, 588), np.float32)
        tmp = out.with_suffix(".tmp.npz")
        np.savez(tmp, tok=tok, y=y, feat=feat.astype(np.float32), n_sec=n_sec, fs=fs, patient=patient)
        os.replace(tmp, out)
        return rec, "ok"
    except Exception as e:  # noqa: BLE001
        return rec, f"error: {type(e).__name__}: {e}"


def main(argv=None):
    from szfusion.catboost.extract import load_config
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="folder containing chbNN/ folders (PhysioNet layout)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--patients", nargs="+", default=chbmit.PATIENTS)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    a = ap.parse_args(argv)
    cfg = load_config()
    out = Path(a.out); (out / "rec").mkdir(parents=True, exist_ok=True)
    jobs = []
    for p in a.patients:
        sz = chbmit.seizures(Path(a.raw) / p / f"{p}-summary.txt")
        for edf in sorted((Path(a.raw) / p).glob("*.edf")):
            jobs.append((str(edf), str(out), sz.get(edf.name, []), p, cfg))
    t = time.time(); status = {}
    from multiprocessing import Pool
    with Pool(a.workers) as pool:
        for k, (rec, st) in enumerate(pool.imap_unordered(process, jobs, chunksize=1), 1):
            status[rec] = st
            print(f"[{k}/{len(jobs)}] {rec}: {st}  {time.time() - t:.0f}s", flush=True)
    sk = out / "skipped.json"
    prev = json.loads(sk.read_text()) if sk.exists() else {}
    prev.update({r: s for r, s in status.items() if not s.startswith(("ok", "cached"))})
    sk.write_text(json.dumps(prev, indent=1))


if __name__ == "__main__":
    main()
