"""Per-second token store: every second of every TUSZ recording, as the input of the LinOSS branch.

  python -m szfusion.tokens build   --split eval --out <work>/tokens/tusz --workers 10   (resumable)
  python -m szfusion.tokens compare --split eval --out <work>/tokens/tusz --ref <other store>
  python -m szfusion.tokens status  --out <work>/tokens/tusz

Preprocessing = the HyBrain authors' (github.com/hhyy0401/seizure @ 694bdd4d), so a 12 s window starting at second
12k equals their clip k:
  1. The 19 electrodes of montage.ELECTRODES_19 are matched by the EDF label before '-' ('EEG FP1-REF' -> 'EEG FP1').
     The recording is skipped if one is missing.
  2. Signals of length getNSamples()[0] are read; the sampling rate fs is that of signal 0.
  3. If fs != 200 Hz, the whole recording is FFT-resampled (scipy.signal.resample) to 200 * floor(n / fs) samples.
  4. Each 1 s step (200 samples, rectangular window, no filtering) gives |FFT| (scipy.fftpack.fft, n = 200);
     bins 0-99 = 0-99 Hz at 1 Hz spacing. Zeros are replaced by 1e-8, then the natural log is taken.
  5. One global z-score: (x - TOKEN_MEAN) / TOKEN_STD (montage.py), stored as float16.
  An EDF channel that cannot be read stays zero, as in the authors' code. Such channels are listed in index.json
  ("zero_channels") and counted in the build log, so their effect is visible.

Layout in <out>/<split>/:
  tokens.npy   (total_sec, 19, 100) float16, recordings concatenated
  y_sec.npy    (total_sec,) int8    1 if the second overlaps a .csv_bi seizure: [floor(start), ceil(stop))
  y_ch.npy     (total_sec, 22) int8 per TCP channel from .csv (same rounding); used only by the evidence figures
  index.json   per recording: rec_id, path, montage, patient, offset, n_sec, fs, seizures, valid_channels, status
  done.npy     per recording: 0 to do, 1 done, -1 failed, -2 skipped (missing electrode)
These files contain TUSZ-derived data and recording IDs: they live under work_dir and are never committed.
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

from szfusion import paths
from szfusion.data import tusz
from szfusion.data.montage import ELECTRODES_19, TOKEN_MEAN, TOKEN_STD

INCLUDED = [f"EEG {e}" for e in ELECTRODES_19]
FS, S = 200, 200


def header(edf_path):
    """EDF header -> sample count, rate, whole seconds, missing electrodes."""
    import pyedflib
    f = pyedflib.EdfReader(edf_path)
    try:
        labels = [l.split("-")[0] for l in f.getSignalLabels()]
        missing = [c for c in INCLUDED if c not in labels]
        n, fs = int(f.getNSamples()[0]), float(f.getSampleFrequency(0))
    finally:
        f.close()
    n_sec = int(n / fs) if fs != FS else n // FS
    return {"n_samples": n, "fs": fs, "n_sec": n_sec, "missing": missing}



def featurize(job):
    """Worker: one EDF -> (n_sec, 19, 100) float16 z-scored tokens, timings, unreadable channels. Never raises."""
    path, n_sec_expected, mean, std = job
    t = {"read": 0.0, "resample": 0.0, "fft": 0.0}
    zero = []
    try:
        import pyedflib
        from scipy.fftpack import fft
        from scipy.signal import resample
        t0 = time.perf_counter()
        f = pyedflib.EdfReader(path)
        try:
            labels = [l.split("-")[0] for l in f.getSignalLabels()]
            order = [labels.index(c) for c in INCLUDED]
            n = f.getNSamples()[0]
            sig = np.zeros((19, n))
            for k, ci in enumerate(order):
                try:
                    sig[k] = f.readSignal(ci)
                except Exception:  # noqa: BLE001  (authors' behaviour: the channel stays zero; reported below)
                    zero.append(ELECTRODES_19[k])
            fs = f.getSampleFrequency(0)
        finally:
            f.close()
        t["read"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        if fs != FS:
            sig = resample(sig, num=int(FS * int(sig.shape[1] / fs)), axis=1)
        t["resample"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        n_sec = sig.shape[1] // S
        steps = sig[:, : n_sec * S].reshape(19, n_sec, S).transpose(1, 0, 2)          # (n_sec, 19, 200)
        a = fft(steps, n=S, axis=-1)[..., : S // 2]
        amp = np.abs(a); amp[amp == 0.0] = 1e-8
        x = ((np.log(amp) - mean) / std).astype(np.float16)
        t["fft"] = time.perf_counter() - t0
        if n_sec != n_sec_expected:
            return path, None, t, f"n_sec {n_sec} != header {n_sec_expected}", zero
        return path, x, t, None, zero
    except Exception as e:  # noqa: BLE001
        return path, None, t, f"{type(e).__name__}: {e}", zero


def build(split, root, workers, flush_every, limit):
    from multiprocessing import Pool
    from numpy.lib.format import open_memmap
    from szfusion.util.monitor import Log, Profiler
    edf_root = paths.get("tusz_root") / "edf"
    out = Path(root) / split; out.mkdir(parents=True, exist_ok=True)
    log = Log(out, name="build"); prof = Profiler(log)
    mean, std = TOKEN_MEAN, TOKEN_STD
    log.event("scaler", mean=mean, std=std)

    ip = out / "index.json"
    if ip.exists():
        index = json.loads(ip.read_text())
        log(f"index loaded: {len(index)} recordings")
    else:
        with prof.stage("index"):
            recs = tusz.list_recordings(edf_root, split)
            if limit:
                recs = recs[:limit]
            index, off = [], 0
            with Pool(workers) as pool:
                heads = pool.map(header, [r.edf_path for r in recs], chunksize=8)
            for r, h in zip(recs, heads):
                ok = not h["missing"]
                index.append({"rec_id": r.rec_id, "path": r.edf_path, "montage": r.montage_dir,
                              "patient": r.patient, "offset": off if ok else -1, "n_sec": h["n_sec"] if ok else 0,
                              "fs": h["fs"], "duration": r.duration, "seizures": r.seizure_intervals,
                              "status": "todo" if ok else "skipped",
                              "error": None if ok else f"missing channels {h['missing']}"})
                off += h["n_sec"] if ok else 0
            ip.write_text(json.dumps(index))
        n_skip = sum(e["status"] == "skipped" for e in index)
        log.event("index_built", recordings=len(index), skipped_missing_channels=n_skip,
                  total_hours=round(sum(e["n_sec"] for e in index) / 3600, 1))
    total = sum(e["n_sec"] for e in index)
    tok_p = out / "tokens.npy"
    X = open_memmap(tok_p, mode="r+" if tok_p.exists() else "w+", dtype=np.float16, shape=(total, 19, 100))
    dp = out / "done.npy"
    done = np.load(dp) if dp.exists() else np.array([0 if e["status"] == "todo" else -2 for e in index], np.int8)

    # labels are cheap; (re)write them every time from the annotation files
    with prof.stage("labels"):
        y_sec = np.zeros(total, np.int8); y_ch = np.zeros((total, 22), np.int8)
        for e in index:
            if e["offset"] < 0:
                continue
            o, n = e["offset"], e["n_sec"]
            for a, b in e["seizures"]:
                s0, s1 = max(int(np.floor(a)), 0), min(int(np.ceil(b)), n)
                y_sec[o + s0:o + s1] = 1
            ev = tusz.read_channel_events(str(Path(e["path"]).with_suffix(".csv")))
            y_ch[o:o + n] = tusz.channel_matrix(ev, 0.0, n)
            no_ear = e["montage"].endswith("_a")                     # _a montages have no A1/A2 electrodes
            e["valid_channels"] = [c for c in tusz.TCP22_NAMES if not (no_ear and c in ("A1-T3", "T4-A2"))]
        np.save(out / "y_sec.npy", y_sec); np.save(out / "y_ch.npy", y_ch)
        ip.write_text(json.dumps(index))
    log.event("labels_written", seizure_seconds=int(y_sec.sum()), seizure_hours=round(y_sec.sum() / 3600, 2))

    todo = [i for i, e in enumerate(index) if done[i] == 0]
    log.event("build_start", todo=len(todo), done=int((done == 1).sum()), failed=int((done == -1).sum()),
              workers=workers, store_gb=round(X.nbytes / 1e9, 2))
    by_path = {index[i]["path"]: i for i in todo}
    jobs = [(index[i]["path"], index[i]["n_sec"], mean, std) for i in todo]
    t_build, sec_done, worker_t = time.time(), 0, {"read": 0.0, "resample": 0.0, "fft": 0.0}
    with Pool(workers) as pool:
        for j, (path, x, t, err, zero) in enumerate(pool.imap_unordered(featurize, jobs, chunksize=1), 1):
            i = by_path[path]; e = index[i]
            for k in worker_t:
                worker_t[k] += t[k]
            e["zero_channels"] = zero
            if zero:
                log(f"  WARNING recording {i}: unreadable channel(s) {zero} kept at zero (authors' behaviour)")
            if err:
                done[i] = -1; e["status"], e["error"] = "failed", err
                log(f"  FAILED recording {i}: {err}")
            else:
                with prof.stage("write"):
                    X[e["offset"]:e["offset"] + e["n_sec"]] = x
                done[i] = 1; e["status"] = "done"; sec_done += e["n_sec"]
            if j % flush_every == 0 or j == len(jobs):
                with prof.stage("flush"):
                    X.flush(); np.save(dp, done); ip.write_text(json.dumps(index))
                el = time.time() - t_build
                rate = sec_done / max(el, 1e-9)
                left = sum(index[k]["n_sec"] for k in todo if done[k] == 0)
                log(f"  {j}/{len(jobs)} recs  {sec_done / 3600:.1f} h done  {rate:.0f} recorded s per s  "
                    f"ETA {left / max(rate, 1e-9) / 60:.1f} min  failed={int((done == -1).sum())}")
    log.table("worker time by stage (summed over workers)", ["stage", "sec", "share"],
              [[k, round(v, 1), f"{100 * v / max(sum(worker_t.values()), 1e-9):.1f}%"] for k, v in worker_t.items()])
    prof.report(f"build {split}")
    log.event("build_done", split=split, done=int((done == 1).sum()), failed=int((done == -1).sum()),
              skipped=int((done == -2).sum()))
    log.close()


def compare(split, root, ref):
    """Bit-level comparison with another store of the same split (the S1 reproduction gate)."""
    a, b = Path(root) / split, Path(ref) / split
    ia = {e["rec_id"]: e for e in json.loads((a / "index.json").read_text()) if e["status"] == "done"}
    ib = {e["rec_id"]: e for e in json.loads((b / "index.json").read_text()) if e["status"] == "done"}
    Xa, Xb = np.load(a / "tokens.npy", mmap_mode="r"), np.load(b / "tokens.npy", mmap_mode="r")
    ya, yb = np.load(a / "y_sec.npy"), np.load(b / "y_sec.npy")
    common = sorted(set(ia) & set(ib)); worst, ydiff = 0.0, 0
    for r in common:
        ea, eb = ia[r], ib[r]
        if ea["n_sec"] != eb["n_sec"]:
            worst = float("inf"); continue
        sa = slice(ea["offset"], ea["offset"] + ea["n_sec"]); sb = slice(eb["offset"], eb["offset"] + eb["n_sec"])
        worst = max(worst, float(np.abs(Xa[sa].astype(np.float32) - Xb[sb].astype(np.float32)).max()))
        ydiff += int((ya[sa] != yb[sb]).sum())
    res = {"split": split, "recordings_here": len(ia), "recordings_ref": len(ib), "common": len(common),
           "max_abs_token_diff": worst, "label_seconds_differing": ydiff}
    print(json.dumps(res))
    return res


def status(root):
    for split in ("train", "dev", "eval"):
        p = Path(root) / split / "done.npy"
        if p.exists():
            d = np.load(p)
            print(f"{split}: done {int((d == 1).sum())}  todo {int((d == 0).sum())}  failed {int((d == -1).sum())}  "
                  f"skipped {int((d == -2).sum())}")
        else:
            print(f"{split}: not started")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "compare", "status"])
    ap.add_argument("--split", choices=["train", "dev", "eval"])
    ap.add_argument("--out", required=True, help="store root")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--flush_every", type=int, default=50)
    ap.add_argument("--limit", type=int, default=None, help="debug: first N recordings")
    ap.add_argument("--ref", help="compare: the other store root")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.split, a.out, a.workers, a.flush_every, a.limit)
    elif a.cmd == "compare":
        compare(a.split, a.out, a.ref)
    else:
        status(a.out)
