"""Per-second summaries of a trained OscSeq run (the H = 96 input of its head), read by the 12 s / 60 s overlookers.

  python -m szfusion.units.embed --run <work>/tusz/linoss/s0 --tokens <work>/tokens/tusz --splits dev eval train

The inference windows follow the run's results.json, exactly as its
prob_{split}.npy was made (train.py): aligned runs (clip12) use windows = the clips themselves (L = S = 12, no tail
window), so each second's summary sees only its own clip; other runs use L-second windows every final_stride seconds,
and overlapping windows are blended with the same weights 0.05 + sin^2(pi (t + 0.5) / L).
Writes emb_{split}.npy (total_sec, H) float16 next to prob_{split}.npy; rows of unused recordings are NaN. For the
train split (no prob file from training) it also writes prob_train.npy = the head applied to the blended summaries.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from szfusion.linoss.data import RecordingWindows, Store
from szfusion.linoss.model import build
from szfusion.util.monitor import Log


@torch.no_grad()
def export(model, ds, total_sec, bs, log, tag):
    model.eval()
    H = model.head.in_features
    acc = np.zeros((total_sec, H), np.float32); wsum = np.zeros(total_sec, np.float32)
    dl = DataLoader(ds, bs, shuffle=False, num_workers=0, generator=torch.Generator())
    t0 = time.time()
    for bi, (x, m, idx) in enumerate(dl):
        f = model.features(x, m).numpy()                                            # (B, L, H)
        for j, i in enumerate(idx.numpy()):
            ri, s0 = ds.w[i]
            e = ds.s.index[ri]
            n = int(min(ds.L, e["n_sec"] - s0)); a = int(e["offset"] + s0)
            acc[a:a + n] += f[j, :n] * ds.weight[:n, None]; wsum[a:a + n] += ds.weight[:n]
        if bi % 100 == 0:
            log(f"  [{tag}] {min((bi + 1) * bs, len(ds))}/{len(ds)} windows  {time.time() - t0:.0f}s")
    out = np.full((total_sec, H), np.nan, np.float32); ok = wsum > 0
    out[ok] = acc[ok] / wsum[ok, None]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory of one LinOSS seed")
    ap.add_argument("--tokens", required=True, help="token store root")
    ap.add_argument("--splits", nargs="+", default=["dev", "eval"])
    ap.add_argument("--bs", type=int, default=None, help="windows per batch (default 512 for 12 s runs, else 32)")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    run = Path(a.run)
    res = json.loads((run / "results.json").read_text())
    L, align, stride = res["L"], res.get("align", False), res.get("final_stride", 120)
    model = build(res["cfg"])
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu", weights_only=True)["model"])
    log = Log(run, name="embed")
    log.event("embed_start", run=str(run), L=L, align=align, stride=L if align else stride, H=model.head.in_features)
    for split in a.splits:
        f = run / f"emb_{split}.npy"
        if f.exists():
            log(f"{f.name} exists, skipping"); continue
        st = Store(split, a.tokens)
        ds = RecordingWindows(st, L, L if align else stride, None, tail=not align)
        t = time.time()
        emb = export(model, ds, len(st.Y), a.bs or (512 if L <= 12 else 32), log, split)
        # check: the head on the blended summaries should track the saved per-second probabilities (dev / eval)
        corr, ok = None, ~np.isnan(emb[:, 0])
        if (run / f"prob_{split}.npy").exists():
            p = np.load(run / f"prob_{split}.npy").astype(np.float64)
            ok &= ~np.isnan(p)
            sub = np.flatnonzero(ok)[:: max(1, ok.sum() // 200_000)]
            with torch.no_grad():
                z = model.head(torch.from_numpy(emb[sub])).squeeze(-1).numpy().astype(np.float64)
            zp = np.log(np.clip(p[sub], 1e-6, 1 - 1e-6) / np.clip(1 - p[sub], 1e-6, None))
            corr = round(float(np.corrcoef(z, zp)[0, 1]), 4)
        else:                                       # train: also save the head's per-second probability
            with torch.no_grad():
                z = torch.cat([model.head(torch.from_numpy(np.nan_to_num(emb[i:i + 500_000]))).squeeze(-1)
                               for i in range(0, len(emb), 500_000)]).numpy()
            pr = (1 / (1 + np.exp(-z.astype(np.float64)))).astype(np.float32); pr[~ok] = np.nan
            np.save(run / f"prob_{split}.npy", pr)
        np.save(f, emb.astype(np.float16))
        log.event("embed_done", split=split, file=str(f), seconds=int(ok.sum()), corr_head_vs_prob=corr,
                  sec=round(time.time() - t, 1))
    log.close()


if __name__ == "__main__":
    main()
