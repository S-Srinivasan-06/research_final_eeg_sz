"""Token-store data for OscSeq: fresh training segments every epoch, and sliding windows for scoring whole recordings.

Store (written by szfusion.tokens): <root>/<split>/
  tokens.npy  (total_sec, N, 100) float16   z-scored log|FFT| per 1 s, N channels
  y_sec.npy   (total_sec,) int8             1 = the second overlaps an annotated seizure
  index.json  [{rec_id, patient, offset, n_sec, status, ...}]  recording r occupies rows [offset, offset + n_sec)
Only recordings with status "done" are used.

TrainSegments: one epoch = n_seg segments of L seconds, drawn afresh from all train recordings and fully determined
by (seed, epoch), so an interrupted and resumed run sees exactly the same batches.
  * Half are "seizure-anchored": a seizure second drawn uniformly from all train seizure seconds, placed at a uniform
    position inside the segment.
  * Half are "random": a recording drawn with probability proportional to its length, and a uniform start. These
    are almost always background.
  * Recordings shorter than L are zero-padded; mask = 0 marks the padding.
  * Augmentation (per segment, as in the HyBrain baseline): hemisphere swap with p = 0.5, then a gain
    s ~ U(0.8, 1.2), applied as + log(s) / TOKEN_STD in the z-scored log-FFT domain (montage.py).
RecordingWindows: every recording cut into windows of L seconds every S seconds. One extra window is aligned to the
  recording end, so every second is covered. Overlapping predictions are blended per second with the weight
  w(t) = 0.05 + sin^2(pi (t + 0.5) / L). This down-weights the window edges, where the bidirectional context is
  one-sided.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from szfusion.data.montage import SWAP_PERM_19, TOKEN_STD


class Store:
    def __init__(self, split, root):
        self.dir = Path(root) / split
        self.index = [e for e in json.loads((self.dir / "index.json").read_text()) if e["status"] == "done"]
        self._X = self._Y = None

    @property
    def X(self):
        if self._X is None:
            self._X = np.load(self.dir / "tokens.npy", mmap_mode="r")
        return self._X

    @property
    def Y(self):
        if self._Y is None:
            self._Y = np.load(self.dir / "y_sec.npy", mmap_mode="r")
        return self._Y

    def __getstate__(self):                       # DataLoader workers re-open the memmaps instead of pickling them
        d = dict(self.__dict__); d["_X"] = d["_Y"] = None
        return d


class TrainSegments(Dataset):
    def __init__(self, store: Store, L=240, n_seg=6000, seed=0, augment=True, pos_frac=0.5, perm=SWAP_PERM_19):
        self.s, self.L, self.n, self.seed, self.augment, self.pos_frac = store, L, n_seg, seed, augment, pos_frac
        self.off = np.array([e["offset"] for e in store.index], np.int64)
        self.nsec = np.array([e["n_sec"] for e in store.index], np.int64)
        self.shape = tuple(store.X.shape[1:])                                  # (N channels, 100 bins)
        y = np.asarray(store.Y)
        rec_of = np.repeat(np.arange(len(self.off)), self.nsec)
        glob = np.concatenate([o + np.arange(n) for o, n in zip(self.off, self.nsec)])
        pos = y[glob] == 1
        self.pos_rec, self.pos_t = rec_of[pos], (glob - np.repeat(self.off, self.nsec))[pos]
        self.perm, self.gain_std = perm, TOKEN_STD
        self.n_pos_sec = int(pos.sum())
        self.set_epoch(1)

    def set_epoch(self, epoch):
        rng = np.random.default_rng((self.seed, epoch, 7))
        n_pos = int(round(self.pos_frac * self.n))
        spec = np.zeros((self.n, 2), np.int64)                                  # (recording, start second)
        k = rng.integers(0, len(self.pos_rec), n_pos)
        r = self.pos_rec[k]; t = self.pos_t[k]
        spec[:n_pos, 0] = r
        spec[:n_pos, 1] = t - rng.integers(0, self.L, n_pos)
        r2 = rng.choice(len(self.nsec), self.n - n_pos, p=self.nsec / self.nsec.sum())
        spec[n_pos:, 0] = r2
        spec[n_pos:, 1] = (rng.random(self.n - n_pos) * np.maximum(self.nsec[r2] - self.L, 0) + 0.5).astype(np.int64)
        hi = np.maximum(self.nsec[spec[:, 0]] - self.L, 0)
        spec[:, 1] = np.clip(spec[:, 1], 0, hi)
        self.spec = spec[rng.permutation(self.n)]
        self.epoch = epoch

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        r, t0 = self.spec[i]
        n = int(min(self.L, self.nsec[r] - t0)); a = int(self.off[r] + t0)
        x = np.zeros((self.L, *self.shape), np.float32); y = np.zeros(self.L, np.float32); m = np.zeros(self.L, np.float32)
        x[:n] = self.s.X[a:a + n]; y[:n] = self.s.Y[a:a + n]; m[:n] = 1
        if self.augment:
            g = np.random.default_rng((self.seed, self.epoch, int(i)))
            if g.random() < 0.5:
                x = x[:, self.perm]
            x[:n] += np.float32(math.log(g.uniform(0.8, 1.2)) / self.gain_std)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(y), torch.from_numpy(m)


class RecordingWindows(Dataset):
    def __init__(self, store: Store, L=240, S=120, rec_subset=None, tail=True):
        """tail=False: no extra end-aligned window (with L = S = 12 each 12 s clip is scored from itself only)."""
        self.s, self.L, self.S = store, L, S
        self.shape = tuple(store.X.shape[1:])
        recs = range(len(store.index)) if rec_subset is None else rec_subset
        w = []
        for ri in recs:
            n = store.index[ri]["n_sec"]
            starts = list(range(0, max(n - L, 0) + 1, S))
            if tail and n > L and starts[-1] != n - L:
                starts.append(n - L)
            w += [(ri, s) for s in starts]
        self.w = np.array(w, np.int64)
        t = np.arange(L)
        self.weight = (0.05 + np.sin(np.pi * (t + 0.5) / L) ** 2).astype(np.float32)

    def __len__(self):
        return len(self.w)

    def __getitem__(self, i):
        ri, t0 = self.w[i]
        e = self.s.index[ri]
        n = int(min(self.L, e["n_sec"] - t0)); a = int(e["offset"] + t0)
        x = np.zeros((self.L, *self.shape), np.float32); m = np.zeros(self.L, np.float32)
        x[:n] = self.s.X[a:a + n]; m[:n] = 1
        return torch.from_numpy(x), torch.from_numpy(m), int(i)

    def blend(self, logits_by_window, total_sec):
        """(n_windows, L) logits -> per-second probability over the whole split (aligned to the store rows); seconds
        of unused recordings stay NaN."""
        acc = np.zeros(total_sec, np.float64); wsum = np.zeros(total_sec, np.float64)
        p = 1 / (1 + np.exp(-logits_by_window.astype(np.float64)))
        for (ri, t0), pw in zip(self.w, p):
            e = self.s.index[ri]
            n = int(min(self.L, e["n_sec"] - t0)); a = int(e["offset"] + t0)
            acc[a:a + n] += pw[:n] * self.weight[:n]; wsum[a:a + n] += self.weight[:n]
        out = np.full(total_sec, np.nan); ok = wsum > 0
        out[ok] = acc[ok] / wsum[ok]
        return out
