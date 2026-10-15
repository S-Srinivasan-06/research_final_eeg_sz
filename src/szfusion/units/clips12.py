"""The 12 s clip test on TUSZ: the clips and labels of the HyBrain authors (Kim et al., CIKM 2026).

The authors list their clips in marker files (github.com/hhyy0401/seizure, src/data/file_markers_detection/
{devSet,testSet}_seq2seq_12s_{sz,nosz}.txt). We do not redistribute these files: scripts/get_hybrain_markers.py
downloads them at a pinned commit. Each line is "<recording>.edf_<k>.h5,<label>".
  * Clip k of a recording covers seconds [12k, 12k + 12) of its 200 Hz resampled signal. In our token store
    (tokens.py, the same preprocessing) these are rows [offset + 12k, offset + 12k + 12).
  * The label is the authors' (1 if the clip contains at least one seizure second).
  * Their dev clips are 73,387 (2,354 positive) and their test clips (= TUSZ eval) 38,384 (2,340 positive). Every
    clip maps to the store. The authors' channel rule (FZ/PZ required) drops the same recordings in both.
Clip scores from a per-second probability p(t): the log-mean-exp of the per-second logits over the clip,
  z_clip = log( (1/12) * sum_t exp(logit p(t)) ),
which lies between the mean logit and the max logit. This is the pooling the overlooker starts from (overlooker.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

STORE_SPLIT = {"dev": "dev", "test": "eval"}


def read_markers(marker_dir, split):
    """-> list of (recording id, clip index, label) for the authors' dev or test clips."""
    name = {"dev": "devSet", "test": "testSet"}[split]
    out = []
    for kind in ("sz", "nosz"):
        for line in (Path(marker_dir) / f"{name}_seq2seq_12s_{kind}.txt").read_text().split():
            key, lab = line.split(",")
            rec, k = key.split(".edf_")
            out.append((rec, int(k.split(".h5")[0]), int(lab)))
    return out


def clips(marker_dir, tokens_root, split):
    """-> start row of each clip in the token store, patient, label, store index (all clips must map)."""
    index = [e for e in json.loads((Path(tokens_root) / STORE_SPLIT[split] / "index.json").read_text())
             if e["status"] == "done"]
    by = {e["rec_id"]: e for e in index}
    mk = read_markers(marker_dir, split)
    missing = [m for m in mk if m[0] not in by or 12 * (m[1] + 1) > by[m[0]]["n_sec"]]
    if missing:
        raise ValueError(f"{len(missing)} of {len(mk)} clips do not map to the token store")
    start = np.array([by[r]["offset"] + 12 * k for r, k, _ in mk], np.int64)
    patient = np.array([by[r]["patient"] for r, _, _ in mk])
    y = np.array([lab for _, _, lab in mk], np.int64)
    return start, patient, y, index


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def lme(z):
    """Row-wise log-mean-exp over the non-NaN entries; NaN if a row has none."""
    ok = ~np.isnan(z)
    m = np.where(ok, z, -np.inf).max(1)
    m_safe = np.where(np.isfinite(m), m, 0.0)
    s = np.where(ok, np.exp(np.where(ok, z, 0.0) - m_safe[:, None]), 0.0).sum(1)
    n = ok.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(n > 0, m_safe + np.log(s / np.maximum(n, 1)), np.nan)


def clip_logits(p_sec, start):
    return lme(logit(p_sec[start[:, None] + np.arange(12)].astype(np.float64)))
