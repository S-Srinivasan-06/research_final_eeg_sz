"""TUSZ v2.0.6: recording index, annotations (.csv_bi term level, .csv per channel) and EDF loading.

Layout: <tusz_root>/edf/{train,dev,eval}/<patient>/<session>/<montage_dir>/<bname>.{edf,csv,csv_bi}, with
montage_dir in {01_tcp_ar, 02_tcp_le, 03_tcp_ar_a, 04_tcp_le_a}.
  .csv_bi  '#' header lines (including '# duration = <s> secs'), then rows channel,start,stop,label,confidence
           with channel TERM and label seiz/bckg. Seizure intervals = the merged 'seiz' rows.
  .csv     the same row format, one row per TCP bipolar channel and seizure, with the seizure type as the label
           (fnsz, gnsz, absz, cpsz, tcsz, tnsz, mysz, spsz; 'bckg' rows are ignored).
load_eeg montages:
  'ref19'  the 19 referential electrodes as recorded (REF or LE reference; no re-referencing).
  'tcp22'  the 22 TCP bipolar channels. A channel touching a missing electrode is zero
           (the _a montages have no A1/A2).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np

ALL_MONTAGES = ("01_tcp_ar", "02_tcp_le", "03_tcp_ar_a", "04_tcp_le_a")

TCP22_PAIRS = [
    ("FP1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
    ("FP2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
    ("A1", "T3"), ("T3", "C3"), ("C3", "CZ"), ("CZ", "C4"), ("C4", "T4"), ("T4", "A2"),
    ("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("FP2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
]
TCP22_NAMES = [f"{a}-{b}" for a, b in TCP22_PAIRS]
REF19 = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T3", "C3", "CZ", "C4", "T4",
         "T5", "P3", "PZ", "P4", "T6", "O1", "O2"]

_LABEL_RE = re.compile(r"^EEG\s+([A-Z0-9]+)-(REF|LE)$")


@dataclass
class Recording:
    split: str
    patient: str
    session: str
    montage_dir: str
    rec_id: str                 # bname, e.g. aaaaaaac_s001_t000 (unique across the corpus)
    edf_path: str
    csv_path: str
    csv_bi_path: str
    duration: float = float("nan")          # seconds, from csv_bi header
    seizure_intervals: list = field(default_factory=list)  # [(start, stop)] seconds, 'seiz' only

    @property
    def has_seizure(self) -> bool:
        return len(self.seizure_intervals) > 0

    @property
    def reference(self) -> str:
        return "LE" if "_le" in self.montage_dir else "REF"

def _merge_intervals(iv: list) -> list:
    iv = sorted((float(a), float(b)) for a, b in iv if b > a)
    out: list = []
    for a, b in iv:
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _parse_csv_bi(csv_bi_path):
    """Single pass over a csv_bi -> (duration_sec from '# duration = ...', merged seiz intervals)."""
    iv, dur = [], float("nan")
    with open(csv_bi_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                m = re.search(r"duration\s*=\s*([0-9.]+)", line)
                if m:
                    dur = float(m.group(1))
                continue
            if not line or line.lower().startswith("channel,"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            if parts[3].lower() == "seiz":
                iv.append((float(parts[1]), float(parts[2])))
    return dur, _merge_intervals(iv)


def list_recordings(edf_root, split="train", montages=ALL_MONTAGES, parse_labels=True) -> list:
    """Index one official split: every <bname>.edf under <edf_root>/<split>, sorted by (patient, session, bname)."""
    montages = set(montages)
    recs = []
    for pdir in sorted((Path(edf_root) / split).iterdir()):
        if not pdir.is_dir():
            continue
        for sdir in sorted(pdir.iterdir()):
            if not sdir.is_dir():
                continue
            for mdir in sorted(sdir.iterdir()):
                if not mdir.is_dir() or mdir.name not in montages:
                    continue
                for edf in sorted(mdir.glob("*.edf")):
                    csv_bi = edf.with_suffix(".csv_bi")
                    rec = Recording(split=split, patient=pdir.name, session=sdir.name, montage_dir=mdir.name,
                                    rec_id=edf.stem, edf_path=str(edf), csv_path=str(edf.with_suffix(".csv")),
                                    csv_bi_path=str(csv_bi))
                    if parse_labels:
                        rec.duration, rec.seizure_intervals = _parse_csv_bi(csv_bi)
                    recs.append(rec)
    return recs


def read_channel_events(csv_path, required=True):
    """-> [(channel, start, stop, seizure_type)] for the seizure rows of a per-channel .csv. With required=True a
    missing file is an error (a silently missing file would make every per-channel label 0)."""
    p = Path(csv_path)
    if not p.exists():
        if required:
            raise FileNotFoundError(csv_path)
        return []
    ev = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.lower().startswith("channel,"):
            continue
        parts = [s.strip() for s in line.split(",")]
        if len(parts) < 4 or parts[3].lower() == "bckg":
            continue
        ev.append((parts[0].upper(), float(parts[1]), float(parts[2]), parts[3].lower()))
    return ev


def channel_matrix(events, t0, n_sec):
    """(n_sec, 22) int8: 1 where TCP channel c carries a seizure label overlapping second [t0+s, t0+s+1)."""
    m = np.zeros((n_sec, len(TCP22_NAMES)), np.int8)
    col = {c: i for i, c in enumerate(TCP22_NAMES)}
    for ch, a, b, _ in events:
        if ch not in col:
            continue
        s0, s1 = max(int(np.floor(a - t0)), 0), min(int(np.ceil(b - t0)), n_sec)
        if s1 > s0:
            m[s0:s1, col[ch]] = 1
    return m




# ----------------------------------------------------------------------------- signal loading
def _resample(sig: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    if abs(fs_in - fs_out) < 1e-6:
        return sig
    from scipy.signal import resample_poly
    fr = Fraction(fs_out / fs_in).limit_denominator(1000)
    return resample_poly(sig, fr.numerator, fr.denominator, axis=-1)


def _filter(x: np.ndarray, fs: float, bandpass, notch) -> np.ndarray:
    from scipy.signal import butter, iirnotch, sosfiltfilt, filtfilt
    if bandpass is not None:
        lo, hi = bandpass
        nyq = fs / 2.0
        if hi is not None and hi >= nyq:
            hi = None
        if lo and hi:
            sos = butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
        elif lo:
            sos = butter(4, lo, btype="highpass", fs=fs, output="sos")
        elif hi:
            sos = butter(4, hi, btype="lowpass", fs=fs, output="sos")
        else:
            sos = None
        if sos is not None:
            x = sosfiltfilt(sos, x, axis=-1)
    if notch:
        for f0 in (notch if isinstance(notch, (list, tuple)) else [notch]):
            if f0 < fs / 2:
                b, a = iirnotch(f0, Q=30.0, fs=fs)
                x = filtfilt(b, a, x, axis=-1)
    return x


def load_eeg(rec, montage: str = "tcp22", fs: float = 256, bandpass=(0.5, None), notch=None,
             return_info: bool = False):
    """Load one recording -> (x float32 [C,T] in microvolts, channel_names).

    montage: 'tcp22' (22 bipolar, TCP) or 'ref19' (19 referential). Missing electrodes are zero-filled.
    bandpass: (lo, hi) Hz, either may be None; None disables. notch: Hz (or list) or None.
    With return_info=True returns (x, names, info) where info has fs_orig, missing electrodes.
    """
    import pyedflib
    edf_path = rec.edf_path if isinstance(rec, Recording) else str(rec)
    with pyedflib.EdfReader(edf_path) as r:
        labels = r.getSignalLabels()
        idx, fs_orig = {}, {}
        for i, lab in enumerate(labels):
            m = _LABEL_RE.match(lab.strip().upper())
            if m and m.group(1) not in idx:
                idx[m.group(1)] = i
                fs_orig[m.group(1)] = r.getSampleFrequency(i)
        need = sorted({e for p in TCP22_PAIRS for e in p}) if montage == "tcp22" else REF19
        if montage not in ("tcp22", "ref19"):
            raise ValueError(f"unknown montage {montage}")
        present = [e for e in need if e in idx]
        if not present:
            raise ValueError(f"no EEG channels found in {edf_path}: {labels[:5]}")
        sigs = {}
        for e in present:
            s = r.readSignal(idx[e], digital=False).astype(np.float64)
            sigs[e] = _resample(s, fs_orig[e], fs)
    T = min(len(s) for s in sigs.values())
    missing = [e for e in need if e not in sigs]
    zero = np.zeros(T)
    get = lambda e: sigs[e][:T] if e in sigs else zero
    if montage == "tcp22":
        x = np.stack([get(a) - get(b) for a, b in TCP22_PAIRS])
        names = list(TCP22_NAMES)
        # zero out any bipolar channel touching a missing electrode (not just half of it)
        for c, (a, b) in enumerate(TCP22_PAIRS):
            if a in missing or b in missing:
                x[c] = 0.0
    else:
        x = np.stack([get(e) for e in REF19])
        names = list(REF19)
    x = _filter(x, fs, bandpass, notch)
    x = np.ascontiguousarray(x, dtype=np.float32)
    if return_info:
        return x, names, {"fs_orig": sorted(set(fs_orig.values())), "missing": missing, "T": T}
    return x, names


# ----------------------------------------------------------------------------- windowing
def window_labels(t0: np.ndarray, win_sec: float, intervals, label_rule: str = "any") -> np.ndarray:
    t0 = np.asarray(t0, dtype=np.float64)
    t1 = t0 + win_sec
    if label_rule == "center":
        c = t0 + win_sec / 2
        y = np.zeros(len(t0), bool)
        for a, b in intervals:
            y |= (c >= a) & (c < b)
        return y.astype(np.int8)
    ov = np.zeros(len(t0))
    for a, b in intervals:
        ov += np.clip(np.minimum(t1, b) - np.maximum(t0, a), 0, None)
    if label_rule == "any":
        return (ov > 0).astype(np.int8)
    if label_rule == "majority":
        return (ov >= 0.5 * win_sec).astype(np.int8)
    raise ValueError(label_rule)


def n_windows(T: int, fs: float, win_sec: float, stride_sec: float) -> int:
    W, S = int(round(win_sec * fs)), int(round(stride_sec * fs))
    return 0 if T < W else (T - W) // S + 1


def window_recording(x: np.ndarray, fs: float, intervals, win_sec: float, stride_sec: float,
                     label_rule: str = "any", copy: bool = False):
    """Slice [C,T] into X [N,C,W] float32, y [N] int8, t0 [N] float (window start, sec).
    copy=False returns a zero-copy READ-ONLY strided view (memory-safe for 1 s strides);
    pass copy=True (or np.array(X)) if you need to modify windows in place."""
    W, S = int(round(win_sec * fs)), int(round(stride_sec * fs))
    x = np.asarray(x, dtype=np.float32)
    N = n_windows(x.shape[1], fs, win_sec, stride_sec)
    if N == 0:
        return (np.zeros((0, x.shape[0], W), np.float32), np.zeros(0, np.int8), np.zeros(0))
    v = np.lib.stride_tricks.sliding_window_view(x, W, axis=1)[:, ::S][:, :N]  # [C,N,W]
    X = v.transpose(1, 0, 2)
    if copy:
        X = np.ascontiguousarray(X)
    t0 = np.arange(N) * S / fs
    y = window_labels(t0, win_sec, intervals, label_rule)
    return X, y, t0
