"""CHB-MIT Scalp EEG Database v1.0.0 (PhysioNet, ODC-By 1.0), under the HyBrain protocol (path.md §1, §4.7).

Data: <chbmit_root>/chbNN/chbNN_MM.edf plus chbNN/chbNN-summary.txt. Patients chb01-chb22 are used:
  * chb23 has no CHB_PATIENTS entry in the authors' code;
  * chb24 has a different montage and is excluded;
  * chb21 is a later re-recording of the chb01 subject, kept as a separate case, as in the authors' code.
Recordings are 256 Hz bipolar (double-banana) EDFs.
Channels: the 22 bipolar channels of the HyBrain code (github.com/hhyy0401/seizure @ 694bdd4d,
src/data/constants_chb.py INCLUDED_CHANNELS_CHB), in their order. Labels are matched after strip().upper(), and for a
duplicated label (T8-P8 appears twice in CHB-MIT EDFs) the first occurrence is used. A recording that misses any of
the 22 channels is skipped (the authors flag it as failed; e.g. chb12 files recorded with another reference).
Note: P7-T7 is the sign-inverted T7-P7, and FT9-FT10 is the only transverse channel. Log-magnitude spectra are
sign-invariant, so P7-T7 carries the same tokens as T7-P7. We keep it to match the authors' input exactly.
Annotations: summary.txt gives, per file, "Seizure [n] Start Time: <s> seconds" / "... End Time: <s> seconds".
A second is labelled 1 if it overlaps a seizure: [floor(start), ceil(end)).
Tokens: the same per-second features as on TUSZ (tokens.py): FFT resampling 256 -> 200 Hz, 1 s steps,
log|FFT| bins 0-99. They are z-scored with the mean and standard deviation of all values in the TRAIN clips of
the HyBrain split (computed by tokens_chbmit.py). The authors use no normalisation. Ours is an affine rescaling of
the input, fitted on train only.
"""
from __future__ import annotations

import re
from pathlib import Path

CHB22 = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
         "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
         "FZ-CZ", "CZ-PZ", "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8"]
PATIENTS = [f"chb{i:02d}" for i in range(1, 23)]
# Hemisphere swap for CHB22 (training augmentation, as SWAP_PERM_19 on TUSZ): each left chain channel with its
# mirror. FP1-F7<->FP2-F8, F7-T7<->F8-T8, T7-P7<->T8-P8, P7-O1<->P8-O2 (temporal chains); FP1-F3<->FP2-F4,
# F3-C3<->F4-C4, C3-P3<->C4-P4, P3-O1<->P4-O2 (parasagittal chains); T7-FT9<->FT10-T8. The midline (FZ-CZ, CZ-PZ),
# FT9-FT10 and P7-T7 (whose mirror P8-T8 is not recorded) stay in place.
SWAP_PAIRS_CHB22 = [("FP1-F7", "FP2-F8"), ("F7-T7", "F8-T8"), ("T7-P7", "T8-P8"), ("P7-O1", "P8-O2"),
                    ("FP1-F3", "FP2-F4"), ("F3-C3", "F4-C4"), ("C3-P3", "C4-P4"), ("P3-O1", "P4-O2"),
                    ("T7-FT9", "FT10-T8")]


def swap_perm_chb22():
    from szfusion.data.montage import swap_perm
    return swap_perm(CHB22, SWAP_PAIRS_CHB22)


_FILE = re.compile(r"File Name:\s*(\S+\.edf)", re.I)
_START = re.compile(r"Seizure(?:\s+\d+)?\s+Start Time:\s*([0-9.]+)\s*seconds", re.I)
_END = re.compile(r"Seizure(?:\s+\d+)?\s+End Time:\s*([0-9.]+)\s*seconds", re.I)


def seizures(summary_txt):
    """chbNN-summary.txt -> {edf file name: [(start_s, end_s), ...]} (files without seizures map to [])."""
    out, cur, start = {}, None, None
    for line in Path(summary_txt).read_text(errors="replace").splitlines():
        m = _FILE.search(line)
        if m:
            cur = m.group(1); out[cur] = []; start = None; continue
        m = _START.search(line)
        if m and cur:
            start = float(m.group(1)); continue
        m = _END.search(line)
        if m and cur:
            end = float(m.group(1))
            out[cur].append((start if start is not None else end - 1, end))   # the authors' fallback
            start = None
    return out


def channel_indices(labels):
    """EDF signal labels -> indices of CHB22 (first occurrence wins); KeyError if a channel is missing."""
    first = {}
    for i, lab in enumerate(labels):
        first.setdefault(lab.strip().upper(), i)
    return [first[c] for c in CHB22]
