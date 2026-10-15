"""TUSZ eval detection by seizure type and by event duration, on the frozen 1 s fusion outputs.

  python -m szfusion.eval.by_type [--fusion_1s <work>/tusz/fusion_1s] [--tokens <work>/tokens/tusz]
                                  [--out results/tusz/by_type.json]

Input: eval_tracks.npz and selection.json of the 1 s fusion stage (fusion/run_1s.py), and the eval token store
(index.json, y_sec.npy, y_ch.npy) for the row mapping and the per-channel .csv files. Default fusion folder: the
official run (<work>/tusz/fusion_1s) when its eval_tracks.npz exists, otherwise the S1-gate copy
(<work>/s1_gate/fusion_1s), which is numerically identical. Nothing is selected here: every operating point is the
dev-frozen one in selection.json (configs.<model>.ev = event point 1, .ev2 = event point 2).

Models: fused_mean (the fusion), catboost, linoss_mean (the 5-seed LinOSS mean).

Events (same code as the scorer, fusion/protocol_1s.py event_scores):
  true event       a run of y = 1 (.csv_bi seizure seconds) inside one recording, on the scored seconds.
  predicted events smoothed track (centred mean over k s) > thr -> runs -> merge gaps <= gap s -> drop runs shorter
                   than min_dur s, with (k, thr, gap, min_dur) frozen on dev.
  detected         at least one predicted second overlaps the true event (any overlap).
  latency          signed: start of the first predicted event that overlaps the true event minus the true onset (s);
                   negative when that predicted event starts before the annotated onset. Median over detected events.
  second recall    seconds of the group's true events that lie inside predicted events / all seconds of those events
                   (pooled over the events of the group).
Seizure type of an event (TYPE_RULE below): the majority TUSZ per-channel .csv type over the event's labelled
channel-seconds. For every .csv seizure row on one of the 22 TCP channels, the labelled channel-seconds are the
seconds s of the event [a, b) whose interval [s, s + 1) overlaps the row (the same rule as the store's y_ch); they are
counted per type and summed over channels; the type with the most channel-seconds wins (ties: alphabetical order).
The type group is the group of that majority type: focal = {fnsz, cpsz, spsz}, generalized = {gnsz, absz, tcsz,
tnsz, mysz} (evidence.py FOCAL_TYPES / GENERALIZED_TYPES); any other type is "other".
Duration bins (DUR_BINS, s): [0, 10), [10, 30), [30, 60), [60, 180), [180, inf).
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from szfusion import paths
from szfusion.data import tusz
from szfusion.data.montage import TCP22
from szfusion.eval.mechanism import provenance
from szfusion.fusion import protocol_1s as P
from szfusion.viz import evidence as E

MODELS = ["fused_mean", "catboost", "linoss_mean"]     # the fusion and its two branches (5-seed LinOSS mean)
POINTS = ["ev", "ev2"]                                 # event point 1 (16.56 FA/24h dev budget) and 2 (33.12)
REPORT_TYPES = ["fnsz", "gnsz", "absz", "cpsz", "tcsz", "tnsz"]   # types reported individually (others listed too)
DUR_BINS = [(0, 10), (10, 30), (30, 60), (60, 180), (180, math.inf)]   # s, half-open [lo, hi)
TYPE_RULE = ("majority TUSZ per-channel .csv seizure type over the event's labelled channel-seconds (seconds of the "
             "event overlapped by a .csv row on one of the 22 TCP channels, counted per type and summed over channels; "
             "ties -> alphabetical); group = group of the majority type: focal {fnsz, cpsz, spsz}, generalized "
             "{gnsz, absz, tcsz, tnsz, mysz}, else other")


# ----------------------------------------------------------------------------- shared input handling
def default_fusion_dir() -> Path:
    work = paths.get("work_dir")
    off = work / "tusz" / "fusion_1s"
    return off if (off / "eval_tracks.npz").exists() else work / "s1_gate" / "fusion_1s"


def source_label(fusion_dir) -> str:
    """'official' for <work>/tusz/fusion_1s, 's1_gate' for <work>/s1_gate/fusion_1s, else 'other'."""
    work = paths.get("work_dir").resolve()
    f = Path(fusion_dir).resolve()
    return {work / "tusz" / "fusion_1s": "official", work / "s1_gate" / "fusion_1s": "s1_gate"}.get(f, "other")


def command(module, source, out, extra=""):
    """The command line, with the work folder written as <work> (no path in results, RULES.md 4)."""
    f = {"official": "<work>/tusz/fusion_1s", "s1_gate": "<work>/s1_gate/fusion_1s"}.get(source, "<fusion_1s>")
    return f"python -m szfusion.eval.{module} --fusion_1s {f} --out {out}{extra}"


def load(fusion_dir, tokens_root):
    """evidence.load (row mapping, asserted) + per-row patient and true events."""
    D = E.load(fusion_dir, tokens_root)
    D["patient"] = np.array([D["index"][r]["patient"] for r in range(len(D["index"]))], object)[D["rec"]]
    D["te"] = P.true_events(D)
    return D


def predicted_mask(D, model, point):
    """(predicted event starts, ends, per-second mask) of one model at one frozen event point."""
    ev = D["sel"][model][point]
    ps = P.smooth(D["T"][model].astype(np.float64), D["rec"], ev["k"])
    st, en = P.events(ps, D["rec"], ev["thr"], ev["gap"], ev["min_dur"])
    diff = np.zeros(len(ps) + 1, np.int64); np.add.at(diff, st, 1); np.add.at(diff, en, -1)
    return st, en, np.cumsum(diff)[:-1] > 0


# ----------------------------------------------------------------------------- event typing
def event_type(rows, a, b):
    """rows: read_channel_events of the recording; event = EDF seconds [a, b) -> (majority type, counts)."""
    cnt = Counter()
    for ch, c0, c1, t in rows:
        if ch not in TCP22:
            continue
        lo, hi = max(a, math.floor(c0)), min(b, math.ceil(c1))     # seconds s with [s, s+1) overlapping [c0, c1)
        if hi > lo:
            cnt[t] += hi - lo
    if not cnt:
        return "none", {}
    top = max(cnt.values())
    return sorted(t for t, v in cnt.items() if v == top)[0], dict(cnt)


def group_of(t):
    return "focal" if t in E.FOCAL_TYPES else "generalized" if t in E.GENERALIZED_TYPES else "other"


def true_event_table(D):
    ts, te = D["te"]
    cache, out = {}, []
    for s, e in zip(ts, te):
        r = int(D["rec"][s]); i0 = D["blocks"][r][0]
        if r not in cache:
            cache[r] = tusz.read_channel_events(str(Path(D["index"][r]["path"]).with_suffix(".csv")))
        a, b = int(s - i0), int(e - i0)
        t, cnt = event_type(cache[r], a, b)
        out.append({"s": int(s), "e": int(e), "dur": b - a, "type": t, "group": group_of(t),
                    "n_types": len(cnt), "majority_share": (max(cnt.values()) / sum(cnt.values())) if cnt else None})
    return out


def detection(D, model, point):
    """Per true event: detected, signed latency, covered seconds."""
    st, en, pm = predicted_mask(D, model, point)
    ts, te = D["te"]
    cm = np.r_[0, np.cumsum(pm)]
    det, lat = [], []
    for s, e in zip(ts, te):
        ov = np.flatnonzero((st < e) & (en > s))
        det.append(len(ov) > 0); lat.append(int(st[ov[0]] - s) if len(ov) else None)
    return np.array(det), lat, (cm[te] - cm[ts]).astype(np.int64)


def summarise(sel_idx, det, lat, cov, dur):
    n = len(sel_idx)
    if n == 0:
        return {"n_events": 0}
    d = det[sel_idx]
    lats = [lat[i] for i in sel_idx if lat[i] is not None]
    return {"n_events": int(n), "n_detected": int(d.sum()), "event_sens": float(d.mean()),
            "median_latency_s": float(np.median(lats)) if lats else None,
            "second_recall": float(cov[sel_idx].sum() / dur[sel_idx].sum()), "n_seconds": int(dur[sel_idx].sum())}


def analyse(D):
    tab = true_event_table(D)
    dur = np.array([t["dur"] for t in tab]); types = [t["type"] for t in tab]; groups = [t["group"] for t in tab]
    all_types = sorted(set(types), key=lambda t: (t not in REPORT_TYPES, REPORT_TYPES.index(t) if t in REPORT_TYPES else 0, t))
    strata = {"all": list(range(len(tab)))}
    strata.update({f"type/{t}": [i for i, x in enumerate(types) if x == t] for t in all_types})
    strata.update({f"group/{g}": [i for i, x in enumerate(groups) if x == g] for g in ("focal", "generalized", "other")})
    for lo, hi in DUR_BINS:
        name = f"duration/{lo}-{'inf' if hi == math.inf else int(hi)}s"
        strata[name] = [i for i in range(len(tab)) if lo <= dur[i] < hi]
    res = {}
    for pt in POINTS:
        for m in MODELS:
            det, lat, cov = detection(D, m, pt)
            ref = P.event_scores(D, P.smooth(D["T"][m].astype(np.float64), D["rec"], D["sel"][m][pt]["k"]),
                                 D["sel"][m][pt]["thr"], D["sel"][m][pt]["gap"], D["sel"][m][pt]["min_dur"], D["te"])
            assert ref["n_detected"] == int(det.sum()), "detection differs from protocol_1s.event_scores"
            res.setdefault(pt, {})[m] = {k: summarise(v, det, lat, cov, dur) for k, v in strata.items()}
    shares = [t["majority_share"] for t in tab if t["majority_share"] is not None]
    return {"n_true_events": len(tab), "events_per_stratum": {k: len(v) for k, v in strata.items()},
            "events_with_more_than_one_type": int(sum(t["n_types"] > 1 for t in tab)),
            "events_without_csv_label": int(sum(t["type"] == "none" for t in tab)),
            "majority_type_share_min_median": [float(np.min(shares)), float(np.median(shares))],
            "results": res}


def main(argv=None):
    work = paths.get("work_dir")
    ap = argparse.ArgumentParser()
    ap.add_argument("--fusion_1s", default=None, help="1 s fusion folder; default: official if present, else s1_gate")
    ap.add_argument("--tokens", default=str(work / "tokens" / "tusz"))
    ap.add_argument("--out", default="results/tusz/by_type.json")
    a = ap.parse_args(argv)
    fdir = Path(a.fusion_1s) if a.fusion_1s else default_fusion_dir()
    src = source_label(fdir)
    D = load(fdir, a.tokens)
    res = analyse(D)
    out = {"provenance": {**provenance(command("by_type", src, a.out)), "source": src},
           "definitions": {
               "data": "TUSZ v2.0.6 eval, scored seconds of the 1 s fusion stage (eval_tracks.npz)",
               "models": {"fused_mean": "fusion", "catboost": "CatBoost branch", "linoss_mean": "LinOSS 5-seed mean"},
               "event_points": {"ev": "event point 1 (dev FA/24h <= 16.56), frozen on dev",
                                "ev2": "event point 2 (dev FA/24h <= 33.12), frozen on dev"},
               "type_rule": TYPE_RULE,
               "duration_bins_s": "[0,10), [10,30), [30,60), [60,180), [180,inf); duration = scored seconds of the event",
               "event_sens": "any-overlap, as fusion/protocol_1s.event_scores",
               "median_latency_s": "signed: start of the first predicted event overlapping the true event minus the "
                                   "true onset; median over detected events",
               "second_recall": "seconds of the stratum's true events inside predicted events / all their seconds"},
           **res}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    for pt in POINTS:
        print(f"== {pt}")
        for k in res["results"][pt]["fused_mean"]:
            row = "  ".join(f"{m}: {r['event_sens']:.3f}/{r['median_latency_s']}/{r['second_recall']:.3f}"
                            if r["n_events"] and r.get("median_latency_s") is not None else
                            f"{m}: {r.get('event_sens', float('nan')):.3f}/-/{r.get('second_recall', float('nan')):.3f}"
                            if r["n_events"] else f"{m}: -"
                            for m, r in ((m, res["results"][pt][m][k]) for m in MODELS))
            print(f"{k:26s} n={res['results'][pt]['fused_mean'][k]['n_events']:4d}  {row}")


if __name__ == "__main__":
    main()
