"""Official SzCORE event scoring (Dan et al. 2024; package timescoring 0.0.7) of the frozen eval outputs.

  python -m szfusion.eval.szcore --fusion_1s O1 --unit12 O12 --unit60 O60 --tokens T --catboost C --feat_dir F
         --markers M --out results/tusz/szcore.json

The timescoring defaults are used unchanged:
  * tolerance of 30 s before and 60 s after each reference seizure;
  * any overlap counts;
  * hypothesis events longer than 5 min are split;
  * hypothesis events less than 90 s apart are merged.
Per recording, timescoring.EventScoring compares the reference events (runs of labelled seconds) with the
hypothesis events (runs of detected seconds) and counts true positives (TP) and false positives (FP).
SampleScoring does the same second by second. The counts are summed over all recordings:
  sensitivity = TP / N_ref,  precision = TP / (TP + FP),  F1 = 2 TP / (2 TP + FP + (N_ref - TP)),
  false positives per day = FP / (scored seconds / 86,400).
SzCORE merges seizures that are less than 90 s apart, so the number of reference events (280 on TUSZ eval) is lower
than under our any-overlap scorer (359).
How each resolution becomes events (every threshold and post-processing setting was frozen on dev):
  1 s   fused per-second track -> centred moving average over k s -> threshold -> merge gaps -> minimum duration
        (operating points 1 and 2, fusion/selection.json)
  12 s  clip score > dev-F1 threshold -> each positive clip marks its 12 seconds
  60 s  the paper's post-processing (windows60) -> predicted intervals in seconds
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from timescoring import scoring
from timescoring.annotations import Annotation

from szfusion.catboost import postprocess
from szfusion.fusion import protocol_1s as P
from szfusion.units import windows60

PARAM = scoring.EventScoring.Parameters()
sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731


class Agg:
    """Sums SzCORE counts over recordings."""

    def __init__(self):
        self.tp = self.fp = self.ref = 0; self.sec = 0.0
        self.stp = self.sfp = self.sref = 0

    def add(self, ref_mask, hyp_mask):
        r, h = Annotation(np.asarray(ref_mask, bool), 1), Annotation(np.asarray(hyp_mask, bool), 1)
        e = scoring.EventScoring(r, h, PARAM)
        self.tp += e.tp; self.fp += e.fp; self.ref += e.refTrue; self.sec += len(ref_mask)
        s = scoring.SampleScoring(r, h)
        self.stp += s.tp; self.sfp += s.fp; self.sref += s.refTrue

    def out(self):
        ss = self.stp / self.sref; sp = self.stp / max(self.stp + self.sfp, 1)
        return {"event_sens": self.tp / self.ref, "event_prec": self.tp / max(self.tp + self.fp, 1),
                "event_f1": 2 * self.tp / (2 * self.tp + self.fp + self.ref - self.tp),
                "fp_per_day": self.fp / (self.sec / 86400), "n_ref": int(self.ref), "tp": int(self.tp),
                "fp": int(self.fp), "sample_sens": ss, "sample_prec": sp, "sample_f1": 2 * ss * sp / max(ss + sp, 1e-9),
                "hours": self.sec / 3600}


def event_mask(rec, ps, thr, gap, min_dur):
    st, en = P.events(ps, rec, thr, gap, min_dur)
    d = np.zeros(len(ps) + 1, np.int64); np.add.at(d, st, 1); np.add.at(d, en, -1)
    return np.cumsum(d)[:-1] > 0


def score_1s(fusion_dir):
    sel = json.loads((Path(fusion_dir) / "selection.json").read_text())["configs"]
    T = np.load(Path(fusion_dir) / "eval_tracks.npz")
    rec, y = T["rec"], T["y"]
    b = np.flatnonzero(np.r_[True, rec[1:] != rec[:-1], True])
    res = {}
    for key, name in (("catboost", "CatBoost branch alone"), ("linoss_mean", "LinOSS branch alone"),
                      ("fused_mean", "CatBoost x LinOSS fusion")):
        p = T[key].astype(np.float64)
        for tag in ("ev", "ev2"):
            e = sel[key][tag]
            m = event_mask(rec, P.smooth(p, rec, e["k"]), e["thr"], e["gap"], e["min_dur"])
            a = Agg()
            for i, j in zip(b[:-1], b[1:]):
                a.add(y[i:j] == 1, m[i:j])
            res[f"1s | {name} | point {1 if tag == 'ev' else 2}"] = a.out()
    return res


def score_12s(unit_dir, tokens, marker_dir):
    from szfusion.units import clips12
    sel = json.loads((Path(unit_dir) / "selection.json").read_text())
    Z = np.load(Path(unit_dir) / "eval_scores.npz")
    _, _, _, index = clips12.clips(marker_dir, tokens, "test")
    Y = np.load(Path(tokens) / "eval" / "y_sec.npy")
    res = {}
    for key, name in (("catboost", "CatBoost branch alone"), ("look_mean", "LinOSS branch alone"),
                      ("look_mean_cb", "CatBoost x LinOSS fusion")):
        z = Z[key]
        p = sig(np.where(np.isnan(z), np.nanmin(z), z))
        hyp = np.zeros(len(Y), bool)
        for s0 in Z["start"][p > sel["thr"][key]]:
            hyp[s0:s0 + 12] = True
        a = Agg()
        for e in index:
            o, n = e["offset"], e["n_sec"]
            a.add(Y[o:o + n] == 1, hyp[o:o + n])
        res[f"12s | {name}"] = a.out()
    return res


def score_60s(unit_dir, tokens, catboost, feat_dir):
    sel = json.loads((Path(unit_dir) / "selection.json").read_text())
    Z = np.load(Path(unit_dir) / "eval_scores.npz")
    _, _, _, _, meta, iv = windows60.windows("eval", tokens, Path(catboost) / "scores_eval.npz", feat_dir)
    meta = meta.reset_index(drop=True)
    durs = meta.groupby("rec_id", sort=False)["rec_duration"].first()
    res = {}
    for key, name in (("catboost", "CatBoost branch alone"), ("look_mean", "LinOSS branch alone"),
                      ("look_mean_cb", "CatBoost x LinOSS fusion")):
        z = Z[key]
        p = sig(np.where(np.isnan(z), np.nanmin(z), z))
        _, _, ev = postprocess.run(meta, p, sel["post"][key], 60.0)
        a = Agg()
        for rid, dur in durs.items():
            n = int(np.ceil(dur)); r = np.zeros(n, bool); h = np.zeros(n, bool)
            for s, e in iv.get(rid, []):
                r[int(s):int(np.ceil(e))] = True
            for s, e in ev.get(rid, []):
                h[int(s):int(np.ceil(min(e, n)))] = True
            a.add(r, h)
        res[f"60s | {name}"] = a.out()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fusion_1s", required=True)
    ap.add_argument("--unit12", required=True)
    ap.add_argument("--unit60", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--catboost", required=True)
    ap.add_argument("--feat_dir", required=True)
    ap.add_argument("--markers", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true", help="only to regenerate the same file (never to re-select)")
    a = ap.parse_args()
    if Path(a.out).exists() and not a.force:
        raise SystemExit(f"{a.out} exists: the frozen outputs were already scored once (RULES.md 3)")
    res = {**score_1s(a.fusion_1s), **score_12s(a.unit12, a.tokens, a.markers),
           **score_60s(a.unit60, a.tokens, a.catboost, a.feat_dir)}
    for k, r in res.items():
        print(f"{k:50s} sens {r['event_sens']:.3f}  prec {r['event_prec']:.3f}  F1 {r['event_f1']:.3f}  "
              f"FP/day {r['fp_per_day']:.2f}  ({r['tp']}/{r['n_ref']})")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"timescoring": "0.0.7", "parameters": vars(PARAM), "results": res}, indent=2))


if __name__ == "__main__":
    main()
