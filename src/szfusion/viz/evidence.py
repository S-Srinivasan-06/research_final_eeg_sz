"""Evidence figures (path.md section 5): what the fused detector marks, second by second and electrode by electrode.

  python -m szfusion.viz.evidence --fusion_1s <work>/tusz/fusion_1s [--tokens <work>/tokens/tusz]
         [--runs <work>/kaggle_out/szfusion-linoss-tusz/runs] [--out docs/figures]

Writes <out>/evidence_panel.png (case a alone), <out>/evidence_4cases.png (cases a-d) and
<out>/evidence_captions.md (the rules and every measured number; no recording or patient ID is printed, rendered
or written anywhere: the case-selection rules below regenerate the same cases deterministically).

Data: TUSZ eval, the 1 s fusion stage (fusion/run_1s.py): eval_tracks.npz and selection.json; event point 1
(selection.json configs.fused_mean.ev).

Row mapping (checked by assertions): eval_tracks rows are the scored seconds in store order. Its `rec` is the position
of the recording among the store's recordings with status "done" (protocol_1s.load_split), and the scored seconds of
a recording are a prefix [0, n_scored) of it (CatBoost covers a prefix, LinOSS every second). So track row b_r + j is
store row offset_r + j, i.e. EDF second [j, j + 1). Asserted: rec is non-decreasing, n_scored <= n_sec, and the
track's y equals the store's y_sec at the mapped rows, for every recording.

Events (same code as the scorer, fusion/protocol_1s.py):
  predicted events  centred moving average of fused_mean over k s -> > thr -> runs -> merge gaps <= gap s -> drop runs
                    shorter than min_dur s, with (k, thr, gap, min_dur) = selection.json configs.fused_mean.ev.
  true events       runs of y = 1 (.csv_bi seizure seconds) within a recording.
  detected          a true event overlapped by at least one predicted second (any overlap).
  false alarm       a predicted event that overlaps no annotated second.
  latency           start of the first predicted event that overlaps the true event minus the true onset (s); negative
                    when the smoothed track crosses the threshold before the annotated onset.
  coverage          share of the true event's seconds inside predicted events.
Event type: the labels of the per-channel .csv rows that overlap the true event. Focal-type: all labels in
FOCAL_TYPES; generalized-type: all labels in GENERALIZED_TYPES (other combinations belong to neither).
Channel share: share of the recording's valid TCP channels (index.json valid_channels; the _a montages have no A1/A2)
that carry a .csv seizure label in at least one second of the event (store y_ch).

Case rules (path.md 5; "median" = the candidate whose value is closest to the median of the candidate values; ties ->
smaller value, then store order, so the choice is deterministic):
  (a) focal-type, 30-180 s, labelled on > 0 and <= 50 % of the valid channels, detected; median latency.
  (b) generalized-type, labelled on all valid channels, detected; median latency.
  (c) false alarm; median duration.
  (d) focal-type, missed (coverage 0) or partially detected (coverage < 0.5); median duration.
Window: 240 s centred on the event (on the false alarm for c), shifted to lie inside the recording (shorter if the
recording is shorter).

Panel (one per case):
  traces    the 19 referential electrodes (data/tusz.py load_eeg 'ref19', resampled to 256 Hz), 0.5-35 Hz display
            filter (Butterworth order 4, zero phase), in DISPLAY_ORDER. Per-channel scaling: each trace is divided by
            its robust SD (1.4826 x median absolute deviation over the window); one row = ROW_RSD robust SD; values
            beyond +-CLIP_ROWS rows are clipped. Traces are rasterised.
  grid      a dotted vertical line every 1 s (the scoring unit).
  top band  per second: blue = annotated (y), green = inside a predicted event that overlaps an annotated second,
            teal = both, red = inside a false-alarm event, white = neither, grey hatch = not scored (recording tail
            without CatBoost cover). Above it: the smoothed fused probability and its threshold (dashed), and a
            bracket over the selected event.
  cells     per electrode and second: blue = annotated, i.e. a TCP channel touching the electrode carries a .csv label
            in that second (OR over the channels; data/montage.py channel_electrodes; FZ and PZ are on no TCP channel,
            so they can never be blue); green = a detected second (green in the top band) where the electrode's LinOSS
            occlusion importance is >= ATTR_FRAC x the window maximum; teal = blue and green; red = a false-alarm
            second where the importance is >= ATTR_FRAC x the window maximum (blue is drawn if a cell is both blue and
            red; the count of such cells is reported).
  occlusion importance  for each of the 5 LinOSS seeds (best.pt), the model is run on the panel window (token store
            rows of the window, padded to 240 s with mask 0 as in linoss/data.py RecordingWindows) once intact and
            once per electrode with that electrode's 100 tokens set to 0 (the z-scored global mean) in every second of
            the window; importance(e, t) = logit(t) - logit_without_e(t), averaged over the seeds. The window maximum
            is the maximum over all 19 electrodes and all seconds of the window.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from szfusion import paths  # noqa: E402
from szfusion.data import tusz  # noqa: E402
from szfusion.data.montage import ELECTRODES_19, TCP22, channel_electrodes  # noqa: E402
from szfusion.fusion import protocol_1s as P  # noqa: E402
from szfusion.linoss.model import build  # noqa: E402

FOCAL_TYPES = {"fnsz", "cpsz", "spsz"}              # TUSZ .csv labels counted as focal-type (path.md 5)
GENERALIZED_TYPES = {"gnsz", "absz", "tcsz", "tnsz", "mysz"}   # ... as generalized-type
WIN = 240                    # panel length, s (path.md 5; = the LinOSS segment length)
FS_DISPLAY = 256             # Hz, display sampling rate of load_eeg
BAND = (0.5, 35.0)           # Hz, display filter (path.md 5)
ROW_RSD = 8.0                # one trace row = 8 robust SD of that channel in the window
CLIP_ROWS = 0.9              # traces clipped at +-0.9 row (so a trace never reaches the next row's baseline)
ATTR_FRAC = 0.5              # occlusion threshold, fraction of the window maximum (path.md 5)
FOCAL_DUR = (30, 180)        # s, duration range of case (a)
FOCAL_MAX_SHARE = 0.5        # case (a): labelled on at most 50 % of the valid channels
PARTIAL_COVER = 0.5          # case (d): "partial" = less than half of the event's seconds detected
DISPLAY_ORDER = ["FP1", "F7", "T3", "T5", "O1", "F3", "C3", "P3", "FZ", "CZ", "PZ",
                 "F4", "C4", "P4", "FP2", "F8", "T4", "T6", "O2"]   # left chains, midline, right chains (top to bottom)

# cell / band categories and colours (light for cells under traces, saturated for the top band)
NONE, BLUE, GREEN, TEAL, RED = 0, 1, 2, 3, 4
CELL = ["#FFFFFF", "#A6CBEA", "#A8D8A0", "#62C6BF", "#F2A3A3"]
BAND_C = ["#FFFFFF", "#3A7DC9", "#3FA34D", "#13948C", "#D63C3C"]
LABELS = {BLUE: "annotated seizure", GREEN: "detected (fused event; cells: occlusion >= 0.5 x max)",
          TEAL: "annotated and detected", RED: "false alarm (event; cells: occlusion >= 0.5 x max)"}
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False})


# ----------------------------------------------------------------------------- data and row mapping
def load(fusion_dir, tokens_root):
    fd, td = Path(fusion_dir), Path(tokens_root) / "eval"
    T = np.load(fd / "eval_tracks.npz")
    sel = json.loads((fd / "selection.json").read_text())["configs"]
    index = [e for e in json.loads((td / "index.json").read_text()) if e["status"] == "done"]
    y_sec = np.load(td / "y_sec.npy", mmap_mode="r")
    y_ch = np.load(td / "y_ch.npy", mmap_mode="r")
    rec, y = T["rec"].astype(np.int64), T["y"].astype(np.int8)
    assert np.all(np.diff(rec) >= 0), "eval_tracks rows are not in store order"
    b = np.flatnonzero(np.r_[True, rec[1:] != rec[:-1], True])
    blocks = {}
    for i, j in zip(b[:-1], b[1:]):
        r = int(rec[i]); e = index[r]; n = j - i
        assert r not in blocks and n <= e["n_sec"], "a recording appears twice or has more scored than stored seconds"
        assert np.array_equal(y[i:j], np.asarray(y_sec[e["offset"]:e["offset"] + n])), "y mismatch at mapped rows"
        blocks[r] = (int(i), int(j))
    assert list(tusz.TCP22_NAMES) == list(TCP22), "y_ch column order differs from montage.TCP22"
    return {"T": T, "sel": sel, "index": index, "y_sec": y_sec, "y_ch": y_ch, "rec": rec, "y": y, "blocks": blocks}


def predicted(D):
    ev = D["sel"]["fused_mean"]["ev"]
    p = D["T"]["fused_mean"].astype(np.float64)
    ps = P.smooth(p, D["rec"], ev["k"])
    st, en = P.events(ps, D["rec"], ev["thr"], ev["gap"], ev["min_dur"])
    return ps, st, en, ev


def event_tables(D):
    """True events (with type, channel share, detection, latency, coverage) and false alarms, in track rows."""
    ps, pst, pen, ev = predicted(D)
    rec, y = D["rec"], D["y"]
    diff = np.zeros(len(y) + 1, np.int64); np.add.at(diff, pst, 1); np.add.at(diff, pen, -1)
    pm = np.cumsum(diff)[:-1] > 0
    cy = np.r_[0, np.cumsum(y)]
    fa = (cy[pen] - cy[pst]) == 0
    ts, te = P.true_events(D)
    csv_cache = {}
    true = []
    for s, e in zip(ts, te):
        r = int(rec[s]); i0 = D["blocks"][r][0]; info = D["index"][r]
        a, b = int(s - i0), int(e - i0)                                   # EDF seconds [a, b)
        if r not in csv_cache:
            csv_cache[r] = tusz.read_channel_events(str(Path(info["path"]).with_suffix(".csv")))
        types = sorted({t for _, c0, c1, t in csv_cache[r] if c0 < b and c1 > a})
        kind = "focal" if types and set(types) <= FOCAL_TYPES else \
            "generalized" if types and set(types) <= GENERALIZED_TYPES else "other"
        valid = [TCP22.index(c) for c in info["valid_channels"]]
        yc = np.asarray(D["y_ch"][info["offset"] + a:info["offset"] + b])[:, valid]
        share = float(yc.max(0).mean())
        ov = np.flatnonzero((pst < e) & (pen > s))
        cover = float(pm[s:e].mean())
        true.append({"rec": r, "a": a, "b": b, "dur": b - a, "types": types, "kind": kind, "share": share,
                     "n_valid": len(valid), "n_labelled": int(yc.max(0).sum()), "detected": len(ov) > 0,
                     "latency": int(pst[ov[0]] - s) if len(ov) else None, "cover": cover})
    fas = []
    for s, e in zip(pst[fa], pen[fa]):
        r = int(rec[s]); i0 = D["blocks"][r][0]
        fas.append({"rec": r, "a": int(s - i0), "b": int(e - i0), "dur": int(e - s)})
    return true, fas, (ps, pst, pen, pm, fa, ev)


def pick_median(cands, key):
    """The candidate closest to the median of `key`; ties -> smaller value, then store order."""
    if not cands:
        return None, None
    v = np.array([c[key] for c in cands], float)
    med = float(np.median(v))
    order = sorted(range(len(cands)), key=lambda i: (abs(v[i] - med), v[i], cands[i]["rec"], cands[i]["a"]))
    return cands[order[0]], med


def select_cases(true, fas):
    rules = {
        "a": ("focal-type event of 30-180 s, labelled on > 0 and <= 50 % of the valid channels, detected; median "
              "detection latency",
              [t for t in true if t["kind"] == "focal" and FOCAL_DUR[0] <= t["dur"] <= FOCAL_DUR[1]
               and 0 < t["share"] <= FOCAL_MAX_SHARE and t["detected"]], "latency"),
        "b": ("generalized-type event labelled on all valid channels, detected; median detection latency",
              [t for t in true if t["kind"] == "generalized" and t["share"] == 1.0 and t["detected"]], "latency"),
        "c": ("false alarm (predicted event overlapping no annotated second); median duration", list(fas), "dur"),
        "d": ("focal-type event missed (coverage 0) or partially detected (coverage < 0.5); median duration",
              [t for t in true if t["kind"] == "focal" and t["cover"] < PARTIAL_COVER], "dur"),
    }
    cases = {}
    for k, (txt, cands, key) in rules.items():
        c, med = pick_median(cands, key)
        cases[k] = {"rule": txt, "n_candidates": len(cands), "key": key, "median": med, "event": c}
    return cases


# ----------------------------------------------------------------------------- one panel's data
def load_models(runs, seeds):
    ms = []
    for s in seeds:
        res = json.loads((Path(runs) / s / "results.json").read_text())
        m = build(res["cfg"])
        m.load_state_dict(torch.load(Path(runs) / s / "best.pt", map_location="cpu", weights_only=True)["model"])
        ms.append(m.eval())
    return ms


@torch.no_grad()
def occlusion(models, tokens):
    """tokens (W, 19, 100) -> importance (19, W): mean over seeds of logit - logit with one electrode zeroed."""
    W = tokens.shape[0]
    x = np.zeros((1, WIN, *tokens.shape[1:]), np.float32); x[0, :W] = tokens
    m = np.zeros((1, WIN), np.float32); m[0, :W] = 1
    x, m = torch.from_numpy(x), torch.from_numpy(m)
    imp = np.zeros((tokens.shape[1], W))
    for mdl in models:
        xs = x.repeat(tokens.shape[1] + 1, 1, 1, 1)
        for el in range(tokens.shape[1]):
            xs[el + 1, :, el] = 0.0
        z = mdl(xs, m.repeat(len(xs), 1)).numpy()[:, :W]
        imp += (z[0][None] - z[1:]) / len(models)
    return imp


def panel_data(D, tabs, case, models, X):
    ps, pst, pen, pm, fa, ev = tabs
    ev_ = case["event"]; r = ev_["rec"]; info = D["index"][r]
    o, n = info["offset"], info["n_sec"]
    i0, i1 = D["blocks"][r]; n_sc = i1 - i0
    W = min(WIN, n)
    w0 = int(np.clip(round((ev_["a"] + ev_["b"]) / 2 - WIN / 2), 0, n - W))
    t = np.arange(w0, w0 + W)                                            # EDF seconds of the window
    scored = t < n_sc
    rows = i0 + np.minimum(t, n_sc - 1)
    y = np.asarray(D["y_sec"][o + w0:o + w0 + W]).astype(bool)
    in_pred = np.where(scored, pm[rows], False)
    # per-second: which predicted event, and is it a false alarm
    ev_id = np.searchsorted(pst, rows, side="right") - 1
    in_fa = in_pred & (ev_id >= 0) & fa[np.clip(ev_id, 0, len(fa) - 1)]
    det = in_pred & ~in_fa
    band = np.full(W, NONE)
    band[y & ~det] = BLUE; band[det & ~y] = GREEN; band[det & y] = TEAL; band[in_fa] = RED
    # electrodes annotated (OR rule over the TCP channels that touch them)
    yc = np.asarray(D["y_ch"][o + w0:o + w0 + W]) > 0
    ann = np.zeros((19, W), bool)
    for c, els in enumerate(channel_electrodes(TCP22, ELECTRODES_19)):
        for el in els:
            ann[el] |= yc[:, c]
    imp = occlusion(models, np.asarray(X[o + w0:o + w0 + W], np.float32))
    vmax = float(imp.max())
    hot = imp >= ATTR_FRAC * vmax if vmax > 0 else np.zeros_like(imp, bool)
    cells = np.full((19, W), NONE)
    cells[ann] = BLUE
    g = hot & det[None]
    cells[g & ~ann] = GREEN; cells[g & ann] = TEAL
    rd = hot & in_fa[None]
    n_blue_red = int((rd & ann).sum())
    cells[rd & ~ann] = RED
    # signals
    x, names = tusz.load_eeg(info["path"], "ref19", fs=FS_DISPLAY, bandpass=BAND)
    sig = x[:, w0 * FS_DISPLAY:(w0 + W) * FS_DISPLAY]
    sig = np.stack([sig[names.index(e)] for e in ELECTRODES_19])         # ELECTRODES_19 order, like tokens/cells
    prob = np.where(scored, ps[rows], np.nan)
    hot_el = sorted({ELECTRODES_19[i] for i in np.flatnonzero((hot & (det | in_fa)[None]).any(1))},
                    key=DISPLAY_ORDER.index)
    ann_el = sorted({ELECTRODES_19[i] for i in np.flatnonzero(ann.any(1))}, key=DISPLAY_ORDER.index)
    stats = {"window_len_s": W, "n_annotated_s": int(y.sum()), "n_detected_s": int(det.sum()),
             "n_false_alarm_s": int(in_fa.sum()), "n_unscored_s": int((~scored).sum()),
             "annotated_electrodes": ann_el, "attributed_electrodes": hot_el,
             "n_cells": {k: int((cells == v).sum()) for k, v in (("blue", BLUE), ("green", GREEN), ("teal", TEAL),
                                                              ("red", RED))},
             "n_cells_blue_and_red": n_blue_red, "occlusion_window_max_logit": vmax}
    return {"w0": w0, "W": W, "sig": sig, "band": band, "cells": cells, "prob": prob, "thr": ev["thr"],
            "scored": scored, "ev_rel": (ev_["a"] - w0, ev_["b"] - w0), "stats": stats}


# ----------------------------------------------------------------------------- drawing
def draw(fig, spec, pd, title, k_smooth):
    gs = spec.subgridspec(3, 1, height_ratios=[0.9, 0.35, 19 * 0.55], hspace=0.04)
    axp, axb, ax = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    W = pd["W"]; t = np.arange(W)
    # probability strip
    axp.plot(t + 0.5, pd["prob"], color="#333333", lw=0.9)
    axp.axhline(pd["thr"], color="#333333", lw=0.7, ls="--")
    a0, a1 = pd["ev_rel"]                                                # the selected event (bracket)
    axp.plot([a0, a0, a1, a1], [0.82, 0.95, 0.95, 0.82], color="#000000", lw=1.2, clip_on=False)
    axp.text((a0 + a1) / 2, 0.97, "selected event", ha="center", va="bottom", fontsize=7.5)
    axp.set_xlim(0, W); axp.set_ylim(0, 1.25); axp.set_yticks([0, 1]); axp.set_xticks([])
    axp.set_ylabel(f"fused P\n(k={k_smooth} s)", fontsize=7.5, rotation=0, ha="right", va="center")
    axp.set_title(title, loc="left", fontsize=9.5)
    axp.spines["bottom"].set_visible(False)
    # top band
    axb.imshow(pd["band"][None], aspect="auto", cmap=ListedColormap(BAND_C), vmin=0, vmax=4,
               extent=[0, W, 1, 0], interpolation="nearest")
    if (~pd["scored"]).any():
        u = np.flatnonzero(~pd["scored"])
        axb.add_patch(plt.Rectangle((u[0], 0), len(u), 1, fill=False, hatch="///", ec="#999999", lw=0))
    axb.set_yticks([0.5]); axb.set_yticklabels(["events"], fontsize=7.5); axb.set_xticks([])
    for s in ("top", "right", "bottom", "left"):
        axb.spines[s].set_visible(True); axb.spines[s].set_linewidth(0.5)
    # cells + traces
    order = [ELECTRODES_19.index(e) for e in DISPLAY_ORDER]
    ax.imshow(pd["cells"][order], aspect="auto", cmap=ListedColormap(CELL), vmin=0, vmax=4,
              extent=[0, W, 19, 0], interpolation="nearest")
    tt = np.arange(pd["sig"].shape[1]) / FS_DISPLAY
    for row, el in enumerate(order):
        s = pd["sig"][el].astype(np.float64)
        rsd = 1.4826 * np.median(np.abs(s - np.median(s)))
        v = np.zeros_like(s) if rsd <= 0 else np.clip((s - np.median(s)) / (rsd * ROW_RSD), -CLIP_ROWS, CLIP_ROWS)
        ax.plot(tt, row + 0.5 - v, color="#111111", lw=0.35, rasterized=True)
    for a in (axp, axb, ax):
        for x0 in range(1, W):
            a.axvline(x0, color="#8C8C8C", lw=0.3, ls=(0, (1, 2)), zorder=0.5 if a is axp else 3)
    ax.set_xlim(0, W); ax.set_ylim(19, 0)
    ax.set_yticks(np.arange(19) + 0.5); ax.set_yticklabels(DISPLAY_ORDER, fontsize=7.5)
    ax.set_xticks(np.arange(0, W + 1, 20)); ax.set_xlabel("time from window start (s)")
    for b in (5, 8, 11, 14):
        ax.axhline(b, color="#555555", lw=0.6)


def legend(fig, y):
    h = [Patch(fc=CELL[k], ec=BAND_C[k], label=LABELS[k]) for k in (BLUE, GREEN, TEAL, RED)]
    h.append(Patch(fc="white", ec="#999999", hatch="///", label="not scored"))
    fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, y))


TITLES = {"a": "(a) Case A: focal-type seizure, detected", "b": "(b) Case B: generalized-type seizure, detected",
          "c": "(c) Case C: false alarm", "d": "(d) Case D: focal-type seizure, missed or partly detected"}


def describe(k, c, pd):
    e = c["event"]; s = pd["stats"]
    if k == "c":
        return f"{e['dur']} s"
    lat = "" if e["latency"] is None else f", latency {e['latency']:+d} s"
    return (f"{'/'.join(e['types'])}, {e['dur']} s, labelled on {e['n_labelled']}/{e['n_valid']} channels"
            f"{lat}, coverage {100 * e['cover']:.0f} %")


def captions(cases, pds, ev, n_true, n_fa, source):
    L = ["# Evidence figures: rules and measured numbers", "",
         f"Generated by `python -m szfusion.viz.evidence` (src/szfusion/viz/evidence.py) from the {source} 1 s fusion "
         "outputs on TUSZ v2.0.6 eval. No recording or patient ID is shown; the rules below select the same cases "
         "again from the same outputs.", "",
         "## Panel rules (path.md section 5)", "",
         f"- 240 s window centred on the selected event and shifted to lie inside the recording. 19 referential "
         f"electrodes (REF or LE reference as recorded), resampled to {FS_DISPLAY} Hz, 0.5-35 Hz display filter "
         "(Butterworth order 4, zero phase). Rows top to bottom: left temporal chain, left parasagittal chain, "
         "midline, right parasagittal chain, right temporal chain.",
         f"- Per-channel scaling: each trace is divided by its robust SD (1.4826 x median absolute deviation over the "
         f"window); one row = {ROW_RSD:g} robust SD; values beyond +-{CLIP_ROWS:g} row are clipped. Amplitudes are "
         "therefore comparable within a trace, not across traces.",
         "- Dotted vertical lines: every 1 s (the scoring unit).",
         f"- Event band: blue = annotated seizure second (.csv_bi); green = second of a predicted event that overlaps "
         f"an annotated second; teal = both; red = second of a false-alarm event (a predicted event overlapping no "
         f"annotated second); hatched = not scored (recording tail without CatBoost cover). Predicted events = event "
         f"point 1 of the fusion, frozen on dev: centred moving average over k = {ev['k']} s, threshold "
         f"{ev['thr']:.4f}, gaps <= {ev['gap']} s merged, events shorter than {ev['min_dur']} s dropped. The strip "
         "above shows that smoothed probability and the threshold (dashed); the black bracket marks the selected "
         "event (other events in the window are drawn by the same rules).",
         "- Electrode cells: blue = a TCP bipolar channel touching the electrode carries a per-channel (.csv) seizure "
         "label in that second (OR over the channels). FZ and PZ lie on no TCP channel, so they can never be blue. "
         f"Green = a detected second (green or teal in the band) in which the electrode's LinOSS occlusion importance "
         f"is >= {ATTR_FRAC:g} x the window maximum; teal = blue and green; red = the same importance rule inside a "
         "false-alarm second.",
         "- Occlusion importance: each of the 5 LinOSS seeds is run on the window (tokens of the window, padded to "
         "240 s with mask 0) intact and with one electrode's 100 tokens set to 0 (the z-scored global mean) in every "
         "second; importance = intact logit - occluded logit, averaged over the 5 seeds; window maximum = over all 19 "
         "electrodes and all seconds of the window. It explains the LinOSS branch only; the CatBoost branch has no "
         "per-electrode attribution here.", "",
         "## Case-selection rules and outcomes", "",
         f"Pool: {n_true} annotated events (runs of seizure seconds) and {n_fa} false alarms at event point 1. "
         "Focal-type = all overlapping .csv labels in {fnsz, cpsz, spsz}; generalized-type = all in {gnsz, absz, "
         "tcsz, tnsz, mysz}. Latency = start of the first overlapping predicted event minus the annotated onset "
         "(negative = before onset; the centred smoothing looks ahead). 'Median' = the candidate closest to the "
         "median of the candidates (ties: smaller value, then store order).", ""]
    for k in "abcd":
        c = cases[k]
        L.append(f"**Case {k.upper()}.** Rule: {c['rule']}. Candidates: {c['n_candidates']}; median {c['key']} "
                 f"{c['median']:.1f} s.")
        if c["event"] is None:
            L.append("No candidate; no panel."); L.append(""); continue
        s = pds[k]["stats"]
        L.append(f"Selected: {describe(k, c, pds[k])}. Window {s['window_len_s']} s: {s['n_annotated_s']} annotated, "
                 f"{s['n_detected_s']} detected, {s['n_false_alarm_s']} false-alarm, {s['n_unscored_s']} unscored "
                 f"seconds. Annotated electrodes: {', '.join(s['annotated_electrodes']) or 'none'}. Electrodes with "
                 f"attributed cells: {', '.join(s['attributed_electrodes']) or 'none'}. Cells blue/green/teal/red: "
                 f"{s['n_cells']['blue']}/{s['n_cells']['green']}/{s['n_cells']['teal']}/{s['n_cells']['red']} "
                 f"(blue-and-red cells drawn blue: {s['n_cells_blue_and_red']}); occlusion window maximum "
                 f"{s['occlusion_window_max_logit']:.2f} logit.")
        L.append("")
    return "\n".join(L)


def main(argv=None):
    work = paths.get("work_dir")
    ap = argparse.ArgumentParser()
    ap.add_argument("--fusion_1s", default=str(work / "tusz" / "fusion_1s"),
                    help="1 s fusion folder (eval_tracks.npz, selection.json); the official run by default")
    ap.add_argument("--tokens", default=str(work / "tokens" / "tusz"))
    ap.add_argument("--runs", default=str(work / "kaggle_out" / "szfusion-linoss-tusz" / "runs"))
    ap.add_argument("--seeds", nargs="+", default=["s0", "s1", "s2", "s3", "s4"])
    ap.add_argument("--source", default="official", help="label written into the captions (official / s1_gate)")
    ap.add_argument("--out", default="docs/figures")
    a = ap.parse_args(argv)
    torch.set_num_threads(4)
    D = load(a.fusion_1s, a.tokens)
    print(f"row mapping verified: {len(D['blocks'])} scored recordings, {len(D['y'])} scored seconds")
    true, fas, tabs = event_tables(D)
    cases = select_cases(true, fas)
    models = load_models(a.runs, a.seeds)
    X = np.load(Path(a.tokens) / "eval" / "tokens.npy", mmap_mode="r")
    ev = tabs[-1]
    pds = {}
    for k in "abcd":
        c = cases[k]
        print(f"case {k}: {c['n_candidates']} candidates, median {c['key']} "
              f"{'-' if c['median'] is None else round(c['median'], 1)}")
        if c["event"] is not None:
            pds[k] = panel_data(D, tabs, c, models, X)
            print(f"   selected: {describe(k, c, pds[k])}")
            print(f"   {json.dumps(pds[k]['stats'])}")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if "a" in pds:
        fig = plt.figure(figsize=(16, 10.5))
        draw(fig, fig.add_gridspec(1, 1, left=0.06, right=0.99, top=0.95, bottom=0.11)[0], pds["a"],
             f"{TITLES['a']}: {describe('a', cases['a'], pds['a'])}", ev["k"])
        legend(fig, 0.0)
        fig.savefig(out / "evidence_panel.png", dpi=150)
        plt.close(fig)
    ks = [k for k in "abcd" if k in pds]
    fig = plt.figure(figsize=(26, 19))
    gs = fig.add_gridspec(2, 2, left=0.04, right=0.995, top=0.97, bottom=0.07, hspace=0.16, wspace=0.07)
    for i, k in enumerate(ks):
        draw(fig, gs[i // 2, i % 2], pds[k], f"{TITLES[k]}: {describe(k, cases[k], pds[k])}", ev["k"])
    legend(fig, 0.005)
    fig.savefig(out / "evidence_4cases.png", dpi=110)
    plt.close(fig)
    (out / "evidence_captions.md").write_text(captions(cases, pds, ev, len(true), len(fas), a.source),
                                              encoding="utf-8")
    print(f"written: evidence_panel.png, evidence_4cases.png, evidence_captions.md in {a.out}")


if __name__ == "__main__":
    main()
