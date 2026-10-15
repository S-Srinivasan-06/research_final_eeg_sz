"""Ablation table of the CatBoost × LinOSS fusion: the fusion against each of its branches, at 1 s, 12 s and 60 s.

  python -m szfusion.eval.ablation [--out results/tusz/ablation.json] [--md docs/tables_ablation.md]

Rows (TUSZ):
  fusion            CatBoost × LinOSS fusion, 5-seed ensemble   1 s: fused_mean     12 s / 60 s: look_mean_cb
  w/o LinOSS        the CatBoost branch alone                   1 s: catboost       12 s / 60 s: catboost
  w/o CatBoost      the LinOSS branch alone, 5-seed ensemble    1 s: linoss_mean    12 s / 60 s: look_mean
  per seed          mean ± sd (ddof = 1) over the 5 seeds of the fusion (fused_s*, look_r*_cb) and of the LinOSS
                    branch (linoss_s*, look_r*); CatBoost has no seeds.
Inputs (frozen outputs; nothing is selected here, every threshold is the dev-frozen one):
  1 s   <fusion_1s>/eval_results.json, dev_results.json, selection.json, eval_tracks.npz   (fusion/run_1s.py)
  12 s  <u12>/eval_results.json, dev_results.json                                         (units/overlooker.py)
  60 s  <u60>/eval_results.json, dev_results.json, selection.json, eval_scores.npz
Source of each folder: the official run (<work>/tusz/...) when its eval_results.json exists, otherwise the S1-gate
copy (<work>/s1_gate/...), which reproduces it numerically (PROGRESS.md S1 entries). The source is written per
resolution into the JSON and the Markdown.

What is copied and what is computed:
  1 s   AUROC, AUPRC, F1, precision, recall are recomputed from eval_tracks.npz at the frozen f1_thr
        (protocol_1s.sec_scores) and asserted equal to eval_results.json (|Δ| <= 1e-6; the tracks are stored as float32); specificity is computed in
        the same pass (seconds with p > f1_thr). Event metrics are copied (ev = point 1, ev2 = point 2).
        Dev: everything is copied; dev specificity is derived exactly from the stored counts (see derive_spec).
  12 s  every metric, specificity included, is copied (window_metrics12 already stores it).
  60 s  AUROC, AUPRC and the events are copied. Precision, recall and specificity at the frozen window threshold
        (post.threshold, applied to the raw window probability with ">=", as eval/metrics.window_metrics that
        produced f1_at_thr) are computed from eval_scores.npz, with F1 asserted equal to f1_at_thr. On dev these three
        are not stored and dev window scores are not kept, so they are left empty.
CHB-MIT (results/chbmit/test_results.json, HyBrain protocol): 12 s clip and 1 s point-wise AUROC, AUPRC, F1,
precision, recall copied; specificity derived exactly from the stored n, pos, precision and recall.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from szfusion import paths
from szfusion.eval.mechanism import provenance
from szfusion.fusion import protocol_1s as P

REPO = Path(__file__).resolve().parents[3]
CONFIG = REPO / "configs" / "tusz_final.yaml"
TOL = 1e-6                     # max |Δ| recomputed vs stored metric: eval_tracks.npz stores float32 tracks, so AUPRC
                               # moves at the 1e-7 level against the float64 scoring run
ROWS = [("fusion", "CatBoost × LinOSS fusion (5-seed ensemble)"),
        ("wo_linoss", "w/o LinOSS (= CatBoost branch)"),
        ("wo_catboost", "w/o CatBoost (= LinOSS branch, 5-seed ensemble)"),
        ("fusion_seeds", "Fusion, per seed (mean ± sd, n = 5)"),
        ("linoss_seeds", "LinOSS branch, per seed (mean ± sd, n = 5)")]
KEYS_1S = {"fusion": "fused_mean", "wo_linoss": "catboost", "wo_catboost": "linoss_mean"}
KEYS_U = {"fusion": "look_mean_cb", "wo_linoss": "catboost", "wo_catboost": "look_mean"}
SEEDS_1S = {"fusion_seeds": [f"fused_s{i}" for i in range(5)], "linoss_seeds": [f"linoss_s{i}" for i in range(5)]}
SEEDS_U = {"fusion_seeds": [f"look_r{i}_cb" for i in range(5)], "linoss_seeds": [f"look_r{i}" for i in range(5)]}
sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731


# ----------------------------------------------------------------------------- inputs
def sources():
    """-> {res: (folder, 'official' | 's1_gate')} for 1s / 12s / 60s."""
    work = paths.get("work_dir")
    out = {}
    for res, name, need in (("1s", "fusion_1s", "eval_tracks.npz"), ("12s", "u12", "eval_scores.npz"),
                            ("60s", "u60", "eval_scores.npz")):
        off = work / "tusz" / name
        ok = all((off / f).exists() for f in ("eval_results.json", "dev_results.json", "selection.json", need))
        out[res] = (off, "official") if ok else (work / "s1_gate" / name, "s1_gate")
    return out


def rd(p):
    return json.loads(Path(p).read_text())


def config_sha256():
    return hashlib.sha256(CONFIG.read_bytes()).hexdigest() if CONFIG.exists() else None


def mean_sd(rows, keys, metric):
    v = [rows[k][metric] for k in keys if rows.get(k) and rows[k].get(metric) is not None]
    return [float(np.mean(v)), float(np.std(v, ddof=1))] if len(v) == len(keys) else None


def derive_spec(n, pos, precision, recall):
    """Exact specificity from stored counts: TP = round(recall × pos), FP = round(TP / precision) - TP,
    spec = 1 - FP / (n - pos). Exact because recall and precision are ratios of integers."""
    tp = int(round(recall * pos))
    fp = int(round(tp / precision)) - tp if precision > 0 else 0
    return 1 - fp / (n - pos)


# ----------------------------------------------------------------------------- 1 s
def one_sec(folder):
    ev, dv, sel = rd(folder / "eval_results.json"), rd(folder / "dev_results.json"), rd(folder / "selection.json")["configs"]
    T = np.load(folder / "eval_tracks.npz")
    y = T["y"].astype(np.int8)
    out = {"eval": {}, "dev": {}, "counts": {"eval_seconds": int(len(y)), "eval_seizure_seconds": int(y.sum())}}
    names = list(KEYS_1S.values()) + sum(SEEDS_1S.values(), [])
    for split, R in (("eval", ev["results"]), ("dev", dv["results"])):
        rows = {}
        for n in names:
            r = R[n]
            row = {k: r[k] for k in ("auroc", "auprc", "f1", "precision", "recall", "f1_thr")}
            if split == "eval":
                p = T[n].astype(np.float64)
                chk = P.sec_scores(y, p, sel[n]["f1_thr"])
                for k in ("auroc", "auprc", "f1", "precision", "recall"):
                    assert abs(chk[k] - r[k]) <= TOL, f"1 s {n} {k}: {chk[k]} vs stored {r[k]}"
                neg = y == 0
                row["specificity"] = float((p[neg] <= sel[n]["f1_thr"]).mean())
                row["specificity_derived_check"] = derive_spec(len(y), int(y.sum()), r["precision"], r["recall"])
                assert abs(row["specificity"] - row["specificity_derived_check"]) <= 1e-12
                del row["specificity_derived_check"]
            else:
                # dev scored seconds N from the stored FA rate: hours = n_fa × 24 / FA_per_24h (fusion row, exact)
                rf = R["fused_mean"]
                N = int(round(rf["ev_n_fa"] * 24 / rf["ev_fa_per_24h"] * 3600))
                P_ = dv["info"]["seizure_seconds_scored"]
                row["specificity"] = derive_spec(N, P_, r["precision"], r["recall"])
                out["counts"]["dev_seconds"] = N; out["counts"]["dev_seizure_seconds"] = P_
            for tag in ("ev", "ev2"):
                for k in ("event_sens", "fa_per_24h", "event_prec", "time_recall", "time_ppv", "n_detected",
                          "n_true", "n_fa"):
                    row[f"{tag}_{k}"] = r[f"{tag}_{k}"]
            rows[n] = row
        tab = {k: rows[v] for k, v in KEYS_1S.items()}
        for k, ks in SEEDS_1S.items():
            tab[k] = {m: mean_sd(rows, ks, m) for m in rows[ks[0]] if m not in ("f1_thr",)}
        out[split] = tab
    # sanity: the eval second count equals the derivation used for dev
    r = ev["results"]["fused_mean"]
    assert int(round(r["ev_n_fa"] * 24 / r["ev_fa_per_24h"] * 3600)) == len(y)
    out["frozen"] = {n: {"f1_thr": sel[n]["f1_thr"], "ev": sel[n]["ev"], "ev2": sel[n]["ev2"]} for n in KEYS_1S.values()}
    return out


# ----------------------------------------------------------------------------- 12 s / 60 s
def unit(folder, u):
    ev, dv, sel = rd(folder / "eval_results.json"), rd(folder / "dev_results.json"), rd(folder / "selection.json")
    names = list(KEYS_U.values()) + sum(SEEDS_U.values(), [])
    out = {"eval": {}, "dev": {}, "epochs": ev.get("epochs")}
    Z = np.load(folder / "eval_scores.npz") if u == 60 else None
    for split, R in (("eval", ev["results"]), ("dev", dv["results"])):
        rows = {}
        for n in names:
            r = R[n]
            if u == 12:
                row = {k: r[k] for k in ("auroc", "auprc", "f1", "precision", "recall", "specificity", "thr")}
                row["n_units"] = r["clips"]
            else:
                row = {k: r[k] for k in ("auroc", "auprc", "event_sens", "fa_per_h", "event_ppv", "time_recall",
                                         "time_ppv", "n_true", "n_detected", "n_false_alarms", "hours")}
                row["f1"] = r["f1_at_thr"]
                row["thr"] = sel["post"][n]["threshold"]
                row["precision"] = row["recall"] = row["specificity"] = None
                if split == "eval":
                    z = Z[n]; yy = Z["y"].astype(int)
                    p = sig(np.where(np.isnan(z), np.nanmin(z), z)) if n.startswith("catboost") else sig(z)
                    pr = p >= row["thr"]
                    tp = int((pr & (yy == 1)).sum()); fp = int((pr & (yy == 0)).sum())
                    fn = int((~pr & (yy == 1)).sum()); tn = int((~pr & (yy == 0)).sum())
                    f1 = 2 * tp / (2 * tp + fp + fn)
                    assert abs(f1 - r["f1_at_thr"]) <= TOL, f"60 s {n} F1 {f1} vs stored {r['f1_at_thr']}"
                    row.update(precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                               specificity=tn / max(tn + fp, 1), n_units=int(len(yy)), n_pos=int(yy.sum()))
            rows[n] = row
        tab = {k: rows[v] for k, v in KEYS_U.items()}
        for k, ks in SEEDS_U.items():
            tab[k] = {m: mean_sd(rows, ks, m) for m in rows[ks[0]] if m not in ("thr", "n_units", "n_pos", "n_true",
                                                                              "hours")}
        out[split] = tab
    return out


# ----------------------------------------------------------------------------- CHB-MIT
def chbmit(path):
    d = rd(path)["results"]; out = {}
    for u in ("clip12", "point1s"):
        tab = {}
        for key, n in (("fusion", "fused_mean"), ("wo_linoss", "catboost"), ("wo_catboost", "linoss_mean")):
            r = d[f"{u}/{n}"]
            tab[key] = {k: r[k] for k in ("auroc", "auprc", "f1", "precision", "recall")}
            tab[key]["specificity"] = derive_spec(r["n"], r["pos"], r["precision"], r["recall"])
            tab[key]["n_units"], tab[key]["n_pos"] = r["n"], r["pos"]
        for key, g in (("fusion_seeds", "fused"), ("linoss_seeds", "linoss")):
            rows = {}
            for i in range(5):
                r = d[f"{u}/{g}_s{i}"]
                rows[i] = {k: r[k] for k in ("auroc", "auprc", "f1", "precision", "recall")}
                rows[i]["specificity"] = derive_spec(r["n"], r["pos"], r["precision"], r["recall"])
            tab[key] = {m: mean_sd(rows, list(range(5)), m) for m in rows[0]}
            st = d.get(f"{u}/{g}_seeds", {})
            for m in ("auroc", "auprc", "f1"):          # the stored per-seed summary must agree
                if m in st:
                    assert max(abs(a - b) for a, b in zip(st[m], tab[key][m])) <= 1e-12
        out[u] = tab
    return out


# ----------------------------------------------------------------------------- Markdown
def f(v, nd=3):
    if v is None:
        return "–"
    if isinstance(v, (list, tuple)):
        return f"{v[0]:.{nd}f} ± {v[1]:.{nd}f}"
    return f"{v:.{nd}f}"


def md_table(title, cols, rows, note=""):
    s = [f"### {title}", ""] + ([note, ""] if note else [])
    s += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(s) + "\n"


def to_md(J):
    src = J["sources"]; parts = ["# Ablation: CatBoost × LinOSS fusion vs its branches", "",
                                 "Generated by `python -m szfusion.eval.ablation` from `results/tusz/ablation.json` "
                                 "(every number below is in that file). Thresholds and event post-processing are the "
                                 "dev-frozen ones; nothing is selected on test.", "",
                                 f"Sources: 1 s = {src['1s']}, 12 s = {src['12s']}, 60 s = {src['60s']} "
                                 "(official = the from-scratch run; s1_gate = the S1-gate outputs, numerically "
                                 "identical to it).", ""]
    for split in ("eval", "dev"):
        A, B, C = J["tusz"]["1s"][split], J["tusz"]["12s"][split], J["tusz"]["60s"][split]
        cols = ["model", "1 s AUROC", "1 s AUPRC", "1 s F1", "1 s prec", "1 s recall", "1 s spec",
                "12 s AUROC", "12 s AUPRC", "12 s F1", "12 s prec", "12 s recall", "12 s spec",
                "60 s AUROC", "60 s AUPRC", "60 s F1", "60 s prec", "60 s recall", "60 s spec"]
        rows = []
        for k, name in ROWS:
            r = [name]
            for T in (A, B, C):
                r += [f(T[k].get(m), 4 if m == "specificity" else 3) for m in ("auroc", "auprc", "f1", "precision", "recall", "specificity")]
            rows.append(r)
        title = "TUSZ eval (test)" if split == "eval" else "TUSZ dev (out-of-fold stacker rows; selection set)"
        parts.append(md_table(f"{title}: unit-level metrics", cols, rows,
                              "prec / recall / spec / F1 at the dev-frozen threshold (1 s: best dev F1; 12 s: best "
                              "dev F1; 60 s: the window threshold of the dev post-processing rule). Recall = "
                              "sensitivity. 60 s dev prec / recall / spec are not stored (–)."))
        cols = ["model", "1 s ev sens (pt 1)", "1 s FA/24h (pt 1)", "1 s ev prec (pt 1)", "1 s ev sens (pt 2)",
                "1 s FA/24h (pt 2)", "1 s ev prec (pt 2)", "60 s ev sens", "60 s FA/h", "60 s ev PPV",
                "60 s time recall", "60 s time PPV"]
        rows = []
        for k, name in ROWS:
            a, c = A[k], C[k]
            rows.append([name, f(a.get("ev_event_sens")), f(a.get("ev_fa_per_24h"), 2), f(a.get("ev_event_prec")),
                         f(a.get("ev2_event_sens")), f(a.get("ev2_fa_per_24h"), 2), f(a.get("ev2_event_prec")),
                         f(c.get("event_sens")), f(c.get("fa_per_h")), f(c.get("event_ppv")),
                         f(c.get("time_recall")), f(c.get("time_ppv"))])
        parts.append(md_table(f"{title}: event metrics", cols, rows,
                              "1 s: any-overlap scorer of fusion/protocol_1s.py at event points 1 and 2 (dev FA/24h "
                              "budgets 16.56 / 33.12). 60 s: the Zabihi et al. 2026 scorer (>= 10 s overlap) with the "
                              "dev post-processing rule. 12 s has no event metric (isolated clips)."))
    for u, title in (("clip12", "12 s clips"), ("point1s", "1 s point-wise")):
        T = J["chbmit"][u]
        cols = ["model", "AUROC", "AUPRC", "F1", "precision", "recall", "specificity"]
        rows = [[name] + [f(T[k].get(m), 4 if m == "specificity" else 3) for m in ("auroc", "auprc", "f1", "precision", "recall", "specificity")]
                for k, name in ROWS]
        parts.append(md_table(f"CHB-MIT test (HyBrain protocol), {title}", cols, rows,
                              f"Source: results/chbmit/test_results.json; specificity derived exactly from its counts "
                              f"(n = {T['fusion']['n_units']}, positives = {T['fusion']['n_pos']})."))
    return "\n".join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/tusz/ablation.json")
    ap.add_argument("--md", default="docs/tables_ablation.md")
    ap.add_argument("--chbmit", default="results/chbmit/test_results.json")
    a = ap.parse_args(argv)
    S = sources()
    J = {"provenance": {**provenance(f"python -m szfusion.eval.ablation --out {a.out} --md {a.md}"),
                        "config_sha256": config_sha256(),
                        "source": {k: v[1] for k, v in S.items()},
                        "inputs": {"1s": f"<work>/{'tusz' if S['1s'][1] == 'official' else 's1_gate'}/fusion_1s",
                                   "12s": f"<work>/{'tusz' if S['12s'][1] == 'official' else 's1_gate'}/u12",
                                   "60s": f"<work>/{'tusz' if S['60s'][1] == 'official' else 's1_gate'}/u60",
                                   "chbmit": a.chbmit}},
         "sources": {k: v[1] for k, v in S.items()},
         "rows": dict(ROWS),
         "keys": {"1s": KEYS_1S, "12s_60s": KEYS_U, "seeds_1s": SEEDS_1S, "seeds_units": SEEDS_U},
         "tusz": {"1s": one_sec(S["1s"][0]), "12s": unit(S["12s"][0], 12), "60s": unit(S["60s"][0], 60)},
         "chbmit": chbmit(a.chbmit)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(J, indent=2))
    Path(a.md).parent.mkdir(parents=True, exist_ok=True)
    Path(a.md).write_text(to_md(J), encoding="utf-8")
    print(to_md(J))


if __name__ == "__main__":
    main()
