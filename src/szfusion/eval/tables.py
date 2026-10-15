"""Markdown tables of the stored results. This module only formats numbers that are already in results/; it computes
no metric (RULES.md 6).

  python -m szfusion.eval.tables [--results results] [--out docs/tables.md]

Every JSON file under <results>/ (recursively) is classified by its content:
  1 s TUSZ      "results" rows with ev_event_sens            (fusion/run_1s.py eval_results.json)
  12 s / 60 s   top-level "unit" 12 or 60                     (units/overlooker.py eval_results.json)
  SzCORE        top-level "timescoring"                       (eval/szcore.py)
  CHB-MIT       "results" keys clip12/... and point1s/...     (eval/chbmit_scores.py test_results.json)
  efficiency    results/efficiency.json
  mechanism     results/tusz/fusion_mechanism.json            (eval/mechanism.py)
Files named dev_results.json or selection.json are dev-stage records and are skipped. Each table names its source
file. A stored [mean, sd] pair is printed as "mean ± sd". Published baselines are constants below, with citations.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ----------------------------------------------------------------------------- published baselines (cited constants)
# [Kim 2026] H. Kim, S. Y. Jhin, N. Park, N. Imam. Spatiotemporal Hyperedges for EEG Seizure Detection and Prediction
# (HyBrain). Proc. ACM CIKM 2026. arXiv:2609.37730. Values = mean ± SD over three seeds, as transcribed in
# docs/DOCUMENT.md section 7.3 (Table 2: window-based; Table 3: point-wise, row "Ours" without the regulariser).
HYBRAIN_TABLE2 = {   # configuration -> {setting: (AUROC, F1)}, each (mean, sd)
    "E_h = 1": {"tusz12": ((0.899, 0.006), (0.551, 0.008)), "tusz60": ((0.885, 0.010), (0.564, 0.041)),
                "chb12": ((0.917, 0.005), (0.182, 0.033))},
    "E_h = 2": {"tusz12": ((0.894, 0.005), (0.541, 0.025)), "tusz60": ((0.892, 0.002), (0.537, 0.041)),
                "chb12": ((0.907, 0.009), (0.164, 0.026))},
    "E_h = 3": {"tusz12": ((0.880, 0.006), (0.540, 0.002)), "tusz60": ((0.908, 0.006), (0.636, 0.023)),
                "chb12": ((0.924, 0.003), (0.191, 0.009))},
}
HYBRAIN_TABLE3 = {"tusz12": ((0.913, 0.005), (0.491, 0.029)), "tusz60": ((0.918, 0.007), (0.512, 0.020)),
                  "chb12": ((0.910, 0.003), (0.228, 0.017))}
# CHB-MIT 1 s point-wise, Table 3 regulariser rows as logged in PROGRESS.md (2026-10-15 12:41 [RESULT] C3 TEST):
# AUROC 0.929 ± 0.006 (lambda = 0.3) and best F1 0.247 ± 0.009 (lambda = 1.0). DOCUMENT.md 7.3 marks the lambda rows
# for re-checking against the camera-ready PDF.
HYBRAIN_CHB_1S = {"auroc": ((0.929, 0.006), "lambda = 0.3"), "f1": ((0.247, 0.009), "lambda = 1.0")}
# [Zabihi 2026] M. Zabihi et al. Scientific Reports 16:11283 (2026), doi:10.1038/s41598-026-41358-w. CatBoost
# ensemble, TUSZ v2.0.3 eval (DOCUMENT.md 7.3): AUROC, AUPRC, event sensitivity, false alarms per hour.
ZABIHI_60S = {"auroc": 0.919, "auprc": 0.753, "event_sens": 0.749, "fa_per_h": 0.682}

ROWS = {"catboost": "CatBoost branch", "linoss_mean": "LinOSS branch (5-seed mean)", "fused_mean": "Fusion",
        "look_mean": "LinOSS branch (5-seed mean)", "look_mean_cb": "Fusion",
        "linoss_seeds": "LinOSS, per seed (mean ± sd)", "fused_seeds": "Fusion, per seed (mean ± sd)"}


def fmt(v, nd=3):
    if v is None:
        return "–"
    if isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
        return f"{v[0]:.{nd}f} ± {v[1]:.{nd}f}"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def md(title, cols, rows, src):
    out = [f"### {title}", "", f"Source: `{src}`", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def classify(p: Path, d: dict):
    if p.name in ("dev_results.json", "selection.json"):
        return None
    if p.name == "efficiency.json":
        return "efficiency"
    if p.name == "fusion_mechanism.json":
        return "mechanism"
    if "timescoring" in d:
        return "szcore"
    if d.get("unit") in (12, 60):
        return f"unit{d['unit']}"
    res = d.get("results")
    if isinstance(res, dict) and res:
        if any(k.startswith(("clip12/", "point1s/")) for k in res):
            return "chbmit"
        if any(isinstance(v, dict) and "ev_event_sens" in v for v in res.values()):
            return "tusz1s"
    return None


def pick(res, names):
    return [(ROWS.get(n, n), res[n]) for n in names if n in res]


def t_tusz1s(d, src):
    cols = ["model", "AUROC", "AUPRC", "F1", "event sens (pt 1)", "FA/24h (pt 1)", "event sens (pt 2)", "FA/24h (pt 2)"]
    keys = ["auroc", "auprc", "f1", "ev_event_sens", "ev_fa_per_24h", "ev2_event_sens", "ev2_fa_per_24h"]
    rows = [[n] + [fmt(r.get(k), 2 if "fa_" in k else 3) for k in keys]
            for n, r in pick(d["results"], ["catboost", "linoss_mean", "fused_mean"])]
    return md("TUSZ eval, 1 s (per second; events: any-overlap scorer)", cols, rows, src)


def t_unit(d, src):
    res = d["results"]; u = d["unit"]
    if u == 12:
        cols = ["model", "AUROC", "AUPRC", "F1", "precision", "recall"]
        keys = ["auroc", "auprc", "f1", "precision", "recall"]
        rows = [[n] + [fmt(r.get(k)) for k in keys] for n, r in pick(res, ["catboost", "look_mean", "look_mean_cb"])]
        for c, v in HYBRAIN_TABLE2.items():
            rows.append([f"HyBrain {c} [Kim 2026, Table 2]", fmt(v["tusz12"][0]), "–", fmt(v["tusz12"][1]), "–", "–"])
        rows.append(["HyBrain point-wise [Kim 2026, Table 3]", fmt(HYBRAIN_TABLE3["tusz12"][0]), "–",
                     fmt(HYBRAIN_TABLE3["tusz12"][1]), "–", "–"])
        return md("TUSZ eval, 12 s clips", cols, rows, src)
    cols = ["model", "AUROC", "AUPRC", "F1", "event sens", "FA/h", "time recall", "time PPV"]
    keys = ["auroc", "auprc", "f1_at_thr", "event_sens", "fa_per_h", "time_recall", "time_ppv"]
    rows = [[n] + [fmt(r.get(k)) for k in keys] for n, r in pick(res, ["catboost", "look_mean", "look_mean_cb"])]
    for c, v in HYBRAIN_TABLE2.items():
        rows.append([f"HyBrain {c} [Kim 2026, Table 2]", fmt(v["tusz60"][0]), "–", fmt(v["tusz60"][1]),
                     "–", "–", "–", "–"])
    rows.append(["HyBrain point-wise [Kim 2026, Table 3]", fmt(HYBRAIN_TABLE3["tusz60"][0]), "–",
                 fmt(HYBRAIN_TABLE3["tusz60"][1]), "–", "–", "–", "–"])
    z = ZABIHI_60S
    rows.append(["CatBoost ensemble, TUSZ v2.0.3 [Zabihi 2026]", fmt(z["auroc"]), fmt(z["auprc"]), "–",
                 fmt(z["event_sens"]), fmt(z["fa_per_h"]), "–", "–"])
    return md("TUSZ eval, 60 s windows (published rows: other TUSZ releases / settings, context only)", cols, rows, src)


def t_szcore(d, src):
    cols = ["setting", "sensitivity", "precision", "F1", "FP/day", "TP / N_ref"]
    rows = [[k.replace(" | ", ", "), fmt(r["event_sens"]), fmt(r["event_prec"]), fmt(r["event_f1"]), fmt(r["fp_per_day"], 2),
             f"{r['tp']} / {r['n_ref']}"] for k, r in d["results"].items()]
    return md(f"TUSZ eval, SzCORE event scoring (timescoring {d.get('timescoring', '?')})", cols, rows, src)


def t_chbmit(d, src):
    res = d["results"]; out = []
    for unit, title, hyb in (("clip12", "12 s clips", "chb12"), ("point1s", "1 s point-wise", None)):
        cols = ["model", "AUROC", "AUPRC", "F1"]
        names = ["catboost", "linoss_mean", "fused_mean", "linoss_seeds", "fused_seeds"]
        rows = [[ROWS[n]] + [fmt(res[f"{unit}/{n}"].get(k)) for k in ("auroc", "auprc", "f1")]
                for n in names if f"{unit}/{n}" in res]
        if hyb:
            for c, v in HYBRAIN_TABLE2.items():
                rows.append([f"HyBrain {c} [Kim 2026, Table 2]", fmt(v[hyb][0]), "–", fmt(v[hyb][1])])
            rows.append(["HyBrain point-wise [Kim 2026, Table 3]", fmt(HYBRAIN_TABLE3[hyb][0]), "–",
                         fmt(HYBRAIN_TABLE3[hyb][1])])
        else:
            a, f = HYBRAIN_CHB_1S["auroc"], HYBRAIN_CHB_1S["f1"]
            rows.append([f"HyBrain [Kim 2026, Table 3] ({a[1]} / {f[1]})", fmt(a[0]), "–", fmt(f[0])])
        out.append(md(f"CHB-MIT test (HyBrain protocol), {title}", cols, rows, src))
    return "\n".join(out)


def t_eff(d, src):
    cols = ["quantity", "TUSZ", "CHB-MIT"]
    tr, mem, inf = d["linoss_train_s_per_epoch"], d["linoss_gpu_peak_MB"], d["linoss_inference_ms_per_12s"]
    rows = [["LinOSS training, s per epoch (median)", fmt(tr["tusz"]["median"], 1), fmt(tr["chbmit"]["median"], 1)],
            ["LinOSS peak GPU memory (MB)", str(mem["tusz"]), str(mem["chbmit"])],
            ["LinOSS inference, ms per 12 s", fmt(inf["tusz"]["value"]), fmt(inf["chbmit"]["value"])],
            ["CatBoost inference, ms per 12 s clip", "–", fmt(d["catboost"]["inference_ms_per_12s_clip"])],
            ["CatBoost committee training (s)", "–", fmt(d["catboost"]["train_total_s_chbmit"], 1)],
            ["Feature extraction, ms per 12 s clip", "–", fmt(d["feature_extraction_ms_per_12s_clip"]["value"], 1)]]
    if "tusz_60s_table" in d:
        t = d["tusz_60s_table"]["inference_ms_per_segment"]
        rows.append(["Inference per 60 s window, full 5-seed system (ms)", fmt(t["total_5seed_system"], 2), "–"])
        rows.append(["Inference per 60 s window, single seed (ms)", fmt(t["single_seed_system"], 2), "–"])
    return md("Efficiency (Tesla T4 unless stated)", cols, rows, src)


def t_mech(d, src):
    c, f = d["logit_correlation"], d["false_alarm_seconds_at_dev_f1_threshold"]
    b = next(v for k, v in d.items() if k.startswith("ictal_caught_at_"))
    g = d["separation_gain_sqrt_2_over_1_plus_rho"]
    rows = [["logit correlation, background seconds", fmt(c["rho_background"])],
            ["logit correlation, ictal seconds", fmt(c["rho_ictal"])],
            ["CatBoost FA seconds that are also LinOSS FA seconds", fmt(f["share_cb_fa_also_linoss"])],
            ["LinOSS FA seconds that are also CatBoost FA seconds", fmt(f["share_linoss_fa_also_cb"])],
            ["ictal seconds above the 1 % background threshold: CatBoost", fmt(b["catboost"]["ictal_frac"])],
            ["... LinOSS (5-seed mean)", fmt(b["linoss_mean"]["ictal_frac"])],
            ["... fusion", fmt(b["fused_mean"]["ictal_frac"])],
            ["separation gain sqrt(2 / (1 + rho)) at rho_background", fmt(g["at_rho_background"])]]
    return md("Fusion mechanism (TUSZ eval, 1 s tracks)", ["quantity", "value"], rows, src)


ORDER = ["tusz1s", "unit12", "unit60", "szcore", "mechanism", "chbmit", "efficiency"]
FUN = {"tusz1s": t_tusz1s, "unit12": t_unit, "unit60": t_unit, "szcore": t_szcore, "chbmit": t_chbmit,
       "efficiency": t_eff, "mechanism": t_mech}


def build(results_dir) -> str:
    root = Path(results_dir)
    found = {k: [] for k in ORDER}
    for p in sorted(root.rglob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        k = classify(p, d) if isinstance(d, dict) else None
        if k:
            found[k].append((p, d))
    parts = ["# Result tables", "", f"Generated by `python -m szfusion.eval.tables` from the JSON files under "
             f"`{root.as_posix()}/`. No number is computed here.", ""]
    for k in ORDER:
        if not found[k]:
            parts.append(f"_{k}: no result file yet._\n")
        for p, d in found[k]:
            parts.append(FUN[k](d, p.relative_to(root.parent).as_posix() if root.parent != p else p.name))
    return "\n".join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", help="also write the Markdown here (e.g. docs/tables.md)")
    a = ap.parse_args(argv)
    s = build(a.results)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(s)
    if a.out:
        Path(a.out).write_text(s, encoding="utf-8")


if __name__ == "__main__":
    main()
