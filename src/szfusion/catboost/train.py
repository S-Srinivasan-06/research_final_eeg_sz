"""Train the CatBoost ensemble on train features, predict EVERY dev/eval window.

  python -m szfusion.catboost.train --feat_dir <work>/cb_feats/tusz --out_dir <work>/tusz/catboost [--gpu] [--smoke]

Steps: load features.parquet per split -> temporal context (per recording, before subsampling) ->
optional train-negative subsampling (paper: none) -> 3 committees [M; GH infer.py]: base (941-equiv base cols,
2x10 GroupKFold), full (+context, 2x10), sens (full, 1x10, Optuna) -> equal mean of committee means ->
scores_{dev,eval}.npz. Early stopping uses the held-out TRAIN fold only; dev/eval never touch training.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from szfusion.catboost.context import add_context
from szfusion.catboost.extract import load_config, load_split

META_COLS = ["rec_id", "patient", "session", "montage_dir", "rec_duration", "t0", "y"]


def feat_cols_of(df):
    return [c for c in df.columns if c not in META_COLS]


def subsample_negatives(df, tcfg, seed):
    rng = np.random.default_rng(seed)
    y = df["y"].to_numpy()
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if tcfg.get("neg_ratio"):           # keep neg_ratio x n_pos negatives
        n = min(len(neg), int(round(tcfg["neg_ratio"] * max(len(pos), 1))))
    elif tcfg.get("neg_fraction"):      # keep this fraction of negatives
        n = int(round(tcfg["neg_fraction"] * len(neg)))
    else:
        return df
    keep = np.sort(np.r_[pos, rng.choice(neg, size=n, replace=False)])
    return df.iloc[keep].reset_index(drop=True)


def patient_holdout(df, frac, seed):
    pats = np.array(sorted(df["patient"].unique()))
    rng = np.random.default_rng(seed)
    hold = set(rng.choice(pats, size=max(1, int(round(frac * len(pats)))), replace=False)) if len(pats) > 1 else set()
    m = df["patient"].isin(hold).to_numpy()
    return df[~m], df[m]


def make_model(mcfg, seed, n_pos, n_neg, clamp=(1.0, 20.0), overrides=None):
    """CatBoost with inverse-frequency seizure weight clamped to `clamp` [M]."""
    from catboost import CatBoostClassifier
    p = dict(mcfg.get("params", {})); p.update(overrides or {})
    p["random_seed"] = int(seed)
    p.setdefault("allow_writing_files", False)  # no catboost_info/ in cwd
    w = float(np.clip(n_neg / max(n_pos, 1), *clamp))
    p["class_weights"] = [1.0, w]
    return CatBoostClassifier(**p)


def _optuna_params(X, y, groups, mcfg, tcfg, n_trials, seed):
    """High-sensitivity committee [M]: Optuna over lr, depth, l2, bagging_temperature, random_strength,
    maximising validation sensitivity at 0.5 on one patient-grouped inner split. Search ranges [DEFAULT]."""
    try:
        import optuna
    except ImportError:
        print("optuna not installed -> sens committee uses default params [DEFAULT]")
        return {}
    from sklearn.model_selection import GroupShuffleSplit
    tr, va = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(X, y, groups))

    def obj(trial):
        o = dict(learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                 depth=trial.suggest_int("depth", 4, 10), l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1, 30, log=True),
                 bagging_temperature=trial.suggest_float("bagging_temperature", 0, 5),
                 random_strength=trial.suggest_float("random_strength", 0, 5), verbose=0)
        m = make_model(mcfg, seed, int(y[tr].sum()), int(len(tr) - y[tr].sum()), tcfg["class_weight_clamp"], o)
        m.fit(X[tr], y[tr], eval_set=(X[va], y[va]), early_stopping_rounds=tcfg.get("early_stopping_rounds"))
        pv = m.predict_proba(X[va])[:, 1] >= 0.5
        return float(pv[y[va] == 1].mean()) if (y[va] == 1).any() else 0.0
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    st.optimize(obj, n_trials=n_trials)
    return st.best_params


def train(cfg, feat_dir, out_dir, smoke=False, dfs=None, resume_dir=None, time_budget_h=None):
    """3 committees x GroupKFold(patient) on TRAIN; dev/eval predicted on EVERY window."""
    from sklearn.model_selection import GroupKFold
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    tcfg, ecfg, mcfg = cfg["train"], cfg["ensemble"], cfg["model"]
    seed = int(cfg.get("seed", 0)); t = time.time()
    if resume_dir:  # bring finished models / tuned params from an earlier (time-limited) session
        import shutil
        for f in list(Path(resume_dir).rglob("cb_*.cbm")) + list(Path(resume_dir).rglob("optuna_*.json")):
            if not (out / f.name).exists():
                shutil.copy2(f, out / f.name)
        print(f"[resume] {len(list(out.glob('cb_*.cbm')))} models available from {resume_dir}", flush=True)
    if dfs is None:
        dfs = {s: load_split(feat_dir, s) for s in ("train", "dev", "eval") if (Path(feat_dir) / s).exists()}
    tr, _ = dfs["train"]
    base_cols = feat_cols_of(tr)
    tr, full_cols = add_context(tr, base_cols, cfg.get("context"))
    tr = subsample_negatives(tr, tcfg, seed)      # paper: none (neg_ratio/neg_fraction null)
    y, groups = tr["y"].to_numpy(), tr["patient"].to_numpy()
    n_folds = min(int(tcfg["n_folds"]), len(np.unique(groups)))
    over = {"iterations": 20, "verbose": 0} if smoke else {}
    committees = {}
    X_cache = {}  # one float32 matrix per feature set; 'sens' reuses 'full' (a second 202k x 7k copy OOMs Kaggle)
    for cname, cc in ecfg["members"].items():
        cols = base_cols if cc["features"] == "base" else full_cols
        if cc["features"] not in X_cache:
            X_cache.clear()
            import gc; gc.collect()
            X_cache[cc["features"]] = tr[cols].to_numpy(np.float32)
        X = X_cache[cc["features"]]
        extra = dict(over)
        if cc.get("optuna") and not smoke:
            op = out / f"optuna_{cname}.json"
            if op.exists():
                best = json.loads(op.read_text())
            else:
                best = _optuna_params(X, y, groups, mcfg, tcfg, int(ecfg.get("optuna_trials", 50)), seed)
                op.write_text(json.dumps(best))
            extra.update(best)
        models = []
        for rep in range(int(cc["repeats"])):
            # different patient permutation per repeat [M]: shuffle group ids before GroupKFold
            rng = np.random.default_rng(seed + 1000 * rep)
            perm = {g: i for i, g in enumerate(rng.permutation(np.unique(groups)))}
            g_rep = np.array([perm[g] for g in groups])
            for f, (i_tr, i_va) in enumerate(GroupKFold(n_folds).split(X, y, g_rep)):
                path = out / f"cb_{cname}_r{rep + 1}_fold{f + 1:02d}.cbm"  # repo naming [GH]
                if path.exists():  # checkpoint: finished in an earlier run
                    from catboost import CatBoostClassifier
                    m = CatBoostClassifier(); m.load_model(str(path)); models.append(m)
                    print(f"{cname} r{rep + 1} fold{f + 1} loaded", flush=True)
                    continue
                if time_budget_h and (time.time() - t) / 3600 > time_budget_h:
                    (out / "INCOMPLETE").write_text(f"time budget {time_budget_h} h reached; rerun with --resume_dir")
                    print(f"[budget] stopping before {path.name}; models so far are saved", flush=True)
                    return None
                m = make_model(mcfg, f + 1 + 100 * rep, int(y[i_tr].sum()), int(len(i_tr) - y[i_tr].sum()),
                               tcfg["class_weight_clamp"], extra)
                kw = {}
                if tcfg.get("early_stopping_rounds") and len(np.unique(y[i_va])) > 1:
                    kw = dict(eval_set=(X[i_va], y[i_va]), early_stopping_rounds=int(tcfg["early_stopping_rounds"]))
                m.fit(X[i_tr], y[i_tr], **kw)
                m.save_model(str(path))
                models.append(m)
                print(f"{cname} r{rep + 1} fold{f + 1} {time.time() - t:.0f}s", flush=True)
        committees[cname] = (cols, models)
    X_cache.clear(); del X, tr
    import gc; gc.collect()
    for name, cols in (("base", base_cols), ("full", full_cols)):
        (out / f"feature_cols_{name}.txt").write_text(chr(10).join(cols))
    for split in ("dev", "eval"):
        if split not in dfs:
            continue
        df, iv = dfs[split]
        df, _ = add_context(df, base_cols, cfg.get("context"))  # every window, no subsampling
        pieces = {c: np.mean([m.predict_proba(df[cols].to_numpy(np.float32))[:, 1] for m in ms], 0)
                  for c, (cols, ms) in committees.items()}
        score = np.mean(np.vstack(list(pieces.values())), 0)  # equal-weight mean of committee means [GH]
        df[["rec_id", "patient", "t0", "rec_duration", "y"]].to_parquet(out / f"meta_{split}.parquet", index=False)
        np.savez(out / f"scores_{split}.npz", score=score.astype(np.float32),
                 **{f"p_{c}": v.astype(np.float32) for c, v in pieces.items()})
        with open(out / f"intervals_{split}.json", "w") as fh:
            json.dump(iv, fh)
    with open(out / "train_summary.json", "w") as fh:
        json.dump({"n_train_rows": int(len(y)), "n_pos": int(y.sum()), "n_base": len(base_cols),
                   "n_full": len(full_cols), "sec": round(time.time() - t, 1), "config": cfg}, fh, indent=2, default=str)
    return committees, base_cols, full_cols


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="default: configs/tusz_final.yaml (catboost section)")
    ap.add_argument("--feat_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--gpu", action="store_true", help="CatBoost task_type=GPU (results differ slightly from CPU)")
    ap.add_argument("--resume_dir", default=None, help="dir with cb_*.cbm / optuna_*.json from an earlier session")
    ap.add_argument("--time_budget_h", type=float, default=None, help="stop cleanly before a new model after N h")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.seed is not None:
        cfg["seed"] = a.seed
    if a.gpu:
        cfg["model"]["params"].update(task_type="GPU", devices="0")
    (Path(a.out_dir) / "INCOMPLETE").unlink(missing_ok=True)
    train(cfg, a.feat_dir, a.out_dir, a.smoke, resume_dir=a.resume_dir, time_budget_h=a.time_budget_h)


if __name__ == "__main__":
    main()
