"""12 s and 60 s outputs of the CatBoost × LinOSS fusion: window overlookers and the window stacker.

  python -m szfusion.units.overlooker dev  --unit 12 --runs S0 ... S4 --tokens T --catboost C --feat_dir F --markers M --out O
  python -m szfusion.units.overlooker eval --unit 12 ... (same arguments; scores the test set once, refuses to re-score)
  (--unit 60: the same, without --markers)

One per-second LinOSS backbone serves every unit. The overlooker reads the per-second summaries h_t (96-d,
units/embed.py) inside each window and corrects the plain pooled window score:

  base(w)   = log-mean-exp over the window of the head's per-second logits   (plain pooling, clips12.lme)
  a_t       = softmax_t( v · g_t ),  g_t = GELU(W1 LayerNorm(h_t)),  W1: 96 -> 32
  z_LinOSS(w) = base(w) + c · [ sum_t a_t g_t ; max_t g_t ] + c0              (attention pool and max pool, 64 -> 1)

c and c0 start at zero, so before training the overlooker equals the plain pooling. It can only learn a
correction from there.
Training (fixed before any result): TRAIN-split windows on the unit's grid with the unit's label rule (12 s:
[12k, 12k + 12), any seizure second; 60 s: every 15 s, overlap >= min(9 s, 20 %)). Each epoch uses all positive
windows plus 4× as many fresh random negatives. AdamW, lr 1e-3, weight decay 0.01, batch 512, at most 15 epochs.
The epoch is chosen by dev window AUPRC (patience 4; epoch 0 = plain pooling is a candidate).
CatBoost fusion: the window stacker (fusion/stacker.WindowStacker), a logistic regression on [z_C, z_LinOSS,
z_C·z_LinOSS] fitted on dev. 5-fold patient-grouped CV gives the dev rows; the refit on all dev is used for test.
Ensemble: the mean of the per-seed window logits.
Thresholds: 12 s, the best dev F1. 60 s, the paper's post-processing rule (windows60.tune).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from szfusion.fusion.protocol_1s import best_f1_threshold, catboost_track
from szfusion.fusion.stacker import WindowStacker
from szfusion.linoss.data import Store
from szfusion.units import clips12, windows60
from szfusion.util.monitor import Log

GRID = {12: (12, "any"), 60: (15, "zabihi")}                  # unit -> (stride, label rule) of the training windows
MAX_EPOCHS, PATIENCE, BS, LR, WD, NEG_RATIO = 15, 4, 512, 1e-3, 0.01, 4


def seizure_runs(y):
    d = np.diff(np.r_[0, np.asarray(y, np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def grid_labels(store, U, S, rule):
    """Per recording, labels of the windows [kS, kS + U) that fit inside it. rule 'any': any seizure second;
    'zabihi': overlap with a seizure run >= min(9 s, 0.2 × the run's length)."""
    Y = np.asarray(store.Y); out = []
    for e in store.index:
        y = Y[e["offset"]:e["offset"] + e["n_sec"]]
        t0 = np.arange(0, max(e["n_sec"] - U, -1) + 1, S)
        if rule == "any":
            c = np.r_[0, np.cumsum(y, dtype=np.int64)]
            lab = (c[t0 + U] - c[t0]) > 0
        else:
            lab = np.zeros(len(t0), bool)
            for a, b in seizure_runs(y):
                ov = np.clip(np.minimum(t0 + U, b) - np.maximum(t0, a), 0, None)
                lab |= (ov > 0) & (ov >= min(9.0, 0.2 * (b - a)))
        out.append(lab.astype(np.float32))
    return out


class Overlooker(nn.Module):
    def __init__(self, H=96, d=32):
        super().__init__()
        self.norm, self.proj, self.score = nn.LayerNorm(H), nn.Linear(H, d), nn.Linear(d, 1)
        self.corr = nn.Linear(2 * d, 1)
        nn.init.zeros_(self.corr.weight); nn.init.zeros_(self.corr.bias)

    def forward(self, E, base):                                                   # E (B, U, H), base (B,)
        h = F.gelu(self.proj(self.norm(E)))
        a = torch.softmax(self.score(h).squeeze(-1), 1)
        return base + self.corr(torch.cat([(a.unsqueeze(-1) * h).sum(1), h.amax(1)], -1)).squeeze(-1)


class Backbone:
    """Per-second summaries of one LinOSS run and its head (the head's logits give the plain pooling)."""

    def __init__(self, run):
        self.dir = Path(run)
        sd = torch.load(self.dir / "best.pt", map_location="cpu", weights_only=True)["model"]
        self.w, self.b = sd["head.weight"].numpy()[0].astype(np.float32), float(sd["head.bias"])
        self.emb = {}

    def windows(self, split, start, U):
        if split not in self.emb:
            self.emb[split] = np.load(self.dir / f"emb_{split}.npy", mmap_mode="r")
        E = np.nan_to_num(np.asarray(self.emb[split][(start[:, None] + np.arange(U)).ravel()], np.float32))
        E = E.reshape(len(start), U, -1)
        z = E @ self.w + self.b
        m = z.max(1, keepdims=True)
        return E, (m + np.log(np.mean(np.exp(z - m), 1, keepdims=True)))[:, 0]


def train_windows(tokens, unit):
    st = Store("train", tokens); S, rule = GRID[unit]
    lab = grid_labels(st, unit, S, rule)
    start = np.concatenate([e["offset"] + S * np.arange(len(l)) for e, l in zip(st.index, lab)]).astype(np.int64)
    return start, np.concatenate(lab).astype(np.int64)


def eval_windows(a, split):
    """-> start rows, labels, patients, CatBoost window logits, extras (store split; meta/intervals for 60 s)."""
    if a.unit == 12:
        start, patient, y, index = clips12.clips(a.markers, a.tokens, split)
        sp = clips12.STORE_SPLIT[split]
        total = len(np.load(Path(a.tokens) / sp / "y_sec.npy", mmap_mode="r"))
        cb, _ = catboost_track(Path(a.catboost) / f"scores_{sp}.npz", index, total)
        return start, y, patient, clips12.clip_logits(cb, start), {"store_split": sp}
    sp = "dev" if split == "dev" else "eval"
    start, y, patient, zc, meta, iv = windows60.windows(sp, a.tokens, Path(a.catboost) / f"scores_{sp}.npz",
                                                        a.feat_dir)
    return start, y, patient, zc, {"store_split": sp, "meta": meta, "iv": iv}


@torch.no_grad()
def predict(model, E, base):
    model.eval()
    return torch.cat([model(torch.from_numpy(E[i:i + 4096]), torch.from_numpy(base[i:i + 4096].astype(np.float32)))
                      for i in range(0, len(E), 4096)]).numpy().astype(np.float64)


def train(bb, unit, tr_start, tr_y, dev_E, dev_base, dev_y, seed, log, tag):
    torch.manual_seed(seed); g = np.random.default_rng(seed)
    model = Overlooker(bb.w.shape[0])
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    pos, neg = np.flatnonzero(tr_y == 1), np.flatnonzero(tr_y == 0)
    best = {"auprc": average_precision_score(dev_y, dev_base), "epoch": 0,
            "state": {k: v.clone() for k, v in model.state_dict().items()}}       # a copy: training updates in place
    log(f"  [{tag}] epoch 0 (= plain pooling): dev AUPRC {best['auprc']:.4f}")
    stale = 0
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        idx = np.concatenate([pos, g.choice(neg, min(len(neg), NEG_RATIO * len(pos)), replace=False)])
        idx = idx[g.permutation(len(idx))]; tot = 0.0; t = time.time()
        for i in range(0, len(idx), BS):
            b = idx[i:i + BS]
            E, base = bb.windows("train", tr_start[b], unit)
            loss = F.binary_cross_entropy_with_logits(model(torch.from_numpy(E), torch.from_numpy(base.astype(np.float32))),
                                                      torch.from_numpy(tr_y[b].astype(np.float32)))
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b)
        z = predict(model, dev_E, dev_base)
        ap, au = average_precision_score(dev_y, z), roc_auc_score(dev_y, z)
        log.event("epoch", tag=tag, epoch=ep, train_loss=round(tot / len(idx), 4), dev_auprc=round(ap, 4),
                  dev_auroc=round(au, 4), sec=round(time.time() - t, 1))
        if ap > best["auprc"]:
            best = {"auprc": ap, "epoch": ep, "state": {k: v.clone() for k, v in model.state_dict().items()}}; stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best["state"])
    log.event("selected", tag=tag, epoch=best["epoch"], dev_auprc=round(best["auprc"], 4))
    return model, best["epoch"]


def window_metrics12(y, p, thr):
    pr = p > thr
    tp = int((pr & (y == 1)).sum()); fp = int((pr & (y == 0)).sum())
    fn = int((~pr & (y == 1)).sum()); tn = int((~pr & (y == 0)).sum())
    return {"auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1), "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "specificity": tn / max(tn + fp, 1), "accuracy": (tp + tn) / len(y), "thr": thr, "clips": int(len(y))}


def metrics_table(unit, ex, y, scores, sel, stage):
    sig = lambda z: 1 / (1 + np.exp(-z))                                         # noqa: E731
    res = {}
    for k, z in scores.items():
        p = sig(np.where(np.isnan(z), np.nanmin(z), z)) if k.startswith("catboost") else sig(z)
        if unit == 12:
            if stage == "dev":
                sel["thr"][k] = best_f1_threshold(y, p)[0]
            res[k] = window_metrics12(y, p, sel["thr"][k])
        else:
            if stage == "dev":
                sel["post"][k], _ = windows60.tune(ex["meta"], p, ex["iv"], (0.44,) if k == "catboost" else ())
            res[k] = windows60.score(ex["meta"], p, sel["post"][k], ex["iv"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["dev", "eval"])
    ap.add_argument("--unit", type=int, choices=[12, 60], required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="LinOSS seed folders (with best.pt and emb_*.npy)")
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--catboost", required=True, help="folder with scores_dev.npz / scores_eval.npz")
    ap.add_argument("--feat_dir", help="CatBoost feature folder (intervals.json); unit 60")
    ap.add_argument("--markers", help="HyBrain 12 s marker folder; unit 12")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=12)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    d = Path(a.out); d.mkdir(parents=True, exist_ok=True)
    log = Log(d, name=f"overlook_{a.stage}")
    split = "dev" if a.stage == "dev" else ("test" if a.unit == 12 else "eval")
    if a.stage == "eval" and (d / "eval_results.json").exists():
        log("test already scored once; not re-scoring"); log.close(); return
    start, y, patient, zc, ex = eval_windows(a, split)
    log.event("windows", unit=a.unit, split=split, n=len(y), pos=int(y.sum()), patients=len(set(patient)))
    if a.stage == "dev":
        sel = {"runs": [Path(r).name for r in a.runs], "unit": a.unit, "models": {}, "fusion": {}, "thr": {}, "post": {}}
        tr_start, tr_y = train_windows(a.tokens, a.unit)
        log.event("train_windows", n=len(tr_y), pos=int(tr_y.sum()))
    else:
        sel = json.loads((d / "selection.json").read_text())
        assert sel["runs"] == [Path(r).name for r in a.runs] and sel["unit"] == a.unit
    z = {"catboost": zc}
    for ri, run in enumerate(a.runs):
        t = time.time(); bb = Backbone(run)
        E, base = bb.windows(ex["store_split"], start, a.unit)
        z[f"pool_r{ri}"] = base.astype(np.float64)
        if a.stage == "dev":
            model, ep = train(bb, a.unit, tr_start, tr_y, E, base, y, 100 + ri, log, f"r{ri}")
            torch.save({"model": model.state_dict()}, d / f"overlooker_r{ri}.pt")
            sel["models"][f"r{ri}"] = {"file": f"overlooker_r{ri}.pt", "epoch": ep}
        else:
            model = Overlooker(bb.w.shape[0])
            model.load_state_dict(torch.load(d / sel["models"][f"r{ri}"]["file"], weights_only=True)["model"])
        z[f"look_r{ri}"] = predict(model, E, base)
        log.event("run_done", run=Path(run).name, sec=round(time.time() - t, 1))
    n = len(a.runs)
    for v in ("pool", "look"):
        if n > 1:
            z[f"{v}_mean"] = np.mean([z[f"{v}_r{i}"] for i in range(n)], 0)
    for k in [k for k in z if k != "catboost"]:                                  # CatBoost fusion of every LinOSS row
        fk = k + "_cb"
        if a.stage == "dev":
            oof = np.zeros(len(y))
            for tr, te in GroupKFold(5).split(y, y, patient):
                oof[te] = WindowStacker().fit(zc[tr], z[k][tr], y[tr]).predict(zc[te], z[k][te])
            z[fk] = clips12.logit(oof)
            sel["fusion"][fk] = WindowStacker().fit(zc, z[k], y).to_dict()
        else:
            z[fk] = clips12.logit(WindowStacker.from_dict(sel["fusion"][fk]).predict(zc, z[k]))
    res = metrics_table(a.unit, ex, y, z, sel, a.stage)
    out = {"runs": sel["runs"], "unit": a.unit, "results": res,
           "epochs": {k: v["epoch"] for k, v in sel["models"].items()}}
    if a.stage == "dev":
        (d / "selection.json").write_text(json.dumps(sel, indent=2, default=float))
        (d / "dev_results.json").write_text(json.dumps(out, indent=2, default=float))
    else:
        np.savez_compressed(d / "eval_scores.npz", start=start, y=y, **z)          # per-window logits; SzCORE + figures
        (d / "eval_results.json").write_text(json.dumps(out, indent=2, default=float))
    keys = ["auroc", "auprc", "f1", "precision", "recall"] if a.unit == 12 else \
        ["auroc", "auprc", "f1_at_thr", "event_sens", "fa_per_h", "time_recall", "time_ppv"]
    log.table(f"overlookers, unit {a.unit} s, {split}", ["config"] + keys,
              [[k] + [round(r[m], 4) for m in keys] for k, r in res.items()])
    log.close()


if __name__ == "__main__":
    main()
