"""Fusion stackers for two per-second (or per-window) scores, stored as plain JSON coefficients (no pickles).

Inputs are logits z = log(p / (1 - p)), with p clipped to [1e-5, 1 - 1e-5].

  late    p = w · Platt_C(z_C) + (1 - w) · Platt_L(z_L), w ∈ {0, 0.1, ..., 1}
          (Platt_X: one-feature logistic regression, i.e. calibration of one branch)
  lr      p = σ(β · s([z_C, z_L, z_C·z_L]) + b)
  lr_ctx  p = σ(β · s([z_C, z_L, z_C·z_L, S15(z_L), S61(z_L)]) + b)

s(·) standardises every feature by the mean and standard deviation of the fitting set. S_k is the centred k-second
moving average within the recording (protocol_1s.smooth). It gives the stacker the LinOSS branch's short-range
(15 s) and long-range (61 s) evidence around each second.
Why a product term: the two branches' errors are only partly correlated. Adding their log-odds (β_C z_C + β_L z_L)
treats them as independent evidence. When both are confident at once, the evidence is partly shared, and the
product weight (fitted negative on TUSZ) stops that agreement from being counted twice.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from szfusion.fusion.protocol_1s import smooth

CANDS = ["late", "lr", "lr_ctx"]


def logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def feats(cand, zc, zl, rec=None):
    if cand == "lr":
        return np.c_[zc, zl, zc * zl]
    if cand == "lr_ctx":
        return np.c_[zc, zl, zc * zl, smooth(zl, rec, 15), smooth(zl, rec, 61)]
    raise ValueError(cand)


class Stacker:
    """fit -> coefficients; predict uses only the stored numbers, so a JSON round-trip predicts identically."""

    def __init__(self, cand, w=None):
        self.cand, self.w, self.p = cand, w, None

    def fit(self, zc, zl, y, rec=None):
        if self.cand == "late":
            a = LogisticRegression(max_iter=500).fit(zc[:, None], y)
            b = LogisticRegression(max_iter=500).fit(zl[:, None], y)
            self.p = {"cb": [float(a.coef_[0, 0]), float(a.intercept_[0])],
                      "lin": [float(b.coef_[0, 0]), float(b.intercept_[0])]}
        else:
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(feats(self.cand, zc, zl, rec), y)
            sc, lr = m[0], m[1]
            self.p = {"mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(), "coef": lr.coef_[0].tolist(),
                      "intercept": float(lr.intercept_[0])}
        return self

    def predict(self, zc, zl, rec=None):
        if self.cand == "late":
            pc = sigmoid(self.p["cb"][0] * zc + self.p["cb"][1])
            pl = sigmoid(self.p["lin"][0] * zl + self.p["lin"][1])
            return self.w * pc + (1 - self.w) * pl
        X = (feats(self.cand, zc, zl, rec) - np.array(self.p["mean"])) / np.array(self.p["scale"])
        return sigmoid(X @ np.array(self.p["coef"]) + self.p["intercept"])

    def to_dict(self):
        return {"cand": self.cand, "w": self.w, "params": self.p}

    @classmethod
    def from_dict(cls, d):
        s = cls(d["cand"], d["w"]); s.p = d["params"]
        return s


def save(obj, path):
    path.write_text(json.dumps(obj, indent=2, default=float))


def load(path):
    return json.loads(path.read_text())


class WindowStacker:
    """12 s / 60 s fusion: p = σ(β · s([z_C, z_L, z_C·z_L]) + b), fitted on dev windows. Windows with no CatBoost
    score (a 12 s clip in a recording shorter than 60 s) use a one-feature logistic regression on z_L, fitted on
    all windows."""

    def fit(self, zc, zl, y):
        ok = ~np.isnan(zc)
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(np.c_[zc[ok], zl[ok], zc[ok] * zl[ok]], y[ok])
        al = LogisticRegression(max_iter=1000).fit(zl[:, None], y)
        self.p = {"mean": m[0].mean_.tolist(), "scale": m[0].scale_.tolist(), "coef": m[1].coef_[0].tolist(),
                  "intercept": float(m[1].intercept_[0]), "alone": [float(al.coef_[0, 0]), float(al.intercept_[0])]}
        return self

    def predict(self, zc, zl):
        out = sigmoid(self.p["alone"][0] * zl + self.p["alone"][1])
        ok = ~np.isnan(zc)
        if ok.any():
            X = (np.c_[zc[ok], zl[ok], zc[ok] * zl[ok]] - np.array(self.p["mean"])) / np.array(self.p["scale"])
            out[ok] = sigmoid(X @ np.array(self.p["coef"]) + self.p["intercept"])
        return out

    def to_dict(self):
        return dict(self.p)

    @classmethod
    def from_dict(cls, d):
        s = cls(); s.p = dict(d)
        return s
