"""Stacker JSON round-trip and the event rule on synthetic data."""
import json

import numpy as np

from szfusion.fusion import protocol_1s as P
from szfusion.fusion.stacker import Stacker


def test_stacker_json_roundtrip():
    g = np.random.default_rng(0)
    n = 4000; rec = np.repeat(np.arange(8), n // 8); y = (g.random(n) < 0.1).astype(int)
    zc = g.normal(size=n) + 2 * y; zl = g.normal(size=n) + 1.5 * y
    for cand, w in (("late", 0.4), ("lr", None), ("lr_ctx", None)):
        s = Stacker(cand, w).fit(zc, zl, y, rec)
        s2 = Stacker.from_dict(json.loads(json.dumps(s.to_dict())))
        assert np.allclose(s.predict(zc, zl, rec), s2.predict(zc, zl, rec), atol=0, rtol=0)


def test_events_any_overlap():
    rec = np.zeros(100, int); y = np.zeros(100, int); y[20:30] = 1; y[60:65] = 1
    D = {"y": y, "rec": rec}
    p = np.zeros(100); p[25:27] = 1; p[80:90] = 1           # one hit inside the first seizure, one false alarm
    r = P.event_scores(D, p, 0.5, 0, 1)
    assert r["n_detected"] == 1 and r["n_fa"] == 1 and r["n_true"] == 2
