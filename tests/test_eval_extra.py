"""Synthetic checks of eval/bootstrap.Ranked (weighted AUROC/AUPRC) and eval/by_type.event_type."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from szfusion.eval.bootstrap import Ranked
from szfusion.eval.by_type import event_type, group_of


def test_ranked_matches_sklearn_with_ties_and_weights():
    rng = np.random.default_rng(0)
    n, n_pat = 5000, 17
    pid = rng.integers(0, n_pat, n)
    y = (rng.random(n) < 0.1).astype(int)
    p = np.round(rng.random(n) * 0.5 + 0.4 * y, 2)          # many ties
    R = Ranked(p, y, pid)
    for w in (np.ones(n_pat), np.bincount(rng.integers(0, n_pat, n_pat), minlength=n_pat).astype(float)):
        au, ap = R.metrics(w)
        assert abs(au - roc_auc_score(y, p, sample_weight=w[pid])) < 1e-12
        assert abs(ap - average_precision_score(y, p, sample_weight=w[pid])) < 1e-12


def test_event_type_majority_channel_seconds():
    rows = [("FP1-F7", 10.0, 20.0, "fnsz"), ("F7-T3", 10.5, 30.0, "gnsz"), ("XX-YY", 0.0, 100.0, "absz")]
    t, cnt = event_type(rows, 10, 25)          # fnsz: s 10..19 -> 10; gnsz: s 10..24 -> 15; XX-YY ignored
    assert t == "gnsz" and cnt == {"fnsz": 10, "gnsz": 15}
    assert event_type([("FP1-F7", 0, 5, "cpsz"), ("F7-T3", 0, 5, "absz")], 0, 5)[0] == "absz"   # tie -> alphabetical
    assert group_of("cpsz") == "focal" and group_of("tcsz") == "generalized" and group_of("atsz") == "other"
