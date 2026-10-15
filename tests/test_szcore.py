"""eval/szcore.py Agg against hand-computed SzCORE counts (timescoring 0.0.7 defaults: tolerance 30 s before and 60 s
after each reference event, any overlap, events < 90 s apart merged, events > 300 s split).

Each case is a synthetic 1 Hz recording; the expected TP / FP / number of reference events is worked out by hand
from those rules in the comment next to it.
"""
import numpy as np
import pytest

from szfusion.eval.szcore import Agg

N = 3600   # s, length of each synthetic recording


def mask(*iv, n=N):
    m = np.zeros(n, bool)
    for a, b in iv:
        m[a:b] = True
    return m


CASES = [
    # name, ref intervals, hyp intervals, (tp, fp, n_ref)
    ("inside", [(100, 160)], [(110, 130)], (1, 0, 1)),
    ("before onset within 30 s", [(100, 160)], [(75, 90)], (1, 0, 1)),         # extended ref starts at 70
    ("before onset beyond 30 s", [(100, 160)], [(60, 65)], (0, 1, 1)),         # ends 5 s before 70
    ("after end within 60 s", [(100, 160)], [(215, 225)], (1, 0, 1)),          # extended ref ends at 220
    ("after end beyond 60 s", [(100, 160)], [(225, 235)], (0, 1, 1)),
    ("refs < 90 s apart merge", [(100, 130), (200, 230)], [(105, 110)], (1, 0, 1)),  # gap 70 s -> one ref
    ("refs >= 90 s apart stay", [(100, 130), (300, 330)], [(105, 110)], (1, 0, 2)),  # gap 170 s -> two refs
    ("hyps < 90 s apart merge", [], [(1000, 1010), (1050, 1060)], (0, 1, 0)),  # gap 40 s -> one false alarm
    ("hyps >= 90 s apart stay", [], [(1000, 1010), (1200, 1210)], (0, 2, 0)),
    ("long hyp split", [], [(1000, 1700)], (0, 3, 0)),                         # 700 s -> 300 + 300 + 100
    ("long ref split", [(100, 800)], [(150, 160)], (1, 0, 3)),                 # 700 s ref -> 3 refs, 1 hit
    ("one hyp covers two refs", [(100, 130), (300, 330)], [(110, 320)], (2, 0, 2)),
    ("nothing", [], [], (0, 0, 0)),
]


@pytest.mark.parametrize("name,ref,hyp,exp", CASES, ids=[c[0] for c in CASES])
def test_event_counts(name, ref, hyp, exp):
    a = Agg()
    a.add(mask(*ref), mask(*hyp))
    assert (a.tp, a.fp, a.ref) == exp


def test_aggregate_over_recordings():
    """Counts are summed over recordings; sensitivity, precision, F1 and FP/day follow from the sums."""
    a = Agg()
    a.add(mask((100, 160)), mask((110, 130)))                  # tp 1
    a.add(mask((100, 160)), mask((2000, 2010)))                # fp 1, missed ref
    a.add(mask(), mask((1000, 1700)))                          # fp 3 (split)
    r = a.out()
    assert (r["tp"], r["fp"], r["n_ref"]) == (1, 4, 2)
    assert r["event_sens"] == pytest.approx(1 / 2)
    assert r["event_prec"] == pytest.approx(1 / 5)
    assert r["event_f1"] == pytest.approx(2 * 1 / (2 * 1 + 4 + 1))
    assert r["fp_per_day"] == pytest.approx(4 / (3 * N / 86400))
    assert r["hours"] == pytest.approx(3.0)


def test_sample_counts():
    """SampleScoring counts seconds: ref 60 s, hyp 20 s inside it plus 10 s outside -> tp 20, fp 10."""
    a = Agg()
    a.add(mask((100, 160)), mask((110, 130), (2000, 2010)))
    assert (a.stp, a.sfp, a.sref) == (20, 10, 60)
