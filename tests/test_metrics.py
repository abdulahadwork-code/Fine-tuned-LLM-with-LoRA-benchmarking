import pytest

from domain_llm.metrics import (
    accuracy,
    bootstrap_ci,
    confusion_matrix,
    macro_f1,
    mcnemar_exact,
    per_class_prf,
    percentile,
)

GOLD = ["yes", "yes", "no", "no", "maybe"]
PRED = ["yes", "no", "no", None, "maybe"]


def test_accuracy():
    assert accuracy(GOLD, PRED) == pytest.approx(3 / 5)


def test_per_class_and_macro():
    prf = per_class_prf(GOLD, PRED)
    assert prf["yes"] == pytest.approx({"precision": 1.0, "recall": 0.5, "f1": 2 / 3, "support": 2})
    assert prf["no"]["precision"] == 0.5 and prf["no"]["recall"] == 0.5
    assert prf["maybe"]["f1"] == 1.0
    assert macro_f1(GOLD, PRED) == pytest.approx((2 / 3 + 0.5 + 1.0) / 3)


def test_confusion_matrix_counts_unparsed():
    cm = confusion_matrix(GOLD, PRED)
    assert cm["no"]["unparsed"] == 1
    assert sum(sum(r.values()) for r in cm.values()) == 5


def test_bootstrap_ci_brackets_point():
    lo, hi = bootstrap_ci(GOLD * 20, PRED * 20, n=300)
    assert lo <= 0.6 <= hi


def test_mcnemar():
    a = [True] * 10 + [False] * 10
    b = [True] * 10 + [True] * 10
    r = mcnemar_exact(a, b)
    assert r["a_only"] == 0 and r["b_only"] == 10
    assert r["p_value"] == pytest.approx(2 / 2**10)
    assert mcnemar_exact([True], [True])["p_value"] == 1.0


def test_percentile():
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([], 0.5) == 0.0
