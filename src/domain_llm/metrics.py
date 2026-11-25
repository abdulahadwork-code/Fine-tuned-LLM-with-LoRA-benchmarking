"""Eval metrics: accuracy, F1, bootstrap CI, McNemar."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Any

from .prompts import LABELS


def confusion_matrix(gold: Sequence[str], pred: Sequence[str | None]) -> dict[str, dict[str, int]]:
    cols = [*LABELS, "unparsed"]
    cm = {g: {p: 0 for p in cols} for g in LABELS}
    for g, p in zip(gold, pred, strict=True):
        cm[g][p if p in LABELS else "unparsed"] += 1
    return cm


def per_class_prf(gold: Sequence[str], pred: Sequence[str | None]) -> dict[str, dict[str, float]]:
    out = {}
    for c in LABELS:
        tp = sum(1 for g, p in zip(gold, pred, strict=True) if g == c and p == c)
        fp = sum(1 for g, p in zip(gold, pred, strict=True) if g != c and p == c)
        fn = sum(1 for g, p in zip(gold, pred, strict=True) if g == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[c] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
    return out


def accuracy(gold: Sequence[str], pred: Sequence[str | None]) -> float:
    return sum(g == p for g, p in zip(gold, pred, strict=True)) / len(gold) if gold else 0.0


def macro_f1(gold: Sequence[str], pred: Sequence[str | None]) -> float:
    """Macro F1 over classes that appear in the gold labels."""
    prf = per_class_prf(gold, pred)
    present = [c for c in LABELS if prf[c]["support"] > 0]
    return sum(prf[c]["f1"] for c in present) / len(present) if present else 0.0


def bootstrap_ci(
    gold: Sequence[str],
    pred: Sequence[str | None],
    metric=accuracy,
    n: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    if not gold or n <= 0:
        return (0.0, 0.0)
    rng = random.Random(seed)
    idx = range(len(gold))
    scores = []
    for _ in range(n):
        s = [rng.choice(idx) for _ in idx]
        scores.append(metric([gold[i] for i in s], [pred[i] for i in s]))
    scores.sort()
    lo = scores[int((alpha / 2) * n)]
    hi = scores[min(n - 1, int((1 - alpha / 2) * n))]
    return (lo, hi)


def mcnemar_exact(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> dict[str, float]:
    """Exact (binomial) McNemar test on paired correctness. Two-sided p-value."""
    b = sum(1 for x, y in zip(correct_a, correct_b, strict=True) if x and not y)
    c = sum(1 for x, y in zip(correct_a, correct_b, strict=True) if y and not x)
    n = b + c
    if n == 0:
        return {"a_only": 0, "b_only": 0, "p_value": 1.0}
    k = min(b, c)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return {"a_only": b, "b_only": c, "p_value": min(1.0, p)}


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] if lo == hi else s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(records: Sequence[dict[str, Any]], bootstrap_samples: int = 1000) -> dict[str, Any]:
    gold = [r["gold"] for r in records]
    pred = [r["pred"] for r in records]
    lat = [r["latency_s"] for r in records if not r.get("error")]
    acc_ci = bootstrap_ci(gold, pred, accuracy, bootstrap_samples)
    f1_ci = bootstrap_ci(gold, pred, macro_f1, bootstrap_samples)
    return {
        "n": len(records),
        "accuracy": accuracy(gold, pred),
        "accuracy_ci95": acc_ci,
        "macro_f1": macro_f1(gold, pred),
        "macro_f1_ci95": f1_ci,
        "parse_rate": sum(p in LABELS for p in pred) / len(pred) if pred else 0.0,
        "error_rate": sum(1 for r in records if r.get("error")) / len(records) if records else 0.0,
        "per_class": per_class_prf(gold, pred),
        "confusion_matrix": confusion_matrix(gold, pred),
        "latency_p50_s": percentile(lat, 0.5),
        "latency_p95_s": percentile(lat, 0.95),
        "total_input_tokens": sum(r.get("input_tokens", 0) for r in records),
        "total_output_tokens": sum(r.get("output_tokens", 0) for r in records),
        "total_cost_usd": sum(r.get("cost_usd", 0.0) for r in records),
    }
