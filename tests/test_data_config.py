import pytest

from domain_llm.config import load_config
from domain_llm.data import (
    DatasetSplits,
    Example,
    assert_no_leakage,
    load_split,
    normalize_record,
    save_splits,
    stratified_split,
)


def test_normalize_record():
    rec = {
        "pubid": 123,
        "question": " Q? ",
        "context": {"contexts": ["a", "", "b"]},
        "long_answer": "LA",
        "final_decision": "Yes",
    }
    ex = normalize_record(rec, "labeled")
    assert ex == Example("123", "Q?", ("a", "b"), "yes", "LA", "labeled")
    assert normalize_record({**rec, "final_decision": "unknown"}, "x") is None
    assert normalize_record({**rec, "context": {"contexts": []}}, "x") is None


def test_stratified_split_deterministic_and_disjoint(examples):
    a = stratified_split(examples, 20, 10, seed=1)
    b = stratified_split(examples, 20, 10, seed=1)
    assert [e.id for e in a[2]] == [e.id for e in b[2]]
    train, val, test = a
    assert len(test) == 20 and len(val) == 10 and len(train) == 30
    assert_no_leakage(DatasetSplits(train, val, test))
    # proportions preserved: 50% yes, 33% no, 17% maybe
    assert sum(e.label == "yes" for e in test) == 10


def test_split_too_large(examples):
    with pytest.raises(ValueError):
        stratified_split(examples, 50, 20, seed=0)


def test_leakage_detected(examples):
    with pytest.raises(ValueError, match="leakage"):
        assert_no_leakage(DatasetSplits(examples[:5], [], examples[:1]))


def test_save_load_roundtrip(tmp_path, examples):
    splits = DatasetSplits(examples[:3], examples[3:4], examples[4:6])
    save_splits(splits, tmp_path)
    assert load_split(tmp_path / "test.jsonl") == examples[4:6]
    assert (tmp_path / "summary.json").exists()


def test_config_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("train:\n  max_steps: 5\n")
    cfg = load_config(p, ["train.learning_rate=1e-4", "benchmark.systems=[base, api]"])
    assert cfg.train.max_steps == 5
    assert cfg.train.learning_rate == 1e-4
    assert cfg.benchmark.systems == ["base", "api"]
    assert cfg.benchmark.adapter_path.replace("\\", "/") == "outputs/qlora/adapter"


def test_config_rejects_unknown_keys():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        load_config(None, ["train.not_a_field=1"])


def test_shipped_configs_valid():
    for name in ("configs/qlora.yaml", "configs/smoke.yaml"):
        load_config(name)
