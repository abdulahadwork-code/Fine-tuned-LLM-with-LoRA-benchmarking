"""PubMedQA loading and splitting. Splits are cached as JSONL so every run uses the same test set."""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import DataConfig
from .prompts import LABELS, build_messages, build_target

log = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Example:
    id: str
    question: str
    contexts: tuple[str, ...]
    label: str
    long_answer: str = ""
    source: str = "labeled"

    def messages(self, max_context_chars: int, with_target: bool = False) -> list[dict[str, str]]:
        target = build_target(self.label, self.long_answer) if with_target else None
        return build_messages(self.question, self.contexts, max_context_chars, target)

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["contexts"] = list(self.contexts)
        return d

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> Example:
        return cls(
            id=str(d["id"]),
            question=d["question"],
            contexts=tuple(d["contexts"]),
            label=d["label"],
            long_answer=d.get("long_answer", ""),
            source=d.get("source", "labeled"),
        )


@dataclass
class DatasetSplits:
    train: list[Example] = field(default_factory=list)
    val: list[Example] = field(default_factory=list)
    test: list[Example] = field(default_factory=list)

    def __getitem__(self, name: str) -> list[Example]:
        return getattr(self, name)

    def summary(self) -> dict[str, dict[str, int]]:
        out = {}
        for s in SPLITS:
            counts: dict[str, int] = defaultdict(int)
            for ex in self[s]:
                counts[ex.label] += 1
            out[s] = {"total": len(self[s]), **dict(sorted(counts.items()))}
        return out


def normalize_record(rec: Mapping[str, Any], source: str) -> Example | None:
    """Convert a raw HF PubMedQA row into an Example. Returns None for unusable rows."""
    label = str(rec.get("final_decision", "")).strip().lower()
    question = str(rec.get("question", "")).strip()
    ctx = rec.get("context") or {}
    contexts = ctx.get("contexts", []) if isinstance(ctx, Mapping) else list(ctx)
    contexts = tuple(str(c).strip() for c in contexts if str(c).strip())
    if label not in LABELS or not question or not contexts:
        return None
    return Example(
        id=str(rec.get("pubid", "")),
        question=question,
        contexts=contexts,
        label=label,
        long_answer=str(rec.get("long_answer", "") or "").strip(),
        source=source,
    )


def stratified_split(
    examples: list[Example], test_size: int, val_size: int, seed: int
) -> tuple[list[Example], list[Example], list[Example]]:
    """Deterministic stratified split preserving label proportions in each split."""
    n = len(examples)
    if test_size + val_size >= n:
        raise ValueError(f"test_size + val_size ({test_size + val_size}) must be < {n}")
    rng = random.Random(seed)
    by_label: dict[str, list[Example]] = defaultdict(list)
    for ex in sorted(examples, key=lambda e: e.id):
        by_label[ex.label].append(ex)

    test, val, train = [], [], []
    for label in sorted(by_label):
        group = by_label[label]
        rng.shuffle(group)
        n_test = round(len(group) * test_size / n)
        n_val = round(len(group) * val_size / n)
        test += group[:n_test]
        val += group[n_test : n_test + n_val]
        train += group[n_test + n_val :]
    for split in (train, val, test):
        rng.shuffle(split)
    return train, val, test


def _load_hf(cfg: DataConfig, subset: str) -> Iterable[Mapping[str, Any]]:
    from datasets import load_dataset

    return load_dataset(cfg.dataset_name, subset, split="train", cache_dir=cfg.cache_dir)


def build_splits(cfg: DataConfig) -> DatasetSplits:
    labeled = [e for r in _load_hf(cfg, cfg.labeled_config) if (e := normalize_record(r, "labeled"))]
    log.info("Loaded %d labelled examples", len(labeled))
    train, val, test = stratified_split(labeled, cfg.test_size, cfg.val_size, cfg.seed)

    if cfg.train_artificial_samples > 0:
        reserved = {e.id for e in labeled}
        ds = _load_hf(cfg, cfg.artificial_config)
        # Sample indices first so we do not materialise 211k rows.
        rng = random.Random(cfg.seed)
        n_total = len(ds)
        want = min(cfg.train_artificial_samples, n_total)
        idx = rng.sample(range(n_total), min(n_total, int(want * 1.05) + 10))
        extra: list[Example] = []
        for i in idx:
            ex = normalize_record(ds[i], "artificial")
            if ex and ex.id not in reserved:
                extra.append(ex)
                reserved.add(ex.id)
            if len(extra) >= want:
                break
        log.info("Added %d artificial training examples", len(extra))
        train = train + extra
        random.Random(cfg.seed + 1).shuffle(train)

    splits = DatasetSplits(train=train, val=val, test=test)
    assert_no_leakage(splits)
    return splits


def assert_no_leakage(splits: DatasetSplits) -> None:
    ids = {s: {e.id for e in splits[s]} for s in SPLITS}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = ids[a] & ids[b]
        if overlap:
            raise ValueError(f"Data leakage: {len(overlap)} ids shared between {a} and {b}")


def save_splits(splits: DatasetSplits, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        with open(out / f"{s}.jsonl", "w", encoding="utf-8") as f:
            for ex in splits[s]:
                f.write(json.dumps(ex.to_json(), ensure_ascii=False) + "\n")
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(splits.summary(), f, indent=2)


def load_split(path: str | Path) -> list[Example]:
    with open(path, encoding="utf-8") as f:
        return [Example.from_json(json.loads(line)) for line in f if line.strip()]


def load_or_build_splits(cfg: DataConfig, rebuild: bool = False) -> DatasetSplits:
    out = Path(cfg.processed_dir)
    if not rebuild and all((out / f"{s}.jsonl").exists() for s in SPLITS):
        log.info("Using cached processed splits in %s", out)
        return DatasetSplits(**{s: load_split(out / f"{s}.jsonl") for s in SPLITS})
    splits = build_splits(cfg)
    save_splits(splits, out)
    log.info("Saved processed splits to %s: %s", out, splits.summary())
    return splits
