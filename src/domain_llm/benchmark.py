"""Benchmark base model vs. fine-tuned model vs. API model on the held-out test split."""

from __future__ import annotations

import gc
import json
import logging
from collections.abc import Callable
from itertools import combinations
from pathlib import Path
from typing import Any

from .config import ProjectConfig
from .data import Example, load_or_build_splits
from .metrics import mcnemar_exact, summarize
from .predictors import Predictor
from .prompts import parse_label
from .report import write_report

log = logging.getLogger(__name__)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_system(
    predictor: Predictor,
    examples: list[Example],
    out_path: Path,
    max_context_chars: int,
    resume: bool = True,
) -> list[dict[str, Any]]:
    """Run one system over the examples, appending to a JSONL file so runs can resume.

    Rows that previously errored (e.g. rate limits, network) are retried on resume.
    """
    done = {r["id"]: r for r in _read_jsonl(out_path) if not r.get("error")} if resume else {}
    if not resume and out_path.exists():
        out_path.unlink()
    elif resume and out_path.exists():
        # Rewrite without the errored rows so they are retried.
        with open(out_path, "w", encoding="utf-8") as f:
            for r in done.values():
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    todo = [e for e in examples if e.id not in done]
    log.info("[%s] %d cached, %d to run", predictor.name, len(done), len(todo))
    bs = max(1, predictor.preferred_batch_size)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        for start in range(0, len(todo), bs):
            chunk = todo[start : start + bs]
            gens = predictor.generate([e.messages(max_context_chars) for e in chunk])
            for ex, g in zip(chunk, gens, strict=True):
                row = {
                    "id": ex.id,
                    "gold": ex.label,
                    "pred": parse_label(g.text),
                    "raw_output": g.text,
                    "latency_s": g.latency_s,
                    "input_tokens": g.input_tokens,
                    "output_tokens": g.output_tokens,
                    "cost_usd": g.cost_usd,
                    "error": g.error,
                }
                done[ex.id] = row
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            log.info("[%s] %d/%d", predictor.name, len(done), len(examples))

    order = {e.id: i for i, e in enumerate(examples)}
    return sorted((r for r in done.values() if r["id"] in order), key=lambda r: order[r["id"]])


def default_factory(cfg: ProjectConfig) -> Callable[[str], Predictor]:
    def make(system: str) -> Predictor:
        if system == "api":
            from .api_client import AnthropicPredictor

            return AnthropicPredictor(cfg.api)
        from .inference import HFPredictor

        adapter = cfg.benchmark.adapter_path if system == "finetuned" else None
        return HFPredictor(cfg.model, cfg.generation, adapter_path=adapter, name=system)

    return make


def _free(predictor: Any) -> None:
    del predictor
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run_benchmark(
    cfg: ProjectConfig,
    factory: Callable[[str], Predictor] | None = None,
    test_examples: list[Example] | None = None,
) -> dict[str, Any]:
    factory = factory or default_factory(cfg)
    out_dir = Path(cfg.benchmark.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = test_examples if test_examples is not None else load_or_build_splits(cfg.data).test
    if cfg.benchmark.limit:
        examples = examples[: cfg.benchmark.limit]
    log.info("Benchmarking %s on %d test examples", cfg.benchmark.systems, len(examples))

    all_records: dict[str, list[dict[str, Any]]] = {}
    for system in cfg.benchmark.systems:
        predictor = factory(system)
        try:
            all_records[system] = run_system(
                predictor,
                examples,
                out_dir / f"predictions_{system}.jsonl",
                cfg.data.max_context_chars,
                cfg.benchmark.resume,
            )
        finally:
            _free(predictor)

    summary: dict[str, Any] = {
        "experiment_name": cfg.experiment_name,
        "base_model": cfg.model.base_model,
        "api_model": cfg.api.model,
        "n_test": len(examples),
        "systems": {
            s: summarize(r, cfg.benchmark.bootstrap_samples) for s, r in all_records.items()
        },
        "pairwise_mcnemar": {},
    }
    for a, b in combinations(all_records, 2):
        ca = [r["pred"] == r["gold"] for r in all_records[a]]
        cb = [r["pred"] == r["gold"] for r in all_records[b]]
        summary["pairwise_mcnemar"][f"{a}_vs_{b}"] = mcnemar_exact(ca, cb)

    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_report(summary, out_dir)
    log.info("Benchmark results written to %s", out_dir)
    return summary
