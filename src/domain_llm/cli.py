"""Command-line entry point: ``domain-llm <command>``."""

from __future__ import annotations

import json
import logging
import os
from typing import Annotated

import typer
from rich.logging import RichHandler

from .config import load_config

app = typer.Typer(add_completion=False, help="QLoRA fine-tuning + benchmarking on PubMedQA.")

ConfigOpt = Annotated[str | None, typer.Option("--config", "-c", help="YAML config path")]
SetOpt = Annotated[
    list[str] | None, typer.Option("--set", "-s", help="Override, e.g. train.max_steps=50")
]


@app.callback()
def _setup(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    handler = RichHandler(rich_tracebacks=True, show_path=False)
    # Drop per-request HTTP chatter from the HF Hub / Anthropic clients unless --verbose.
    if not verbose:
        handler.addFilter(
            lambda r: r.levelno >= logging.WARNING or not r.name.startswith(_NOISY_LOGGERS)
        )
    logging.basicConfig(
        level=logging.DEBUG if verbose else os.getenv("LOG_LEVEL", "INFO"),
        format="%(message)s",
        handlers=[handler],
    )


_NOISY_LOGGERS = ("httpx", "httpcore", "urllib3", "filelock", "huggingface_hub")


@app.command("prepare-data")
def prepare_data(config: ConfigOpt = None, set_: SetOpt = None, rebuild: bool = True) -> None:
    """Download PubMedQA and write leakage-free train/val/test JSONL splits."""
    from .data import load_or_build_splits

    cfg = load_config(config, set_)
    splits = load_or_build_splits(cfg.data, rebuild=rebuild)
    typer.echo(json.dumps(splits.summary(), indent=2))


@app.command()
def train(config: ConfigOpt = None, set_: SetOpt = None) -> None:
    """Fine-tune the base model with (Q)LoRA."""
    from .train import train as run_train

    run_train(load_config(config, set_))


@app.command()
def benchmark(config: ConfigOpt = None, set_: SetOpt = None) -> None:
    """Evaluate base vs fine-tuned vs API model on the test split."""
    from .benchmark import run_benchmark

    summary = run_benchmark(load_config(config, set_))
    for name, m in summary["systems"].items():
        typer.echo(
            f"{name:>10}: acc={m['accuracy']:.3f} macroF1={m['macro_f1']:.3f} "
            f"parse={m['parse_rate']:.2f} cost=${m['total_cost_usd']:.4f}"
        )


@app.command()
def merge(
    out_dir: Annotated[str, typer.Option(help="Where to write the merged model")] = "outputs/merged",
    config: ConfigOpt = None,
    set_: SetOpt = None,
) -> None:
    """Merge the LoRA adapter into the base weights for standalone deployment."""
    from .inference import merge_adapter

    cfg = load_config(config, set_)
    merge_adapter(cfg.model, cfg.benchmark.adapter_path, out_dir)


@app.command()
def predict(
    question: Annotated[str, typer.Option(help="Research question")],
    context: Annotated[list[str], typer.Option(help="Context passage (repeatable)")],
    adapter: Annotated[str | None, typer.Option(help="LoRA adapter (omit for base)")] = None,
    config: ConfigOpt = None,
    set_: SetOpt = None,
) -> None:
    """Answer a single question from the command line."""
    from .inference import HFPredictor
    from .prompts import build_messages

    cfg = load_config(config, set_)
    p = HFPredictor(cfg.model, cfg.generation, adapter_path=adapter)
    gen = p.generate([build_messages(question, context, cfg.data.max_context_chars)])[0]
    typer.echo(gen.text)


@app.command()
def serve(
    host: str = "0.0.0.0",
    port: int = 8000,
    config: ConfigOpt = None,
    adapter: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Serve the fine-tuned model over HTTP (FastAPI)."""
    import uvicorn

    from .serve import create_app

    if config:
        os.environ["DOMAIN_LLM_CONFIG"] = config
    if adapter:
        os.environ["DOMAIN_LLM_ADAPTER"] = adapter
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    app()
