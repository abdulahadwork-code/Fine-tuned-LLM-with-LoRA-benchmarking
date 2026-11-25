"""Config loaded from YAML, with key=value overrides from the CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(_Strict):
    dataset_name: str = "qiaojin/PubMedQA"
    labeled_config: str = "pqa_labeled"
    artificial_config: str = "pqa_artificial"
    # Extra (auto-labelled, yes/no only) training examples sampled from pqa_artificial.
    train_artificial_samples: int = Field(10_000, ge=0)
    test_size: int = Field(500, gt=0)
    val_size: int = Field(100, ge=0)
    max_context_chars: int = Field(3000, gt=0)
    processed_dir: str = "data/processed"
    cache_dir: str | None = None
    seed: int = 42


class ModelConfig(_Strict):
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    quantization: Literal["4bit", "8bit", "none"] = "4bit"
    dtype: Literal["auto", "bfloat16", "float16", "float32"] = "auto"
    trust_remote_code: bool = False
    attn_implementation: str | None = None


class LoraSettings(_Strict):
    r: int = Field(16, gt=0)
    alpha: int = Field(32, gt=0)
    dropout: float = Field(0.05, ge=0.0, lt=1.0)
    target_modules: list[str] | str = "all-linear"
    bias: Literal["none", "all", "lora_only"] = "none"


class TrainConfig(_Strict):
    output_dir: str = "outputs/qlora"
    num_train_epochs: float = 2.0
    max_steps: int = -1
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    max_seq_length: int = 1024
    logging_steps: int = 10
    eval_steps: int = 100
    save_total_limit: int = 2
    gradient_checkpointing: bool = True
    early_stopping_patience: int | None = 3
    report_to: str = "none"  # "none" | "tensorboard" | "wandb"
    dataloader_num_workers: int = 0


class GenerationSettings(_Strict):
    max_new_tokens: int = Field(64, gt=0)
    batch_size: int = Field(8, gt=0)


class ApiModelConfig(_Strict):
    provider: Literal["anthropic"] = "anthropic"
    model: str = "claude-opus-5-5"
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    max_tokens: int = 2048
    # Server-side refusal fallback (beta). Set false on platforms that do not support it.
    enable_fallbacks: bool = True
    concurrency: int = Field(4, gt=0)
    timeout_s: float = 120.0
    max_retries: int = 4
    # USD per million tokens, used for the cost column of the report.
    input_price_per_mtok: float = 4.0
    output_price_per_mtok: float = 20.0


class BenchmarkConfig(_Strict):
    output_dir: str = "outputs/benchmark"
    systems: list[Literal["base", "finetuned", "api"]] = ["base", "finetuned", "api"]
    adapter_path: str | None = None  # defaults to <train.output_dir>/adapter
    limit: int | None = None
    bootstrap_samples: int = 1000
    resume: bool = True


class ProjectConfig(_Strict):
    experiment_name: str = "pubmedqa-qlora"
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    lora: LoraSettings = LoraSettings()
    train: TrainConfig = TrainConfig()
    generation: GenerationSettings = GenerationSettings()
    api: ApiModelConfig = ApiModelConfig()
    benchmark: BenchmarkConfig = BenchmarkConfig()

    @model_validator(mode="after")
    def _resolve_adapter(self) -> ProjectConfig:
        if self.benchmark.adapter_path is None:
            self.benchmark.adapter_path = str(Path(self.train.output_dir) / "adapter")
        return self


def _apply_override(tree: dict[str, Any], dotted: str) -> None:
    if "=" not in dotted:
        raise ValueError(f"Override must look like key.sub=value, got: {dotted!r}")
    key, raw = dotted.split("=", 1)
    parts = key.strip().split(".")
    node = tree
    for p in parts[:-1]:
        node = node.setdefault(p, {})
        if not isinstance(node, dict):
            raise ValueError(f"Cannot override {key!r}: {p!r} is not a section")
    node[parts[-1]] = yaml.safe_load(raw)


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> ProjectConfig:
    """Load a YAML config, apply ``a.b=value`` overrides, and validate it."""
    tree: dict[str, Any] = {}
    if path is not None:
        with open(path, encoding="utf-8") as f:
            tree = yaml.safe_load(f) or {}
    for ov in overrides or []:
        _apply_override(tree, ov)
    return ProjectConfig.model_validate(tree)


def dump_config(cfg: ProjectConfig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.model_dump(mode="json"), f, sort_keys=False)
