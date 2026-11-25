"""QLoRA fine-tuning. Loss is only computed on the answer tokens (prompt is masked with -100)."""

from __future__ import annotations

import dataclasses
import json
import logging
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from .config import ProjectConfig, dump_config
from .data import Example, load_or_build_splits
from .modeling import count_parameters, load_base_model, load_tokenizer

log = logging.getLogger(__name__)
IGNORE_INDEX = -100


def tokenize_example(ex: Example, tokenizer, max_seq_length: int, max_context_chars: int) -> dict:
    """Tokenise one example into input_ids / attention_mask / labels with prompt masking."""
    prompt_msgs = ex.messages(max_context_chars, with_target=False)
    full_msgs = ex.messages(max_context_chars, with_target=True)

    prompt_text = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(full_msgs, tokenize=False)
    if full_text.startswith(prompt_text):
        completion_text = full_text[len(prompt_text) :]
    else:  # template does not render the generation prompt as a prefix; be robust
        completion_text = full_msgs[-1]["content"] + (tokenizer.eos_token or "")

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]

    # Keep the whole prompt; truncate the completion (rationale) if the sequence is too long.
    room = max_seq_length - len(prompt_ids)
    if room < 8:
        # Prompt alone is too long: trim from the *left* of the prompt (keeps question + answer).
        keep = max_seq_length - min(len(completion_ids), 32)
        prompt_ids = prompt_ids[-keep:]
        room = max_seq_length - len(prompt_ids)
    completion_ids = completion_ids[:room]

    input_ids = prompt_ids + completion_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + completion_ids
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def _run_metadata(cfg: ProjectConfig, extra: dict[str, Any]) -> dict[str, Any]:
    import peft
    import torch
    import transformers

    return {
        "experiment_name": cfg.experiment_name,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        **extra,
    }


def train(cfg: ProjectConfig) -> Path:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )
    from transformers.trainer_utils import get_last_checkpoint

    set_seed(cfg.data.seed)
    out_dir = Path(cfg.train.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, out_dir / "config.yaml")

    splits = load_or_build_splits(cfg.data)
    tokenizer = load_tokenizer(cfg.model.base_model, cfg.model.trust_remote_code, "right")

    def to_ds(examples: list[Example]) -> Dataset:
        rows = [
            tokenize_example(e, tokenizer, cfg.train.max_seq_length, cfg.data.max_context_chars)
            for e in examples
        ]
        return Dataset.from_list(rows)

    train_ds = to_ds(splits.train)
    eval_ds = to_ds(splits.val) if splits.val else None
    log.info("Tokenised %d train / %d val examples", len(train_ds), len(eval_ds or []))

    model, quantized = load_base_model(cfg.model, quantize=True)
    model.config.use_cache = False
    if quantized:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.train.gradient_checkpointing
        )
    elif cfg.train.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lora = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        bias=cfg.lora.bias,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    params = count_parameters(model)
    log.info(
        "Trainable params: %s / %s (%.3f%%)",
        f"{params['trainable']:,}",
        f"{params['total']:,}",
        100 * params["trainable"] / max(params["total"], 1),
    )

    use_cuda = torch.cuda.is_available()
    bf16 = use_cuda and torch.cuda.is_bf16_supported()
    has_eval = eval_ds is not None and len(eval_ds) > 0
    early_stop = has_eval and cfg.train.early_stopping_patience is not None

    # transformers >=5 folds `warmup_ratio` into `warmup_steps` (a float in [0, 1) is a ratio).
    arg_fields = {f.name for f in dataclasses.fields(TrainingArguments)}
    warmup = (
        {"warmup_ratio": cfg.train.warmup_ratio}
        if "warmup_ratio" in arg_fields
        else {"warmup_steps": cfg.train.warmup_ratio}
    )

    args = TrainingArguments(
        **warmup,
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=cfg.train.num_train_epochs,
        max_steps=cfg.train.max_steps,
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.train.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        learning_rate=cfg.train.learning_rate,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        weight_decay=cfg.train.weight_decay,
        max_grad_norm=cfg.train.max_grad_norm,
        logging_steps=cfg.train.logging_steps,
        eval_strategy="steps" if has_eval else "no",
        eval_steps=cfg.train.eval_steps if has_eval else None,
        save_strategy="steps",
        save_steps=cfg.train.eval_steps,
        save_total_limit=cfg.train.save_total_limit,
        load_best_model_at_end=early_stop,
        metric_for_best_model="eval_loss" if early_stop else None,
        greater_is_better=False if early_stop else None,
        bf16=bf16,
        fp16=use_cuda and not bf16,
        optim="paged_adamw_8bit" if quantized else "adamw_torch",
        gradient_checkpointing=False,  # already configured on the model above
        report_to=[] if cfg.train.report_to == "none" else [cfg.train.report_to],
        run_name=cfg.experiment_name,
        dataloader_num_workers=cfg.train.dataloader_num_workers,
        remove_unused_columns=False,
        seed=cfg.data.seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds if has_eval else None,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, padding=True, label_pad_token_id=IGNORE_INDEX, pad_to_multiple_of=8
        ),
        callbacks=(
            [EarlyStoppingCallback(early_stopping_patience=cfg.train.early_stopping_patience)]
            if early_stop
            else []
        ),
    )

    # Pick up from the newest checkpoint if a previous run was interrupted (e.g. Colab disconnect).
    ckpt_dir = out_dir / "checkpoints"
    last_ckpt = get_last_checkpoint(str(ckpt_dir)) if ckpt_dir.is_dir() else None
    if last_ckpt:
        log.info("Resuming training from %s", last_ckpt)

    t0 = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=last_ckpt)
    train_seconds = time.perf_counter() - t0

    adapter_dir = out_dir / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    metrics: dict[str, Any] = {"train": result.metrics, "train_seconds": train_seconds}
    if has_eval:
        metrics["eval"] = trainer.evaluate()
    metrics["log_history"] = trainer.state.log_history
    with open(out_dir / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    with open(out_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            _run_metadata(
                cfg,
                {
                    "quantized": quantized,
                    "parameters": params,
                    "n_train": len(train_ds),
                    "n_val": len(eval_ds or []),
                },
            ),
            f,
            indent=2,
        )
    log.info("Saved LoRA adapter to %s (%.1f min)", adapter_dir, train_seconds / 60)
    return adapter_dir
