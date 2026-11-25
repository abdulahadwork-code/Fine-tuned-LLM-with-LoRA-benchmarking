"""Model / tokenizer loading with optional 4-bit (QLoRA) or 8-bit quantisation."""

from __future__ import annotations

import logging
from typing import Any

from .config import ModelConfig

log = logging.getLogger(__name__)


def resolve_dtype(name: str):
    import torch

    if name == "auto":
        if torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def device_supports_quantization() -> bool:
    import torch

    if not torch.cuda.is_available():
        return False
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        return False
    return True


def load_tokenizer(model_name: str, trust_remote_code: bool = False, padding_side: str = "right"):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = padding_side
    if not getattr(tok, "chat_template", None):
        raise ValueError(
            f"Tokenizer for {model_name!r} has no chat template; use an instruct/chat model."
        )
    return tok


def load_base_model(cfg: ModelConfig, quantize: bool = True):
    """Load the base causal LM. Quantisation silently falls back to full precision on CPU."""
    import torch
    from transformers import AutoModelForCausalLM

    dtype = resolve_dtype(cfg.dtype)
    kwargs: dict[str, Any] = {"dtype": dtype, "trust_remote_code": cfg.trust_remote_code}
    if cfg.attn_implementation:
        kwargs["attn_implementation"] = cfg.attn_implementation

    want_quant = quantize and cfg.quantization != "none"
    if want_quant and not device_supports_quantization():
        log.warning(
            "Quantization=%s requested but CUDA/bitsandbytes unavailable; loading in %s instead.",
            cfg.quantization,
            dtype,
        )
        want_quant = False

    if want_quant:
        from transformers import BitsAndBytesConfig

        if cfg.quantization == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
        else:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"

    log.info("Loading %s (quantized=%s, dtype=%s)", cfg.base_model, want_quant, dtype)
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **kwargs)
    return model, want_quant


def count_parameters(model) -> dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total}
