"""Local inference for the base model and the base model + LoRA adapter."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from .config import GenerationSettings, ModelConfig
from .modeling import load_base_model, load_tokenizer
from .predictors import Generation

log = logging.getLogger(__name__)


def _tokenizer_source(model_cfg: ModelConfig, adapter_path: str | None) -> str:
    """Prefer the tokenizer saved next to the adapter (identical to the training one)."""
    if adapter_path and Path(adapter_path, "tokenizer_config.json").exists():
        return adapter_path
    return model_cfg.base_model


class HFPredictor:
    """Batched greedy generation with a Hugging Face causal LM (optionally + LoRA adapter)."""

    def __init__(
        self,
        model_cfg: ModelConfig,
        gen_cfg: GenerationSettings,
        adapter_path: str | None = None,
        name: str | None = None,
        quantize: bool = True,
    ) -> None:
        import torch

        self.gen_cfg = gen_cfg
        self.name = name or ("finetuned" if adapter_path else "base")
        tok_src = _tokenizer_source(model_cfg, adapter_path)
        self.tokenizer = load_tokenizer(tok_src, model_cfg.trust_remote_code, padding_side="left")

        model, _ = load_base_model(model_cfg, quantize=quantize)
        if adapter_path:
            from peft import PeftModel

            if not Path(adapter_path).exists():
                raise FileNotFoundError(f"Adapter not found: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
            log.info("Loaded LoRA adapter from %s", adapter_path)
        model.eval()
        self.model = model
        self.device = next(model.parameters()).device
        self._torch = torch
        # HF generate is not re-entrant on a shared model; serialise calls from the web server.
        self._lock = threading.Lock()

    @property
    def preferred_batch_size(self) -> int:
        return self.gen_cfg.batch_size

    def generate(self, batch: list[list[dict[str, str]]]) -> list[Generation]:
        torch = self._torch
        prompts = [
            self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in batch
        ]
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with self._lock, torch.inference_mode():
            t0 = time.perf_counter()
            out = self.model.generate(
                **enc,
                max_new_tokens=self.gen_cfg.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

        prompt_len = enc["input_ids"].shape[1]
        new_tokens = out[:, prompt_len:]
        texts = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        per_item = elapsed / max(len(batch), 1)
        pad = self.tokenizer.pad_token_id
        results = []
        for i, text in enumerate(texts):
            n_in = int(enc["attention_mask"][i].sum())
            n_out = int((new_tokens[i] != pad).sum())
            results.append(
                Generation(text=text.strip(), latency_s=per_item, input_tokens=n_in, output_tokens=n_out)
            )
        return results


def merge_adapter(model_cfg: ModelConfig, adapter_path: str, out_dir: str) -> Path:
    """Merge LoRA weights into a full-precision copy of the base model for deployment."""
    from peft import PeftModel

    base, _ = load_base_model(model_cfg, quantize=False)  # merge needs un-quantised weights
    merged = PeftModel.from_pretrained(base, adapter_path).merge_and_unload()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out), safe_serialization=True)
    load_tokenizer(_tokenizer_source(model_cfg, adapter_path)).save_pretrained(str(out))
    log.info("Merged model written to %s", out)
    return out
