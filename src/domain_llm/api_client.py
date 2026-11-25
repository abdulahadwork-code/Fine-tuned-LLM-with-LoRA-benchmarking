"""API-model baseline (Claude via the official Anthropic SDK)."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .config import ApiModelConfig
from .predictors import Generation

log = logging.getLogger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicPredictor:
    """Claude baseline. Uses the same prompt as the local models; the SDK handles retries."""

    def __init__(self, cfg: ApiModelConfig, client: Any | None = None) -> None:
        self.cfg = cfg
        self.name = "api"
        if client is None:
            import anthropic

            client = anthropic.Anthropic(timeout=cfg.timeout_s, max_retries=cfg.max_retries)
        self.client = client

    @property
    def preferred_batch_size(self) -> int:
        return self.cfg.concurrency * 4

    def _cost(self, in_tok: int, out_tok: int) -> float:
        return (
            in_tok * self.cfg.input_price_per_mtok + out_tok * self.cfg.output_price_per_mtok
        ) / 1e6

    def _one(self, messages: list[dict[str, str]]) -> Generation:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        chat = [m for m in messages if m["role"] != "system"]
        kwargs: dict[str, Any] = {
            "model": self.cfg.model,
            "max_tokens": self.cfg.max_tokens,
            "system": system,
            "messages": chat,
            "output_config": {"effort": self.cfg.effort},
        }
        t0 = time.perf_counter()
        try:
            if self.cfg.enable_fallbacks:
                resp = self.client.beta.messages.create(
                    betas=[FALLBACK_BETA], fallbacks="default", **kwargs
                )
            else:
                resp = self.client.messages.create(**kwargs)
        except Exception as e:
            latency = time.perf_counter() - t0
            err = _describe_error(e)
            log.warning("API call failed: %s", err)
            return Generation(text="", latency_s=latency, error=err)
        latency = time.perf_counter() - t0

        usage = getattr(resp, "usage", None)
        in_tok = int(getattr(usage, "input_tokens", 0) or 0)
        out_tok = int(getattr(usage, "output_tokens", 0) or 0)
        text = "".join(
            getattr(b, "text", "") for b in resp.content if getattr(b, "type", None) == "text"
        ).strip()
        error = None
        if resp.stop_reason == "refusal":
            details = getattr(resp, "stop_details", None)
            error = f"refusal:{getattr(details, 'category', None)}"
        elif resp.stop_reason == "max_tokens" and not text:
            error = "max_tokens"
        return Generation(
            text=text,
            latency_s=latency,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=self._cost(in_tok, out_tok),
            error=error,
        )

    def generate(self, batch: list[list[dict[str, str]]]) -> list[Generation]:
        with ThreadPoolExecutor(max_workers=self.cfg.concurrency) as pool:
            return list(pool.map(self._one, batch))


def _describe_error(e: Exception) -> str:
    try:
        import anthropic
    except ImportError:
        return f"{type(e).__name__}: {e}"
    if isinstance(e, anthropic.AuthenticationError):
        return "authentication_error: set ANTHROPIC_API_KEY or run `ant auth login`"
    if isinstance(e, anthropic.NotFoundError):
        return f"not_found: check api.model ({e.message})"
    if isinstance(e, anthropic.BadRequestError):
        return f"bad_request: {e.message}"
    if isinstance(e, anthropic.RateLimitError):
        return "rate_limited (retries exhausted)"
    if isinstance(e, anthropic.APIStatusError):
        return f"api_status_{e.status_code}: {e.message}"
    if isinstance(e, anthropic.APIConnectionError):
        return "connection_error"
    return f"{type(e).__name__}: {e}"
