"""Common predictor interface implemented by the local HF model and the API model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class Generation:
    text: str
    latency_s: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None


class Predictor(Protocol):
    name: str

    def generate(self, batch: list[list[dict[str, str]]]) -> list[Generation]:
        """Generate one completion per chat-message list."""
        ...

    @property
    def preferred_batch_size(self) -> int: ...
