"""Prompt + answer parsing. All three systems get the same prompt."""

from __future__ import annotations

import re
from collections.abc import Sequence

LABELS: tuple[str, ...] = ("yes", "no", "maybe")

SYSTEM_PROMPT = (
    "You are a biomedical research assistant. You answer research questions using only the "
    "evidence in the provided abstract excerpts. Reply in exactly this format:\n"
    "Answer: <yes|no|maybe>\n"
    "Rationale: <one or two sentences grounded in the context>"
)


def format_context(contexts: Sequence[str], max_chars: int) -> str:
    """Join context passages and truncate deterministically to ``max_chars`` characters."""
    text = "\n".join(c.strip() for c in contexts if c and c.strip())
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    space = cut.rfind(" ")
    if space > max_chars * 0.8:
        cut = cut[:space]
    return cut.rstrip() + " ..."


def build_user_prompt(question: str, contexts: Sequence[str], max_context_chars: int) -> str:
    ctx = format_context(contexts, max_context_chars)
    return f"Context:\n{ctx}\n\nQuestion: {question.strip()}"


def build_target(label: str, long_answer: str | None) -> str:
    label = label.strip().lower()
    if label not in LABELS:
        raise ValueError(f"Unknown label {label!r}")
    rationale = (long_answer or "").strip() or "See context."
    return f"Answer: {label}\nRationale: {rationale}"


def build_messages(
    question: str,
    contexts: Sequence[str],
    max_context_chars: int,
    target: str | None = None,
) -> list[dict[str, str]]:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(question, contexts, max_context_chars)},
    ]
    if target is not None:
        msgs.append({"role": "assistant", "content": target})
    return msgs


_ANSWER_RE = re.compile(r"answer\s*[:\-]\s*\**\s*(yes|no|maybe)\b", re.IGNORECASE)
_WORD_RE = re.compile(r"\b(yes|no|maybe)\b", re.IGNORECASE)
_RATIONALE_RE = re.compile(r"rationale\s*[:\-]\s*(.+)", re.IGNORECASE | re.DOTALL)


def parse_label(text: str | None) -> str | None:
    """Extract yes/no/maybe from model output. Returns None if nothing parseable."""
    if not text:
        return None
    m = _ANSWER_RE.search(text)
    if m:
        return m.group(1).lower()
    # Fallback: first label word in the first non-empty line (handles bare "Yes." replies).
    for line in text.strip().splitlines():
        if line.strip():
            w = _WORD_RE.search(line)
            return w.group(1).lower() if w else None
    return None


def parse_rationale(text: str | None) -> str | None:
    if not text:
        return None
    m = _RATIONALE_RE.search(text)
    return m.group(1).strip() if m else None
