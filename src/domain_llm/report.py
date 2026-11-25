"""Markdown + PNG report for benchmark results."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .prompts import LABELS

log = logging.getLogger(__name__)

DISPLAY = {"base": "Base model (zero-shot)", "finetuned": "Fine-tuned (QLoRA)", "api": "API model"}


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def render_markdown(summary: dict[str, Any]) -> str:
    systems = summary["systems"]
    lines = [
        f"# Benchmark: {summary['experiment_name']}",
        "",
        f"- Base model: `{summary['base_model']}`",
        *([f"- API model: `{summary['api_model']}`"] if "api" in systems else []),
        f"- Test examples: {summary['n_test']} (PubMedQA expert-labelled, held out)",
        "",
        "## Headline results",
        "",
        "| System | Accuracy (95% CI) | Macro-F1 (95% CI) | Parse rate | p50 latency | p95 latency | Cost |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, m in systems.items():
        alo, ahi = m["accuracy_ci95"]
        flo, fhi = m["macro_f1_ci95"]
        lines.append(
            f"| {DISPLAY.get(name, name)} | {_pct(m['accuracy'])} ({_pct(alo)}–{_pct(ahi)}) "
            f"| {_pct(m['macro_f1'])} ({_pct(flo)}–{_pct(fhi)}) | {_pct(m['parse_rate'])} "
            f"| {m['latency_p50_s'] * 1000:.0f} ms | {m['latency_p95_s'] * 1000:.0f} ms "
            f"| ${m['total_cost_usd']:.4f} |"
        )

    lines += ["", "## Per-class F1", "", "| System | " + " | ".join(LABELS) + " |"]
    lines.append("|---|" + "---|" * len(LABELS))
    for name, m in systems.items():
        row = " | ".join(f"{m['per_class'][c]['f1']:.3f}" for c in LABELS)
        lines.append(f"| {DISPLAY.get(name, name)} | {row} |")

    if summary.get("pairwise_mcnemar"):
        lines += [
            "",
            "## Significance (exact McNemar test on paired correctness)",
            "",
            "| Comparison | Only A correct | Only B correct | p-value |",
            "|---|---|---|---|",
        ]
        for k, v in summary["pairwise_mcnemar"].items():
            lines.append(f"| {k.replace('_vs_', ' vs ')} | {v['a_only']} | {v['b_only']} | {v['p_value']:.4g} |")

    lines += ["", "## Confusion matrices (rows = gold, cols = predicted)", ""]
    for name, m in systems.items():
        cm = m["confusion_matrix"]
        cols = [*LABELS, "unparsed"]
        lines += [f"**{DISPLAY.get(name, name)}**", "", "| gold \\ pred | " + " | ".join(cols) + " |"]
        lines.append("|---|" + "---|" * len(cols))
        for g in LABELS:
            lines.append(f"| {g} | " + " | ".join(str(cm[g][c]) for c in cols) + " |")
        lines.append("")

    lines += [
        "## Notes",
        "",
        "- All systems receive the identical system prompt, context (truncated to the same "
        "character budget) and question.",
        "- Unparseable outputs and API errors count as incorrect.",
        "- Local-model latency is per example within a batch on the benchmark machine; API "
        "latency is wall-clock per request and includes network time.",
    ]
    return "\n".join(lines) + "\n"


def _plot(summary: dict[str, Any], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.info("matplotlib not installed; skipping chart")
        return
    systems = summary["systems"]
    names = [DISPLAY.get(s, s) for s in systems]
    acc = [systems[s]["accuracy"] * 100 for s in systems]
    f1 = [systems[s]["macro_f1"] * 100 for s in systems]
    err = [
        [systems[s]["accuracy"] * 100 - systems[s]["accuracy_ci95"][0] * 100 for s in systems],
        [systems[s]["accuracy_ci95"][1] * 100 - systems[s]["accuracy"] * 100 for s in systems],
    ]
    x = range(len(names))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    w = 0.38
    ax.bar([i - w / 2 for i in x], acc, w, yerr=err, capsize=4, label="Accuracy", color="#2a6fdb")
    ax.bar([i + w / 2 for i in x], f1, w, label="Macro-F1", color="#9bbcf0")
    ax.set_xticks(list(x), names)
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)
    ax.set_title("PubMedQA test set")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(summary: dict[str, Any], out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md = out / "report.md"
    md.write_text(render_markdown(summary), encoding="utf-8")
    _plot(summary, out / "accuracy.png")
    return md
