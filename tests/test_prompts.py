import pytest

from domain_llm.prompts import (
    build_messages,
    build_target,
    format_context,
    parse_label,
    parse_rationale,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Answer: yes\nRationale: because", "yes"),
        ("answer - NO", "no"),
        ("**Answer:** maybe", "maybe"),
        ("Answer: **Yes**", "yes"),
        ("Yes. The study shows...", "yes"),
        ("The data suggest no effect.\nAnswer: no", "no"),
        ("I cannot determine this.", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_label(text, expected):
    assert parse_label(text) == expected


def test_parse_rationale():
    assert parse_rationale("Answer: yes\nRationale: It helps.") == "It helps."
    assert parse_rationale("Answer: yes") is None


def test_build_target_validates():
    assert build_target("YES", "x").startswith("Answer: yes")
    with pytest.raises(ValueError):
        build_target("perhaps", "x")


def test_format_context_truncates():
    ctx = ["word " * 500]
    out = format_context(ctx, 100)
    assert len(out) <= 104 and out.endswith("...")
    assert format_context(["a", " ", "b"], 100) == "a\nb"


def test_build_messages_roles():
    msgs = build_messages("Q?", ["c"], 100, target="Answer: yes")
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert "Question: Q?" in msgs[1]["content"]
