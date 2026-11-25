"""Pipeline tests with fake predictors / fake API client (no GPU, no network)."""

import json
from types import SimpleNamespace

import pytest

from domain_llm.config import ApiModelConfig, load_config
from domain_llm.predictors import Generation


class FakePredictor:
    def __init__(self, name, answer_fn):
        self.name = name
        self.answer_fn = answer_fn
        self.calls = 0

    @property
    def preferred_batch_size(self):
        return 4

    def generate(self, batch):
        self.calls += len(batch)
        return [
            Generation(text=self.answer_fn(m), latency_s=0.01, input_tokens=10, output_tokens=5)
            for m in batch
        ]


def test_run_benchmark_end_to_end(tmp_path, examples):
    from domain_llm.benchmark import run_benchmark

    cfg = load_config(
        None,
        [f"benchmark.output_dir={tmp_path.as_posix()}", "benchmark.systems=[base, finetuned]",
         "benchmark.bootstrap_samples=50"],
    )
    gold = {e.question: e.label for e in examples}

    def oracle(msgs):
        q = msgs[1]["content"].split("Question: ")[1]
        return f"Answer: {gold[q]}\nRationale: ok"

    preds = {
        "base": FakePredictor("base", lambda m: "Answer: yes"),
        "finetuned": FakePredictor("finetuned", oracle),
    }
    summary = run_benchmark(cfg, factory=lambda s: preds[s], test_examples=examples)

    assert summary["systems"]["finetuned"]["accuracy"] == 1.0
    assert summary["systems"]["base"]["accuracy"] == pytest.approx(30 / 60)
    assert summary["pairwise_mcnemar"]["base_vs_finetuned"]["p_value"] < 0.001
    assert (tmp_path / "report.md").read_text(encoding="utf-8").startswith("# Benchmark")
    assert json.loads((tmp_path / "results.json").read_text())["n_test"] == 60

    # Resume: a second run must not call the predictor again.
    preds["base"] = FakePredictor("base", lambda m: "Answer: yes")
    run_benchmark(cfg, factory=lambda s: preds[s], test_examples=examples)
    assert preds["base"].calls == 0


def _fake_response(text="Answer: no\nRationale: x", stop_reason="end_turn"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        stop_details=SimpleNamespace(category="bio") if stop_reason == "refusal" else None,
        usage=SimpleNamespace(input_tokens=1000, output_tokens=100),
    )


def test_anthropic_predictor_with_fallbacks():
    from domain_llm.api_client import FALLBACK_BETA, AnthropicPredictor

    seen = {}

    def create(**kw):
        seen.update(kw)
        return _fake_response()

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
    p = AnthropicPredictor(ApiModelConfig(), client=client)
    g = p.generate([[{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]])[0]
    assert g.text.startswith("Answer: no") and g.error is None
    assert g.cost_usd == pytest.approx((1000 * 4 + 100 * 20) / 1e6)
    assert seen["system"] == "S" and seen["messages"] == [{"role": "user", "content": "U"}]
    assert seen["betas"] == [FALLBACK_BETA] and seen["fallbacks"] == "default"
    assert seen["output_config"] == {"effort": "low"}
    assert "temperature" not in seen and "thinking" not in seen


def test_anthropic_predictor_refusal_and_exception():
    from domain_llm.api_client import AnthropicPredictor

    client = SimpleNamespace(
        messages=SimpleNamespace(create=lambda **kw: _fake_response("", "refusal"))
    )
    p = AnthropicPredictor(ApiModelConfig(enable_fallbacks=False), client=client)
    assert p.generate([[{"role": "user", "content": "U"}]])[0].error == "refusal:bio"

    def boom(**kw):
        raise RuntimeError("network down")

    p.client = SimpleNamespace(messages=SimpleNamespace(create=boom))
    g = p.generate([[{"role": "user", "content": "U"}]])[0]
    assert g.text == "" and "network down" in g.error


def test_serve_endpoints(monkeypatch):
    from fastapi.testclient import TestClient

    from domain_llm.serve import create_app

    fake = FakePredictor("ft", lambda m: "Answer: maybe\nRationale: mixed evidence")
    with TestClient(create_app(predictor=fake, model_label="test")) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/ready").json()["model"] == "test"

        r = client.post("/v1/predict", json={"question": "Does X help?", "contexts": ["ctx"]})
        assert r.status_code == 200
        body = r.json()
        assert body["answer"] == "maybe" and body["rationale"] == "mixed evidence"
        assert r.headers["x-request-id"] == body["request_id"]

        assert client.post("/v1/predict", json={"question": "Q", "contexts": []}).status_code == 422

        monkeypatch.setenv("DOMAIN_LLM_API_KEY", "secret")
        payload = {"question": "Does X help?", "contexts": ["c"]}
        assert client.post("/v1/predict", json=payload).status_code == 401
        ok = client.post("/v1/predict", json=payload, headers={"X-API-Key": "secret"})
        assert ok.status_code == 200
