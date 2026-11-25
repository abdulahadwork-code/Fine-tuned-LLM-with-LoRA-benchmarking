"""Production HTTP API for the fine-tuned model.

Environment variables:
  DOMAIN_LLM_CONFIG        path to YAML config (default: configs/qlora.yaml if present)
  DOMAIN_LLM_ADAPTER       LoRA adapter dir (default: benchmark.adapter_path from config)
  DOMAIN_LLM_MERGED_MODEL  path to a merged model (takes precedence over base + adapter)
  DOMAIN_LLM_API_KEY       if set, requests must send header ``X-API-Key`` with this value
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from . import __version__
from .config import load_config
from .predictors import Predictor
from .prompts import build_messages, parse_label, parse_rationale

log = logging.getLogger("domain_llm.serve")


class PredictRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    contexts: list[str] = Field(..., min_length=1, max_length=50)


class PredictResponse(BaseModel):
    answer: str | None
    rationale: str | None
    raw_output: str
    latency_ms: float
    model: str
    request_id: str


def _load_default_predictor() -> tuple[Predictor, int, str]:
    from .inference import HFPredictor

    cfg_path = os.getenv("DOMAIN_LLM_CONFIG") or (
        "configs/qlora.yaml" if Path("configs/qlora.yaml").exists() else None
    )
    cfg = load_config(cfg_path)
    merged = os.getenv("DOMAIN_LLM_MERGED_MODEL")
    if merged:
        cfg.model.base_model = merged
        predictor = HFPredictor(cfg.model, cfg.generation, adapter_path=None, name="finetuned-merged")
        label = merged
    else:
        adapter = os.getenv("DOMAIN_LLM_ADAPTER") or cfg.benchmark.adapter_path
        predictor = HFPredictor(cfg.model, cfg.generation, adapter_path=adapter, name="finetuned")
        label = f"{cfg.model.base_model}+{adapter}"
    return predictor, cfg.data.max_context_chars, label


def create_app(
    predictor: Predictor | None = None,
    max_context_chars: int = 3000,
    model_label: str = "custom",
    max_concurrency: int = 4,
) -> FastAPI:
    state: dict = {"predictor": predictor, "ctx": max_context_chars, "label": model_label}
    sem = asyncio.Semaphore(max_concurrency)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state["predictor"] is None:
            log.info("Loading model ...")
            p, ctx, label = await run_in_threadpool(_load_default_predictor)
            state.update(predictor=p, ctx=ctx, label=label)
            log.info("Model ready: %s", label)
        yield

    app = FastAPI(title="Domain LLM (PubMedQA)", version=__version__, lifespan=lifespan)

    def check_key(x_api_key: str | None = Header(default=None)) -> None:
        expected = os.getenv("DOMAIN_LLM_API_KEY")
        if expected and not (x_api_key and hmac.compare_digest(x_api_key, expected)):
            raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = rid
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = rid
        log.info(
            "%s %s %s %.1fms rid=%s",
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - t0) * 1000,
            rid,
        )
        return response

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": __version__}

    @app.get("/ready")
    async def ready():
        if state["predictor"] is None:
            raise HTTPException(status_code=503, detail="model loading")
        return {"status": "ready", "model": state["label"]}

    @app.post("/v1/predict", response_model=PredictResponse, dependencies=[Depends(check_key)])
    async def predict(req: PredictRequest, request: Request):
        p = state["predictor"]
        if p is None:
            raise HTTPException(status_code=503, detail="model loading")
        messages = build_messages(req.question, req.contexts, state["ctx"])
        t0 = time.perf_counter()
        async with sem:
            try:
                gen = (await run_in_threadpool(p.generate, [messages]))[0]
            except Exception as e:
                log.exception("generation failed")
                raise HTTPException(status_code=500, detail="generation failed") from e
        if gen.error:
            raise HTTPException(status_code=502, detail=gen.error)
        return PredictResponse(
            answer=parse_label(gen.text),
            rationale=parse_rationale(gen.text),
            raw_output=gen.text,
            latency_ms=(time.perf_counter() - t0) * 1000,
            model=state["label"],
            request_id=request.state.request_id,
        )

    return app
