# GPU image for training, benchmarking and serving.
#   docker build -t domain-llm .
#   docker run --gpus all -p 8000:8000 -v $PWD/outputs:/app/outputs domain-llm
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[all]"

COPY configs ./configs

RUN useradd --create-home --uid 1000 app && mkdir -p /app/outputs /app/data /app/.cache \
    && chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/ready').status==200 else 1)"

ENV DOMAIN_LLM_CONFIG=configs/qlora.yaml
CMD ["domain-llm", "serve", "--host", "0.0.0.0", "--port", "8000"]
