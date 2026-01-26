FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-install-project --no-dev

COPY src/ ./src/
COPY README.md ./

RUN uv sync --frozen --no-dev

RUN mkdir -p /app/model_registry /app/data

ENV PYTHONPATH=/app

CMD ["uv", "run", "train"]
