FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    stockfish git curl ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy HF_HUB_ENABLE_HF_TRANSFER=0
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen
COPY scripts ./scripts
COPY tests ./tests

ENV PATH="/app/.venv/bin:$PATH" STOCKFISH_PATH=/usr/games/stockfish
CMD ["python", "scripts/pretrain.py", "--help"]
