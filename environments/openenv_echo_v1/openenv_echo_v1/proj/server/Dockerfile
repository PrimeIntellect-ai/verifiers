FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

ENV UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app/env

COPY pyproject.toml .
RUN uv sync --no-dev --no-install-project

COPY . .
