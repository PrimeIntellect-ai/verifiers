# Development

Source: `docs/development.md`.

Repository contributors should install hooks with `uv run pre-commit install`
before changing code. Common checks include `uv run ruff check --fix .`,
`uv run pytest tests/`, and `uv run pre-commit run --all-files`.

Environment changes should stay aligned with the documented repository layout:
`verifiers/`, `environments/`, `configs/`, `tests/`, and `docs/`.
