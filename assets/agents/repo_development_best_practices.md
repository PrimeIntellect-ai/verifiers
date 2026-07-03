## Repository Development Notes

Use this guidance when contributing to the `verifiers` repository itself.

- Ensure that `uv` is installed, see [uv docs](https://docs.astral.sh/uv/getting-started/installation/) for further information.
- Always run `uv run pre-commit install` before making any changes.
- Run the documented contributor checks for touched areas: `uv run ruff check --fix .`, `uv run pytest tests/`, and `uv run pre-commit run --all-files` as needed. (See `docs/development.md`.)
- The documentation (in `docs/`, `skills/`, `configs/`) is intentionally kept minimal. Do not touch them unless your changes break with these assumptions.
- Verifiers has two API surfaces under `verifiers/`: `verifiers/v1`, which is referred to as "v1", while the rest is "legacy". Unless specificially requested, always use and assume v1.
- Verifiers v1 uses Pydantic objects everywhere, so you should, too, when working in v1. Mimic the style of the existing code.
- Do not add unit tests to your PRs. v1 has a few end to end tests, which are sufficient. Unit tests clog up the repository. You can, however, write temporary scripts to test.
- The repository uses Python 3.11 or older, so you are encouraged to use modern Python function to cut down on lines.
- Do not add dependencies, optional extras etc. to the main pyproject.toml.
