# AGENTS.md

## Writing code

- **Code is the source of truth**: before writing anything, read the v1 code for existing helpers, harnesses, and interfaces instead of reinventing them. Prefer verifiers-native task, trace, server, harness, and runtime interfaces over repeated path/import/discovery plumbing in user packages.
- **Minimal config surface**: expose as few knobs as possible, but as many as needed.
- **Keep tasksets small**: a basic taskset fits in a few dozen idiomatic lines — typed data/task/config classes, `load()`, and decorated scoring on the task. Don't override `Taskset.__init__` (implement `load()`); don't override `Harness.__init__` (use `setup()`).

## Running code

- **Always use uv**: run code and commands with `uv run`, never raw `python`. Make sure `uv` is installed ([docs](https://docs.astral.sh/uv/getting-started/installation/)).
- **Run evals**: run a taskset with `uv run vf-eval <taskset>`.
- **Validate TOML first**: validate config `.toml` files before running them.
- **Don't add dependencies**: never add dependencies or optional extras to the top-level `pyproject.toml`.

## Docs

- **Kept intentionally minimal**: `docs/` and `skills/` are deliberately sparse. Don't touch them unless your change breaks their assumptions.
- **Docs reflect `main`, not history**: describe the current state of the codebase only — no removed/legacy fields, migration paths, or "this used to be X" anecdotes.

## Skills

- **Use bundled skills first**: the `release` skill in `skills/` documents the stable-version release workflow, and the `bump` skill documents how to bump a pinned harness version.

## Testing

- **Two suites**: `tests/unit` runs one component in isolation; `tests/integration` runs real evals (a live model, real runtimes) per behavior across the harness x runtime grid in `tests/integration/grid.py`. Prefer an integration cell over a unit test; a unit test earns its place by stating one component's intended behavior, not by asserting an implementation detail. To check your own work, write a temporary script instead of committing new tests.
- **Run the contributor checks**: run `uv run pre-commit install` once, then for touched areas `uv run ruff check --fix .`, `uv run pytest tests/unit`, and `uv run pre-commit run --all-files`.
