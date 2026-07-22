# AGENTS.md

## Writing code

- **Code is the source of truth**: before writing anything, read the v1 code for existing helpers, harnesses, and interfaces instead of reinventing them. Prefer verifiers-native task, trace, server, harness, and runtime interfaces over repeated path/import/discovery plumbing in user packages.
- **Minimal config surface**: expose as few knobs as possible, but as many as needed.
- **Keep tasksets small**: a basic taskset fits in a few dozen idiomatic lines — typed data/task/config classes, `load()`, and decorated scoring on the task. Don't override `Taskset.__init__` (implement `load()`); don't override `Harness.__init__` or `User.__init__` (use `setup()`).

## Running code

- **Always use uv**: run code and commands with `uv run`, never raw `python`. Make sure `uv` is installed ([docs](https://docs.astral.sh/uv/getting-started/installation/)).
- **Scaffold environments**: create a new taskset/environment with `uv run init <name>` (`uv run init -h` lists options like `-T`/`-U`/`-H`), and run evals with `uv run eval <taskset>`.
- **Validate TOML first**: validate config `.toml` files before running them.
- **Don't add dependencies**: never add dependencies or optional extras to the top-level `pyproject.toml`.

## Docs

- **Kept intentionally minimal**: `docs/`, `skills/`, and `configs/` are deliberately sparse. Don't touch them unless your change breaks their assumptions.
- **Docs reflect `main`, not history**: describe the current state of the codebase only — no removed/legacy fields, migration paths, or "this used to be X" anecdotes.

## Skills

- **Use bundled skills first**: the skills in `skills/` cover the core workflows — `create-environments` (build or migrate a v1 taskset/environment/harness), `evaluate-environments` (configure and run evals), and `brainstorm` (ideation and research planning). Reach for them before doing the work by hand.

## Testing

- **Prefer e2e tests over unit tests**: v1's end-to-end tests are sufficient — extra unit tests clog the repo. Editing existing tests is fine; to check your own work, write a temporary script instead of committing new tests.
- **Run the contributor checks**: run `uv run pre-commit install` once, then for touched areas `uv run ruff check --fix .`, `uv run pytest tests/`, and `uv run pre-commit run --all-files` (see `docs/v0/development.md`).
