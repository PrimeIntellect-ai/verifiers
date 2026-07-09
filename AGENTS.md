# AGENTS.md

<!-- Generated for repository development workflows. Do not edit directly. -->

## Shared Best Practices (All Contexts)

- Create new environments with `prime env init <MY_ENV_NAME>`.
- Use `prime <command> --plain` to get better formatted command outputs.
- Use the bundled skills first for create, browse, review, eval, optimization, training, and brainstorming workflows.
- Go through the verifiers code as the source of truth.
- Use toml files and validate them first before you run them.
- Always look at the verifiers v1 code first to see whether helper functions or harnesses exist for your given task.
- Never mix v1 and legacy verifiers functions in one environment. Always prefer and use v1.* code.

## Style Rules

Use these rules when shaping public v1 APIs, configs, and environment files.

- Prefer verifiers-native task, trace, server, harness, and runtime interfaces over repeated path/import/discovery plumbing in user packages.
- Expose as few knobs in the configs as possible, but as many as needed.
- Use strict Pydantic models for structured config, tasks, messages, and state.
- A basic taskset should fit in a few dozen idiomatic lines: typed task/config classes, `load_tasks()`, and decorated scoring.
- Do not override `Taskset.__init__`, `Harness.__init__`, or `User.__init__`. Use `setup()` instead.
- Refer to the code as the source of truth.
- Avoid mutable module globals. Process-level locks/rate limiters and immutable constants are the narrow exceptions.

## Repository Development Notes

Use this guidance when contributing to the `verifiers` repository itself.

- Ensure that `uv` is installed, see [uv docs](https://docs.astral.sh/uv/getting-started/installation/) for further information.
- Always run `uv run pre-commit install` before making any changes.
- Run the documented contributor checks for touched areas: `uv run ruff check --fix .`, `uv run pytest tests/`, and `uv run pre-commit run --all-files` as needed. (See `docs/v0/development.md`.)
- The documentation (in `docs/`, `skills/`, `configs/`) is intentionally kept minimal. Do not touch them unless your changes break with these assumptions.
- verifiers has two API surfaces under `verifiers/`: `verifiers/v1`, which is referred to as "v1", while the rest is "legacy". Unless specifically requested, always use and assume v1.
- verifiers v1 uses Pydantic objects everywhere, so you should, too, when working in v1. Mimic the style of the existing code.
- Do not add unit tests to your PRs. v1 has a few end to end tests, which are sufficient. Unit tests clog up the repository. You can, however, write temporary scripts to test.
- The repository uses Python 3.11 or older, so you are encouraged to use modern Python functions to cut down on lines.
- Do not add dependencies, optional extras etc. to the main pyproject.toml.
