# AGENTS.md

<!-- Generated for repository development workflows. Do not edit directly. -->

## Shared Best Practices (All Contexts)

These points are direct restatements of Verifiers docs so agents can follow the same golden-path workflows.

- Environments are expected to expose `load_environment(...) -> vf.Environment` and be installable with `prime env install <env-name>`. (See `docs/overview.md` and `docs/environments.md`.)
- Validate environment behavior with `prime eval run <env-name> ...` before sharing/publishing changes. (See `docs/overview.md` and `docs/development.md`.)
- Use `ToolEnv`/`MCPEnv` for stateless tools and `StatefulToolEnv` when per-rollout state must persist (sandbox/session/db handles). (See `docs/environments.md`.)
- If external API keys are required, validate them in `load_environment()` with `vf.ensure_keys(...)` so failures are explicit and early. (See `docs/environments.md`.)

## Repository Development Notes

Use this guidance when contributing to the `verifiers` repository itself.

- Always run `uv run pre-commit install` before making any changes.
- Run the documented contributor checks for touched areas: `uv run ruff check --fix .`, `uv run pytest tests/`, and `uv run pre-commit run --all-files` as needed. (See `docs/development.md`.)
- Keep changes aligned with documented architecture (`verifiers/`, `environments/`, `configs/`, `tests/`, `docs/`) and update docs when behavior changes. (See `docs/development.md`.)
- Prefer a single clear path over maintaining parallel approaches by default; if two options exist, preserve both only when there is an explicit long-term reason.
- Aggressively deprecate/remove inferior paths when they are not part of an intended multi-option contract, especially in repo-internal development workflows.
