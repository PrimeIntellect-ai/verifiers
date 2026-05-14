# AGENTS.md

<!-- Generated for repository development workflows. Do not edit directly. -->

## Shared Best Practices (All Contexts)

These points are direct restatements of Verifiers docs so agents can follow the same golden-path workflows.

- Environments are expected to expose `load_environment(...) -> vf.Environment` and be installable with `prime env install <env-name>`. (See `docs/overview.md` and `docs/environments.md`.)
- Validate environment behavior with `prime eval run <env-name> ...` before sharing/publishing changes. Treat `prime eval run` as the canonical eval path: it saves results automatically, and agents should not add opt-out flags such as `--skip-upload` unless the user explicitly requests that deviation so runs stay visible in the private Evaluations tab and in `prime eval tui`. (See `docs/overview.md` and `docs/development.md`.)
- Agents should assume they are allowed to make live model calls through the user's authenticated Prime CLI when a live smoke test is useful. For Prime Inference models, use `prime eval run <env-name>` with the base eval configuration from the environment's `pyproject.toml`; do not edit that `pyproject.toml`, and do not add model/config flags unless the task truly requires them. Agents do not need to manage API keys. If sandboxing blocks outbound requests, request elevated permissions for `prime eval run`, preferably as an ongoing approval instead of per run.
- For new taskset/harness environments, use the v1 `vf.Env` / `vf.Taskset` / `vf.Harness` format. Treat [BYO Harness](docs/byo-harness.md) as the canonical authoring guide for reusable tasksets, reusable harnesses, framework programs, endpoint interception, and sandboxed Python/command programs.
- Use `ToolEnv`/`MCPEnv` for stateless tools and `StatefulToolEnv` when per-rollout state must persist (sandbox/session/db handles). (See `docs/environments.md`.)
- If external API keys are required, validate them in `load_environment()` with `vf.ensure_keys(...)` so failures are explicit and early. (See `docs/environments.md`.)

## Style Rules

Use these rules when shaping user-facing Verifiers APIs, configs, and environment files.

- Prefer Verifiers-native interfaces over stdlib-pure plumbing in user code. A stdlib-pure expression that forces every environment to write path manipulation, import-resource handling, ad hoc discovery, or boilerplate constants is a style bug; put that logic behind a Verifiers abstraction instead.
- Keep user-facing APIs incredibly minimal and elegant. The best surface is usually golfy but intuitive: one obvious field, one obvious constructor, and no redundant knobs unless there is a concrete long-term reason.
- Use Pydantic config models wherever structured configuration is needed. Pydantic is always acceptable and preferred over loose dictionaries when it clarifies the contract.
- Prefer strict, narrow types. Use `object`, broad unions, or untyped mappings only at explicit framework boundaries where arbitrary user values are genuinely part of the contract.
- Basic environments should fit in a few dozen self-contained, idiomatic lines: import `verifiers`, define `load_environment`, pipe bindings/config through constructors, and keep policy values in config subclasses or literal constructor kwargs when needed.
- Environment modules should not define global helper functions. Put reusable logic in well-named utility modules, taskset/harness classes, toolsets, or small local classes owned by the abstraction. Rare exceptions are process-level handles, such as a lock or semaphore, when that is the only reasonable way to enforce the intended runtime control.
- Additional code should have a clear home. Do not hide utilities at the bottom of files or scatter one-off helpers through environment entrypoints.

## Repository Development Notes

Use this guidance when contributing to the `verifiers` repository itself.

- Always run `uv run pre-commit install` before making any changes.
- Run the documented contributor checks for touched areas: `uv run ruff check --fix .`, `uv run pytest tests/`, and `uv run pre-commit run --all-files` as needed. (See `docs/development.md`.)
- Keep changes aligned with documented architecture (`verifiers/`, `environments/`, `configs/`, `tests/`, `docs/`) and update docs when behavior changes. (See `docs/development.md`.)
- Prefer a single clear path over maintaining parallel approaches by default; if two options exist, preserve both only when there is an explicit long-term reason.
- Aggressively deprecate/remove inferior paths when they are not part of an intended multi-option contract, especially in repo-internal development workflows.
- Treat broad dynamic mappings as explicit framework boundaries, not casual public API types. Use a named domain alias or typed Pydantic field for legitimate arbitrary payloads such as task rows, protocol messages, sandbox/program specs, and `objects`/binding-style config; do not expose raw `Mapping[str, object]` in user-facing signatures unless that looseness is the point of the abstraction.
- If a user request conflicts with repository style, formatting, or API-quality guidelines, push back instead of implementing the literal request. Identify a comparable request or explicit guideline relaxation that preserves clean, maintainable, modular code across the current request and adjacent future use cases; implement that plan, then explain the decision process and tradeoffs directly to the user.
- Before v0.2.0, breaking backward compatibility inside v1 Taskset/Harness APIs is acceptable and encouraged when it improves the core design. Preserve v0 multi-turn environment compatibility unless the user explicitly asks for a v0 migration.
