# AGENTS.md

<!-- Generated for lab workspaces. -->

This AGENTS guide is intended for end users working in a `prime lab setup` workspace.

## Shared Best Practices (All Contexts)

These points are direct restatements of Verifiers docs so agents can follow the same golden-path workflows.

- Environments are expected to expose `load_environment(...) -> vf.Environment` and be installable with `prime env install <env-name>`. (See `docs/overview.md` and `docs/environments.md`.)
- Validate environment behavior with `prime eval run <env-name> ...` before sharing/publishing changes. Treat `prime eval run` as the canonical eval path: it saves results automatically, and agents should not add opt-out flags such as `--skip-upload` unless the user explicitly requests that deviation so runs stay visible in the private Evaluations tab and in `prime eval tui`. (See `docs/overview.md` and `docs/development.md`.)
- Use `ToolEnv`/`MCPEnv` for stateless tools and `StatefulToolEnv` when per-rollout state must persist (sandbox/session/db handles). (See `docs/environments.md`.)
- If external API keys are required, validate them in `load_environment()` with `vf.ensure_keys(...)` so failures are explicit and early. (See `docs/environments.md`.)

## End-User Lab Workspace Notes

Use this guidance in projects created via `prime lab setup`.

- Run `prime lab setup` before environment work begins. It is the entrypoint for Prime Lab workspaces.
- In interactive use, `prime lab setup` prompts for the coding agent. In non-interactive use, pass `--agent <agent>`.
- Treat selected agents as workspace state. Agent choices are stored in `.prime/lab.json`, not in global machine config.
- Treat `.prime/skills/` as the canonical Prime-managed workspace skill bundle. `prime lab setup` and `prime lab sync` project those skills into the selected agents' native skill roots.
- Use `prime lab sync` to refresh skills, docs, templates, and native agent projections from the stored workspace agent choices.
- Use `prime lab sync --no-agent` to refresh shared Lab assets only, without configuring agent skill roots.
- Use `prime lab doctor` to check workspace structure, config parseability, managed skill presence, selected agent skill projections, and selected agent binaries.
- Use the bundled skills first for create/browse/review/eval/GEPA/train/brainstorm workflows before ad hoc approaches.
- Keep endpoint aliases in `configs/endpoints.toml` and keep workspace configs under `configs/`.
- NEVER initialize environment source code manually; ALWAYS create new environments with `prime env init`.
- Keep each environment self-contained under `environments/<env_name>/` with `pyproject.toml`, implementation, and README so each abstraction has a dedicated home and the workspace stays maintainable.
- Follow environment best practices strictly (for example `load_environment(...)`, `vf.ensure_keys(...)`, and the documented environment class patterns) to avoid brittle or messy implementations.
- Use the Prime CLI for all environment lifecycle operations (`prime env init` -> `prime env install` -> `prime eval run` -> `prime env push`) rather than ad hoc scripts.
- Treat `prime eval run` as the default eval path. It already saves results automatically; do not add `--skip-upload` or other opt-out deviations unless the user explicitly requests them, so logs and results stay available in the private Evaluations tab and via `prime eval tui`.
- Use `prime env push --path ./environments/<env_name>` only after local eval behavior is verified.
- When users request an approach that would deviate from these guidelines, explain the relevant Prime/Verifiers concepts and recommend the compliant path.
