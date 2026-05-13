## Repository Development Notes

Use this guidance when contributing to the `verifiers` repository itself.

- Always run `uv run pre-commit install` before making any changes.
- Run the documented contributor checks for touched areas: `uv run ruff check --fix .`, `uv run pytest tests/`, and `uv run pre-commit run --all-files` as needed. (See `docs/development.md`.)
- Keep changes aligned with documented architecture (`verifiers/`, `environments/`, `configs/`, `tests/`, `docs/`) and update docs when behavior changes. (See `docs/development.md`.)
- Prefer a single clear path over maintaining parallel approaches by default; if two options exist, preserve both only when there is an explicit long-term reason.
- Aggressively deprecate/remove inferior paths when they are not part of an intended multi-option contract, especially in repo-internal development workflows.
- Treat public configuration and docs as part of the API. Keep TOML shapes consistent across eval, GEPA, RL, and Hosted Training; normalize legacy inputs at the ingestion boundary instead of spreading compatibility branches through examples.
- For v1 Taskset/Harness work, make the taskset own task data, task tools, user behavior, metrics, rewards, and task-specific configuration. Use the base `vf.Harness` unless the harness really owns a reusable execution mechanism.
- When renaming or deleting an environment/module path, update package metadata, README/docs references, tests, build includes, and generated AGENTS output in the same change.
- For environment changes, validate the install/load/eval path, not just imports. Prefer `prime eval run` for user-visible behavior and `tests/test_envs.py` for package-install coverage when the change affects packaged examples.
- When fixing a PR review, Bugbot issue, CI failure, or release blocker, inspect the live thread/check/log first and address the exact failure. Do not infer the root cause from stale local context.
- Before changing dependencies, optional extras, lockfiles, or config fields consumed by `prime-cli`, `prime-rl`, Hosted Training, or public docs, trace the downstream consumer and update the matching docs/skills in the same patch.
- Keep generated artifacts out of commits. Remove bytecode, coverage files, local eval outputs, and temporary build products unless they are explicitly part of the release artifact.
