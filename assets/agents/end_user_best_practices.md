## End-User Lab Workspace Notes

Use this guidance in projects created via `prime lab setup`.

- Treat `.prime/skills/` as the canonical skill entrypoint in Lab workspaces. Use the bundled skills first for create/browse/review/eval/GEPA/train/brainstorm workflows before ad hoc approaches.
- Keep endpoint aliases in `./configs/endpoints.toml` and use `endpoint_id`/model shortcuts in commands and configs.
- Use the documented workspace flow: `prime env init` → `prime env install` → `prime eval run`.
- Keep each environment self-contained under `environments/<env_name>/` with `pyproject.toml`, implementation, and README.
- Document required environment variables in README and validate missing keys early with `vf.ensure_keys(...)`.
- Use `prime env push --path ./environments/<env_name>` only after local eval behavior is verified.
