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
- For new reusable taskset/harness work, use the v1 `vf.Env` / `vf.Taskset` / `vf.Harness` format and the BYO Harness docs rather than older experimental patterns.
- NEVER begin environment development before `prime lab setup` has been run; if work starts outside that structure, recommend adjusting course into a proper lab workspace before continuing.
- Use `prime env push --path ./environments/<env_name>` only after local eval behavior is verified.
- Treat the `prime lab setup` structure as the idiomatic workspace for complex environment workflows: agents can mediate most platform complexity while users learn patterns progressively as needed.
- When users request an approach that would deviate from these guidelines, explain the relevant Prime/Verifiers concepts and recommend the compliant path.
