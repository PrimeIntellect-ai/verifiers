## Shared Best Practices (All Contexts)

These points are direct restatements of Verifiers docs so agents can follow the same golden-path workflows.

- Environments are expected to expose `load_environment(...) -> vf.Environment` and be installable with `prime env install <env-name>`. (See `docs/overview.md` and `docs/environments.md`.)
- Validate environment behavior with `prime eval run <env-name> ...` before sharing/publishing changes. (See `docs/overview.md` and `docs/development.md`.)
- Use `ToolEnv`/`MCPEnv` for stateless tools and `StatefulToolEnv` when per-rollout state must persist (sandbox/session/db handles). (See `docs/environments.md`.)
- If external API keys are required, validate them in `load_environment()` with `vf.ensure_keys(...)` so failures are explicit and early. (See `docs/environments.md`.)
