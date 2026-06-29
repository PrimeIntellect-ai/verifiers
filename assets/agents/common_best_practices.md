## Shared Best Practices (All Contexts)

These points are direct restatements of Verifiers docs so agents can follow the same golden-path workflows.

- V1 packages export taskset and optional harness classes; V0 packages expose `load_environment(...) -> vf.Environment`. Both are installable with `prime env install <env-name>`. (See `docs/overview.md` and `docs/environments.md`.)
- Validate behavior with `prime eval run <taskset-id> ...` for v1 or `prime eval run --id <env-id> ...` for V0 before publishing. Local runs write `config.toml` and `results.jsonl`; use `prime eval push` explicitly when results should appear on the platform. (See `docs/evaluation.md`.)
- Agents should assume they may make live model calls through the user's authenticated Prime CLI when a smoke test is useful. Prime supplies the selected API and inference context to Verifiers. If sandboxing blocks outbound requests, request the required permission for the eval command.
- Start new taskset/harness environments with `prime env init <name>` and add `--add-tool`, `--add-user`, or `--add-harness` only when needed. Edit the generated taskset config, `load_tasks()`, optional `Toolset`/`User` classes, and `@vf.*` lifecycle methods. Keep the generated package exports and README structure. Use `prime env init <name> --v0` only for the legacy API. Treat [BYO Harness](docs/byo-harness.md) as the canonical v1 authoring guide.
- Use `ToolEnv`/`MCPEnv` for stateless tools and `StatefulToolEnv` when per-rollout state must persist (sandbox/session/db handles). (See `docs/environments.md`.)
- Validate required external credentials at the component that owns them. V0 loaders can use `vf.ensure_keys(...)` for an explicit early failure. (See `docs/environments.md`.)

## Style Rules

Use these rules when shaping user-facing Verifiers APIs, configs, and environment files.

- Prefer Verifiers-native interfaces over stdlib-pure plumbing in user code. A stdlib-pure expression that forces every environment to write path manipulation, import-resource handling, ad hoc discovery, or boilerplate constants is a style bug; put that logic behind a Verifiers abstraction instead.
- Keep user-facing APIs incredibly minimal and elegant. The best surface is usually golfy but intuitive: one obvious field, one obvious constructor, and no redundant knobs unless there is a concrete long-term reason.
- Use Pydantic config models wherever structured configuration is needed. Pydantic is always acceptable and preferred over loose dictionaries when it clarifies the contract.
- Prefer strict, narrow types. Use `object`, broad unions, or untyped mappings only at explicit framework boundaries where arbitrary user values are genuinely part of the contract.
- Basic v1 environments should fit in a few dozen self-contained, idiomatic lines: import `verifiers as vf`, define typed taskset/harness config classes when needed, keep policy values in config subclasses, and put implementation logic on the owning taskset or harness class. Static prompts should usually be `system_prompt` config fields; override `load_system_prompt` only for computed prompt loading. Use bindings for shared resources owned by tasksets, toolsets, users, programs, or harnesses; object entries should be loader specs, not pre-initialized resources.
- Avoid module globals. Acceptable globals are imports, immutable literals, factory functions, and carefully managed process-level resource constraints such as locks or semaphores. Put all other behavior and state in well-named utility modules, taskset/harness classes, toolsets, users, programs, or user code.
- Additional code should have a clear home. Do not hide utilities at the bottom of files or scatter one-off helpers through environment entrypoints.
