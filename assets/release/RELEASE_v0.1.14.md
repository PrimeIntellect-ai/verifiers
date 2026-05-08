# Verifiers v0.1.14 Release Notes

*Date:* 05/07/2026

## Highlights since v0.1.13.dev8

- **Composable v1 Taskset/Harness API.** Adds the `verifiers.v1` authoring surface around serializable `Task`/`State` data, composable `Taskset` and `Harness` objects, and the `vf.Env(taskset, harness)` adapter for existing eval and training workers. The release includes lifecycle decorators, typed config objects, endpoint routing, toolsets, MCP tools, sandbox/program utilities, nested harness support, v1 docs, migration notes, and several v1 example environments, including new OpenAI Agents, LangChain Deep Agents, and DSPy RLM harness examples.
- **Consistent v1 environment configuration.** Eval and RL/Hosted Training TOML now share the same public projection shape through `[*.args]`, `[*.taskset]`, and `[*.harness]` sections. v1 loaders accept both mapping and model-backed config objects through a common access helper, and strict child config parsing strips loader-local routing keys at the boundary.
- **Model-family starter configs.** Restructures bundled eval, RL, and GEPA starter configs around model families such as Qwen 3.5, Qwen 3.5 MoE, Nemotron 3, and Llama 3, with setup mirroring the new config set into Lab workspaces.
- **New client and rendering paths.** Adds an OpenAI Responses API client and a renderer-backed client path for exact token rendering and multi-turn bridge metrics. The renderer implementations now live in the external `renderers` package, exposed from Verifiers through an optional `renderers` extra and client integration. Renderer clients also forward `preserve_all_thinking` and `preserve_thinking_between_tool_calls` flags into the underlying renderer.
- **More rollout observability and artifacts.** Adds per-turn timing through eval outputs and TUI display, token-id preservation for Nemotron client responses, GEPA system-prompt artifact export plus path-based prompt loading, and Lean guard markers with tamper-aware `LeanRubric` scoring.
- **Release and infrastructure hardening.** Adds universal locks and a 7-day PyPI freshness cooldown, scopes Hub install freshness filtering to registry packages, skips secret-backed environment tests on fork PRs, points the composable RLM harness at `rlm-harness`, and routes opencode `AGENT_WORKDIR` per rollout.

## Changes included in v0.1.14 (since v0.1.13.dev8)

### Features and enhancements

- Taskset Harness (v1) (#1277)
- ApiEnv examples for OpenAI Agents, LangChain Deep Agents, and DSPy RLM (#1121)
- Refactor tau2 bench into a taskset-owned v1 environment (#1293)
- Restructure example configs around model families (#1297)
- add openai responses client (#1261)
- Renderer-backed client integration via the external `renderers` package (#1068, #1279, #1282)
- feat(renderer-client): forward preserve_*_thinking config flags (#1298)
- feat: per-turn timing (#1182)
- Add GEPA system prompt export and path-based prompt loading (#1268)
- feat(lean): lean-guard markers + tamper-aware LeanRubric (#1271)
- token id support for Nemotron client responses (#1231)

### Fixes and maintenance

- Fix v1 env config projection and typed child loader boundaries (#1294)
- Skip secret-backed environment tests for fork PRs (#1292)
- Scope uv freshness filtering to PyPI for Hub installs (#1286)
- opencode harness: route AGENT_WORKDIR per-rollout instead of baked-in (#1280)
- chore: add 7-day supply chain cooldown via uv exclude-newer (#1274)
- chore: point DEFAULT_RLM_REPO_URL to rlm-harness (#1267)
- Update Lab workspace setup guidance (#1299)

**Full Changelog**: https://github.com/PrimeIntellect-ai/verifiers/compare/v0.1.13.dev8...v0.1.14
