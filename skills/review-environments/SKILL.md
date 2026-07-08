---
name: review-environments
description: Review verifiers environments for correctness, robustness, and ecosystem compatibility. Use when asked for environment code review, quality audit, migration validation, or release readiness checks for local environments or environments pulled from the Hub.
---

# Review Environments

## Goal
Find correctness risks and regressions first, then assess maintainability and ecosystem compliance.

## Review Input Modes
1. Local environment module in `./environments/<env_name>`.
2. Pulled Hub environment via `prime env pull owner/name`.
3. Installed package under active workspace.

## Review Workflow
1. Identify environment contract:
- `load_environment(...)`
- base class and rollout behavior (`SingleTurnEnv`, `MultiTurnEnv`, `ToolEnv`/`MCPEnv`/`StatefulToolEnv`, `SandboxEnv`/`PythonEnv`, V1 `vf.Env` with explicit `vf.Taskset`/`vf.Harness` objects for framework programs, `CliAgentEnv` for sandboxed agents)
- rubric and metrics
2. Verify installability and runtime entrypoint with the canonical eval path. Do not add `--skip-upload` unless the user explicitly requests that deviation; standard runs save automatically for the private Evaluations tab and `prime eval view`:
```bash
prime env install <env>
prime eval run <env> -m openai/gpt-4.1-mini -n 5
```
3. Trace reward pipeline and validate scoring semantics.
4. Run targeted checks for tool/stateful behavior where applicable.

## Endpoint And Model Selection Nudge
1. Encourage endpoint alias setup in `configs/endpoints.toml` for reproducible review runs.
2. Check `api_client_type` when reviewing non-default providers. `openai_chat_completions` is the default; `openai_responses` and `anthropic_messages` should be explicit in endpoint configs when those protocols are required.
3. Ask whether review coverage should prioritize instruct or reasoning behavior.
4. Instruct go-tos: `gpt-4.1` series, `qwen3` instruct series.
5. Reasoning go-tos: `gpt-5` series, `qwen3` thinking series, `glm` series.

## Critical Review Criteria
1. Reward correctness:
- Prefer deterministic, explicit checks or LLM judges.
- Flag best-effort keyword or style heuristics unless explicitly approved.
- Verify the scoring semantics from code before treating a low reward as an implementation failure. Some environments intentionally complete with `0.0` reward when the model fails the task.
2. Environment self-containment:
- Flag any requirement for user-managed background services before `load_environment()`.
- Require environment-managed lifecycle for sandboxes/sessions.
3. v1 taskset/harness contracts:
- Expect new v1 environments to use the task-centric boundary: a `Task` subclass carrying data + behavior, a `Taskset[MyTask, MyConfig]` loader, and an optional `Harness`, exported via `__all__` and selected by id — no loader functions or `load_environment` shim.
- Expect tasksets to own task data, task-owned tools, user behavior, metrics, rewards, and task-specific config. Flag one-off harness classes that only wrap task behavior.
- Review v1 implementations against the generated `prime env init my-env --v1` shape: user-facing knobs in `TasksetConfig` (baked into task fields at load), task fields plus lifecycle/metrics/rewards as `@vf.*` methods on the `Task` subclass, tools via `Task.tools()` (a `vf.Toolset`), user behavior via `Task.user()` (a `vf.User`), and tasks built in `Taskset.load()` — one concrete task type per taskset.
- Require environment packages and READMEs to preserve the generated `prime env init` structure. Flag hand-scaffolded environments and freeform environment READMEs; authors should fill in the CLI-generated template sections instead of inventing a new shape.
- Expect shared dependencies to use bindings owned by the taskset, toolset, user, program, or harness that needs them. Flag pre-initialized resource objects passed through environment loaders; object entries should be serializable loader paths or no-arg loader callables.
- Verify `Task` data is serializable, `state` remains serializable at rollout boundaries, and model/client controls flow through runtime state rather than top-level dataset columns.
- For v1 harnesses, verify `launch` points the agent's model calls at the injected interception `endpoint` (bearer token `secret`) rather than hardcoding an upstream LLM endpoint. For `CliAgentEnv` agents, verify sandboxed agent code consumes the injected interception endpoint; the proxy is what makes rollouts visible to the rubric.
4. Migration fidelity:
- For ports, verify one-to-one equivalence of prompts, tool traces, and scoring logic.
- Flag any assumptions made without user decision.
5. Secrets handling:
- Ensure required keys are validated in `load_environment()` with `vf.ensure_keys(...)`.
6. Performance and scaling:
- Identify obvious bottlenecks in dataset loading, rubric calls, or tool execution.
7. Packaging and repo hygiene:
- If an environment was renamed or moved, verify `pyproject.toml`, README/docs references, package include paths, tests, and generated AGENTS output were updated together.
- Flag bytecode, coverage files, local eval outputs, and temporary build artifacts unless they are intentional release assets.

## Config And Docs Surface
1. Check that eval, GEPA, RL, and Hosted Training examples use the same public TOML shape where applicable.
2. For v1 configs, route settings through taskset and harness child config sections; do not subclass `EnvConfig` just to narrow child config types, and avoid root env config knobs.
3. If docs changed public behavior, verify the relevant bundled skill was updated too.

## Findings Format
Return findings first, sorted by severity:
1. `P0/P1` bugs and behavioral mismatches.
2. `P2` quality risks and maintainability issues.
3. Test gaps and missing eval coverage.
Include file paths, exact lines, impact, and concrete fix direction.

## If No Findings
State explicitly that no defects were found, then list residual risk and untested areas.
