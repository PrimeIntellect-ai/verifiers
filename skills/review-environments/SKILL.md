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
- base class and rollout behavior (`SingleTurnEnv`, `MultiTurnEnv`, `ToolEnv`/`MCPEnv`/`StatefulToolEnv`, `SandboxEnv`/`PythonEnv`, `ApiEnv` for local Python `agent_fn` with LLM-API interception, `CliAgentEnv`/`ComposableEnv` for sandboxed agents — note that `CliAgentEnv` inherits from `ApiEnv` and shares its interception/proxy lifecycle)
- rubric and metrics
2. Verify installability and runtime entrypoint with the canonical eval path. Do not add `--skip-upload` unless the user explicitly requests that deviation; standard runs save automatically for the private Evaluations tab and `prime eval tui`:
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
2. Environment self-containment:
- Flag any requirement for user-managed background services before `load_environment()`.
- Require environment-managed lifecycle for sandboxes/sessions.
3. v1 taskset/harness contracts:
- Verify `Task` data is serializable, `state` remains serializable at rollout boundaries, and model/client controls flow through runtime state rather than top-level dataset columns.
- For V1 harness programs, verify framework clients consume `state.get_endpoint_config(api="chat")` rather than hardcoding an upstream LLM endpoint. For `CliAgentEnv` agents, verify sandboxed agent code consumes the injected interception endpoint; the proxy is what makes rollouts visible to the rubric.
4. Migration fidelity:
- For ports, verify one-to-one equivalence of prompts, tool traces, and scoring logic.
- Flag any assumptions made without user decision.
5. Secrets handling:
- Ensure required keys are validated in `load_environment()` with `vf.ensure_keys(...)`.
6. Performance and scaling:
- Identify obvious bottlenecks in dataset loading, rubric calls, or tool execution.

## Findings Format
Return findings first, sorted by severity:
1. `P0/P1` bugs and behavioral mismatches.
2. `P2` quality risks and maintainability issues.
3. Test gaps and missing eval coverage.
Include file paths, exact lines, impact, and concrete fix direction.

## If No Findings
State explicitly that no defects were found, then list residual risk and untested areas.
