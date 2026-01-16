# Verifiers v0.1.9 Release Notes

*Date:* 01/08/2026

Verifiers v0.1.9 introduces several new experimental environments, monitor rubrics for automatic metrics collection, improved error handling, and documentation overhaul.

**Post-release update:**
- Tweaks to setup script (post1).
- Fix for exporting setup script (post0).
- Fix for gitignore section in setup script (post2).

## Highlights

- **RLMEnv (Experimental)**: New environment implementing the Recursive Language Model (RLM) inference strategy, where language models decompose and recursively interact with input data through sandboxed REPL environments. Supports sub-LLM calls via `llm_batch()` function intercepted through HTTP proxy. See [RLM paper](https://www.alexzhang.dev/blog/recursive-language-models).

- **GymEnv (Experimental)**: Universal Gym-compatible environment runner for standard RL gymnasium environments. Enables training on classic control tasks and custom Gym environments.

- **CliAgentEnv & HarborEnv (Experimental)**: New environments for running custom agent code in sandboxes (`CliAgentEnv`) and loading Harbor-format tasks (`HarborEnv`).

- **MCPEnv (Experimental)**: Environment for Model Context Protocol (MCP) server integration.

- **Monitor Rubrics**: Each environment now automatically includes a monitor rubric that tracks environment-specific metrics without affecting rewards (weight=0). For example:
  - `MultiTurnEnv`: `num_turns`
  - `ToolEnv`: `tool_call_count`
  - `SandboxEnv`: `sandbox_call_count`, `sandbox_total_time_seconds`, `sandbox_mean_time_seconds`
  - `PythonEnv`: `repl_call_count`, `repl_total_time_seconds`, `repl_mean_time_seconds`
  - `RLMEnv`: Sub-LLM metrics and more

- **Improved Error Handling**: New error chain helpers, better error propagation through rollouts, and `abort_on_code_timeout` support for sandbox environments.

- **Documentation Overhaul**: Complete reorganization of documentation with new Mintlify-based docs, improved examples, and automatic docs sync workflow.

## New Features

### Environments
- Add `get_sandbox_request` hook for per-rollout sandbox customization (#699)
- Expose `render_completion` and `add_trajectory_step` methods with private/final guardrails (#679)
- Add `final_messages` pattern for cleaner message handling (#677)
- Support for token-in vLLM endpoint (#626)
- Static `make_dataset` function for environments (#683)
- Add `alphabet-sort` example environment (#695)
- `system_prompt` is now prepended to existing prompts that don't already start with a system message

### Evaluation & Training
- Optionally enable independent per-rollout scoring: run and score rollouts independently rather than only in groups (#694)
- `vf-tui` improvements: regex search modal and run details panel (#705)
- Log eventloop lag during `vf-eval` (#687)
- Log timings in `vf-eval` (#686)
- Show rolling average as tqdm postfix (#693)
- Option to bypass scoring for faster iteration (#645)
- Add `trajectory_id` to TrajectoryStep (#675)

### Rubrics
- Add RLM monitor rubric for sub-LLM metrics (#698)
- Improvements to math rubric with better timeout handling (#657)
- JudgeRubric now accepts optional `state` argument (#684)

### Error Handling
- Helpers for error chains (#649)
- Better error handling with `abort_on_code_timeout` (#659)
- Handle all truncation cases (#637)
- Raise `ModelError` when `response.choices` is `None` (#640)
- Apply `stop_errors` pattern to StatefulToolEnv for parse/call errors (#618)
- Normalize messages from sub-LLM calls to prevent errors (#664)

## Bug Fixes

- Fix tool duplication when calling `add_tool` on `ToolEnv` with shared list reference
- Fix `args_to_skip` validation failure for dict type parameters in `StatefulToolEnv` (#674)
- Fix empty slice handling (#701)
- Fix wiki-search environment (#697)
- Fix tool test environment (#692)
- Fix PythonEnv deadlock (#652)
- Fix auto-format dataset for `message_type=completions` (#624)
- Fix math verify timeout (#620)
- Fix sub-LLM metrics and context warnings
- `pip_install_packages=""` no longer breaks sandbox (#633)
- Remove prompt logprobs to reduce memory usage (#666)
- Warn when ignoring system prompt/few-shot with `prompt` present (#668)
- Handle empty completions in `parse_answer` (#672)

## Infrastructure & Documentation

- Ensure integrations can be installed via full path (#704)
- Reorganize third-party env integrations (TextArena, ReasoningGym, etc.) (#682)
- Experimental folder structure for newer environments (#643)
- Overhaul docs with example configs (#700)
- Update docs for v0.1.8 API (#670)
- Add automatic docs sync workflow (#628)
- Redirect RTD to shared Mintlify docs (#654)
- Dynamic logger names (#639)
- Use threading for sandbox client (#638)
- Bump `prime-sandboxes>=2.7.0` (#660)

### `vf-setup` Command

The `vf-setup` command bootstraps a verifiers training workspace:

**Default behavior** (no flags):
- Creates `configs/` and `environments/` directories
- Downloads `AGENTS.md`, `CLAUDE.md`, and `environments/AGENTS.md` for AI coding assistants
- Downloads `configs/endpoints.py` (API endpoint configuration)
- Downloads lab configs for quick experimentation (`configs/lab/*.toml`)

**With `--prime-rl`**:
- Installs [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) and syncs dependencies
- Installs all environments from `environments/` into the prime-rl workspace
- Downloads prime-rl-specific configs to `configs/prime-rl/`

**With `--vf-rl`**:
- Downloads `configs/zero3.yaml` (DeepSpeed config)
- Downloads vf-rl configs to `configs/vf-rl/`

**With `--skip-agents-md`**:
- Skips downloading `AGENTS.md`, `CLAUDE.md`, and `environments/AGENTS.md`

## Migration Notes

- Environments now automatically include monitor rubrics which track default class-specific metrics. If you were manually adding metrics for `num_turns`, `tool_call_count`, etc., these are now provided automatically.
- Third-party integrations (TextArena, ReasoningGym) have been moved to `verifiers.envs.integrations`.
- Experimental environments (GymEnv, MCPEnv, CliAgentEnv, HarborEnv, RLMEnv) are now in `verifiers.envs.experimental` and require explicit imports or `verifiers[all]` installation.

**Full Changelog**: https://github.com/PrimeIntellect-ai/verifiers/compare/v0.1.8.post2...v0.1.9

