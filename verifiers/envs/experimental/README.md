# Experimental Environments

Newer and more experimental environment classes that may have some sharper edges + change more frequently.

## SandboxMixin

`SandboxMixin` (in `sandbox_mixin.py`) provides sandbox lifecycle management
(creation with retries + rate limiting, tracking, cleanup, and typed errors)
to `Environment` and `Rubric` subclasses. See `docs/environments.md` for the
full programming model.

### VM sandboxes

Opt in by setting `vm=True` on the `CreateSandboxRequest` passed to
`create_sandbox`. VM sandboxes are required for GPU attachments; the
`CreateSandboxRequest` pydantic validator enforces that `gpu_count > 0`
implies `vm=True` and a non-null `gpu_type`.

Once created, `create_sandbox` populates the following state fields for
downstream rubrics and helpers:

- `state["sandbox_is_vm"]: bool`
- `state["sandbox_gpu_count"]: int`
- `state["sandbox_gpu_type"]: str | None`

VM and container sandboxes share the same `sandbox_creations_per_minute`
rate limiter and `sandbox_wait_for_creation_max_attempts` readiness cap.
If VM boots are slow or quota-constrained, raise both via
`init_sandbox_client`.

Port exposure (`expose` / `unexpose` / `list_exposed_ports`) and SSH
sessions are not supported by the sandbox gateway on VM-backed sandboxes;
if you call them via `self.sandbox_client`, the SDK will raise
`APIError`. Subclasses that need to fail fast on their own may raise
`SandboxVMUnsupportedError` (exported from `sandbox_mixin`) when
`state["sandbox_is_vm"]` is true.

`SandboxMonitorRubric` reports two VM-aware metrics — `sandbox_is_vm` and
`sandbox_gpu_count` — in addition to the existing `sandbox_oom` /
`sandbox_timeout` metrics.

## GymEnv

Universal runner for Gym-compatible environments. Wraps any environment that implements `reset(seed)` and `step(action)` methods (following the OpenAI Gym / Gymnasium API). Supports both old-style 4-tuple and new-style 5-tuple step returns.

## MCPEnv

Environment for integrating MCP (Model Context Protocol) servers as tools. Connects to one or more MCP servers via stdio transport and exposes their tools to the model. Useful for giving models access to external services like web search, file fetching, or any MCP-compatible tool server.

## CliAgentEnv

Environment for running custom agent code inside sandboxes. Intercepts the agent's OpenAI API requests via an HTTP proxy server, with each request triggering one `MultiTurnEnv` rollout step.

## HarborEnv

`CliAgentEnv` subclass that loads Harbor-format tasks. Harbor is a task format for agent benchmarks with structured task directories containing `task.toml` configuration and `instruction.md` prompts, along with test scripts for computing rewards.

## RLMEnv

Environment implementing [Recursive Language Models](https://alexzhang13.github.io/blog/2025/rlm/) (RLMs), an inference strategy where language models can decompose and recursively interact with input context of unbounded length through REPL environments.
