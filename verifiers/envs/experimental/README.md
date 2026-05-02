# Experimental Environments

Newer and more experimental environment classes that may have some sharper edges + change more frequently.

## SandboxMixin

`SandboxMixin` works with both container and VM sandboxes. If your environment needs a VM, pass `CreateSandboxRequest(..., vm=True)` to `create_sandbox`. For a GPU VM, also set `gpu_count` and `gpu_type`. Everyday sandbox operations like file upload, file reads, background jobs, and cleanup work the same way. Port exposure and SSH are currently container-only.

## GymEnv

Universal runner for Gym-compatible environments. Wraps any environment that implements `reset(seed)` and `step(action)` methods (following the OpenAI Gym / Gymnasium API). Supports both old-style 4-tuple and new-style 5-tuple step returns.

## MCPEnv

Environment for integrating MCP (Model Context Protocol) servers as tools. Connects to one or more MCP servers via stdio transport and exposes their tools to the model. Useful for giving models access to external services like web search, file fetching, or any MCP-compatible tool server.

## ApiEnv

Base environment for running agent code that makes API calls through an interception proxy. Executes a user-provided Python callable (`agent_fn`) as a background task while the rollout loop intercepts, forwards, and records all LLM calls. The agent receives a `base_url` and can use any HTTP client or agent framework (OpenAI SDK, DSPy, LangChain, etc). Provides lifecycle hooks (`launch_agent`, `cleanup_agent`, `compute_base_url`) for subclasses.

## CliAgentEnv

`ApiEnv` subclass for running agent code inside remote sandboxes. Overrides the agent lifecycle to create a sandbox and start the agent as a CLI background job, with API requests intercepted via a tunnel to the interception proxy.

## HarborEnv

`CliAgentEnv` subclass that loads Harbor-format tasks. Harbor is a task format for agent benchmarks with structured task directories containing `task.toml` configuration and `instruction.md` prompts, along with test scripts for computing rewards.

## RLMEnv

Environment implementing [Recursive Language Models](https://alexzhang13.github.io/blog/2025/rlm/) (RLMs), an inference strategy where language models can decompose and recursively interact with input context of unbounded length through REPL environments.
