# Experimental Environments

Newer and more experimental environment classes that may have some sharper edges + change more frequently.

## GymEnv

Universal runner for Gym-compatible environments. Wraps any environment that implements `reset(seed)` and `step(action)` methods (following the OpenAI Gym / Gymnasium API). Supports both old-style 4-tuple and new-style 5-tuple step returns.

## MCPEnv

Environment for integrating MCP (Model Context Protocol) servers as tools. Connects to one or more MCP servers via stdio transport and exposes their tools to the model. Useful for giving models access to external services like web search, file fetching, or any MCP-compatible tool server.

## CliAgentEnv

Environment for running custom agent code inside sandboxes. Intercepts the agent's OpenAI API requests via an HTTP proxy server, with each request triggering one `MultiTurnEnv` rollout step.

## HarborEnv

`CliAgentEnv` subclass that loads Harbor-format tasks. Harbor is a task format for agent benchmarks with structured task directories containing `task.toml` configuration and `instruction.md` prompts, along with test scripts for computing rewards.

## RLMEnv

Environment implementing [Recursive Language Models](https://alexzhang13.github.io/blog/2025/rlm/) (RLMs), an inference strategy where language models can decompose and recursively interact with input context of unbounded length through REPL environments. The root model interacts with a REPL (`repl_language="bash"` by default, or `repl_language="python"` for the Python REPL) and can spawn sub-LLM calls to process chunks of the context recursively. Execution supports both local and sandbox backends via `execution_backend` (`"local"` default, `"sandbox"` to run inside a Prime Sandbox). Extra context is still provided as a filesystem (either a copied `context_dir` or JSON-serializable `context` written to `context.json`/`context.txt`). The RLM scaffolding prompt is injected into the first user message; the model-visible prompt is stored in `state["prompt"]`, while the original input prompt is preserved in `state["raw_prompt"]`. Sandbox interception for sub-LLM/root-tool calls is routed through a Prime Tunnel unless `interception_url` is provided.

Tool split:

- `tools`: shared between root and sub-LLMs
- `root_tools`: REPL-only tools (host-executed)
- `sub_tools`: tools exposed to sub-LLMs

`llm_batch` is a fixed root tool and always available (callable as a shell command in Bash mode, or a Python function in Python mode).
