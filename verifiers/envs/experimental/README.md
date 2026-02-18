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

Environment implementing [Recursive Language Models](https://alexzhang13.github.io/blog/2025/rlm/) (RLMs), an inference strategy where language models can decompose and recursively interact with input context of unbounded length through REPL environments. The root model interacts with a REPL (`repl_language="bash"` by default, or `repl_language="python"` for the Python REPL) and can spawn sub-LLM calls to process chunks of the context recursively. Code execution runs inside a Prime Sandbox. Extra context is provided as a filesystem (either a copied `context_dir` or JSON-serializable `context` written to `context.json`/`context.txt`). The RLM scaffolding prompt is injected into the first user message; the model-visible prompt is stored in `state["prompt"]`, while the original input prompt is preserved in `state["raw_prompt"]`. Interception for sub-LLM/root-tool calls is routed through a Prime Tunnel unless `interception_url` is provided.

Notes:
- The sandbox and worker are started eagerly during `setup_state`.
  Environments can pre-set `state["rlm_fs_root_remote"]` (and optionally `state["rlm_control_dir_remote"]`)
  before calling `super().setup_state` to point the worker at an existing filesystem path in the sandbox.
  You can also override `get_sandbox_request`, `on_sandbox_ready`, and `customize_worker_script` on `RLMEnv`
  to customize sandbox creation, run setup steps (e.g., repo initialization), or tweak the worker script.
- Package installation in sandboxes is best-effort: packages are only installed if they are not importable, which avoids unnecessary installs on images that already include them.

Tool split:

- `tools`: shared between root and sub-LLMs
- `root_tools`: REPL-only tools (host-executed)
- `sub_tools`: tools exposed to sub-LLMs

`llm_batch` is a fixed root tool and always available (callable as a shell command in Bash mode, or a Python function in Python mode).

## SRLMEnv

Simpler RLM environment that uses the [rlms](https://pypi.org/project/rlms/) PyPI library (from the [RLM GitHub repo](https://github.com/alexzhang13/rlm)). Single-turn: one `rlm.completion(prompt)` per task, no sandbox or custom tools. Install with `pip install rlms`. When you run eval (e.g. `prime eval run`), the RLM library’s logger is used to save trajectory steps under the eval results folder in `outputs/.../rlm_trajectories/` (`.jsonl` files viewable with the RLM visualizer). The module docstring in `srlm_env.py` describes the ideal return shape for the RLM client (for when you modify the client) so it maps cleanly into verifiers' `Response` and usage tracking.
