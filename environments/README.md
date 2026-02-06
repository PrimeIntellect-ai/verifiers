# Environments

This folder contains installable example environments that showcase common usage patterns in Verifiers. Each module exposes a `load_environment(...)` function that returns a ready-to-use `vf.Environment` object.

## Quick start

- **Install an environment from this GitHub repo**: `prime env install math-python --from-repo`
- **Evaluate**: `prime eval run math-python` (defaults to gpt-4.1-mini, small sample)

## Common usage patterns and examples

### SingleTurnEnv (prompt → single response)
- **gsm8k**: Classic QA with exact-match reward and optional response-format reward.
- **reverse_text**: XML formatting with non-binary LCS reward + format reward.
- **continuation_quality**: Completion-style generation (`message_type="completion"`) judged for prose quality with `JudgeRubric`.
- **mmmu**: Multimodal inputs (image + text) packed in chat content; single-turn boxed-answer check.

### SingleTurnEnv subclass (custom dataset/scoring wrappers)
- **reasoning_gym_env**: Wraps `reasoning_gym` procedural datasets, converts to HF datasets, and applies task-specific scoring.

### MultiTurnEnv (custom interaction protocols)
- **alphabet_sort**: Multi-turn task requiring the model to maintain and update an alphabetically sorted list of names across turns; uses per-turn sequence similarity rewards.
- **doublecheck**: Simple follow-up turn ("Are you sure?") with math rewards; minimal `is_completed`/`env_response` implementation.
- **sentence_repeater**: Multi-turn Q/A over a paragraph; rewards compare assistant messages to expected answers.
- **wordle**: Game-style interaction via `TextArenaEnv`; multiple rewards (correctness, partial credit, few-turn bonus) and XML formatting.

### Tool use
- **ToolEnv (native function-calling)**
  - **tool_test**: Validates parallel tool calls and checks exact tool usage via `ToolRubric` + custom reward.
  - **wiki_search**: Multi-tool retrieval (search/view/read) with `ToolEnv`; final judgment combined via `RubricGroup` with a `JudgeRubric`.

### Sandboxes
- **PythonEnv (ipython-style REPL)**
  - **math_python**: Solve math problems using Python in a sandbox environment.

### GymEnv (external gym environments)
- **gem_wordle**: Multi-turn Wordle game powered by the GEM framework; models must guess a 5-letter word using `\boxed{}` format.

### Experimental environments
- **MCPEnv (MCP server integration)**
  - **mcp_search_env**: Example environment demonstrating `vf.MCPEnv` for Model Context Protocol server integration.

- **RLMEnv (Recursive Language Model)**
  - **rlm_secrets**: Puzzle environment testing RLM functionality including root-level tools, sub-LLM tool use, and file operations.

- **HarborEnv / CliAgentEnv (CLI agent sandboxes)**
  - **opencode_harbor**: Runs the OpenCode CLI agent on Harbor tasks with API interception via Prime Tunnel.
  - **terminus_harbor**: Runs the Terminus agent on Harbor tasks with API interception via Prime Tunnel.

### Composition
- **EnvGroup**
  - **math_group**: Groups two `SingleTurnEnv` tasks (GSM8K + Math) into one environment with shared interface.

- **RubricGroup**
  - **math_python**: `ToolRubric` (tool adherence) + `MathRubric` (answer correctness).
  - **wiki_search**: Merges judge scoring with the tool-use rubric.

### Judge-based evaluation (LLM-as-judge)
- **continuation_quality**: Judge rubric extracts `<grade>` and maps A–F to a continuous score.
- **toxicity_explanation**: Judge rubric returns 0–10 normalized score for both classification correctness and explanation quality.
- **self_reward**: Pattern for `SingleTurnEnv` with only a `JudgeRubric` over a dataset that supplies `question`/`answer`; intended for online RL where model acts as its own judge.

### Multimodal inputs
- **mmmu**: Demonstrates passing images via chat `content` items with `{type: "image_url", image_url: {url: ...}}` and standard answer parsing.

## What to look at for each pattern
- **Minimal SingleTurnEnv**: `reverse_text`, `gsm8k`
- **JudgeRubric end-to-end**: `continuation_quality`, `toxicity_explanation`, `self_reward`
- **ToolEnv with real tools**: `wiki_search`, `math_python`
- **Custom MultiTurnEnv**: `alphabet_sort`, `doublecheck`, `sentence_repeater`, `wordle`
- **GymEnv integration**: `gem_wordle`
- **CLI agent sandboxes**: `opencode_harbor`, `terminus_harbor`
- **MCP integration**: `mcp_search_env`
- **RLM (recursive LLM)**: `rlm_secrets`
- **Environment and rubric composition**: `math_group`, `math_python`, `wiki_search`
- **Procedural datasets**: `reasoning_gym_env`
- **Multimodal**: `mmmu`

## Running examples
All environments export `load_environment(...)`. 

In-line usage:
```python
import verifiers as vf
from openai import AsyncOpenAI
vf_env = vf.load_environment("reverse-text")
results = vf_env.evaluate(client=AsyncOpenAI(), model="gpt-4.1-mini", num_examples=25)
```

CLI usage:
```bash
prime env install reverse-text --from-repo
prime eval run reverse-text -n 50 -r 1
```

If you are building a new environment, prefer starting from `prime env init` and consult the top-level README and docs for dataset format, rubric design, and environment class specifications.
