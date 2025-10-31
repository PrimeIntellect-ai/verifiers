<p align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/40c36e38-c5bd-4c5a-9cb3-f7b902cd155d">
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/6414bc9b-126b-41ca-9307-9e982430cde8">
    <img alt="Prime Intellect" src="https://github.com/user-attachments/assets/6414bc9b-126b-41ca-9307-9e982430cde8" width="312" style="max-width: 100%;">
  </picture>
</p>

---

<h3 align="center">
Verifiers: Environments for LLM Reinforcement Learning
</h3>

---

<p align="center">
  <a href="https://github.com/PrimeIntellect-ai/verifiers/actions/workflows/style.yml">
    <img src="https://github.com/PrimeIntellect-ai/verifiers/actions/workflows/style.yml/badge.svg" alt="Style" />
  </a>
  <a href="https://github.com/PrimeIntellect-ai/verifiers/actions/workflows/test.yml">
    <img src="https://github.com/PrimeIntellect-ai/verifiers/actions/workflows/test.yml/badge.svg" alt="Test" />
  </a>
  <a href="https://github.com/PrimeIntellect-ai/verifiers/actions/workflows/publish-environments.yml">
    <img src="https://github.com/PrimeIntellect-ai/verifiers/actions/workflows/publish-environments.yml/badge.svg" alt="Envs" />
  </a>
</p>

## News & Updates

- [10/28/25] Verifiers v0.1.7 is released! This includes an improved quickstart configuration for training with [prime-rl], a new included "nano" trainer (`vf.RLTrainer`, replacing `vf.GRPOTrainer`), and a number of bug fixes and improvements to the documentation.
- [10/27/25] A new iteration of the Prime Intellect [Environments Program]() is live!  

## Overview

Verifiers is a library of modular components for creating RL environments and training LLM agents. Environments built with Verifiers can be used directly as LLM evaluations, synthetic data pipelines, or agent harnesses for any OpenAI-compatible model endpoint, in addition to RL training. Verifiers includes an async GRPO implementation built around the `transformers` Trainer, is supported by `prime-rl` for large-scale FSDP training, and can easily be integrated into any RL framework which exposes an OpenAI-compatible inference client.

Full documentation is available [here](https://verifiers.readthedocs.io/en/latest/). 

Verifiers is also the native library used by Prime Intellect's [Environments Hub](https://app.primeintellect.ai/dashboard/environments?ex_sort=most_stars); see [here](https://docs.primeintellect.ai/tutorials-environments/environments) for information about publishing your Environments to the Hub.

## Quick Start

Install the `prime` [CLI](https://github.com/PrimeIntellect-ai/prime-cli):
```bash
uv tool install prime
``` 

Install `verifiers` in your project:
```bash
uv add verifiers
```

Select an environment from the [Environments Hub](https://app.primeintellect.ai/dashboard/environments?ex_sort=most_stars) to install:
```bash
prime env install will/wordle
```

Run a quick evaluation with OpenAI models:
```bash
uv run vf-eval wordle -m gpt-5-nano
```

For RL training, do:
```bash
uv add 'verifiers[rl] @ git+https://github.com/PrimeIntellect-ai/verifiers.git@main' # after 0.1.7 release, this will be: uv add 'verifiers[rl]'
uv run vf-setup # creates default RL TOMLs under configs/
```

## Setup

We recommend using `verifiers` along with [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management in your own project:
```bash
# install uv (first time only)
curl -LsSf https://astral.sh/uv/install.sh | sh
# create a fresh project -- 3.11 + 3.12 supported
uv init && uv venv --python 3.12 
```

For local (CPU) development and evaluation with API models, do:
```bash
uv add verifiers
```

For RL training on GPUs with the included trainer, do:
```bash
uv add 'verifiers[rl]'
```

To use the latest `main` branch with RL extras, do:
```bash
uv add 'verifiers[rl] @ git+https://github.com/PrimeIntellect-ai/verifiers.git@main'
```

To use with `prime-rl`, see [here](https://github.com/PrimeIntellect-ai/prime-rl).

To install `verifiers` from source for core library development, install with:
```bash
curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/verifiers/main/scripts/install.sh | bash
```

If you want to develop with RL extras enabled in this repo, do:
```bash
uv sync --extra rl
```

In general, we recommend that you build and train Environments *with* `verifiers`, not *in* `verifiers`. If you find yourself needing to clone and modify the core library in order to implement key functionality for your project, we'd love for you to open an issue so that we can try and streamline the development experience. Our aim is for `verifiers` to be a reliable toolkit to build on top of, and to minimize the "fork proliferation" which often pervades the RL infrastructure ecosystem.

## Environments

Environments in Verifiers are installable Python modules which can specify dependencies in a `pyproject.toml`, and which expose a `load_environment` function for instantiation by downstream applications (e.g. trainers). See `environments/` for examples. 

To initialize a blank Environment module template, do:
```bash
uv run vf-init environment-name # -p /path/to/environments (defaults to "./environments")
```

To an install an Environment module into your project, do:
```bash
uv run vf-install environment-name # -p /path/to/environments (defaults to "./environments") 
```

To install an Environment module from this repo's `environments` folder, do:
```bash
uv run vf-install math-python --from-repo # -b branch_or_commit (defaults to "main")
```

Once an Environment module is installed, you can create an instance of the Environment using `load_environment`, passing any necessary args:
```python
import verifiers as vf
vf_env = vf.load_environment("environment-name", **env_args)
```

To run a quick evaluation of your Environment with an API-based model, do:
```bash
uv run vf-eval environment-name -s # run and save eval results locally
# vf-eval -h for config options; defaults to gpt-4.1-mini, 5 prompts, 3 rollouts for each
```

If you're using Prime Intellect infrastructure, the [`prime` CLI](https://github.com/PrimeIntellect-ai/prime-cli) provides first-class commands for working with Verifiers environments through the [Environments Hub](https://docs.primeintellect.ai/tutorials-environments/environments). Install it with `uv tool install prime`, authenticate via `prime login`, then use `prime env push` to publish your package and `prime env install owner/name` (optionally pinning a version) to consume it from pods or local machines.

The core elements of Environments are:
- Datasets: a Hugging Face `Dataset` with a `prompt` column for inputs, and optionally `answer (str)` or `info (dict)` columns for evaluation (both can be omitted for environments that evaluate based solely on completion quality)
- Rollout logic: interactions between models and the environment (e.g. `env_response` + `is_completed` for any `MultiTurnEnv`)
- Rubrics: an encapsulation for one or more reward functions
- Parsers: optional; an encapsulation for reusable parsing logic

We support both `/v1/chat/completions`-style and `/v1/completions`-style inference via OpenAI clients, though we generally recommend `/v1/chat/completions`-style inference for the vast majority of applications. Both the included `GRPOTrainer` as well as `prime-rl` support the full set of [SamplingParams](https://docs.vllm.ai/en/stable/api/vllm/sampling_params.html#vllm.sampling_params.SamplingParams) exposed by vLLM (via their OpenAI-compatible [server](https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html) interface), and leveraging this will often be the appropriate way to implement rollout strategies requiring finer-grained control, such as interrupting and resuming generations for interleaved tool use, or enforcing reasoning budgets.

The primary constraint we impose on rollout logic is that token sequences must be *increasing*, i.e. once a token has been added to a model's context in a rollout, it must remain as the rollout progresses. Note that this causes issues with some popular reasoning models such as the Qwen3 and DeepSeek-R1-Distill series; see [Troubleshooting](https://verifiers.readthedocs.io/en/latest/training.html#common-issues) for guidance on adapting these models to support multi-turn rollouts.  

### SingleTurnEnv

For tasks requiring only a single response from a model for each prompt, you can use `SingleTurnEnv` directly by specifying a Dataset and a Rubric. Rubrics are sets of reward functions, which can be either sync or async.

```python
from datasets import load_dataset
import verifiers as vf

dataset = load_dataset("my-account/my-dataset", split="train")

def reward_A(prompt, completion, info) -> float:
	# reward fn, e.g. correctness
	...

def reward_B(parser, completion) -> float:
	# auxiliary reward fn, e.g. format
	...

async def metric(completion) -> float:
	# non-reward metric, e.g. proper noun count
	...

rubric = vf.Rubric(funcs=[reward_A, reward_B, metric], weights=[1.0, 0.5, 0.0])

vf_env = SingleTurnEnv(
	dataset=dataset,
	rubric=rubric
)
results = vf_env.evaluate_sync(client=OpenAI(), model="gpt-4.1-mini", num_examples=100, rollouts_per_example=1)
vf_env.make_dataset(results) # HF dataset format
```

Datasets should be formatted with columns for:
- `'prompt' (List[ChatMessage])` OR `'question' (str)` fields
	- `ChatMessage` = e.g. `{'role': 'user', 'content': '...'}`
	- if `question` is set instead of `prompt`, you can also pass `system_prompt (str)` and/or `few_shot (List[ChatMessage])`
- `answer (str)` AND/OR `info (dict)` (both optional, can be omitted entirely)
- `task (str)`: optional, used by `EnvGroup` and `RubricGroup` for orchestrating composition of Environments and Rubrics

The following named attributes available for use by reward functions in your Rubric:
- `prompt`: sequence of input messages
- `completion`: sequence of messages generated during rollout by model and Environment
- `answer`: primary answer column, optional (defaults to empty string if omitted)
- `state`: can be modified during rollout to accumulate any metadata (`state['responses']` includes full OpenAI response objects by default)
- `info`: auxiliary info needed for reward computation (e.g. test cases), optional (defaults to empty dict if omitted)
- `task`: tag for task type (used by `EnvGroup` and `RubricGroup`)
- `parser`: the parser object declared. Note: `vf.Parser().get_format_reward_func()` is a no-op (always 1.0); use `vf.ThinkParser` or a custom parser if you want a real format adherence reward.

**Note**: Some environments can fully evaluate using only `prompt`, `completion`, and `state` without requiring ground truth `answer` or `info` data. Examples include format compliance checking, completion quality assessment, or length-based rewards.

For tasks involving LLM judges, you may wish to use `vf.JudgeRubric()` for managing requests to auxiliary models.

Note on concurrency: environment APIs accept `max_concurrent` to control parallel rollouts. The `vf-eval` CLI currently exposes `--max-concurrent-requests`; ensure this maps to your environment’s concurrency as expected.

`vf-eval` also supports specifying `sampling_args` as a JSON object, which is sent to the vLLM inference engine:

```bash
uv run vf-eval vf-environment-name --sampling-args '{"reasoning_effort": "low"}'
```

Use `vf-eval -s` to save outputs as dataset-formatted JSON, and view all locally-saved eval results with `vf-tui`. Pass `--save-every N` to checkpoint long runs incrementally when interleaving generation and scoring.

### ToolEnv

For many applications involving tool use, you can use `ToolEnv` to leverage models' native tool/function-calling capabilities in an agentic loop. Tools must be stateless and idempotent—each call should be fully determined by the provided arguments—because the environment will automatically terminate once the assistant responds without tool calls. Tools can be specified as generic Python functions (with type hints and docstrings), which will then be passed in JSON schema form to each inference request.


```python
import verifiers as vf
vf_env = vf.ToolEnv(
	dataset= ... # HF Dataset with 'prompt'/'question' and optionally 'answer'/'info' columns
	rubric= ... # Rubric object; vf.ToolRubric() can be optionally used for counting tool invocations in each rollout
	tools=[search_tool, read_article_tool, python_tool], # python functions with type hints + docstrings
	max_turns=10
)
```

In cases where your tools require heavy computational resources, we recommend hosting your tools as standalone servers (e.g. MCP servers) and creating lightweight wrapper functions to pass to `ToolEnv`. Parallel tool call support is enabled by default. If you need to inject per-rollout or cross-call state (IDs, credentials, cached resources), promote the environment to `StatefulToolEnv` and populate that state through `setup_state`/`update_tool_args` instead of hiding globals.

#### StatefulToolEnv

`StatefulToolEnv` extends `ToolEnv` for workflows where tool calls must incorporate dynamic state (for example, sandbox handles or per-user secrets). Implement `setup_state` to seed the state dict and override `update_tool_args` to merge state into each tool invocation. Any arguments you strip from the OpenAI schema via `args_to_skip` should be tracked in `skipped_args` so the model never sees sensitive parameters. Avoid storing global state; keep everything in the provided `state` dict.

#### SandboxEnv & PythonEnv

`SandboxEnv` builds on `StatefulToolEnv` to coordinate long-running sandboxes. Queue heavyweight provisioning inside `setup_state` (without awaiting) and gate tool execution on readiness inside `update_tool_args` or the tools themselves. `PythonEnv` is a concrete sandboxed executor that demonstrates the pattern: it spins up a Prime sandbox, injects the sandbox ID into each tool call, and tears down resources when the rollout finishes. Treat both environments as references when building similar stateful tool workflows.

For training, or self-hosted endpoints, you'll want to enable auto tool choice in [vLLM](https://docs.vllm.ai/en/stable/features/tool_calling.html#automatic-function-calling) with the appropriate parser. If your model does not support native tool calling, you may find the `XMLParser` abstraction useful for rolling your own tool call parsing on top of `MultiTurnEnv`; see `environments/xml_tool_env` for an example.

### MultiTurnEnv

Both `SingleTurnEnv` and `ToolEnv` are instances of `MultiTurnEnv`, which exposes an interface for writing custom Environment interaction protocols. Override `is_completed` and `env_response`, and make sure any custom completion logic defers to the base class so turn limits and other shared guards keep working.


```python
from typing import Tuple
import verifiers as vf
from verifiers.types import Messages, State
class YourMultiTurnEnv(vf.MultiTurnEnv):
    def __init__(self,
                 dataset: Dataset,
                 rubric: Rubric,
				 max_turns: int,
                 **kwargs):
	
  async def is_completed(self, messages: Messages, state: State, **kwargs) -> bool:
    # Always call the base check so max_turns and shared guards are respected
    if await super().is_completed(messages, state, **kwargs):
        return True
    # return whether or not a rollout is completed
    return state.get("task_complete", False)

  async def env_response(self, messages: Messages, state: State, **kwargs) -> Tuple[Messages, State]:
    # return new environment message(s) + updated state
```

If your application requires more fine-grained control than is allowed by `MultiTurnEnv`, you may want to inherit from the base `Environment` functionality directly and override the `rollout` method.

### ToolEnv
For many applications involving tool use, you can use `ToolEnv` to leverage models' native tool/function-calling capabilities in an agentic loop. Tools must be stateless and idempotent—each call should be fully determined by the provided arguments—because the environment will automatically terminate once the assistant responds without tool calls.

#### StatefulToolEnv
`StatefulToolEnv` extends `ToolEnv` for workflows where tool calls must incorporate dynamic state ...
#### SandboxEnv & PythonEnv
`SandboxEnv` builds on `StatefulToolEnv` to coordinate long-running sandboxes ... `PythonEnv` is a concrete sandboxed executor that demonstrates the pattern ...

## Training

Verifiers is supported by a number of trainers, including:
- `verifiers.RLTrainer` (included with `verifiers[train]`, `transformers`-based, ZeRO3 + vLLM)
- `prime-rl`
- `SkyRL`
- `tinker_cookbook` trainer

### RL

Use `vf-rl` with a single TOML to orchestrate both inference (vLLM) and training panes inside tmux.

```bash
# create or edit a config, e.g. configs/rl/wordle.toml
# vf-rl will launch a tmux session with vLLM (top) and trainer (bottom)
uv run vf-rl @ configs/rl/config.toml
```

Minimal TOML:

```toml
model = "Qwen/Qwen3-4B-Instruct-2507"

[env]
id = "kalomaze/alphabet-sort" # auto-installed from hub if given as user/env-id, or from local project if given as env-id

[inference]
gpus = 1

[inference.args]
enforce_eager = true

[trainer]
gpus = 1

[trainer.args]
run_name = "alphabet-sort"
use_lora = true
learning_rate = 1e-5
micro_batch_size = 4
rollouts_per_example = 16
batch_size = 512
max_steps = 100
max_tokens = 512
max_seq_len = 2048
```

Alternatively, you can train environments with the external `prime-rl` project (FSDP-first orchestration). See the `prime-rl` README for installation and examples. For example:

```toml
# orchestrator config (prime-rl)
[environment]
id = "vf-math-python"  # or your environment ID
```

```bash
# run (prime-rl)
uv run rl \
  --trainer @ configs/your_exp/train.toml \
  --orchestrator @ configs/your_exp/orch.toml \
  --inference @ configs/your_exp/infer.toml
```

### Troubleshooting 
- Ensure your `wandb` and `huggingface-cli` logins are set up (or set `report_to=None` in `training_args`). You should also have something set as your `OPENAI_API_KEY` in your environment (can be a dummy key for vLLM). 
- If using high max concurrency, increase the number of allowed open sockets (e.g. `ulimit -n 4096`)
- On some setups, inter-GPU communication can [hang](https://github.com/huggingface/trl/issues/2923) or crash during vLLM weight syncing. This can usually be alleviated by setting (or unsetting) `NCCL_P2P_DISABLE=1` in your environment (or potentially `NCCL_CUMEM_ENABLE=1`). Try this as your first step if you experience NCCL-related issues.
- If problems persist, please open an [issue](https://github.com/PrimeIntellect-ai/verifiers/issues).

### Resource Requirements
`GRPOTrainer` is optimized for setups with at least 2 GPUs, scaling up to multiple nodes. 2-GPU setups with sufficient memory to enable small-scale experimentation can be [rented](https://app.primeintellect.ai/dashboard/create-cluster?image=ubuntu_22_cuda_12) for <$1/hr.

### PRIME-RL
If you do not require LoRA support, you may want to use the `prime-rl` trainer, which natively supports Environments created using `verifiers`, is more optimized for performance and scalability via FSDP, includes a broader set of configuration options and user experience features, and has more battle-tested defaults. Both trainers support asynchronous rollouts, and use a one-step off-policy delay by default for overlapping training and inference. See the `prime-rl` [docs](https://github.com/PrimeIntellect-ai/prime-rl) for usage instructions.

## Further Documentation

See the full [docs](https://verifiers.readthedocs.io/en/latest/) for more information.

## Contribution Guidelines

Verifiers warmly welcomes community contributions! Please open an issue or PR if you encounter bugs or other pain points during your development, or start a discussion for more open-ended questions.

Please note that the core `verifiers/` library is intended to be a relatively lightweight set of reusable components rather than an exhaustive catalog of RL environments. Consider sharing any environments you create to the [Environments Hub](https://app.primeintellect.ai/dashboard/environments) 🙂

## Citation

Originally created by Will Brown ([@willccbb](https://github.com/willccbb)).

If you use this code in your research, please cite:

```bibtex
@misc{brown_verifiers_2025,
  author       = {William Brown},
  title        = {{Verifiers}: Environments for LLM Reinforcement Learning},
  howpublished = {\url{https://github.com/willccbb/verifiers}},
  note         = {Commit abcdefg • accessed DD Mon YYYY},
  year         = {2025}
}
```
