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

<p align="center">
  <a href="https://docs.primeintellect.ai/verifiers">Documentation</a> •
  <a href="https://app.primeintellect.ai/dashboard/environments?ex_sort=most_stars">Environments Hub</a> •
  <a href="https://github.com/PrimeIntellect-ai/prime-rl">PRIME-RL</a>
</p>

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

- [11/07/25] Verifiers v0.1.7 is released! This includes an improved quickstart configuration for training with [prime-rl], a new included "nano" trainer (`vf.RLTrainer`, replacing `vf.GRPOTrainer`), and a number of bug fixes and improvements to the documentation.
- [10/27/25] A new iteration of the Prime Intellect [Environments Program](https://docs.google.com/spreadsheets/d/13UDfRDjgIZXsMI2s9-Lmn8KSMMsgk2_zsfju6cx_pNU/edit?gid=0#gid=0) is live!  

## Overview

Verifiers is a library of modular components for creating RL environments and training LLM agents. Environments built with Verifiers can be used directly as LLM evaluations, synthetic data pipelines, or agent harnesses for any OpenAI-compatible model endpoint, in addition to RL training. Verifiers is supported by `prime-rl` for large-scale performance-optimized async RL training, includes a minimal `transformers`-based trainer (`vf.RLTrainer`) for simple algorithmic experiments, and can easily be integrated into any RL training stack which exposes an OpenAI-compatible inference client.

Full documentation is available [here](https://verifiers.readthedocs.io/en/latest/). 

Verifiers is the native library used by Prime Intellect's [Environments Hub](https://app.primeintellect.ai/dashboard/environments?ex_sort=most_stars); see [here](https://docs.primeintellect.ai/tutorials-environments/environments) for information about publishing your Environments to the Hub, and [here](https://github.com/PrimeIntellect-ai/prime-environments) for a collection of Environments built with Verifiers.

## Quick Start

Verifiers supports CPU-based environment development and evaluation with API models, as well as large-scale GPU-based RL training with [`prime-rl`](https://github.com/PrimeIntellect-ai/prime-rl) and several other trainers. Environments built with Verifiers are standalone Python packages that can be installed and used in your own projects, or shared with the community through the [Environments Hub](https://app.primeintellect.ai/dashboard/environments?ex_sort=most_stars).


To get started, install `uv` and the `prime` [CLI](https://github.com/PrimeIntellect-ai/prime-cli), and add `verifiers` to your project:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv init && uv venv --python 3.12    # to create a new project if needed
uv tool install prime
uv add verifiers
```

Select an environment from the [Environments Hub](https://app.primeintellect.ai/dashboard/environments?ex_sort=most_stars) to install:
```bash
prime env install will/wiki-search
```

Or install an environment from this repo:
```bash
uv run vf-install wordle --from-repo
```

Run a quick evaluation with OpenAI models:
```bash
uv run vf-eval wordle -m gpt-5-nano
```

For advanced evaluation configurations with the `prime` [CLI](https://github.com/PrimeIntellect-ai/prime-cli), see [here](https://docs.primeintellect.ai/tutorials-environments/evaluating)

## RL Training

### `prime-rl`
We recommend using the [`prime-rl`](https://github.com/PrimeIntellect-ai/prime-rl) trainer, and provide a basic setup guide below. See the [prime-rl documentation](https://github.com/PrimeIntellect-ai/prime-rl) for more information.

To get started, do: 

```bash
uv run vf-setup
```

This will clone and install the `prime-rl` trainer and its dependencies, and set up a default configuration for training with the included `wiki-search` Environment.

Then, you can start training with:

```bash
uv run prime-rl @ configs/prime-rl/wiki-search.toml
```

This will launch a tmux session with separate panes for the trainer, orchestrator, and inference server.

### vf.RLTrainer

The included `RLTrainer` is a minimal, hackable training loop based on `transformers.Trainer` that supports both full-parameter finetuning and LoRA training. `RLTrainer` can be viewed as a "baby" `prime-rl` that adopts a similar default training recipe (async CISPO with one-step off-policy overlap), intended for single-node test runs with dense models. The primary files (`trainer.py` and `orchestrator.py`, located in `verifiers/rl/trainer/`) are under 1000 lines of code, and are designed to be a convenient starting point for writing your own training loop.

The feature set is intentionally kept minimal and focused. Users seeking maximum performance, MoE support, multi-node training, multidimensional parallelism, and other advanced features should use the `prime-rl` trainer. 

To use `vf.RLTrainer` in your own project, install with RL extras:
```bash
uv add 'verifiers[rl]'
```

Then, create a training configuration file, e.g. `configs/vf-rl/wiki-search.toml`, and do:

```bash
uv run vf-rl @ configs/vf-rl/wiki-search.toml
```

Example configuration files can be created in your project by running `uv run vf-setup`.

### Other Trainers

`verifiers` is intended to be largely trainer-agnostic. It is supported by [SkyRL] and [Tinker], and is straightforward to support for any trainer which can expose an OpenAI-compatible inference client for rollouts. See the [integrations](https://github.com/PrimeIntellect-ai/verifiers/tree/main/integrations) directory for more information.


## Development

To install `verifiers` from source for core library development, or to use the latest `main` branch, install with:
```bash
curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/verifiers/main/scripts/install.sh | bash
```

If you want to develop with RL extras enabled in this repo, do:
```bash
uv sync --extra rl
```

Please use the [Environments Hub](https://app.primeintellect.ai/dashboard/environments?ex_sort=most_stars) to share your Environments with the community, rather than PRs to this repo. If you find yourself needing to clone and modify the core library in order to implement key functionality for your project, please open an issue or PR so that we can help you.


## Environments

Environments in Verifiers are installable Python modules which can specify dependencies in a `pyproject.toml`, and which expose a `load_environment` function for instantiation by downstream applications (e.g. trainers). See `environments/` for examples. 

To initialize a blank Environment module template, do:
```bash
uv run vf-init environment-name # -p /path/to/environments (defaults to "./environments")
```

To install an Environment module into your project, do:
```bash
uv run vf-install environment-name # -p /path/to/environments (defaults to "./environments") 
```

To install an Environment module from the [Environments Hub](https://app.primeintellect.ai/dashboard/environments?ex_sort=most_stars), do:
```bash
prime env install user/environment-name
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

We support both `/v1/chat/completions`-style and `/v1/completions`-style inference via OpenAI clients, though we generally recommend `/v1/chat/completions`-style inference for the vast majority of applications. Both  `prime-rl` as well as the included `vf.RLTrainer` support the full set of [SamplingParams](https://docs.vllm.ai/en/stable/api/vllm/sampling_params.html#vllm.sampling_params.SamplingParams) exposed by vLLM (via their OpenAI-compatible [server](https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html) interface), and leveraging this will often be the appropriate way to implement rollout strategies requiring finer-grained control, such as interrupting and resuming generations for interleaved tool use, or enforcing reasoning budgets.


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

vf_env = vf.SingleTurnEnv(
	dataset=dataset,
	rubric=rubric
)

# Async evaluation (recommended)
from openai import AsyncOpenAI
results = await vf_env.evaluate(client=AsyncOpenAI(), model="gpt-4.1-mini", num_examples=100, rollouts_per_example=1)

# Sync evaluation
from openai import OpenAI
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


### Troubleshooting 
- Ensure your `wandb` and `huggingface-cli` logins are set up (or set `report_to=None` in `training_args`). You should also have something set as your `OPENAI_API_KEY` in your environment (can be a dummy key for vLLM). 
- If using high max concurrency, increase the number of allowed open sockets (e.g. `ulimit -n 4096`)
- On some setups, inter-GPU communication can [hang](https://github.com/huggingface/trl/issues/2923) or crash during vLLM weight syncing. This can usually be alleviated by setting (or unsetting) `NCCL_P2P_DISABLE=1` in your environment (or potentially `NCCL_CUMEM_ENABLE=1`). Try this as your first step if you experience NCCL-related issues.
- If problems persist, please open an [issue](https://github.com/PrimeIntellect-ai/verifiers/issues).

### Resource Requirements
`prime-rl` can be run on a single GPU by allocating only a fraction of the available memory to the inference server (see [here](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples/alphabet_sort) for an example configuration), and can also be scaled to hundreds of GPUs for large-scale training. A wide range of competitively-priced cluster configurations are available on [Prime Intellect](https://app.primeintellect.ai/dashboard/create-cluster?image=ubuntu_22_cuda_12).


## Citation

Originally created by Will Brown ([@willccbb](https://github.com/willccbb)).

If you use this code in your research, please cite:

```bibtex
@misc{brown_verifiers_2025,
  author       = {William Brown},
  title        = {{Verifiers}: Environments for LLM Reinforcement Learning},
  howpublished = {\url{https://github.com/PrimeIntellect-ai/verifiers}},
  note         = {Commit abcdefg • accessed DD Mon YYYY},
  year         = {2025}
}
```
