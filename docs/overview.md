# Overview

Verifiers is our library for creating environments to train and evaluate LLMs.

Environments contain everything required to run and evaluate a model on a particular task:
- A *dataset* of task inputs
- A *harness* for the model (tools, sandboxes, context management, etc.)
- A reward function or *rubric* to score the model's performance

Environments can be used for training models with reinforcement learning (RL), evaluating capabilities, generating synthetic data, experimenting with agent harnesses, and more. 

Verifiers is tightly integrated with the [Environments Hub](https://app.primeintellect.ai/dashboard/environments?ex_sort=most_stars), as well as our training framework [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) and our [Hosted Training](https://app.primeintellect.ai/dashboard/training) platform.

## Getting Started

Ensure you have `uv` installed, as well as the `prime` [CLI](https://docs.primeintellect.ai/cli-reference/introduction) tool:
```bash
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# install the prime CLI
uv tool install prime
# log in to the Prime Intellect platform
prime login
```
To set up a new workspace for developing environments, do:
```bash
# ~/dev/my-lab
prime lab setup 
```

This sets up a Python project if needed (with `uv init`), installs `verifiers` (with `uv add verifiers`), creates the recommended workspace structure, and downloads useful starter files:
```
configs/
├── endpoints.toml      # OpenAI-compatible API endpoint configuration
├── rl/                 # Example configs for Hosted Training
├── eval/               # Example multi-environment eval configs
└── gepa/               # Example configs for prompt optimization
.prime/
└── skills/             # Bundled workflow skills for create/browse/review/eval/GEPA/train/brainstorm
environments/
└── AGENTS.md           # Documentation for AI coding agents
AGENTS.md               # Top-level documentation for AI coding agents
CLAUDE.md               # Claude-specific pointer to AGENTS.md
```

Alternatively, add `verifiers` to an existing project:
```bash
uv add verifiers && prime lab setup --skip-install
```

Environments built with Verifiers are self-contained Python modules. To initialize a fresh environment template, do:
```bash
prime env init my-env # creates a v0 stub in ./environments/my_env
```
For the v1 Taskset/Harness authoring path:
```bash
prime env init my-env --v1
prime env init my-env --v1 --with-harness
```

This will create a new module called `my_env` with a runnable environment
template. For v1 templates, start by editing the generated `TasksetConfig`,
`Taskset.load_tasks()`, and `@vf.reward` methods. Use `--with-harness` when the
environment also owns reusable execution behavior.
```
environments/my_env/
├── my_env.py           # Main implementation
├── pyproject.toml      # Dependencies and metadata
└── README.md           # Documentation
```

Environment modules should expose a `load_environment` function which returns an
environment object. For simple legacy environments, this can still be a direct
constructor:
```python
# my_env.py
import verifiers as vf

def load_environment(dataset_name: str = 'gsm8k') -> vf.Environment:
    dataset = vf.load_example_dataset(dataset_name) # 'question'
    async def correct_answer(completion, answer) -> float:
        completion_ans = completion[-1]['content']
        return 1.0 if completion_ans == answer else 0.0
    rubric = vf.Rubric(funcs=[correct_answer])
    env = vf.SingleTurnEnv(dataset=dataset, rubric=rubric)
    return env
```

For new environments with reusable tasksets, toolsets, custom programs, or
custom harnesses, use the v1 Taskset/Harness path:
```python
# my_env.py
import verifiers as vf


class MyTasksetConfig(vf.TasksetConfig):
    system_prompt: vf.SystemPrompt = "Answer exactly."


class MyTaskset(vf.Taskset[MyTasksetConfig]):
    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        """Return serializable task records as a list, generator, or Dataset."""
        if split == "eval":
            return []
        return [
            {
                "prompt": [{"role": "user", "content": "Reverse abc."}],
                "answer": "cba",
                "max_turns": 1,
            }
        ]

    @vf.reward(weight=1.0)
    async def correct_answer(self, task: vf.Task, state: vf.State) -> float:
        messages = vf.get_messages(state.get("completion") or [], role="assistant")
        if not messages:
            return 0.0
        response = str(messages[-1].content or "").strip()
        return float(response == task["answer"])


def load_taskset(config: MyTasksetConfig) -> MyTaskset:
    return MyTaskset(config=config)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    """Loader pattern for all Taskset/Harness environments."""
    return vf.Env(
        taskset=vf.load_taskset(config=config.taskset),
        harness=vf.load_harness(config=config.harness),
    )
```
See [BYO Harness](byo-harness.md) for the advanced v1 taskset/harness API.
The child loader annotation is the config contract: keep root
`load_environment` typed as `vf.EnvConfig`, put task settings on
`TasksetConfig`, and add `load_harness(config: MyHarnessConfig)` only when the
environment owns a reusable harness.
Reusable v1 taskset and harness packages live in `tasksets` and `harnesses`.
Install them with `uv add "verifiers[packages]"`, or with the narrower
`verifiers[tasksets]`, `verifiers[harnesses]`, and backend-specific extras such
as `verifiers[nemogym]`. For example, Harbor task directories can run through
the bundled OpenCode CLI harness with:

```python
from harnesses import OpenCode, OpenCodeConfig
from tasksets import HarborTaskset, HarborTasksetConfig

env = vf.Env(
    taskset=HarborTaskset(config=HarborTasksetConfig(bundle_package=__name__)),
    harness=OpenCode(config=OpenCodeConfig()),
)
```

The packaged `Taskset` and `Harness` classes can also be constructed directly in
ordinary Python code; loaders are the environment/config composition path.
`NeMoGymTaskset` and `NeMoGymHarness` expose packaged NeMo Gym rows and rollout
collection through the same taskset/harness composition boundary.

To run a local evaluation with any OpenAI-compatible model, do:
```bash
prime eval run my-env -m openai/gpt-5-nano # run and save eval results locally
```
Evaluations use [Prime Inference](https://docs.primeintellect.ai/inference/overview) by default; configure your own API endpoints in `./configs/endpoints.toml`.

View local evaluation results in the terminal UI:
```bash
prime eval view
```
The TUI opens a single run browser (`environment -> model -> run`). Press `Enter` on a run to open rollout details, `b` to go back, `tab` to cycle panes, `e` and `x` to expand or collapse history, `pageup` and `pagedown` to scroll history, and `c` for Copy Mode.

To publish the environment to the [Environments Hub](https://app.primeintellect.ai/dashboard/environments?ex_sort=most_stars), do:
```bash
prime env push my-env # equivalent to --path ./environments/my_env
```

To run an evaluation directly from the Environments Hub, do:
```bash
prime eval run primeintellect/math-python
```

## Documentation

**[Environments](environments.md)** — Create datasets, rubrics, and custom multi-turn interaction protocols.

**[BYO Harness](byo-harness.md)** — Build v1 Taskset/Harness environments with custom tools, sandboxes, users, and custom programs.

**[Evaluation](evaluation.md)** - Evaluate models using your environments.

**[Training](training.md)** — Train models in your environments with reinforcement learning.

**[Development](development.md)** — Contributing to verifiers

**[API Reference](reference.md)** — Understanding the API and data structures

**[FAQs](faqs.md)** - Other frequently asked questions.
