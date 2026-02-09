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
prime env init my-env # creates a new template in ./environments/my_env
```

This will create a new module called `my_env` with a basic environment template.
```
environments/my_env/
├── my_env.py           # Main implementation
├── pyproject.toml      # Dependencies and metadata
└── README.md           # Documentation
```

Environment modules should expose a `load_environment` function which returns an instance of the Environment object, and which can accept custom arguments. For example: 
```python
# my_env.py
import verifiers as vf

def load_environment(dataset_name: str = 'gsm8k') -> vf.Environment:
    dataset = vf.load_example_dataset(dataset_name) # 'question'
    async def correct_answer(completion, answer) -> float:
        completion_ans = completion[-1]['content']
        return 1.0 if completion_ans == answer else 0.0
    rubric = Rubric(funcs=[correct_answer])
    env = vf.SingleTurnEnv(dataset=dataset, rubric=rubric)
    return env
```

To install the environment module into your project, do:
```bash
prime env install my-env # installs from ./environments/my_env
```

To install an environment from the Environments Hub into your project, do:
```bash
prime env install primeintellect/math-python
```

To run a local evaluation with any OpenAI-compatible model, do:
```bash
prime eval run my-env -m gpt-5-nano # run and save eval results locally
```
Evaluations use [Prime Inference](https://docs.primeintellect.ai/inference/overview) by default; configure your own API endpoints in `./configs/endpoints.toml`.

View local evaluation results in the terminal UI:
```bash
prime eval tui
```
In the TUI, press `c` to open Copy Mode for prompt/completion text; highlight and press `c` again to copy.

To publish the environment to the [Environments Hub](https://app.primeintellect.ai/dashboard/environments?ex_sort=most_stars), do:
```bash
prime env push --path ./environments/my_env
```

To run an evaluation directly from the Environments Hub, do:
```bash
prime eval run primeintellect/math-python
```

## Documentation

**[Environments](environments.md)** — Create datasets, rubrics, and custom multi-turn interaction protocols.

**[Evaluation](evaluation.md)** - Evaluate models using your environments.

**[Training](training.md)** — Train models in your environments with reinforcement learning.

**[Development](development.md)** — Contributing to verifiers

**[API Reference](reference.md)** — Understanding the API and data structures

**[FAQs](faqs.md)** - Other frequently asked questions.
