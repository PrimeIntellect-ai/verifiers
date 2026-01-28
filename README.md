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
  <a href="https://github.com/PrimeIntellect-ai/verifiers/actions/workflows/publish-envs.yml">
    <img src="https://github.com/PrimeIntellect-ai/verifiers/actions/workflows/publish-envs.yml/badge.svg" alt="Envs" />
  </a>
</p>

## News & Updates

- [01/08/26] v0.1.9 is released, featuring a number of new experimental environment class types, monitor rubrics for automatic metric collection, improved workspace setup flow, improved error handling, bug fixes, and a documentation overhaul.
- [11/19/25] v0.1.8 is released, featuring a major refactor of the rollout system to use trajectory-based tracking for token-in token-out training across turns, as well as support for truncated or branching rollouts.
- [11/07/25] Verifiers v0.1.7 is released! This includes an improved quickstart configuration for training with [prime-rl], a new included "nano" trainer (`vf.RLTrainer`, replacing `vf.GRPOTrainer`), and a number of bug fixes and improvements to the documentation.
- [10/27/25] A new iteration of the Prime Intellect [Environments Program](https://docs.google.com/spreadsheets/d/13UDfRDjgIZXsMI2s9-Lmn8KSMMsgk2_zsfju6cx_pNU/edit?gid=0#gid=0) is live!  


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
├── endpoints.py        # OpenAI-compatible API endpoint configuration
└── lab/                # Example configs for Hosted Training
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
    rubric = vf.Rubric(funcs=[correct_answer])
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
prime eval run my-env -m openai/gpt-5-nano # run and save eval results locally
```
Evaluations use [Prime Inference](https://docs.primeintellect.ai/inference/overview) by default; configure your own API endpoints in `./configs/endpoints.py`.

View local evaluation results in the terminal UI:
```bash
prime eval tui
```

To publish the environment to the [Environments Hub](https://app.primeintellect.ai/dashboard/environments?ex_sort=most_stars), do:
```bash
prime env push --path ./environments/my_env
```

To run an evaluation directly from the Environments Hub, do:
```bash
prime eval run primeintellect/math-python
```

## Documentation

**[Environments](docs/environments.md)** — Create datasets, rubrics, and custom multi-turn interaction protocols.

**[Evaluation](docs/evaluation.md)** - Evaluate models using your environments.

**[Training](docs/training.md)** — Train models in your environments with reinforcement learning.

**[Development](docs/development.md)** — Contributing to verifiers

**[API Reference](docs/reference.md)** — Understanding the API and data structures

**[FAQs](docs/faqs.md)** - Other frequently asked questions.


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
