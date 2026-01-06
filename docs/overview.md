# Overview

Verifiers is our library for creating environments to train and evaluate LLMs.

Environments contain everything required to run and evaluate a model on a particular task:
- A *dataset* of task inputs
- A *harness* for the model (tools, sandboxes, context management, etc.)
- A reward function or *rubric* to score the model's performance

Environments can be used for training models with reinforcement learning (RL), evaluating capabilities, generating synthetic data, experimenting with agent harnesses, and more. 

Verifiers is tightly integrated with the [Environments Hub](https://app.primeintellect.ai/dashboard/environments?ex_sort=most_stars), as well as our training framework [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl).

## Getting Started

Ensure you have `uv` installed, as well as the `prime` CLI tool:
```bash
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# create a new project if needed
uv init && uv venv --python 3.12
# install the prime CLI
uv tool install prime
```

Then, add `verifiers` to your project:
```bash
uv add verifiers
```

Environments built with Verifiers are self-contained Python modules. To initialize a fresh environment template, do:
```bash
uv run vf-init my-env # creates a new template in ./environments/my_env
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
uv run vf-install my-env
```

To run a quick local evaluation with OpenAI-compatible models, do:
```bash
uv run vf-eval my-env -m gpt-4.1-mini -n 5
```

To publish the environment to the [Environments Hub](https://app.primeintellect.ai/dashboard/environments?ex_sort=most_stars), do:
```bash
prime env push --path ./environments/my_env
```

To run an evaluation directly from the Environments Hub, do:
```bash
prime eval primeintellect/math-python
``` 

To install an environment from the Environments Hub into your project, do:
```bash
prime env install primeintellect/math-python
```

## Documentation

**[Environments](environments.md)** — Create datasets, rubrics, and custom multi-turn interaction protocols.

**[Evaluation](evaluation.md)** - Evaluate models using your environments.

**[Training](training.md)** — Train models in your environments with reinforcement learning.

**[Contributing](development.md)** — Contributing to verifiers

**[API Reference](api_reference.md)** — Understanding the API and data structures

**[FAQs](faqs.md)** - Other frequently asked questions.