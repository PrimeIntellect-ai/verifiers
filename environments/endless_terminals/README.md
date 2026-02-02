# Endless Terminals Environment

An environment for [verifiers](https://github.com/PrimeIntellect-ai/verifiers) that dynamically generates terminal-use tasks using the [endless-terminals](https://github.com/kcoopermiller/endless-terminals) pipeline.

## Overview

Endless Terminals is a fully autonomous pipeline that procedurally generates terminal-use tasks. Since it requires Linux and Apptainer, this environment uses **Prime Sandbox** to run the generation pipeline remotely, then downloads the generated tasks locally for use with `LocalHarborEnv`.

## Task Generation

### CLI

Generate tasks using the CLI:

```bash
uv run generator.py --num-tasks 10 --out-dir ./tasks --model gpt-4o-mini
```

Options:

- `--num-tasks`: Number of tasks to generate (default: 10)
- `--out-dir`: Output directory for generated tasks (default: ./tasks)
- `--model`: OpenAI model for task generation (default: gpt-4o-mini)
- `--sandbox-timeout`: Sandbox timeout in minutes (default: 60)
- `--no-cleanup`: Don't delete sandbox after generation

### Programmatic

Generate tasks from Python:

```python
import asyncio
from endless_terminals.generator import generate_tasks

# Generate 10 tasks
task_dirs = asyncio.run(generate_tasks(
    num_tasks=10,
    out_dir=Path("./tasks"),
    model="gpt-4o-mini",
))
```

## Using the Environment

### With Pre-Generated Tasks

```python
from endless_terminals import load_environment

env = load_environment(dataset_path="./tasks")
```

### With Dynamic Generation

```python
from endless_terminals import load_environment

# Generate tasks on initialization
env = load_environment(
    generate_on_init=True,
    num_tasks=5,
    openai_api_key="sk-...",  # Or set OPENAI_API_KEY env var
)
```

## How It Works

1. **Create Prime Sandbox**: Spins up an Ubuntu sandbox with network access
2. **Install Dependencies**: Installs Apptainer and clones endless-terminals
3. **Generate Tasks**: Runs the endless-terminals pipeline to create tasks
4. **Convert to Harbor**: Converts tasks to Harbor format (Dockerfile, task.toml, etc.)
5. **Download Locally**: Downloads the generated tasks to your local machine
6. **Load Environment**: Uses `LocalHarborEnv` to load and run the tasks

## Task Format

Generated tasks follow the Harbor format:

```
task_name/
├── environment/
│   └── Dockerfile           # Container environment
├── instruction.md           # Task description
├── task.toml                # Configuration
└── tests/
    ├── test_state.py        # Pytest tests
    └── test.sh              # Verifier script
```

## Citation

If you use this environment, please cite:

```bibtex
@article{gandhi2025endless,
    title={Endless Terminals: Scaling RL Environments for Terminal Agents},
    author={Gandhi, Kanishk and Garg, Shivam and Goodman, Noah D. and Papailiopoulos, Dimitris},
    journal={arXiv preprint arXiv:2601.16443},
    year={2025}
}
```
