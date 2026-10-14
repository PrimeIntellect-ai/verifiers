<p align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/40c36e38-c5bd-4c5a-9cb3-f7b902cd155d">
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/6414bc9b-126b-41ca-9307-9e982430cde8">
    <img alt="Prime Intellect" src="https://github.com/user-attachments/assets/6414bc9b-126b-41ca-9307-9e982430cde8" width="312" style="max-width: 100%;">
  </picture>
</p>

---

# Overview

verifiers is a library of modular building blocks for training and evaluating LLMs: a **Task** bundles one row of data with its scoring, a **Taskset** loads and configures tasks, an **Agent** is a harness × model × runtime that turns a task into a **Trace**, and an **Env** composes multiple agents into multi-agent environments.

verifiers is tightly integrated with the [Environments Hub](https://app.primeintellect.ai/dashboard/environments?ex_sort=most_stars), our training framework [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl), and our [Hosted Training](https://app.primeintellect.ai/dashboard/training) platform.

## Installation

Use it as a library:

```bash
uv add verifiers
```

To interact with the Environments Hub, Hosted Training, and hosted evals, install the Prime CLI:

```bash
uv tool install prime
```

## Building blocks

A taskset in a few dozen lines — data, scoring, and config:

```python
import verifiers.v1 as vf


class AdditionData(vf.TaskData):
    answer: int


class AdditionTask(vf.Task[AdditionData]):
    @vf.reward
    async def exact_match(self, trace: vf.Trace) -> float:
        return float(trace.last_reply == str(self.data.answer))


class AdditionConfig(vf.TasksetConfig):
    num_tasks: int = 100


class AdditionTaskset(vf.Taskset[AdditionTask, AdditionConfig]):
    def load(self) -> list[AdditionTask]:
        return [
            AdditionTask(
                AdditionData(prompt=f"What is {i} + {i}?", answer=2 * i),
                self.config.task,
            )
            for i in range(self.config.num_tasks)
        ]
```

One task, one agent, one trace:

```python
import verifiers.v1 as vf

solver = vf.make_agent(vf.AgentConfig(model="z-ai/glm-5.3"))
task = vf.Task(vf.TaskData(prompt="What is 2+2? Answer with just the number."))

async with solver:
    trace = await solver.run(task)
```

The harness — Claude Code, Codex, mini-swe-agent, a plain bash shell, or your own — and the runtime it executes in (a Prime sandbox, Docker, subprocess) are pure config on the agent. Everything else extends the same way: a `vf.Toolset` installs custom tools as MCP servers into supporting harnesses, a `vf.Env` composes multiple agents with control flow and judging, and the `vf.Trace` records the full rollout — messages, tool calls, rewards, and usage.

## Evaluate

```bash
uv run vf-eval addition-v1 -n 100
```

One command per workflow: `vf-validate` checks a taskset without an agent, `vf-debug` runs setup plus one command, `vf-replay` re-scores a finished run's saved traces, and `vf-gepa` optimizes prompts with GEPA.

## Train and evaluate

**Locally with [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl)** — configs reference a taskset by id:

```toml
[[orchestrator.train.source]]
name = "wordle"
env.taskset.id = "wordle"
env.player.harness.id = "null"
env.player.runtime.type = "subprocess"
```

```bash
uv run rl @ examples/basic/wordle/rl.toml   # async RL training
uv run eval gsm8k -n 32 -r 4 -c 8           # standalone eval
```

**Hosted with the [prime](https://github.com/PrimeIntellect-ai/prime) CLI**:

```bash
prime env push          # publish your taskset to the Environments Hub
prime eval wordle       # run a hosted evaluation
prime train config.toml # launch a Hosted Training run
```

## Further reading

- The [docs](docs/) contain short, human-written guides and overviews about the architecture.
- The [AGENTS.md](AGENTS.md) and [skills](skills/) are for coding agents and go into more details.

## Contributing

We welcome community contributions. We use [issues](https://github.com/PrimeIntellect-ai/verifiers/issues) to track bugs and feature requests. If you hit a bug, a pain point, or have an idea for a feature, please open an issue.

Contributions are welcome via PR. Please follow these guidelines:

1. Install the [pre-commit hooks](#pre-commit-hooks) so your code is formatted and type-checked.
2. Keep your PR in "Draft" until it is ready for review.
3. If your PR resolves an issue, link the issue in the PR description.
4. Run the [tests](#tests) for the areas you touched.

### Pre-commit hooks

```bash
uv run pre-commit install
```

### Tests

Unit tests run one component in isolation, with no network, and mirror the `verifiers/v1` layout. Integration tests run real evals: they need `PRIME_API_KEY` and skip without it.

```bash
uv run pytest tests/unit -n auto                       # unit
uv run pytest tests/integration -n 8                   # integration
uv run pytest tests/integration/test_single_turn.py    # one behavior
uv run pytest tests/integration -m "docker and codex"  # one runtime and harness
```

## Citation

Originally created by Will Brown ([@willccbb](https://github.com/willccbb)).

```bibtex
@misc{brown_verifiers_2025,
  author       = {William Brown},
  title        = {{Verifiers}: Environments for LLM Reinforcement Learning},
  howpublished = {\url{https://github.com/PrimeIntellect-ai/verifiers}},
  year         = {2025}
}
```
