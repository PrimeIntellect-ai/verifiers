<p align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/40c36e38-c5bd-4c5a-9cb3-f7b902cd155d">
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/6414bc9b-126b-41ca-9307-9e982430cde8">
    <img alt="Prime Intellect" src="https://github.com/user-attachments/assets/6414bc9b-126b-41ca-9307-9e982430cde8" width="312" style="max-width: 100%;">
  </picture>
</p>

---

# Overview

verifiers is our library for creating environments to train and evaluate LLMs.

verifiers is tightly integrated with the [Environments Hub](https://app.primeintellect.ai/dashboard/environments?ex_sort=most_stars), as well as our training framework [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) and our [Hosted Training](https://app.primeintellect.ai/dashboard/training) platform.

## Installation

We recommend installing the [Prime CLI](https://github.com/PrimeIntellect-ai/prime) to interact with the environments.

```bash
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# install the prime CLI
uv tool install prime
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
