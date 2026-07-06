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
