# Installation

verifiers runs locally with `uv`. Install it, clone the repo, and sync dependencies:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/PrimeIntellect-ai/verifiers.git
cd verifiers
uv sync
```

Scaffold new tasksets with `uv run init <name>` and check them model-free with `uv run validate <taskset-id>`. Evaluations run through prime-rl: `uv run evals <taskset-id>` there (see prime-rl's `docs/training.md`).

## Skills

To equip your agent with the necessary knowledge, we highly recommend the skills in this repository's [`skills/`](https://github.com/PrimeIntellect-ai/verifiers/tree/main/skills) directory (alongside [`AGENTS.md`](https://github.com/PrimeIntellect-ai/verifiers/blob/main/AGENTS.md)). They are more comprehensive than these docs, which are meant for human consumption.
