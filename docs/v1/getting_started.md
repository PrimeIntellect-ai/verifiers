# Installation

verifiers runs locally with `uv`. Install it, clone the repo, and sync dependencies:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/PrimeIntellect-ai/verifiers.git
cd verifiers
uv sync
```

Scaffold new tasksets with `uv run vf-init <name>` and check them model-free with `uv run vf-validate <taskset-id>`. Run evaluations through [Prime-RL](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/docs/eval.md) with `uv run eval <taskset-id>`.

## Skills

To equip your agent with the necessary knowledge, we highly recommend the skills in this repository's [`skills/`](https://github.com/PrimeIntellect-ai/verifiers/tree/main/skills) directory (alongside [`AGENTS.md`](https://github.com/PrimeIntellect-ai/verifiers/blob/main/AGENTS.md)). They are more comprehensive than these docs, which are meant for human consumption.
