# Installation

verifiers runs locally with `uv`. Install it, clone the repo, and sync dependencies:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/PrimeIntellect-ai/verifiers.git
cd verifiers
uv sync
```

You can now run tasksets directly, e.g. `uv run vf-eval <taskset-id>`.

## Skills

The [`skills/release`](https://github.com/PrimeIntellect-ai/verifiers/tree/main/skills/release) skill documents the stable-version release workflow (alongside [`AGENTS.md`](https://github.com/PrimeIntellect-ai/verifiers/blob/main/AGENTS.md)).
