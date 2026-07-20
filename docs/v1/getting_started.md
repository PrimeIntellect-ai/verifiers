### Installation
verifiers runs locally with `uv`. Install it, clone the repo, and sync dependencies:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/PrimeIntellect-ai/verifiers.git
cd verifiers
uv sync
```

You can now run tasksets directly, e.g. `uv run eval <taskset-id>`, and scaffold new ones with `uv run init <name>`.

### Skills
To equip your agent with the necessary knowledge, we highly recommend the skills in this repository's [`skills/`](../../skills/) directory (alongside [`AGENTS.md`](../../AGENTS.md)). They are more comprehensive than these docs, which are meant for human consumption.
