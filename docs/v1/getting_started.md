# Installation

verifiers runs locally with `uv`. Install it, clone the repo, and sync dependencies:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/PrimeIntellect-ai/verifiers.git
cd verifiers
uv sync
```

The checkout installs its example tasksets. First, check that an example's
configuration loads:

```bash
uv run vf-eval reverse-text --env.agent.harness.id null \
  --env.agent.runtime.type subprocess --no-push --dry-run
```

This dry run does not call a model. To run the evaluation, provide model
credentials (Prime uses `PRIME_API_KEY` or `prime login`) and replace `--dry-run`
with `-n 3 -r 1 -c 1`. The default runtime is remote Prime; `subprocess` runs this
example locally. `--no-push` keeps the results local. See [evaluation](evaluation.md)
for other endpoints and runtimes.

Create new packages with `uv run vf-init <name>` and install them with
`uv pip install -e environments/<package_dir>`. Other tasksets must also be
installed before `vf-eval` can load their IDs. Harbor needs Python 3.12+ and the
`harbor` extra (`uv sync --extra harbor` in this checkout).

## Skills

The repository's [`skills/`](../../skills) and [`AGENTS.md`](../../AGENTS.md) give coding agents instructions for building, evaluating, and debugging environments. These docs explain the same features for people.
