# Installation

verifiers runs locally with `uv`. Install `uv`, clone the repository, and sync dependencies:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/PrimeIntellect-ai/verifiers.git
cd verifiers
uv sync
```

The checkout installs the example tasksets right away, so you can verify your setup with a local check that doesn't call a model:

```bash
uv run vf-validate reverse-text -n 1
```

To run a few episodes against a model, use `uv run vf-run reverse-text -n 2 -r 1`. For full model evaluations, use [`uv run eval` in prime-rl](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/docs/eval.md).

To work with other tasksets, including packages you build yourself following [building environments](building-environments.md#create-a-package), install them into the environment first (for example, `uv pip install -e environments/<package_dir>`) so the CLI can resolve their IDs. Harbor-based tasksets require Python 3.12+ and the `harbor` extra (`uv sync --extra harbor`).

## Commands

verifiers installs four CLI commands, which you run with `uv run` in the environment where verifiers and your taskset package are installed:

| Command | Use |
| --- | --- |
| [`vf-validate <taskset>`](debugging.md#before-running-an-agent) | Check task setup, a known gold solution, or the untouched task |
| [`vf-debug <taskset>`](debugging.md#before-running-an-agent) | Run a command or script inside the prepared task runtime |
| `vf-run <taskset>` | Run a few episodes in-process and save their traces, with a one-line summary |
| [`vf-gepa <taskset>`](gepa.md) | Optimize a system prompt against task rewards |

Pass `--help` to any command to inspect its flags. Because tasksets and harnesses register their own options dynamically, include the taskset ID before `--help` to see the taskset's flags.
