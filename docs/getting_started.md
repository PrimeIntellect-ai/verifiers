# Installation

verifiers runs locally with `uv`. Install `uv`, clone the repository, and sync dependencies:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/PrimeIntellect-ai/verifiers.git
cd verifiers
uv sync
```

The checkout installs the example tasksets right away, so you can verify your setup with a local dry run that doesn't call a model:

```bash
uv run vf-eval reverse-text --env.agent.harness.id null \
  --env.agent.runtime.type subprocess --no-push --dry-run
```

For full model evaluations, we recommend [`uv run eval` in prime-rl](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/docs/eval.md), which is replacing the bundled `vf-eval` CLI and uses its own run config, resume behavior, and output layout. If you use `vf-eval` directly, see the [evaluation guide](evaluation.md) for its flags and output files.

To work with other tasksets, including packages you build yourself following [building environments](building-environments.md#create-a-package), install them into the environment first (for example, `uv pip install -e environments/<package_dir>`) so the CLI can resolve their IDs. Harbor-based tasksets require Python 3.12+ and the `harbor` extra (`uv sync --extra harbor`).

## Commands

verifiers installs five CLI commands, which you run with `uv run` in the environment where verifiers and your taskset package are installed:

| Command | Use |
| --- | --- |
| [`vf-validate <taskset>`](debugging.md#before-running-an-agent) | Check task setup, a known gold solution, or the untouched task |
| [`vf-debug <taskset>`](debugging.md#before-running-an-agent) | Run a command or script inside the prepared task runtime |
| [`vf-replay <run-dir>`](debugging.md#re-score-without-rerunning-the-agent) | Re-score saved `vf-eval` traces without recreating their sandbox |
| [`vf-gepa <taskset>`](gepa.md) | Optimize a system prompt against task rewards |
| [`vf-eval <taskset>`](evaluation.md) | Run the bundled evaluator |

Pass `--help` to any command to inspect its flags. Because tasksets and harnesses register their own options dynamically, include the taskset and harness IDs before `--help` when checking evaluation flags (note that `vf-validate` and `vf-debug` take `--taskset.*` and `--runtime.*`, whereas `vf-eval` nests them under `--env.taskset.*` and `--env.<agent>.runtime.*`).

## Skills

The [release skill](../skills/release/SKILL.md) documents the stable-version release workflow, alongside [`AGENTS.md`](../AGENTS.md) for coding agents working in this repository.
