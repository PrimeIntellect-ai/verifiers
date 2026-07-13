"""Score a v1 taskset for an external optimizer: one eval, a parseable scalar summary.

Registered as the `weco-eval` console script — the `--eval-command` half of pairing a v1
environment with Weco (https://github.com/wecoai/weco-cli), an optimizer that improves a
text artifact by rewriting a file and re-running an eval command whose output ends in
`metric: value` lines:

    uv run weco-eval <taskset-id> --system-prompt-path prompt.txt --init-prompt
    weco run --source prompt.txt \\
        --eval-command "uv run weco-eval <taskset-id> --system-prompt-path prompt.txt -n 20" \\
        --metric reward --goal maximize

Each step Weco rewrites `prompt.txt`; this command re-scores the taskset with that prompt
(`--system-prompt-path` is the env-level override — see `EnvConfig.system_prompt_path`) and
prints per-component means plus a final `reward: <mean>` line for the optimizer to parse.
Unlike `eval`, stdout carries only that summary (traces still stream to the output dir), the
live dashboard is off, and nothing is pushed — a candidate eval is an optimizer step, not a
run to publish. CLI resolution mirrors `eval`/`gepa` (`verifiers.v1.cli.resolve`): a leading
bare token is the taskset id, `--taskset.*` / `--harness.*` stay typed, `@ file.toml` loads.
v1-native tasksets only, in-process only.
"""

import asyncio
import logging
import sys

from pydantic_config import cli

import verifiers.v1 as vf
from verifiers.v1.cli.eval.runner import run_eval
from verifiers.v1.cli.output import output_path, write_config
from verifiers.v1.cli.resolve import (
    extract_id,
    narrow_config,
    references_config_file,
    with_positional_taskset,
)
from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.trace import Trace
from verifiers.v1.utils.interrupt import install_interrupt
from verifiers.v1.utils.logging import setup_logging

logger = logging.getLogger(__name__)

USAGE = (
    "usage: uv run weco-eval [<taskset-id>] [--harness.id <id>] "
    "[--system-prompt-path <file>] [options] [@ file.toml]"
)


class WecoEvalConfig(EvalConfig):
    """An `eval` re-defaulted for optimizer consumption: quiet stdout, no publishing."""

    rich: bool = False
    """Off by default here: stdout is the optimizer's input, so the summary must be the
    only thing printed. Logs still tee to the run's `eval.log`."""
    push: bool = False
    """Off by default here: every optimizer step is a throwaway candidate eval, and pushing
    each one would flood the platform's Evaluations tab. Opt back in with `--push`."""
    init_prompt: bool = False
    """Write the taskset's own baseline system prompt to `--system-prompt-path` and exit —
    the seed file `weco run --source` starts optimizing from."""


def summary(traces: list[Trace]) -> str:
    """Mean scores in `name: value` lines. Components are prefixed (`reward/lcs`,
    `metric/has_boxed_answer`) so the final unprefixed `reward:` line — the scalar the
    optimizer tracks — is unambiguous, whatever the tasks name their parts."""
    errors = sum(1 for trace in traces if trace.has_error)
    lines = [f"rollouts: {len(traces)}", f"errors: {errors}"]
    for kind, prefix in (("rewards", "reward"), ("metrics", "metric")):
        keys = sorted({key for trace in traces for key in getattr(trace, kind)})
        for key in keys:
            mean = sum(getattr(t, kind).get(key, 0.0) for t in traces) / len(traces)
            lines.append(f"{prefix}/{key}: {mean:.4f}")
    lines.append(f"reward: {sum(t.reward for t in traces) / len(traces):.4f}")
    return "\n".join(lines)


def write_seed_prompt(config: WecoEvalConfig) -> None:
    """Materialize the taskset's baseline system prompt as the optimizer's seed file."""
    from verifiers.v1.loaders import load_taskset

    path = config.system_prompt_path
    if path is None:
        raise SystemExit(
            "--init-prompt writes the seed prompt file; pass --system-prompt-path <file> "
            "to name it"
        )
    if path.exists():  # never clobber — the file may already hold an optimized prompt
        raise SystemExit(f"{path} already exists; delete it first to re-seed")
    tasks = load_taskset(config.taskset).load()
    seed = next(
        (t.data.system_prompt for t in tasks if t.data.system_prompt is not None), None
    )
    if seed is None:
        raise SystemExit(
            "no task in this taskset sets system_prompt — some tasksets bake instructions "
            "into `prompt` instead and can't be optimized this way; write the seed file "
            "yourself to override anyway"
        )
    path.write_text(seed)
    print(f"wrote the taskset's baseline system prompt to {path}")


def main(argv: list[str] | None = None) -> None:
    argv = with_positional_taskset(list(sys.argv[1:]) if argv is None else list(argv))

    if not argv or any(arg in ("-h", "--help") for arg in argv):
        print(USAGE)
        sys.argv = [sys.argv[0], "--help"]
        cli(
            narrow_config(WecoEvalConfig, argv)
        )  # full option help, narrowed to the given ids
        return
    if not extract_id(argv, "taskset") and not references_config_file(argv):
        raise SystemExit(
            USAGE
        )  # need a taskset (positional / --taskset.id) or a @ file.toml

    config_type = narrow_config(WecoEvalConfig, argv)
    sys.argv = [sys.argv[0], *argv]  # let prime-pydantic-config render help/errors
    config = cli(config_type)
    if config.is_legacy:
        raise SystemExit(
            "weco-eval scores native v1 tasksets; legacy (v0) environments aren't supported"
        )
    if config.server:
        raise SystemExit(
            "weco-eval runs in-process; drop --server (use `uv run eval --server` for the "
            "worker pool)"
        )
    if config.resume is not None:
        raise SystemExit(
            "weco-eval scores one candidate per invocation and doesn't resume; use "
            "`uv run eval --resume` to complete an interrupted run"
        )
    if config.init_prompt:
        write_seed_prompt(config)
        return
    log_file = str(output_path(config) / "eval.log")
    level = "DEBUG" if config.verbose else "INFO"
    if (
        config.rich
    ):  # someone explicitly re-enabled the dashboard; mirror eval's handling
        setup_logging(level, log_file=log_file, console=False)
        logging.lastResort = None
    else:
        setup_logging(level, log_file=log_file, console=True)
    if config.dry_run:  # resolved + validated; write it to the output dir and exit
        logger.info("wrote config to %s", write_config(config, output_path(config)))
        return
    # First Ctrl-C / SIGTERM warns and raises KeyboardInterrupt so a killed/timed-out eval
    # still runs each rollout's `finally`; further signals during cleanup are swallowed.
    install_interrupt()

    env = vf.Environment(config)
    try:
        traces = asyncio.run(run_eval(env, config))
    except KeyboardInterrupt:
        # Graceful cleanup already ran; partial results are on disk. Exit on the
        # conventional Ctrl-C code without a traceback.
        raise SystemExit(130)
    if not traces:
        raise SystemExit("no rollouts ran — nothing to score")
    if all(trace.has_error for trace in traces):
        # Every rollout failing is an infrastructure problem, not a bad candidate — exit
        # non-zero so the optimizer marks the step failed instead of scoring it 0.0.
        first = next(trace for trace in traces if trace.has_error)
        raise SystemExit(
            f"all {len(traces)} rollouts errored (first: {first.error}); refusing to "
            "report a reward for a broken eval"
        )
    if config.push:
        from verifiers.v1.push import push_traces

        push_traces(traces, config)
    print(summary(traces))
