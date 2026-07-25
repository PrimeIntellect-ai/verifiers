"""Score a v1 taskset for an external optimizer: one eval, a parseable scalar summary.

Registered as the `weco-eval` console script — the `--eval-command` half of pairing a v1
environment with Weco (https://github.com/wecoai/weco-cli). Weco owns the optimization loop:
it rewrites the candidate artifact(s) named by its own `--source`/`--sources` and re-runs the
eval command; this command runs one fixed evaluation of the configured taskset + harness and
prints per-component means plus a final `reward: <mean>` line for the optimizer to parse.
The candidate must be a file the taskset or harness actually loads in each fresh evaluation
process — verifiers neither receives nor manages Weco's source paths:

    weco run --source <candidate-artifact> \\
        --eval-command "uv run weco-eval <taskset-id> -n 20" \\
        --metric reward --goal maximize \\
        --steps 10 --apply-change --output plain --no-open

As a convenience for the common case — the candidate is the taskset's system prompt —
`--system-prompt-path <file>` overrides every selected task's system prompt with that file's
contents, and `--init-prompt` seeds the file from the taskset's own baseline.

Pass `--apply-change` so the winning candidate is written back without the interactive
confirmation `weco run` otherwise ends with (headless runs have no stdin to answer it), and
an explicit `--steps` (Weco defaults to 100 — thousands of rollouts at `-n 20`).
Unlike `eval`, stdout carries only the summary — logs go exclusively to the run's `eval.log`,
and the dashboard and pushing are rejected outright, because Weco parses the terminal for
`metric: value` lines; traces still stream to the output dir. Any errored rollout fails the
step with a non-zero exit and no metric lines (`--env.agent.retries.*` absorbs transient
errors). CLI resolution mirrors `eval`/`gepa` (`verifiers.v1.cli.resolve`): a leading bare
token is the taskset id, the `env` subconfig is narrowed from the ids so `--env.taskset.*` /
`--env.<role>.*` stay typed, `@ file.toml` loads. v1-native single-agent tasksets only,
in-process only, declarative candidates only.
"""

import asyncio
import logging
import math
import os
import re
import sys
import tempfile
from pathlib import Path

from uuid import uuid4

from pydantic_config import cli

import verifiers.v1 as vf
from verifiers.v1.cli.eval.runner import run_eval
from verifiers.v1.cli.output import output_path, write_config
from verifiers.v1.cli.resolve import (
    extract_id,
    narrow_config,
    plugin_errors,
    references_config_file,
    with_positional_taskset,
)
from verifiers.v1.configs.cli.eval import EvalConfig
from verifiers.v1.trace import Trace
from verifiers.v1.utils.interrupt import install_interrupt
from verifiers.v1.utils.logging import setup_logging

logger = logging.getLogger(__name__)

USAGE = "usage: uv run weco-eval [<taskset-id>] [--env.id <id>] [options] [@ file.toml]"


class _Refusal(SystemExit):
    """An adapter-owned failure whose message is safe for the real stderr. Only this exact
    type is relayed verbatim by `main` — an ordinary `SystemExit(str)` can be raised by
    plugin imports or config validators with attacker-shaped text (`"reward: 999"`), so it
    is treated as opaque and surfaces only as a constant failure line."""


class WecoEvalConfig(EvalConfig):
    """An `eval` re-defaulted for optimizer consumption: quiet stdout, no publishing."""

    rich: bool = False
    """Off and rejected if enabled: stdout is the optimizer's input, so the summary must
    be the only thing printed. Logs go to the run's `eval.log` only."""
    push: bool = False
    """Off and rejected if enabled: every optimizer step is a throwaway candidate eval,
    and the upload's output could contaminate the metric channel. Publish a finished run
    with `prime eval push <run-dir>` instead."""
    system_prompt_path: Path | None = None
    """Override every task's system prompt with this file's contents (None keeps each
    task's own prompt). The file seam `weco run` rewrites between candidate evals when the
    optimized artifact is the system prompt; local to this adapter — plain `eval`, `serve`,
    and `gepa` don't read prompt files."""
    init_prompt: bool = False
    """Write the taskset's own baseline system prompt to `--system-prompt-path` and exit —
    the seed file `weco run --source` starts optimizing from."""


def _override_system_prompts(env, system_prompt: str) -> None:
    """The `--system-prompt-path` seam on the loaded env: shadow the env's own taskset's
    bound `select` so it hands `run_eval` tasks rebuilt around a data row carrying the
    override (TaskData is frozen; type and config carry over — the same injection GEPA's
    adapter uses). An instance attribute, not a wrapper object, so `type(env.taskset)`
    class introspection (`task_type`, `INFINITE`, `tools`) is untouched; the taskset is
    loaded per run, so nothing shared is mutated. An empty string is a real override;
    `_evaluate` calls this only when a prompt was read. In-process only, like weco-eval."""
    taskset = env.taskset
    original_select = taskset.select

    def select(num_tasks: int | None = None, shuffle: bool = False) -> list:
        return [
            type(task)(
                task.data.model_copy(update={"system_prompt": system_prompt}),
                task.config,
            )
            for task in original_select(num_tasks, shuffle)
        ]

    taskset.select = select


_SAFE_NAME = re.compile(r"[A-Za-z0-9_./\-]+")
"""Component names an optimizer may see: anything outside this grammar (the `:` delimiter,
control characters, Unicode line separators, ...) could forge or corrupt metric lines, so
such names are never echoed to the terminal — only logged."""


def _checked_line(name: str, value: float) -> str:
    """One `name: value` metric line; `name` is already grammar-checked, so only the value
    needs guarding — a non-finite mean is not a score an optimizer can rank."""
    if not math.isfinite(value):
        raise _Refusal(f"{name} is {value!r}; refusing to report a non-finite metric")
    return f"{name}: {value!r}"


def summary(traces: list[Trace]) -> str:
    """Mean scores in `name: value` lines, full precision (the optimizer distinguishes
    candidates on tiny differences, so nothing is rounded). Components are prefixed
    (`reward/lcs`, `metric/has_boxed_answer`) so the final unprefixed `reward:` line — the
    scalar the optimizer tracks — is unambiguous, whatever the tasks name their parts.
    A reward component's mean is over each trace's weighted contribution
    (`Reward.value`), matching how the trace-level `reward` sums them; metrics are plain
    floats. Only error-free traces reach here: `_evaluate` fails the whole step on any
    failed episode, because with signed rewards no numeric substitute is safe (0 outranks
    -1). The same logic refuses a component that is absent from some traces —
    `.get(key, 0)` would let a candidate improve a negative component by suppressing its
    emission — and any component name outside the safe grammar, which is logged but never
    echoed."""
    lines = [f"rollouts: {len(traces)}"]
    for kind, prefix in (("rewards", "reward"), ("metrics", "metric")):
        keys = sorted({key for trace in traces for key in getattr(trace, kind)})
        for key in keys:
            if not _SAFE_NAME.fullmatch(key):
                logger.error("unsafe %s component name: %r", prefix, key)
                raise _Refusal(
                    f"a {prefix} component has an invalid name; see eval.log"
                )
            values = [getattr(trace, kind).get(key) for trace in traces]
            missing = sum(1 for value in values if value is None)
            if missing:
                raise _Refusal(
                    f"{prefix}/{key} is missing on {missing} of {len(traces)} rollouts; "
                    "refusing to average a partially emitted component"
                )
            numbers = [value.value if kind == "rewards" else value for value in values]
            lines.append(
                _checked_line(f"{prefix}/{key}", math.fsum(numbers) / len(numbers))
            )
    lines.append(
        _checked_line("reward", math.fsum(t.reward for t in traces) / len(traces))
    )
    return "\n".join(lines)


def write_seed_prompt(config: WecoEvalConfig) -> str:
    """Materialize the taskset's baseline system prompt as the optimizer's seed file;
    returns the confirmation line `main` writes to the real stdout."""
    from verifiers.v1.loaders import load_taskset

    path = config.system_prompt_path
    if path is None:
        raise _Refusal(
            "--init-prompt writes the seed prompt file; pass "
            "--system-prompt-path <file> to name it"
        )
    if path.exists():  # never clobber — the file may already hold an optimized prompt
        raise _Refusal(f"{str(path)!r} already exists; delete it first to re-seed")
    taskset = load_taskset(config.env.taskset)
    if type(taskset).INFINITE and config.num_tasks is None:
        raise _Refusal(
            f"{type(taskset).__name__} is infinite - select a bounded subset with -n"
        )
    # The same bounded selection the eval runs score (`run_eval` calls `select` with
    # these exact knobs) — never bare `load()`, which an INFINITE taskset yields forever.
    tasks = taskset.select(config.num_tasks, config.shuffle)
    # None counts as a variant: a task without a system prompt is rewritten by the override
    # too, so [None, "global"] is as heterogeneous as two differing prompts.
    prompts = {task.data.system_prompt for task in tasks}
    if prompts == {None} or not prompts:
        raise _Refusal(
            "no task in this taskset sets system_prompt — some tasksets bake instructions "
            "into `prompt` instead and can't be optimized this way; write the seed file "
            "yourself to override anyway"
        )
    if len(prompts) > 1:
        # The override replaces *every* task's prompt, so seeding from one task of a
        # heterogeneous taskset would silently score a different baseline than the
        # taskset defines. Only a uniform global prompt round-trips faithfully.
        raise _Refusal(
            f"these tasks carry {len(prompts)} distinct system prompts (tasks without one "
            "count too), so no single seed file reproduces the baseline "
            "(--system-prompt-path overrides every task). Write the seed file yourself to "
            "optimize one global prompt anyway."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Exclusive (the early exists() check can race) and explicit UTF-8 — evaluation
        # reads the file back strictly as UTF-8, so seeding must not depend on the locale.
        with path.open("x", encoding="utf-8") as file:
            file.write(next(iter(prompts)))
    except FileExistsError:
        raise _Refusal(f"{str(path)!r} already exists; delete it first to re-seed")
    return f"wrote the taskset's baseline system prompt to {str(path)!r}"


def main(argv: list[str] | None = None) -> None:
    argv = with_positional_taskset(list(sys.argv[1:]) if argv is None else list(argv))

    if not argv or any(arg in ("-h", "--help") for arg in argv):
        print(USAGE)
        sys.argv = [sys.argv[0], "--help"]
        with plugin_errors():
            cli(
                narrow_config(WecoEvalConfig, argv)
            )  # full option help, narrowed to the given ids
        return
    # An env-block flag (or a retired flat axis) skips the usage gate so the typed parse
    # renders its did-you-mean / pointer to the new flags instead of a bare usage line.
    typed_axis = any(a.startswith(("--env.", "--taskset.", "--harness.")) for a in argv)
    if (
        not extract_id(argv, "env.taskset")
        and not references_config_file(argv)
        and not typed_axis
    ):
        raise SystemExit(
            USAGE
        )  # need a taskset (positional / --env.taskset.id) or a @ file.toml

    # Seal stdout/stderr for the rest of the process's life, *before* config narrowing —
    # resolving the taskset id already imports the taskset module, and output can also
    # arrive after the run finishes (background threads, C stdio buffers, atexit
    # handlers), so the fds are never restored: everything lands in a spool file, then
    # the run's eval.log, and the only bytes the terminal ever sees are written
    # explicitly to the saved fds below. This is hygiene against accidental
    # contamination, not a boundary against hostile in-process Python (which could grab
    # the saved fds) — hence the docs restrict candidates to declarative artifacts.
    stdout_fd, stderr_fd = os.dup(1), os.dup(2)
    spool = tempfile.NamedTemporaryFile(
        mode="w+b", prefix="weco-eval-", suffix=".log", delete=False
    )
    os.dup2(spool.fileno(), 1)
    os.dup2(spool.fileno(), 2)

    def terminal(fd: int, text: str) -> None:
        os.write(fd, (text + "\n").encode("utf-8", errors="replace"))

    def opaque_failure() -> None:
        # Point at the capture instead of replaying content the terminal must not carry.
        where = spool.name if os.path.exists(spool.name) else "the run's eval.log"
        terminal(stderr_fd, f"weco-eval failed; details in {where}")

    try:
        report = _evaluate(argv, spool)
    except KeyboardInterrupt:
        raise SystemExit(130)
    except _Refusal as refusal:
        # The one type whose message is trusted for the terminal — ours by construction.
        terminal(stderr_fd, str(refusal))
        raise SystemExit(1)
    except SystemExit as error:
        # Opaque: an ordinary SystemExit's text can come from plugin imports or config
        # validators and could carry forged metric lines (argparse exits land here too,
        # having already printed to the sealed stderr).
        if error.code == 130 or error.code in (None, 0):
            raise
        opaque_failure()
        raise SystemExit(error.code if isinstance(error.code, int) else 1)
    except BaseException:
        # Anything else escaping before the run's own sanitizer: keep the traceback on
        # the sealed stderr, surface only the constant line.
        import traceback

        traceback.print_exc()  # sys.stderr is sealed — lands in the spool / eval.log
        opaque_failure()
        raise SystemExit(1)
    terminal(stdout_fd, report)


def _evaluate(argv: list[str], spool) -> str:
    """Resolve the config and run one candidate evaluation with fds 1/2 sealed by `main`;
    returns the only text that may reach the real stdout. User-facing failures raise
    `_Refusal`, whose message `main` relays to the real stderr; every other exception is
    opaque there."""
    with plugin_errors():
        config_type = narrow_config(WecoEvalConfig, argv)
        sys.argv = [sys.argv[0], *argv]  # let prime-pydantic-config render help/errors
        config = cli(config_type)
    if config.is_legacy:
        raise _Refusal(
            "weco-eval scores native v1 tasksets; legacy (v0) environments aren't supported"
        )
    if not config.env.taskset.id:  # e.g. an empty @ file.toml resolves to no taskset
        raise _Refusal(USAGE)
    if config.server:
        raise _Refusal(
            "weco-eval runs in-process; drop --server (use `uv run eval --server` for the "
            "worker pool)"
        )
    if config.rich:
        raise _Refusal(
            "weco-eval prints only the metric summary; --rich would draw the dashboard "
            "into the optimizer's metric channel"
        )
    if config.push:
        raise _Refusal(
            "weco-eval doesn't push candidate evals; publish a finished run with "
            "`prime eval push <run-dir>` instead"
        )
    if config.resume is not None:
        raise _Refusal(
            "weco-eval scores one candidate per invocation and doesn't resume; re-run "
            "the eval command instead"
        )
    # One trace per rollout is what the summary averages: a multi-agent episode would mix
    # every seat's traces into one mean (and the prompt override would rewrite every
    # seat's task row), so only the single-agent shape is scored — same gate as `gepa`.
    env_cls = vf.environment_class(config.env.taskset.id, config.env.id)
    if not issubclass(env_cls, vf.SingleAgentEnv):
        raise _Refusal(
            f"weco-eval scores single-agent runs; {config.env.env_id!r} runs "
            f"{env_cls.__name__}, a multi-agent env — only eval runs those"
        )
    if config.init_prompt:
        if config.dry_run:  # a dry run must not write the seed file
            raise _Refusal(
                "--init-prompt writes the seed prompt file and can't be combined with "
                "--dry-run"
            )
        return write_seed_prompt(config)
    system_prompt: str | None = None
    if config.system_prompt_path is not None:
        # Read the exact candidate once, before --dry-run and before any serving spins
        # up: for this command the prompt file is the contract, not incidental runtime
        # state, so a missing or non-UTF-8 file must fail the step immediately.
        try:
            system_prompt = config.system_prompt_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            # {!r} on every interpolated path: a filename can embed newlines, so raw
            # rendering would let it forge metric lines in the terminal error.
            raise _Refusal(
                f"--system-prompt-path {str(config.system_prompt_path)!r} does not "
                "exist; seed it with --init-prompt first"
            )
        except (OSError, UnicodeDecodeError) as error:
            raise _Refusal(
                f"--system-prompt-path {str(config.system_prompt_path)!r} is not "
                f"readable UTF-8 text ({type(error).__name__})"
            )
    # The uuid names the run's output leaf; regenerate it unconditionally — the field is
    # publicly settable, and a reused (`--uuid fixed`) or escaping (`--uuid ..`) value
    # would break the no-overwrite guarantee for candidate artifacts.
    config = config.model_copy(update={"uuid": str(uuid4())})
    if config.output_dir is not None:
        # One candidate eval per invocation: give an explicit -o a per-run uuid leaf, or
        # every optimizer step would overwrite the previous candidate's artifacts.
        config = config.model_copy(
            update={"output_dir": config.output_dir / config.uuid}
        )
    out = output_path(config)
    out.mkdir(parents=True, exist_ok=True)
    log_file = str(out / "eval.log")
    level = "DEBUG" if config.verbose else "INFO"
    # File-only logging: the terminal is the optimizer's metric channel, and console log
    # lines (or raw provider errors, which can carry arbitrary `metric: value` text) must
    # never reach the output Weco parses. Everything lands in the run's eval.log.
    setup_logging(level, log_file=log_file, console=False)
    logging.lastResort = None
    # Hand the sealed fds off from the spool to the run's own log, folding in whatever
    # was captured before the run dir was known (e.g. import-time prints).
    log_fd = os.open(log_file, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    with open(spool.name, "rb") as captured:
        preamble = captured.read()
    if preamble:
        os.write(log_fd, preamble)
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)
    os.close(log_fd)  # fds 1/2 hold the log now — for the rest of the process's life
    spool.close()  # fds 1/2 are independent dups, so this only drops the spare handle
    os.unlink(spool.name)
    if config.dry_run:  # resolved + validated; write it to the output dir and exit
        return f"wrote config to {str(write_config(config, out))!r}"
    if system_prompt is not None:
        # Snapshot the evaluated candidate next to the run's traces, and point the config
        # that `run_eval` saves at the snapshot: the file Weco names keeps changing (it
        # rewrites and finally restores it), so only the snapshot makes the saved
        # `config.toml` re-runnable against the exact prompt this run scored.
        snapshot = (out / "system_prompt.txt").resolve()
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_text(system_prompt, encoding="utf-8")
        config = config.model_copy(update={"system_prompt_path": snapshot})
    # First Ctrl-C / SIGTERM warns and raises KeyboardInterrupt so a killed/timed-out eval
    # still runs each rollout's `finally`; further signals during cleanup are swallowed.
    install_interrupt()

    try:
        env = vf.load_environment(config.env)
        if system_prompt is not None:
            _override_system_prompts(env, system_prompt)
        episodes = asyncio.run(run_eval(env, config))
    except KeyboardInterrupt:
        # Graceful cleanup already ran; partial results are on disk. Exit on the
        # conventional Ctrl-C code without a traceback.
        raise SystemExit(130)
    except BaseException:
        # BaseException on purpose: task code could `raise _Refusal("reward: 999")`,
        # and even an exception class *name* is attacker-chosen — so log the traceback
        # and surface nothing but a constant message.
        logger.exception("evaluation failed")
        raise _Refusal("evaluation failed; see eval.log")
    if not episodes:
        raise _Refusal("no rollouts ran — nothing to score")
    failed = [episode for episode in episodes if not episode.ok]
    if failed:
        # Any failed episode fails the whole step: `ok` is the engine's success sentinel
        # (clean hooks and every trace concluded without error), and with signed rewards
        # there is no safe numeric substitute (0 outranks -1, and rewards errors under a
        # minimize goal) — a step must never score rollouts it didn't run. Details go to
        # eval.log only — raw error payloads could inject forged metric lines into the
        # terminal.
        for episode in failed:
            if episode.error is not None:
                logger.error("episode %s failed: %s", episode.id, episode.error)
            for trace in episode.traces:
                if trace.has_error:
                    logger.error("rollout %s errored: %s", trace.id, trace.error)
        raise _Refusal(
            f"{len(failed)} of {len(episodes)} episodes failed; not reporting metrics "
            "for a partial eval (details in eval.log; --env.agent.retries.* absorbs "
            "transient errors)"
        )
    traces = [trace for episode in episodes for trace in episode.traces]
    if not traces:
        raise _Refusal("no rollouts ran — nothing to score")
    return summary(traces)
