# Validation and trace inspection

Start with the check closest to the failure. A config check, setup check, and
full agent run test different things.

## Before running an agent

```bash
uv run vf-eval my-task --dry-run
uv run vf-validate my-task --runtime.type docker -n 1 --only-setup
uv run vf-validate my-task --runtime.type docker -n 1 --only-gold
uv run vf-validate my-task --runtime.type docker -n 1 --only-noop
uv run vf-debug my-task --runtime.type docker -n 1 --command 'pwd && ls -la'
```

`--dry-run` loads and saves config. It does not load dataset rows or check that
images, credentials, or services work.

`vf-validate` starts the selected runtime and runs task setup. With `--only-gold`,
it also calls `Task.validate(runtime)` to check a known answer or solution. That
method returns `True` for valid, `False` for invalid, or `None` if no check exists.
`None` is reported as `unchecked`, not a pass.

`--only-noop` runs setup, finalization, and scoring without an agent changing the
task. A reward of at least 1.0 is flagged as an invalid task because it already
passes untouched. Lower rewards are recorded and pass this check; no reward is
reported as `unchecked`. Inspect partial rewards too if they should require work.

Without an `--only-*` flag, gold and noop checks run in separate runtimes.
`--only-setup` just checks setup. Validation does not run the harness or apply
its execution network restrictions. The noop check includes configured judges,
so it can call models.

Validation writes `results.jsonl`, `summary.json`, `logs/validate.log`, and
`configs/resolved/validate.json`. Resume missing, errored, or timed-out checks with:

```bash
uv run vf-validate @ <run-dir>/configs/resolved/validate.json --resume
```

`vf-debug` runs a shell command or `--script-path check.sh` after task setup, then
removes the runtime. Both tools use `--taskset.*` and `--runtime.*`, rather than
eval's `--env.taskset.*` and `--env.agent.runtime.*`. Both accept
[`--select.*`](tasksets.md#selecting-tasks) to choose tasks. They run setup code
and may create paid sandboxes; validation may also call judges.

## Read saved episodes

Each line of `traces.jsonl` holds one **episode**, which can contain several
agent traces. Read it with `WireEpisode`; this does not require the original
taskset or harness to be installed:

```python
from pathlib import Path

from verifiers.v1.episode import WireEpisode

with Path("<run-dir>/traces.jsonl").open() as saved:
    for line in saved:
        episode = WireEpisode.model_validate_json(line)
        for trace in episode.traces:
            print(trace.task.key, trace.agent.name, trace.reward, trace.stop_condition)
            print(trace.last_reply)
            if trace.has_error:
                print(trace.errors)
```

Read the task data and agent settings alongside the messages, tool results,
`stop_condition`, `timing`, and `errors`. Useful fields include:

- `trace.branches`: conversation paths, reconstructed from saved messages.
- `trace.last_reply`: the final assistant reply, also computed from messages.
- `trace.calls`: each model call's settings, token usage, timing, and provider errors.
- `trace.info["judge_calls"]`: recorded judge calls. Their token usage is included through `extra_usage`.

`branches` and `last_reply` are Python properties, not separate fields in the JSON.

Each named reward has a `score` and `weight`; `trace.reward` sums their products.
A `None` reward or metric means it was not scored, not that it scored zero.
`trace.state` is not saved. Put evidence needed later in `trace.info` during
`finalize`. Archives transferred to a separate verifier are not saved in traces.

Report wrong answers, runs cut short by limits, and execution errors separately.
Check `logs/latest/eval.log` for the first failure before attributing a low score
to the model. An episode can fail before producing any traces, so count episodes
as well as traces.

## Re-score without rerunning the agent

```bash
uv run vf-replay <run-dir> --dry-run
uv run vf-replay <run-dir> -n 3 -r 2
```

Replay loads the saved config and task data, clears the copied scores, and runs
scoring methods that need only saved data. It can also call configured judges.
It writes a new run and keeps the original unchanged.

`-n` selects traces; `-r` repeats scoring to check how much judge results vary.
Override task settings with `--taskset.task.*`. For example,
`--taskset.task.judges.0.model` changes the first configured judge's model.

Judge calls can incur charges. Replay cannot recreate a removed sandbox: it
skips scores that need a runtime and does not rerun setup or finalization. It
requires `SingleAgentEnv` or a subclass; scoring traces separately cannot
reproduce another environment's interactions or scoring.
