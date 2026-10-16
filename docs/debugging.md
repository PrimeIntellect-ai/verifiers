# Validation and trace inspection

verifiers includes CLI tools to validate task setup and scoring before spending model tokens (`vf-validate`, `vf-debug`), inspect saved episodes (`WireEpisode`), and re-score finished runs offline (`vf-replay`).

## Before running an agent

Before launching a full evaluation, you can verify your configuration, task setup, reference solutions, and scoring logic directly against a runtime:

```bash
uv run vf-eval my-task --dry-run
uv run vf-validate my-task --runtime.type docker -n 1 --only-setup
uv run vf-validate my-task --runtime.type docker -n 1 --only-gold
uv run vf-validate my-task --runtime.type docker -n 1 --only-noop
uv run vf-debug my-task --runtime.type docker -n 1 --command 'pwd && ls -la'
```

`vf-eval --dry-run` resolves and saves the configuration without loading dataset rows or starting sandboxes.

`vf-validate` boots the configured runtime and runs task checks without an agent:

- `--only-setup` runs `Task.setup(runtime)` to verify that containers, files, and background services start cleanly.
- `--only-gold` runs `setup` followed by `Task.validate(runtime)` to check a reference answer or solution. `validate` returns `True` (valid), `False` (invalid), or `None` when the task defines no gold check (reported as `unchecked` rather than a pass).
- `--only-noop` runs `setup`, `finalize`, and task scoring on the untouched environment. Any task that already scores `1.0` or higher with no agent actions is flagged as `invalid` (lower partial rewards pass the check and are recorded in the results, while tasks with no reward are marked `unchecked`).

When you run `vf-validate` without an `--only-*` flag, it runs both the gold and noop checks in separate runtimes. Because validation runs directly against the task rather than through a harness or `Env`, it does not apply harness network restrictions, transfer artifacts, or invoke `stage_verifier`. If your gold check relies on private tests, call your staging helper inside `Task.validate`, and use a real `vf-eval` run to test the full isolated-grading lifecycle.

Validation writes `results.jsonl`, `summary.json`, `logs/validate.log`, and `configs/resolved/validate.json`. You can re-run missing, errored, or timed-out checks in place with `--resume`:

```bash
uv run vf-validate @ <run-dir>/configs/resolved/validate.json --resume
```

For interactive troubleshooting, `vf-debug` runs `Task.setup` and then executes a one-off shell `--command` or uploaded `--script-path check.sh` inside the runtime before tearing it down. Because neither `vf-validate` nor `vf-debug` wraps the task in an `Env`, both take top-level `--taskset.*` and `--runtime.*` flags (rather than `vf-eval`'s `--env.taskset.*` and `--env.agent.runtime.*`), alongside [`--select.*`](tasksets.md#selecting-tasks) to filter tasks. Note that both commands execute real task setup (which may spin up paid sandboxes), and `vf-validate`'s noop check will call any configured judges.

## Read saved episodes

Each line in `traces.jsonl` is a serialized **episode** containing one or more agent traces. You can parse it with `WireEpisode` without needing the original taskset or harness package installed:

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

Each `trace` records the task metadata and agent config alongside the message history, tool outputs, `stop_condition`, `timing`, and `errors`:

- `trace.branches` and `trace.last_reply` reconstruct the conversation paths and final assistant response from the saved messages (both are computed Python properties on the loaded object rather than separate keys in the raw JSON).
- `trace.calls` lists every model call with its sampling parameters, token usage, latency, and any provider errors.
- `trace.info["judge_calls"]` captures requests and responses from task judges, whose token usage is also rolled into `extra_usage`.

Each named entry in `trace.rewards` stores a `score` and `weight`, and `trace.reward` is the sum of their products. A `None` reward or metric indicates that the hook did not run (for example, after an earlier error), not that the model scored zero. Because runtime-only state (`trace.state`) and artifact archives transferred to an isolated verifier are not persisted to `traces.jsonl`, store any evidence you want to inspect after the run in `trace.info` during `finalize`.

When diagnosing low scores, check `logs/latest/eval.log` as well as `traces.jsonl` to distinguish model mistakes from rollouts cut short by limits or infrastructure errors; an episode that fails during setup or environment initialization may not produce any agent traces at all.

## Re-score without rerunning the agent

`vf-replay` re-computes offline metrics, rewards, and judge scores against a finished run's saved transcripts without re-running the agent:

```bash
uv run vf-replay <run-dir> --dry-run
uv run vf-replay <run-dir> -n 3 -r 2
```

Replay loads the source run's config and saved `TaskData`, clears the copied scores, and runs any scoring hooks and judges that only depend on the saved `Trace` (writing the results to a new output directory so the original run is never overwritten). Use `-n` to limit how many saved traces are re-scored, `-r` to score each trace multiple times (useful for measuring judge variance), and `--taskset.task.*` to override scoring settings (for example, `--taskset.task.judges.0.model` swaps the model on the first configured judge).

Because the original sandboxes are already gone, `vf-replay` skips `setup`, `finalize`, and any scoring hooks that request a `runtime`. It only supports runs produced by `SingleAgentEnv` (or its subclasses), since multi-agent environments score interactions across traces rather than per trace.
