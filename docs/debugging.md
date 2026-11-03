# Debugging

verifiers includes CLI tools to validate task setup and scoring before spending model tokens (`vf-validate`, `vf-debug`).

## Before running an agent

Before launching a full evaluation, you can verify your task setup, reference solutions, and scoring logic directly against a runtime:

```bash
uv run vf-validate my-task -n 1 --only-setup
uv run vf-validate my-task -n 1 --only-gold
uv run vf-validate my-task -n 1 --only-noop
uv run vf-debug my-task -n 1 --command 'pwd && ls -la'
```

`vf-validate` boots the configured runtime and runs task checks without an agent:

- `--only-setup` runs `Task.setup(runtime)` to verify that containers, files, and background services start cleanly.
- `--only-gold` runs `setup` followed by `Task.validate(runtime)` to check a reference answer or solution; `validate` can also request the check's `trace` by name. `validate` returns `True` (valid), `False` (invalid), or `None` when the task defines no gold check (reported as `unchecked` rather than a pass).
- `--only-noop` runs `setup`, `finalize`, and task scoring on the untouched environment. Any task that already scores `1.0` or higher with no agent actions is flagged as `invalid` (lower partial rewards pass the check and are recorded in the results, while tasks with no reward are marked `unchecked`).

When you run `vf-validate` without an `--only-*` flag, it runs both the gold and noop checks in separate runtimes. Because validation runs directly against the task rather than through a harness or `Env`, it does not apply harness network restrictions, transfer artifacts, or invoke `stage_verifier`. If your gold check relies on private tests, call your staging helper inside `Task.validate`, and use a real `vf-run` rollout to test the full isolated-grading lifecycle.

Validation writes `results.jsonl`, `summary.json`, `logs/validate.log`, and `configs/resolved/validate.json`. You can re-run missing, errored, or timed-out checks in place with `--resume`:

```bash
uv run vf-validate @ <run-dir>/configs/resolved/validate.json --resume
```

For interactive troubleshooting, `vf-debug` runs `Task.setup` and then executes a one-off shell `--command` or uploaded `--script-path check.sh` inside the runtime before tearing it down. Because neither `vf-validate` nor `vf-debug` wraps the task in an `Env`, both take top-level `--taskset.*` and `--runtime.*` flags, alongside [`--select.*`](tasksets.md#selecting-tasks) to filter tasks. Note that both commands execute real task setup (which may spin up paid sandboxes), and `vf-validate`'s noop check will call any configured judges.

## Re-score without rerunning the agent

`Task.rescore(trace)` re-computes the offline metrics, rewards, and judge scores of a finished `Trace` without re-running the agent. It clears the trace's scores, then runs the scoring hooks and judges that need only the saved `Trace`. The original sandbox is gone, so it skips `setup`, `finalize`, and any scoring hooks that request a `runtime`.
