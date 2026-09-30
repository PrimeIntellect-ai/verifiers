---
name: debug-environments
description: Find the cause of failed or unexpected Verifiers results. Use logs, saved traces, validation, vf-debug, and replay to check task loading, setup, execution, and scoring.
---

# Debug Environments

Start with the failed run and the code that produced it. See
[validation and trace inspection](../../docs/v1/debugging.md) for commands and
[runtimes](../../docs/v1/runtimes.md) for files, containers, and network rules.

## Find what failed

1. Read `configs/resolved/eval.json` and `logs/latest/eval.log`. Record the taskset,
   env, harness, runtime, model, client, sampling, limits, and installed revisions.
2. Read `traces.jsonl` as `WireEpisode` records. Inspect episode errors and each
   trace's task data, `errors`, `stop_condition`, `timing`, `calls`, tool results,
   rewards, and `info["judge_calls"]` when present.
3. Find the first failed operation. A later cleanup error or zero reward may be
   a consequence. Distinguish wrong answers, limits reached, and failures in the
   provider, harness, sandbox, or task setup.

Keep the original taskset, split, image, harness, and provider when reproducing a
failure. If you change one, state that difference. Check package versions when
an external environment imports an API missing from this checkout.

## Run the smallest useful check

| What failed | First check |
| --- | --- |
| Package import or config | `uv run vf-eval <id> --dry-run`; help with the same package IDs |
| Image, files, service setup | `uv run vf-validate <id> --runtime.type docker -n 1 --only-setup` |
| Ground truth | `uv run vf-validate <id> --runtime.type docker -n 1 --only-gold` |
| Reward without any work | `uv run vf-validate <id> --runtime.type docker -n 1 --only-noop` |
| Command inside the prepared task | `uv run vf-debug <id> --runtime.type docker -n 1 --command 'pwd && ls -la'` |
| Scoring from saved data or a judge | `uv run vf-replay <run-dir> -n 1` |
| Harness or model interaction | A small eval using the original settings |

Use the original runtime when it matters to the failure; Docker above is an
example. Validation/debug flags use `taskset.*` and `runtime.*`; eval uses
`env.taskset.*` and `env.<agent>.runtime.*`. These tools run setup code and may
create paid sandboxes. Replay may call judge models. Use existing authorization
and ask only for missing scope before an action that needs it.

Validation normally runs gold and noop checks. Noop runs finalization and full
task scoring on the untouched task, including configured judges, so it may call
models too. It flags rewards of at least 1.0; inspect partial rewards separately.

A dry run checks config and imports only. Gold `unchecked` means there was no
model-free verdict. Validation does not run the harness or apply its execution
network restrictions, so it cannot show that an agent works offline. For network
failures, find where the request runs: in the sandbox, a separate tool server,
or Python on the evaluator. Also check whether it happens before or after setup.

## Check scoring and saved evidence

- Compare the prompt with what the reward function checks. Inspect the saved
  file or reply and each named reward before changing the scorer.
- `trace.reward` sums `score * weight`; `None` means unscored. Inspect judge parse
  failures separately from a valid negative verdict, and check the judge's own
  model and endpoint.
- Runtime commands return an exit code; task code must check it. Plain Python
  filesystem calls run on the evaluator, not in the sandbox.
- `trace.state` disappears after the run. Replay uses saved task data and `info`,
  skips checks that need a runtime, and requires `SingleAgentEnv` or a subclass.
  It cannot check container grading.
- Keep the source run intact; replay writes a new run. Use the exact resolved
  config with `--resume` for interrupted evals, not `--clean`.

Report the cause with a task or episode ID and an error, trace excerpt, or output
file that supports it. State what you reproduced and what remains untested. Fix
the component causing the failure, check the fix, then continue any evaluation
the user already requested.

Use [audit-envs](../audit-envs/SKILL.md) when the question is whether the task or
grader is sound, including suspicious passes and rejected valid answers.
