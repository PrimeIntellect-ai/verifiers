# SIGTERM Handling And Sandbox Cleanup

## Current Understanding

`asyncio.shield()` only protects an awaited coroutine from asyncio task
cancellation. It does not make cleanup reliable when the Python process is
terminating.

For rollout-local sandbox deletion, this pattern is useful:

```python
try:
    await asyncio.shield(sandbox_client.delete(sandbox_id))
except Exception as exc:
    raise vf.SandboxDeleteError(
        f"Failed to delete sandbox {sandbox_id}"
    ) from exc
```

If the rollout task is cancelled while the delete is in flight,
`asyncio.shield()` prevents that cancellation from cancelling the inner delete
operation. The outer await still raises `asyncio.CancelledError`, and because
`CancelledError` is a `BaseException`, not an `Exception`, the `except
Exception` block does not catch or mask cancellation.

This helps with ordinary async cancellation while the process keeps running. It
does not guarantee cleanup after `SIGTERM`.

## Current SIGTERM Behavior

The environment setup registers a synchronous signal handler that calls a helper
like this:

```python
def _sync_teardown():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(self._teardown())
    else:
        loop.create_task(self._teardown())
```

For `SIGTERM`, the handler then exits with status 143.

When no event loop is running, `asyncio.run(self._teardown())` runs teardown to
completion before returning.

During normal async rollout execution, an event loop is already running. In that
case, the handler only schedules teardown with `loop.create_task(...)`. It does
not await that task, and then the process exits. The scheduled teardown may get
little or no time to run.

`SIGKILL` cannot be handled by Python at all. No signal handler, shielded task,
or teardown code can run after `SIGKILL`.

## Implications

The sandbox-delete observability PR should not claim to solve process-level
termination. It can make delete failures visible when rollout cleanup actually
runs and returns or raises through normal Python control flow.

On `SIGTERM`, the process may exit before rollout cleanup or environment
teardown can complete. Any reliable fix for that belongs at the runner boundary,
not in each sandbox delete call site.

## Recommendation

Keep the sandbox-delete rollout-error PR focused:

- Use `asyncio.shield()` around rollout-local delete calls where cancellation
  during cleanup would otherwise cancel the delete request.
- Raise a non-retryable `vf.SandboxDeleteError` for ordinary delete failures.
- Do not catch `BaseException`.
- Do not catch or mask `asyncio.CancelledError`.
- Keep teardown-only paths log/backstop-only unless they have a live rollout
  state and can return a rollout output.

Defer SIGTERM-safe teardown to a separate runner-level change:

- Replace direct `exit(143)` from the signal handler with a shutdown request.
- Have the async main task observe that request, stop dispatching new work, and
  cancel or drain active work according to the runner's policy.
- Run environment teardown from an awaited `finally` block.
- Put a bounded timeout around teardown so shutdown cannot hang forever.
- Ensure the timeout fits inside the deployment platform's SIGTERM grace period,
  before any forced SIGKILL.

That design keeps cleanup semantics explicit: normal rollout delete failures can
flow to rollout error accounting, while process termination uses a separate
graceful-shutdown path.

## Deferred Cleanup Surfacing Scope

The current sandbox-delete rollout-error PR should not broaden cleanup behavior
across the whole library. These cleanup paths were considered but intentionally
removed from the PR so the first change stays focused on the v1 sandbox lease
path that Prime-RL observes:

- `Rubric.cleanup` handler exception collection and sibling-continuation
  behavior.
- `RubricGroup.cleanup` handler exception collection and sibling-continuation
  behavior.
- `Environment.cleanup` handler exception collection and sibling-continuation
  behavior.
- `SandboxEnv.destroy_sandbox` rollout-local delete surfacing.
- `OpenEnvEnv` normal cleanup delete surfacing.

Those broader changes can be revisited in follow-up PRs after the smaller v1
delete-error path lands and the desired cleanup semantics are agreed on.
