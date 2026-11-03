# Agent

An `Agent` pairs a configured harness and model with a `Runtime`, configured through `AgentConfig`. Calling `agent.run(task)` executes a full rollout on a `Task` and returns its `Trace`:

```python
import verifiers.v1 as vf


async with vf.make_agent(vf.AgentConfig(model="z-ai/glm-5.2")) as solver:
    trace = await solver.run(vf.Task(vf.TaskData(prompt="What is 2+2?")))
```

If you want to drive a conversation turn by turn instead of running to completion in one shot, `agent.interaction(task)` holds the rollout open as an async context manager. The caller acts as the user, and each call to `turn()` runs one harness segment (which may span several model and tool calls) and returns a `Segment`. If the task already defines a `prompt`, call `turn()` with no arguments to run the opening turn; if `prompt=None`, pass the first user message to `turn(message)`.

You can keep sending follow-up messages until `segment.terminated` is true (subsequent calls after termination return an empty terminated segment without consuming the message), or exit the `async with` block early to end the exchange yourself. Leaving the block finalizes and scores the rollout so `interaction.trace` is ready to inspect:

```python
import verifiers.v1 as vf


task = vf.Task(vf.TaskData(prompt=None))
config = vf.AgentConfig(model="z-ai/glm-5.2", max_turns=3)
async with vf.make_agent(config) as solver:
    async with solver.interaction(task) as interaction:
        segment = await interaction.turn("hello")
        while not segment.terminated:
            segment = await interaction.turn(f"you said: {segment.last_reply}")

trace = interaction.trace
```

See [Runtimes](runtimes.md) for execution handles, resource ownership, and deployment of the harness, task workspace, and supporting services.

## Messages during an active turn

`turn()` is designed for sequential exchanges where each user message waits for the previous segment to finish. When you need to send input while the agent is still working, or wake an idle interaction with new input, use `interaction.send()`:

```python
receipt = await interaction.send(
    "Also check the empty-input case.", mode="queue", message_id="followup-1"
)
```

- `mode="steer"` (the default) injects the message at the harness's next safe point during active work, which requires a harness that supports live steering.
- `mode="queue"` holds the message until the current turn yields, then delivers any pending messages together in order at the start of the next turn. Because it waits for a turn boundary rather than interrupting an active tool or model call, it does not require steering support.

For tasks that define an initial `prompt`, start the opening `turn()` before calling `send()`. Note that the returned receipt confirms whether the message was accepted for delivery, not that the agent has finished acting on it. Keep the `interaction` context open while the agent works, since exiting the block cancels outstanding deliveries and finishes the rollout.

## Verification placement

`agent.scoring.mode` selects whether task scoring runs in the agent's workspace (`shared`) or in a fresh container (`isolated`). Tasks can declare their own defaults on `TaskData.scoring` via `ScoringConfig`, and any fields explicitly set on `agent.scoring` override the task defaults (falling back to `shared` when neither sets a mode).

When `mode="isolated"`, verifiers restores the task's collected artifacts into a fresh runtime before running task metrics, rewards, and judge plugins:

- `agent.scoring.runtime` optionally overrides the runtime configuration used for the isolated verifier.
- `agent.scoring.env` merges over `TaskData.scoring.env` in both `shared` and `isolated` modes, resolving `${VAR}` and `${VAR:-default}` templates from the host environment when scoring starts.
- `agent.scoring.network` sets the task's network policy while scoring runs, replacing `TaskData.scoring.network`; see [network policies](runtimes.md#network-policies).
- `agent.scoring.retries` configures `RetryConfig` for isolated scoring (defaulting to `max_retries = 2`), reusing the completed submission in a fresh scoring environment whenever a retryable error occurs. Shared scoring is not retried.

### Skip scoring for cut-off runs

A truncated or timed-out run is often not worth grading. Set `agent.scoring.skip_on_truncated` to skip task scoring when `Trace.is_truncated` is true (a turn or token cap, or a length-finished response). Set `agent.scoring.skip_on_timeout` to skip it when a stage deadline expired before scoring (`Trace.is_timeout`). A skipped trace keeps no task rewards, metrics, or judge scores, so its reward is 0. `trace.info["scoring_skipped"]` records the reason (`"truncated"` or `"timeout"`). Task finalization and harness metrics still run.

`Trace.is_truncated` only checks the run's last model response. Some harnesses keep going after a response hits its per-turn token cap: they compact, or answer a cut-off tool call. Set `agent.max_turn_tokens` to cap each response and stop the run before the next model call once a response hits that cap. The stop condition is `max_turn_tokens`. The run is then truncated, so `skip_on_truncated` applies.

```toml
[env.agent]
max_turns = 10
max_turn_tokens = 4096

[env.agent.scoring]
skip_on_truncated = true
skip_on_timeout = true
```

## Retries and checkpoints

An agent retries a whole rollout up to twice on a `SandboxError` and twice on a `ProviderError` by default (separate from the turn retries below and from HTTP retries inside the provider SDK). You can configure retry budgets per error type under `[env.agent.retries]`, or under `[env.retries]` for environments that manage shared resources such as Harbor Compose. When an agent rollout fails, `rules` are checked in order and the first rule where all fields match sets the retry budget for that error; unmatched errors use the top-level `max_retries` budget.

A model call that fails gets turn retries first (`[env.agent.turn_retries]`): the interception retries the call with backoff, and the harness waits on it and never sees an error that a retry recovered. By default it retries 408, 429, 5xx and transport failures 5 times; another status goes to the harness at once. When the turn retries run out, the rollout checkpoints its box and fails the attempt with the error, and the rollout retry restores the checkpoint:

- **Filesystem checkpoints** (`[env.agent.checkpoint]`, on by default) snapshot the box's files. The retry starts a new box from the snapshot and a new harness process, which continues the trace and is told that the box was restored; processes, sub-agents and in-memory state are gone. They need a runtime that takes snapshots (prime) and a harness that supports them (`SUPPORTS_FILESYSTEM_CHECKPOINTS`).
- **Memory checkpoints** (`[env.agent.memory_checkpoint]`, off by default) also hold the box's processes and memory. The retry resumes the box, and the harness program carries on as it was, with its background processes, REPL state and sub-agents. They need a runtime and a harness that support them (`SUPPORTS_MEMORY_CHECKPOINTS`). On prime, a memory checkpoint pauses the sandbox and resumes it once, on the same node; when it no longer resumes, the retry restores its files as a filesystem checkpoint would. Where both kinds apply, memory checkpoints are taken.

A failed box (`SandboxError`) takes no checkpoint: its retry starts over in a new box. Elsewhere the agent logs a warning once, and each retry starts a fresh attempt from the beginning. `after` sets whether a checkpoint is deleted (`"delete"`) or kept (`"keep"`) when the run is final; a kept memory checkpoint bills as a paused sandbox until it is deleted. Set `--no-env.agent.checkpoint` to turn filesystem checkpoints off.

The rollout's owner can also checkpoint it. `Rollout.interrupt` checkpoints the rollout and ends it with `RolloutInterrupted`, for another process to continue; `interrupt_all` interrupts every open rollout of the process. The trace's `restorable` checkpoint and `RestorePoint.of` give what that process needs.

To bound how long individual stages or rollouts can run, configure stage timeouts and token limits on the [runtime](runtimes.md); see [debugging](debugging.md) for diagnosing failed episodes.
