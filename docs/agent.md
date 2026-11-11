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

`[env.agent.retries]` sets a retry budget per error type: `rules` are checked in order, and the first rule where all fields match sets the budget for that error; unmatched errors use the top-level `max_retries` budget.

Rollout checkpointing is on by default: a rollout can be snapshotted, and a retry restores the newest snapshot of the box and continues the trace from there, instead of starting over.

By default a rollout takes a snapshot only when its owner saves it (`save_all`, below). Two settings add snapshots that the rollout takes by itself:

- `on_error`: when a model call fails with an error that the run retries (`[env.agent.retries]`). The snapshot is taken before the harness gets the error, so the box that the retry restores still waits for the reply and asks for the call again. This helps only when the harness would end on the error. A harness that handles a failed call itself never uses the snapshot.
- `interval`: at every n-th model turn. This covers failures that leave no box to snapshot: the box died, or the harness program crashed.

Without a snapshot, a retry starts over.

A snapshot holds the box's files and its memory. The restored box continues every process where the snapshot stopped it: the harness program, its sub-agents and shells, and the task's own processes. The harness gets no notice and starts no new process; a model call that was open is asked again. The snapshot stops the box for a few seconds.

A rollout takes snapshots only when all of these hold. Otherwise a retry starts over, with a warning once per harness and runtime:

- The runtime takes memory snapshots (`SUPPORTS_MEMORY_SNAPSHOTS`: prime).
- The harness runs detached from the host (`SUPPORTS_MEMORY_RESTORE`: null, bash and the ACP harnesses).
- The rollout runs in a box of its own, not in a borrowed one.
- The rollout has no tool servers, no shared tools and no services of its own. They reach the host at an address that does not follow the box to its restore.

A retry also starts over when a restore fails before the restored attempt's first new turn, unless a model call failed again. Every snapshot is deleted once the run is final. Set `--no-env.agent.checkpoint` to turn it off.

```toml
[env.agent.checkpoint]
# on_error = true  # snapshot when a model call fails with an error the run retries (off by default)
# interval = 10    # snapshot at the 10th, 20th, ... model turn (off by default)
keep = "last"  # snapshots kept while the rollout runs: "last", or every k-th plus the newest
```

An owner that outlives its process, such as a training run that stops and resumes, saves its running rollouts instead of waiting for a failure. `verifiers.v1.restore.save_all()` takes a memory snapshot of every running rollout of the process now and returns the trace of each as plain data the owner stores. The trace holds everything a restore needs, on the entry of its checkpoint. A later process continues a rollout from its saved trace with `RunSlot(task, id=..., restore=trace)` (or `Agent.run(task, restore=trace)`): same trace, a new box, and the harness program still running. The owner holds these snapshots; it deletes them with `PrimeBackend.discard` once nothing will restore them. A saved trace continues one rollout: an episode with more than one agent run cannot continue from the trace of one of them, and starts over.
