# Agent

An `Agent` is a configured `harness` with a model running in a `Runtime`. It can be configured with an `AgentConfig`. An agent is given a `Task` and produces a `Trace`.

```python
async with vf.make_agent(vf.AgentConfig(model="z-ai/glm-5.2")) as solver:
    trace = await solver.run(vf.Task(vf.TaskData(prompt="What is 2+2?")))
```

`agent.interaction(task)` holds a rollout open turn by turn. The caller acts as the
user, and each `turn()` runs one harness segment (i.e., a message, tool call, tool result etc.). A `Segment` therefore contains messages, tool calls etc.

```python
async with agent.interaction(task) as interaction:
    segment = await interaction.turn("hello")
    if not segment.terminated:
        segment = await interaction.turn(f"you said: {segment.last_reply}")

trace = interaction.trace
```

## Shared boxes

`agent.provision(task)` hands out a box that runs borrow with `run(task, runtime=box)` or
`interaction(task, runtime=box)`; several seats can share it (a control agent and its
reviewer). The box comes with an owner, `vf.box_of(box)`, and the owner does what a
borrowing run may not:

- **Checkpoints.** A borrowed run checkpoints the box through the owner, one checkpoint at
  a time across all seats: per call when the run's `save = "per_call"` or the provisioning
  agent's is, and on `interaction.checkpoint()`. Each checkpoint is recorded on the run's
  own trace, matching that run's conversation.
- **Restore.** With `interaction_retries`, a turn that loses a borrowed box has the owner
  start a fresh box from the latest checkpoint any seat took and swap it in under the same
  runtime object, so every seat holding the box continues on the restored one. The turn
  needs a checkpoint of its own trace to continue from.

A box made any other way (`provision_runtime`) has no owner: runs on it are never
checkpointed or retried.

## Durable calls

With a `vf.Durable` installed (an env's `durable(task)` for its episodes, or
`vf.use_durable`), every agent call survives a crash: `run`, `interaction` and
`provision`. A call's identity is the durable's scope (for an episode, the env's
`trace_labels`), the agent's name (its seat) and its position: the key `<seat>/<n>` names
the agent's n-th call since the durable was installed, taken when the call starts (so
`asyncio.gather` keeps the order). `key="draft"` names the call instead (`<seat>/draft`),
for a call whose order is not stable; `durable=False` opts a call out. Its trace is
labelled with the key and a fingerprint of what the result depends on
(`trace.info["labels"]["call"]`), and before it runs, the latest earlier trace with the
same scope and key is looked up in the run's trace store:

- a finished ok trace with the same fingerprint is returned as it is, without running
  (`trace.kept` is True);
- an unfinished or failed one with the same fingerprint and checkpoints is continued from
  its latest checkpoint;
- anything else (nothing found, a changed fingerprint) runs fresh.

Without a durable installed, calls are plain, exactly as before: RL rollouts stay fresh.

```python
durable = vf.Durable(vf.TraceDir("traces"), match={"job": "j1"})
with vf.use_durable(durable):
    trace = await solver.run(task)                      # solver/0
    async with solver.interaction(task) as exchange:    # solver/1
        reply = await exchange.turn("hello")
    async with solver.provision(task) as box:           # solver/2 (the box)
        await solver.run(task, runtime=box)             # solver/3
        await reviewer.run(review_task, runtime=box)    # reviewer/0
    await solver.run(task, key="lookup")                # solver/lookup: no position
    await solver.run(resume=trace, note="and now?")     # solver/4: continues solver/0

with vf.use_durable(durable):                           # a crashed attempt, run again
    trace = await solver.run(task)                      # solver/0: kept, nothing runs
    async with solver.interaction(task) as exchange:    # solver/1: replays
        reply = await exchange.turn("hello")            #   the recorded segment
        notes = await exchange.runtime.read("notes.md") #   restores the box at this turn
```

See [Durable episodes](env.md#durable-episodes) for the rule host code between the calls
must follow.

- **The store.** `Durable.lookup` is a `vf.TraceLookup`: `latest(labels)` returns the
  stored trace whose labels contain `labels` (a nested subset) that moved last (its latest
  checkpoint, else its start), a partial one as of its latest save, or None. A training
  or eval run's store is the caller's (prime-rl's live and finished trace stores, which
  hold every trace call by call); `vf.TraceDir` is a directory of traces for tests and
  standalone use, which the calls also write (a `vf.TraceSink`).
- **The scope.** `Durable.match` holds the labels an earlier trace must carry besides the
  key; left None, an episode takes its `trace_labels(task)`. The calls stamp it on their
  traces too. `durable.within(match)` is the same store under another scope (a call
  shared by every stage of a job), and `durable=` on a call replaces the installed one.
- **The fingerprint.** The task's class, code (its source outside verifiers; pin it with
  `__code_version__`), data and config; the agent's model, sampling, harness and client;
  `Durable.parts` (a pipeline version); and the call's `depends=` (a `Path` by its
  contents). Budgets are not parts: a
  raised timeout, more turns or retries keep a finished call, while a changed prompt,
  model or harness runs it again.
- **Interactions.** A finished exchange replays: each `turn(message)` that matches the
  recorded one returns the recorded segment. A cut-off one replays the turns its latest
  checkpoint covers and continues live from there. A driver that sends something else
  continues live from the checkpoint at the end of the turns it got (with per-call saves,
  every final reply has one); without one, the turn raises (before any turn was
  replayed, the exchange runs fresh).
- **The box during a replay.** An exchange that owns its box has no live box while it
  replays, so `exchange.runtime` is a `vf.ReplayBox`: the first awaited method on it
  (`await exchange.runtime.read(path)`), or `await exchange.box()`, restores the box from
  the checkpoint recorded at the end of the turns given so far and keeps it. A later
  replayed turn moves the kept box to its own checkpoint in place, so a reference to it
  stays valid. When the replay hands over to the live exchange, the live exchange
  continues on the kept box (as a borrowed box), with whatever the driver wrote to it
  since; it is moved first only when the live exchange continues from another checkpoint
  (a cut-off turn). A replay that never uses the box restores nothing, and the live
  exchange restores its own box at the hand-over, as before. An exchange on a borrowed
  box (`runtime=`) has that box throughout.
- **Boxes.** Every run on a durable box is labelled with the box's key and the checkpoint
  the box was at when the run started, and a later `provision` under the key starts the
  box from the latest checkpoint of the run on it that moved last. Durable runs on it then
  continue from their own latest checkpoint no newer than the box's: seats that take turns
  on a box resume exactly. A durable run on a box that was not restored this way runs
  fresh.
- **Continuations.** A call passing `resume=` (with `at=` and `note=`) is durable too, by
  its position or `key=`: a stage that continues a seat's conversation from an earlier
  stage keeps its own turns across a crash. Its fingerprint holds the trace it continues
  (its origin) by id, with `at` and `note`. A finished continuation is kept, a cut-off one
  continues from its own latest checkpoint (not the origin's), and a continuation of
  another origin (the earlier stage ran again) or with another note runs fresh from that
  origin. Without `task=`, the origin's recorded task is fingerprinted. A call passing
  `checkpoint=` (or `at=`/`note=` without `resume=`) is plain without `key=` and refused
  with it.

A kept trace restores the conversation, not the box: it is not run again, so whatever it
did to a box it ran on is not redone. Code that needs the box as the calls left it uses a
durable box.
