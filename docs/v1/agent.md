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
`provision`. Each call is keyed by its position, `<agent name>/<n>` for the agent's n-th
call since the durable was installed, taken when the call starts (so `asyncio.gather`
keeps the order); `key=` names it instead, for a call whose order is not stable, and
`durable=False` opts a call out. Its trace is labelled with the key and a fingerprint of
what the result depends on (`trace.info["labels"]["call"]`), and before it runs, the
newest earlier trace of the key is looked up in the run's trace store:

- a finished ok trace with the same fingerprint is returned as it is, without running
  (`trace.kept` is True);
- an unfinished or failed one with checkpoints is continued from its latest checkpoint;
- anything else (nothing found, a changed fingerprint) runs fresh.

Without a durable installed, calls are plain, exactly as before.

```python
durable = vf.Durable(vf.TraceDir("traces"), match={"job": "j1"}, parts={"version": "3"})
with vf.use_durable(durable):
    trace = await solver.run(task)                      # solver's call 0
    async with solver.interaction(task) as exchange:    # call 1
        reply = await exchange.turn("hello")
    async with solver.provision(task) as box:           # call 2
        await solver.run(task, runtime=box)             # call 3
        await reviewer.run(review_task, runtime=box)    # reviewer's call 0
    await solver.run(task, key="lookup")                # named: takes no position
```

**The rule for env authors.** Run again after a crash, the same code makes the same calls
in the same order, because every result it saw comes back as recorded: each call finds
its own trace. Code changed so that positions shift meets traces whose fingerprints
differ, and those calls run fresh, which is safe. So the host code between agent calls
must be deterministic given those results: anything nondeterministic or with side
effects (a clock, a random draw, a write to a database, a request to another service)
must be idempotent or go through a durable step of its own; otherwise give the calls
whose order depends on it a `key=`.

- **The store.** `Durable.lookup` is a `vf.TraceLookup`: `find(labels)` returns every
  stored trace whose labels contain `labels`, partial ones as of their latest save. A
  training or eval run's store is prime-rl's, which writes every trace call by call;
  `vf.TraceDir` is a directory of traces for tests and standalone use, which the calls
  also write (a `vf.TraceSink`).
- **The scope.** `Durable.match` holds the labels an earlier trace must carry besides the
  key; the calls stamp them on their traces too. `durable.within(match)` is the same
  store under another scope, and `durable=` on a call replaces the installed one.
- **The fingerprint.** The task's class, code (its source outside verifiers; pin it with
  `__code_version__`), data and config; the agent's model, sampling, harness and client;
  `Durable.parts`; and the call's `depends=` (a `Path` by its contents).
  `cache_inputs=` replaces the task and agent parts. Budgets are not parts: a raised
  timeout, more turns or retries keep a finished call, while a changed prompt, model or
  harness runs it again.
- **Interactions.** A finished exchange replays: each `turn(message)` that matches the
  recorded one returns the recorded segment. A cut-off one replays the turns its latest
  checkpoint covers and continues live from there. A driver that sends something else
  continues live from the checkpoint at the end of the turns it got (with per-call saves,
  every final reply has one); without one, the turn raises.
- **Boxes.** Every run on a durable box is labelled with the box's key, and a later
  `provision` under the key starts the box from the latest checkpoint those runs
  recorded. Durable runs on it then continue from their own latest checkpoint no newer
  than the box's: seats that take turns on a box resume exactly. A durable run on a box
  that was not restored this way runs fresh.
- **Explicit continuations.** A call passing `resume=`, `at=`, `note=` or `checkpoint=`
  without `key=` is a plain call.

A kept trace restores the conversation, not the box: it is not run again, so whatever it
did to a box it ran on is not redone. Code that needs the box as the calls left it uses a
durable box.
