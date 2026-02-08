# AgentEnv + Harness + Taskset: Contract Sketch (Draft)

## Goal

Start a practical refactor path where:

- `Taskset` is the primary unit for task collections and grading metadata.
- `Harness` is the execution/runtime wrapper around model behavior.
- `AgentEnv` composes `taskset + harness` and remains in Verifiers orchestration.
- The **main rubric** is compiled from `taskset` + `harness`, while users can still add extra rubrics.

This is a contract sketch, not a full implementation.

---

## Proposed user-facing shape

```python
from tasksets import HarborTaskset
from harnesses import OpenCode
import verifiers as vf

env = vf.AgentEnv(
    taskset=HarborTaskset(taskset_id="my-taskset"),
    harness=OpenCode(),
)
```

And direct harness usage outside Verifiers:

```python
from tasksets import Task
from harnesses import OpenCode

task = Task("my-taskset:task-1")
oc = OpenCode(llm="gpt-5")
result = await oc.run(task)
```

---

## Core contracts

## 1) `Task`

Minimal, portable task reference + payload.

- `task_id: str` (globally unique or namespaced, e.g. `taskset_id:item_id`)
- `input`: prompt/messages/structured input
- `target`: optional expected output / checker target
- `metadata: dict[str, Any]`
- `grading_spec: dict[str, Any] | None` (optional per-task override)
- **No model/provider fields** (these belong to agent run context)
- Runtime code should hard-reject reserved agent keys in task payload metadata

### Footgun

**Ambiguous task identity** when mixing sources.

### Resolution

- Require canonical `task_id` strings.
- Add helper `Task.parse_task_id(...)` and enforce deterministic formatting.

---

## 2) `Taskset` (note spelling)

Taskset should be an object, not just a dataset wrapper.

- `taskset_id: str`
- iteration / random access to `Task`
- `get_task(task_id) -> Task`
- optional `default_grading_spec`
- optional execution hints (tools/docs/sandbox)

### Footgun

**Taskset trying to own runtime behavior** (tool loop, retries, etc.).

### Resolution

- Keep taskset declarative.
- Keep runtime control in harness/env.

---

## 3) `Harness`

Harness is the execution protocol + model integration.

- `run(task, context?, on_event?) -> HarnessResult`
- optional `supports(taskset/task)` capability checks
- standardized trace/events surface + callback for streaming observers
- optional rubric contributions via `compile_rubric(...)`

### Footgun

**Harnesses diverge in return formats**, breaking orchestration.

### Resolution

- Require `HarnessResult` schema (messages/output/status/error/usage/events).
- Provide adapters for provider-specific fields under `result.raw`.

---

## 4) `AgentEnv`

`AgentEnv` composes `Taskset + Harness` under existing Verifiers lifecycle.

- takes `taskset` and `harness` in constructor
- compiles a base rubric from taskset+harness
- allows user-added rubrics with `add_rubric(...)`
- keeps existing eval/generate orchestration behavior

### Footgun

**Double-scoring or metric name collisions** between compiled and user rubrics.

### Resolution

- Namespace compiled rubric metrics (e.g. `core.*`).
- Namespace harness telemetry metrics (e.g. `harness.*`).
- Validate duplicate reward function names at env init.

---

## Rubric compilation model

Main idea: users should not need to manually wire the foundational rubric for common harness+taskset stacks.

## Source-of-truth ordering

1. `taskset.default_grading_spec`
2. `harness.compile_rubric(taskset)`
3. optional env-level normalizers
4. user-added rubrics (`env.add_rubric(...)`)

This should produce one `RubricGroup` with deterministic order.

### Footgun

**Hidden rubric logic** makes behavior hard to debug.

### Resolution

- `AgentEnv.describe_rubric()` returns compiled structure.
- Save compiled rubric manifest in run metadata.

---


## Streaming-awareness (first-class)

The baseline contract should make live trace viewing easy:

- Harness emits structured trace events during execution.
- `AgentEnv` can forward events to a callback while also storing them in state.
- Final `HarnessResult.events` remains the durable post-run summary.

This gives both real-time observability and deterministic persisted artifacts.

### Footgun

**Only storing final events** makes live debugging impossible.

### Resolution

- Add `on_event` callback to `Harness.run(...)`.
- Ensure callback ordering is deterministic per rollout.
- Keep callback side effects out of scoring-critical logic.

### Minimal live-trace watcher shape

```python
async def print_event(event):
    print(f"[{event.event_type}] {event.task_id}")

env = vf.AgentEnv(
    taskset=my_taskset,
    harness=my_harness,
    trace_callback=print_event,
)
```

---

## Contract boundaries (important)

## Taskset should own

- task identity + payload
- grading metadata
- static task docs/specs

## Harness should own

- model execution loop
- tool/runtime actions
- trace emission

## AgentEnv should own

- dataset/taskset orchestration
- rollout grouping and scoring
- retries/checkpointing/persistence
- rubric execution and aggregation

If a feature touches all three, define an explicit handoff object rather than leaking internals.

---

## Draft compatibility strategy

- Keep current environments unchanged.
- Introduce `AgentEnv` as additive (initially experimental).
- Add adapters:
  - `Taskset.from_dataset(...)`
  - `Taskset.to_dataset(...)`
  - `HarnessResult.to_state_patch(...)`
- Keep `example_id` compatibility but start writing `task_id` in parallel.

---

## Immediate implementation sketch

1. Add minimal `tasksets` package with `Task`, `Taskset`, `HarborTaskset` placeholders.
2. Add minimal `harnesses` package with `Harness`, `HarnessResult`, `OpenCode` placeholder.
3. Add experimental `AgentEnv` class in Verifiers:
   - constructor accepts `taskset`, `harness`, optional `rubric`
   - base rubric compiled if `rubric` omitted
   - rollout calls harness and merges result into state
4. Add docs + examples for `AgentEnv(taskset=..., harness=...)` and direct harness execution.

---

## Open questions to settle early

- Should `RunContext` live in `harnesses` as a hard type, or stay protocol-only initially?
- Should `Taskset` be fully materialized, iterable lazy, or both?
- Where does token-usage accounting live when harness can run outside Verifiers?
- What is the minimum required `HarnessResult` schema for stable orchestration?
- Should Harbor-specific grading live in `HarborTaskset`, `OpenCode`, or compiled rubric helpers?

---

## Recommended defaults

- **Task IDs are required and stable**.
- **Harness result schema is strict**.
- **Compiled base rubric is visible and inspectable**.
- **Taskset metadata is declarative only**.
- **AgentEnv is additive and experimental until two production tasksets migrate cleanly**.
