# Deferred topology work

This is the living list of topology work that is known, intentionally deferred, and not
required for the first trainable `AgentGraph` path. It records gaps so they do not become
implicit contracts or disappear into follow-up discussions.

It does not track work that is already complete: topology selection is shared through
`EnvConfig`, taskset and harness syntax lowers to the built-in single-agent topology, one
invocation returns and persists one `AgentGraph`, and native v1 group rewards and `Episode`
have been removed.

## Execution and scalability

### Bound session-driven model concurrency

`TopologyRun.run_agent()` acquires the configured rollout semaphore, but
`TopologyRun.interact_agent()` deliberately bypasses it. In-process evaluation also starts all
topology invocations eagerly. A session topology can therefore create unbounded model pressure
even when `max_concurrent` is configured. Server-backed execution is bounded by the outer graph
request pool, but that does not bound turns or agents inside a graph.

The eventual solution must limit active model work without holding a permit for an entire
suspended session, which could deadlock multi-seat topologies. A turn-level permit, or another
model-call-level admission mechanism, is preferable to a session-lifetime permit.

Done when direct and server-backed execution have an explicit, tested concurrency bound for
session topologies without deadlocking games or debates.

### Account for variable topology cost

The server currently charges every topology invocation as one unit of capacity regardless of
how many agents, turns, fan-outs, runtimes, or model calls it creates. This preserves the simple
and scalable existing server lifecycle, but intentionally leaves scheduling blind to internal
graph cost.

Defer cost-aware admission, per-agent/model telemetry, and dynamic capacity accounting until
there is enough operational evidence to choose a useful abstraction.

### Add explicit run-level state if needed

There is no canonical state store for information that must survive beyond one `go(task)`
invocation, such as cumulative chess wins and losses. Storing mutable data on a `Topology` or
`TopologyRunner` instance is not a contract: workers have separate copies, processes restart,
and distributed execution cannot observe a single authoritative value.

If this becomes necessary, define the ownership, persistence, consistency, and worker-sharing
semantics explicitly. Do not accidentally introduce run-level state through topology object
mutation.

## Self-contained graph semantics

### Persist the topology completion verdict

`Topology.complete(graph)` lets a topology accept a graph whose child traces include handled
failures. In-process resume can call the live topology and respects this policy. Server-backed
resume and other consumers without the topology object fall back to conservative
`graph_complete()`, so they can redo a graph the topology already considered valid.

Stamp the topology's final completion verdict onto the returned and persisted `AgentGraph` after
topology scoring. Consumers should read that self-contained verdict; older records without it
can retain the conservative fallback.

Done when resume produces the same decision in process, through an environment server, and from
a graph record alone.

## Replay, presentation, and platform upload

### Make replay graph-native

Replay is trace-native today. It flattens graph records, assumes the seed taskset's one task
type, invokes only `Task.score()`, and writes flat trace records. It cannot reconstruct topology
judgement or correctly replay derived task types produced by other agents.

The short-term safe behavior is to reject explicit-topology replay clearly rather than emit a
plausible but incorrect result. Full support should load one `AgentGraph` at a time, recover each
trace's task behavior, recompute eligible task rewards, run topology-level scoring, and persist
another graph.

### Make the evaluation dashboard graph-aware

The dashboard still presents explicit topologies through taskset and harness fields and streams
individual traces as though they were independent invocations. It does not expose topology id,
graph-level completion/error state, agent roles, parent links, or the distinction between one
graph and its internal traces.

Until the dashboard has a graph view, it should avoid silently describing an explicit topology
as a single taskset-harness rollout.

### Make platform push graph-native

`push_traces()` receives flattened graph traces and uploads each as an independent rollout. This
loses graph identity, topology id, agent and parent structure, graph completion semantics, and
correct replica numbering. Multi-agent topologies are consequently misrepresented on the
platform.

Prefer a graph-native upload contract. Until the platform can accept it, explicit-topology push
should fail or warn clearly instead of silently flattening graphs.

## Configuration boundaries

### Normalize agent execution configuration before running

Agent model, client, and sampling settings are currently resolved while binding a topology run:
agent-specific values override request-level defaults. The behavior is correct, but the layered
fallbacks remain visible in execution code and leave more than one representation of a resolved
agent configuration.

A future cleanup can introduce one normalized, fully resolved agent execution context before any
agent runs. This is not required for correctness and should not be combined with unrelated server
lifecycle changes.

## Compatibility isolation

### Separate the legacy protocol from native v1

Native v1 execution uses only `run -> AgentGraph`, but the public `EnvClient` and shared protocol
module still expose `run_rollout`, `run_group`, `requires_group_scoring`, and their legacy
request/response types. This is intentional compatibility leakage, not part of the native graph
contract.

Move the bridge routes and types behind an explicitly legacy client/module, or remove them when
v0 support is retired. Native imports and generated client interfaces should then expose only the
graph route.

## Documentation and skills

The topology authoring guide is substantially in place, but the v1 overview, architecture,
evaluation docs, and evaluation skills still describe taskset plus harness as the primary
execution model. They do not consistently explain the canonical lowering to a topology, the
`Topology.complete()` contract, or the graph limitations of replay, dashboard, and push.

Update these surfaces after the corresponding contracts settle. Documentation should describe
the current model only, not preserve a migration narrative.
