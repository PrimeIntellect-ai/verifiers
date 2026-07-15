# Deferred topology work

This is the living list of topology work that is known, intentionally deferred, and not
required for the first trainable `AgentGraph` path. It records gaps so they do not become
implicit contracts or disappear into follow-up discussions.

It does not track work that is already complete: topology selection is shared through
`EnvConfig`, taskset and harness syntax lowers to the built-in single-agent topology, one
invocation returns and persists one `AgentGraph`, and native v1 group rewards and `Episode`
have been removed.

## Execution and scalability

### Choose the interception at rollout time, not config time

The run's one `Interception` is built in `TopologyRunner.serving()` with a single
run-level `requires_tunnel` verdict, decided before any instance runs. That forces a
prediction: agent placements are knowable from config, but the tool/user servers of
tasks minted inside `run()` are not — `taskset_server_configs()` (env.py) guesses from
the *seed* taskset's task class and conservatively assumes remote when the pairing
isn't statically derivable. Two consequences: in a mixed placement (remote solver +
local judge) the local agent's model calls transit the tunnel needlessly, and an
all-local topology whose `run()` mints a task with a remote tool server gets a
localhost-only interception that server cannot reach.

The agreed future shape: `serving()` holds *two* interception pools built from the same
`EnvConfig.interception` config — one plain, one tunneled — and each `Rollout` picks at
acquire time from facts it already has in hand (its actual runtime plus its actual
tool/user servers; the exact `requires_tunnel` computation its no-injection fallback
already performs). The default `elastic` shape spawns servers only on acquire, so an
all-local run never mints a tunnel and the idle pool costs nothing; the `server`/`static`
shapes need an explicit don't-start-until-first-acquire tweak. This deletes
`taskset_server_configs()` and the whole statically-knowable hedge, and is really an
upstream interception-layer improvement (main's `Environment` used the same static
logic) — best done as its own small PR, coordinated with main.

Done when the guesser is gone, a mixed-placement run keeps local agents on loopback
(tested), and the minted-remote-tool case is served correctly.

### Account for variable topology cost

The server currently charges every topology invocation as one unit of capacity regardless of
how many agents, turns, fan-outs, runtimes, or model calls it creates. This preserves the simple
and scalable existing server lifecycle, but intentionally leaves scheduling blind to internal
graph cost.

Defer cost-aware admission, per-agent/model telemetry, and dynamic capacity accounting until
there is enough operational evidence to choose a useful abstraction.

### Retry whole instances, not just rollouts

`retries.rollout` reruns a single failed agent run, but there is no graph-level analogue:
an invocation that comes out invalid (by the topology's own `Topology.complete()` verdict)
is persisted as-is and only redone by a later `--resume`. `Topology.complete()` is already
the right predicate for an online instance-level retry (`retries.instance`), and the
`RetryConfig` wrapper deliberately keeps that slot open. Defer until a real training or
eval run demonstrates the need; when added, cap and attribute retries so a topology whose
instances are *never* complete fails loudly instead of looping.

### Right-size graph wire payloads

`RunResponse` now serializes graphs with full per-node training tensors
(`routed_experts`, `multi_modal_data`) so the trainable path survives the env-server hop —
correct for training, but eval-mode server runs now ship tensors nobody reads, inflating
ZMQ payloads and decode time for large graphs. Defer until payload size shows up in
practice; the likely shape is a per-request or per-server "strip training tensors" switch
mirroring what `to_record()` already does for JSON persistence.

### Add explicit run-level state if needed

There is no canonical state store for information that must survive beyond one
`run(task, agents)` invocation (cumulative match results, cross-instance curricula).
Storing mutable data on a `Topology` or `TopologyRunner` instance is not a contract —
`setup()`/`teardown()` scope per-worker resources, not authoritative shared state:
workers have separate copies, processes restart, and distributed execution cannot observe
a single authoritative value.

If this becomes necessary, define the ownership, persistence, consistency, and worker-sharing
semantics explicitly. Do not accidentally introduce run-level state through topology object
mutation.

## Self-contained graph semantics

### Persist the topology completion verdict

`Topology.complete(graph)` lets a topology accept a graph whose child traces include handled
failures. Eval resume is trace-flat and does not consume this verdict today; in-process
callers and future graph-native resume should. Stamp the topology's final completion verdict
onto the returned `AgentGraph` after topology scoring so consumers can read a self-contained
verdict without loading topology code.

### Recover relative signals as a built-in best-of-N topology

Deleting `@group_reward` removed the only native home for pairwise/relative eval signals
(shortest-of-group, preference comparisons); they were demoted to per-trace metrics with
the comparison left to "the training algorithm". For *training* that is the right owner,
but eval-mode relative signals have a clean replacement the framework should ship rather
than describe: a built-in best-of-N topology — one seed, `run` fans out N runs of the same
agent, a `@reward(agent=...)` compares `graph.children(...)`. That recovers everything
group rewards did, explicitly, on the canonical path, and doubles as the documented
migration story for the removal.

Done when a `best-of-n` built-in exists and the docs name it as the group-reward successor.

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

`push_traces()` receives flattened graph traces and uploads each as an independent rollout.
Explicit topologies force `--no-push` today (and still reject `--resume`), so multi-agent
graphs are not pushed. Prefer a graph-native upload contract before re-enabling push for
`--topology.id`.

## Configuration boundaries

### Normalize agent execution configuration before running

Agent model, client, and sampling settings are currently resolved while binding a topology run:
agent-specific values override request-level defaults. The behavior is correct, but the layered
fallbacks remain visible in execution code and leave more than one representation of a resolved
agent configuration.

A future cleanup can introduce one normalized, fully resolved agent execution context before any
agent runs. This is not required for correctness and should not be combined with unrelated server
lifecycle changes.

### Settle the config layer's structure

A reviewed set of config cleanups, deferred as one pass (analysis 2026-07-11):

- **Split `topology.py` into its authoring and runner halves.** The authoring surface
  (`AgentConfig`, `TopologyConfig`, `AgentGraph`, `Topology`, `SingleAgentTopology`,
  `graph_complete`) imports nothing from `env.py`; only the runner half does. Splitting
  turns the layering into a clean DAG (`topology ← env ← runner`) and deletes the
  forward-referenced `EnvConfig.topology` annotation plus the `model_rebuild()` call that
  currently papers over the cycle — the one trick in the config layer.
- **Declare `RolloutLimits` once.** `EnvConfig` carries four flat `max_*` fields that are
  hand-copied into `RolloutLimits` at the runner; a `limits` property on `EnvConfig`
  states the correspondence once while keeping the flat CLI flags.
- **Align timeout vocabulary.** `--timeout.rollout` maps onto the task-authored
  `TaskTimeout.harness`; post-consolidation "rollout" is the word — rename the task field
  while it is still a five-file change.
- **Delete the dead `EnvConfig.env_id` property** (no v1 consumers) and, riding the
  split, consider moving the `EnvConfig` family into `configs/` so that package holds all
  process-level configs (plugin configs stay colocated with their plugin classes — that
  idiom is deliberate and load-bearing for id-narrowing).

Explicitly considered and rejected (do not re-open without new evidence): nesting
`model`/`client`/`sampling` into a ModelContext-shaped sub-config (kills `-m` /
`--topology.<agent>.model` ergonomics), a shared CLI mixin for `verbose`/`dry_run`
(saves ~6 lines, adds a class), quarantining the legacy `--id` block (it *is* the compat
interface), and dropping the `group_size`/`rollouts_per_example` aliases (trainer-facing).

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

### Reconcile PR #1939 (agent programs)

The standalone Agent-facade PR (#1939, open against main) is the ancestor of this branch's
agent layer. Its machinery is fully absorbed or deliberately superseded here (interception ownership
moved to the runner's serving scope; `TaskData.sources`/`relation` lineage replaced by trace-field links;
the per-run `ctx=` override replaced by per-agent routing), and its three correctness fixes
(borrowed-box tombstone, mid-run teardown attribution, resolved-runtime pairing) have been
ported. What remains unique to it is `docs/v1/agent-programs.md` plus four
`examples/agent_programs/` scripts — written against its API variant, so merging it as-is
would reintroduce a conflicting Agent surface and a duplicate lineage mechanism.

Done when the bare-`vf.Agent` scripting surface is documented against *this* branch's API
(an agent-programs page or a section of the topology docs, examples rewritten) and #1939 is
closed or rebased to nothing.
