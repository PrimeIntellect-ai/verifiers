# v1 PRs: #1559 vs #1576

Two open v1 refactors of verifiers (the `Taskset × Harness × Runtime` model), both against `main`.

|  | **#1559** — `codex/v1-nano-refactor-draft` | **#1576** — `feat/nano-as-v1` |
|---|---|---|
| Code diff (`verifiers/`) | +6,906 / −11,817, 69 files | +6,010 / −13,585, 93 files (re-vendors vf-nano) |
| Thesis | Broad v1 surface — many harnesses, in-tree advantages, nested subagents | v0↔v1 bridge + training-readiness — legacy bridge, message graph, multiplexing, benchmarked |
| Rollout record | `State` + flat `Turn` list (serializable, no graph) | delta-native `MessageNode` graph (branches via leaves→root) |
| RL contract | token-level **advantages computed in-lib** (`@advantage`) | trainer (prime-rl) computes advantages; lib exposes trainable `Trace` |

## Parity — supported by both

- Core `Taskset × Harness × Runtime` over a typed (pydantic) rollout model
- Runtimes: **subprocess, docker, prime**
- Harnesses: a **default chat harness** + **rlm**
- Taskset authoring: `load_tasks` + `@reward` / `@metric` / `@stop`, **group rewards**, **runtime-based (in-sandbox) scoring**, per-task **image + resources**
- **MCP tool servers** exposed to the model + a first-class **user simulator** (framework-injected user turns)
- **Eval CLI + TOML config** (runtime/harness selected by config)
- **Trainable rollouts** — per-turn token ids + logprobs + mask
- **Interception server** proxying model calls; SIGTERM → graceful teardown
- v1 **unit tests** + live **eval reward** checks

## Only in #1559

- Harness ecosystem: **`CommandHarness`** (agentic-CLI base) + **MiniSWEAgent / OpenCode / Pi / Terminus2 / Replay / NeMoGym**
- **In-tree token-level advantages** — `@advantage` (grpo / rloo / reinforce / sft) writing `Turn.tokens.*_advantages`; `advantage=None` defers to a trainer
- **Nested harnesses / subagents** — `Harness.run(context=parent)` reuses the parent's runtime, clients, toolsets
- **Richer MCP** — placement `dedicated` / `colocated` / `remote` × scope `rollout` / `env` (refcounted, start-once) + **bound-arg tools** (`args`/`sets`/`extends`) that hide state plumbing from the model
- **Replay harness** (SFT)

## Only in #1576

- **Legacy v0 bridge** (`LegacyEnvServer`) — runs classic v0 envs over the **same ZMQ protocol** as native v1, indistinguishable to the trainer; token ids/logprobs carried 1:1; group scoring; eval-split fallback; renderer (train) vs chat-completions (eval) client dispatch. *The headline ("nano bridge").*
- **Delta-native message graph** — each message stored once, branches recovered leaves→root (linear, not quadratic in turns); one training sample per branch
- **Interception multiplexing** (`InterceptionPool`, `multiplex=32`) — N rollouts share servers + tunnels, to beat prime's 512/min tunnel cap
- **ZMQ env server is the v1 training path** (native + bridge both serve over it); prime-rl drives it [#1559's ZMQ is v0-only; its v1 eval runs in-process]
- **Modal runtime functional** (4 working runtimes vs 3 + 2 stubs)
- Framework-enforced limits (`max_turns` / token budgets / `@stop`) applied **harness-agnostically** in the interception server
- **Runtime + multiplex benchmark** (`bench/`) with committed numbers

> Excluded from #1576's tip via reverts (in separate review — verifiers#1618): multimodal/VLM, user-sim colocation, color-codeword taskset.

---

*Net:* **#1559** is the broader feature surface (harness ecosystem, in-lib advantages, nested subagents, richer MCP). **#1576** is narrower but is the only one that bridges v0→v1.
