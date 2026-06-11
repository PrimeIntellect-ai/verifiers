# v1 PRs: #1559 vs #1576

Two open v1 refactors of verifiers (the `Taskset × Harness × Runtime` model), both against `main`.

|  | **#1559** — `codex/v1-nano-refactor-draft` (willccbb) | **#1576** — `feat/nano-as-v1` (mika) |
|---|---|---|
| Size | +17.6k / −34.1k, 422 files | +11.5k / −50k, 401 files (re-vendors vf-nano) |
| Thesis | Broad v1 surface — many harnesses, engine bridges, in-tree advantages, a bundled trainer | v0↔v1 bridge + training-readiness — legacy bridge, message graph, multiplexing, benchmarked |
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

- Runtime configs for **modal + daytona** (stubs — `NotImplementedError`)
- Harness ecosystem: **`CommandHarness`** (agentic-CLI base) + **MiniSWEAgent / OpenCode / Pi / Terminus2 / Replay / NeMoGym**
- **In-tree token-level advantages** — `@advantage` (grpo / rloo / reinforce / sft) writing `Turn.tokens.*_advantages`; `advantage=None` defers to a trainer
- **Nested harnesses / subagents** — `Harness.run(context=parent)` reuses the parent's runtime, clients, toolsets
- **Richer MCP** — placement `dedicated` / `colocated` / `remote` × scope `rollout` / `env` (refcounted, start-once) + **bound-arg tools** (`args`/`sets`/`extends`) that hide state plumbing from the model
- **External-engine tasksets** — OpenEnv, NeMo-Gym, OpenReward, TextArena, Harbor, Replay
- **~30 bundled v1 example envs** (bfcl, tau2, dspy, openai-agents, langchain, self-judge, parallel-sandbox, …)
- Bundled **"baby prime-rl" trainer** (`verifiers-rl`, `vf-rl`/`vf-train`) — targets the v0 `generate` API
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

## Validation done

**#1559** — CI green (ruff / ty / semgrep / codeql); ~110-test v1 unit suite (`test_v1_core.py`); live `prime eval` **reward matrix** (reverse-text 0.948, alphabet-sort 0.801, mcp-search 1.0, math-python 1.0, hello-group-reward 0.645, tau2 0.533; openenv echo/textarena sandbox-cleanup verified); prime-rl **smoke-load** of 2 envs. *No throughput benchmark; no RL reward-curve / parity numbers.*

**#1576** — ruff / ty + v0 CPU suite; v1 unit (graph / legacy / configs) + **6-test e2e matrix** (harness × runtime, asserts `reward==1`); eval reward (gsm8k 1.0, code_golf 1.5, glossary 1.0); **runtime+multiplex benchmark** with committed results (prime gsm8k n=32/64/128, 0 errors, gen-p50/p90 methodology); **training validation with numbers** — prime-rl native + bridge **parity** (reverse-text: v0-bridge 0.14→0.79, v1-native 0.08→0.76, 20 steps, 128/128 trainable, 0% error) and **LoRA** (alphabet-sort r32/a64, 0.43→0.46, 0% error/truncation). SWE `scaleswe-v1` ~50% solve on glm-5.1 (validated post-merge of #1616).

---

*Net:* **#1559** is the broader feature surface (harness/engine ecosystem, in-lib advantages, subagents, a bundled trainer), validated mainly by CI + an eval reward matrix. **#1576** is narrower but **bridges v0→v1 and is the more training/perf-validated** path (real reward curves, LoRA, throughput benchmark, e2e matrix).
