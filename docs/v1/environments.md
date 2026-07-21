# Multi-agent environments

One eval rollout doesn't have to be one agent run. `Environment` is abstract, and
every run gets a concrete subclass: plain tasksets resolve to the bundled
`SingleAgentEnv` (one `agent` playing the taskset), and a package can export
its own (via `__all__`, alongside its [`Taskset`](tasksets.md) — the same plugin
idiom as a bundled harness). An env declares its config as an `EnvConfig` subclass —
each agent an `AgentConfig` field, plus its own knobs — writes `rollout()`, and
optionally overrides `brief()` and `score()`:

```python
class DebateConfig(vf.EnvConfig):
    pro: vf.AgentConfig = vf.AgentConfig()
    con: vf.AgentConfig = vf.AgentConfig()
    judge: vf.AgentConfig = vf.AgentConfig(model="openai/gpt-5-mini")


class VerdictTask(vf.Task):
    @classmethod
    def from_traces(cls, task: vf.Task, pro: vf.Trace, con: vf.Trace) -> "VerdictTask":
        prompt = (
            f"Question: {task.data.prompt_text}\n\n"
            f"PRO argued:\n{pro.last_reply}\n\nCON argued:\n{con.last_reply}\n\n"
            "Who won? Reply with exactly 'pro' or 'con'."
        )
        return cls(vf.TaskData(idx=task.data.idx, prompt=prompt))


class DebateEnv(vf.Environment[DebateConfig]):
    def brief(self, agents: vf.Agents) -> None:
        """Per-agent standing the env hardcodes: the judge grades the debate,
        so its tokens are never training data."""
        agents.judge.trainable = False

    async def rollout(self, task: vf.Task, agents: vf.Agents) -> None:
        """How the agents interact on one task: imperative Python over Agent
        values. A loop is rounds, a TaskGroup is fan-out, a Task classmethod is
        chaining. Returns nothing — every finished run joins the episode
        automatically, stamped with its standing."""
        pro, con = await asyncio.gather(agents.pro.run(task), agents.con.run(task))
        await agents.judge.run(VerdictTask.from_traces(task, pro, con))

    async def score(self, task: vf.Task, traces: list[vf.Trace]) -> None:
        """Sibling-dependent judgement over the finished traces (per-trace
        judgement already ran on each trace's own task); each trace's
        `agent_name` stamp names its agent. Attach via record_reward/record_metric."""
        by_agent = {t.agent_name: t for t in traces}
        winner = (by_agent["judge"].last_reply or "").strip().lower()
        by_agent["pro"].record_reward("won", float(winner == "pro"))
        by_agent["con"].record_reward("won", float(winner == "con"))
```

- **The declared fields ARE the agents.** Every top-level `AgentConfig` field on the
  env's config plays under its field name — the config is the only naming site, so
  there is no separate declaration to drift from what `rollout()` actually does.
  The base scrapes them into the attribute-addressed `Agents` container
  (`agents.judge`) it hands the hooks.
- **Agents are typed config** (`Environment[DebateConfig]` binds it; `self.config`
  reads it), so the CLI addresses them for free: `--env.pro.model ...`,
  `--env.judge.client.base_url ...`, `--env.con.max_turns 4` — the framework
  narrows the run's `env` field to the selected env's config class by the env id
  (else the taskset id), and a partial override deep-merges with the declared
  default (`--env.judge.sampling.temperature 0` doesn't reset the judge's pinned
  model). An `AgentConfig`'s model context defaults to the run's own —
  `AgentConfig()` is "the policy under evaluation/training" (the serve protocol
  carries model/client/sampling per rollout request, which is what makes self-play
  trainable). Its harness does not: an unpinned agent runs the taskset's default
  harness (its bundled one, else `bash`) — there is no run-level harness. An agent
  pins only what makes it a different actor: its own harness or runtime
  (`--env.judge.harness.runtime.type docker`), a frozen model, an off-train
  endpoint, tighter limits. Per-run caps (turns, tokens, the
  setup/rollout/finalize/scoring timeouts) and whole-run `retries` live only on
  agents; the env keeps just its own hooks' bounds (`--env.timeout.episode` for
  `rollout()`, `--env.timeout.score` for `score()`) and its own coarse fallback
  `--env.retries` (below).
- **Task x agent fit validates on ground truth, per run.** Tasks require (declared
  `tools`, `NEEDS_CONTAINER`), harnesses support — and `Agent.run` checks the pair
  on every task it's actually given, before any work. An env-minted task carries
  its own needs, which is why a bare verdict task pairs the judge with *any*
  taskset; the taskset's shared tool servers ride only its own tasks (a run may
  pass `shared_tools=` to override). `SingleAgentEnv` still refuses an impossible
  pairing at construction: its one agent definitionally plays the taskset, so the
  mismatch is knowable before any rollout.
- **`brief()` is env truth, not config.** Whether an agent trains is decided by the
  env's design — a judge that grades the policy must never be trainable, no matter
  what a run config says — so it is set in place on the initialized agents
  (default: everyone trains) rather than exposed as a per-agent knob. An env that
  legitimately wants the flip exposes its *own* switch: the proposer-solver
  example's `--env.train_solver false` is a config field its `brief()` consults.
- **The base builds the agents** — one per field, fresh for every env-rollout,
  riding the eval's serving resources (shared interception pool, shared tool
  servers, per-endpoint clients — all env-owned and borrowed, so an agent is a
  cheap per-rollout value and concurrent episodes share no agent state) — and
  hands them into `rollout()`. The hook never constructs agents.
- **Traces are flat and self-contained; the episode is the durability atom.**
  Every trace carries its own `EpisodeInfo` stamp (`trace.episode` — id, env,
  episode-level errors; siblings share it) next to its `agent` info (name,
  trainability), so a flat bag of traces reconstitutes its episodes without a
  nested schema. On disk, one env-rollout is one `traces.jsonl` line (an
  `EpisodeRecord`: the episode's traces plus the shared stamp), so an episode
  persists whole or not at all — a torn line is the whole episode owed on resume,
  and a failure before any trace minted still leaves its errors on disk. An agent
  failure is data on its trace (the hook decides what a failed participant
  means); an exception in `rollout()`/`score()` is the env-rollout failing, and
  every trace that completed before it is still captured.
- **Retries are agent-first.** Each agent reruns its own rollout while its trace
  ends with a retryable error (`--env.judge.retries.max_retries 2` retries a
  flaky judge without re-burning the solver). The env-level `--env.retries.*` is
  the opt-in coarse fallback for faults no agent owns — the env's own hooks,
  cross-agent state — and reruns the episode whole: a half-played sibling context
  isn't reproducible.
- **Cross-agent signals can be declarative.** The default `score()` runs the env's
  own decorated `@vf.reward`/`@vf.metric` methods: each is invoked once per target
  trace and records there, with the finished set in reach (`trace` — the target,
  `traces` — every trace in the episode, `task`). `agent=` narrows the targets to
  one agent's traces; unset means every trace (a shared team signal). The bundled
  best-of-n's whole judgement is two such metrics:

  ```python
      @vf.metric
      async def pass_at_n(self, trace, traces):
          return float(max(t.reward for t in traces) >= self.config.threshold)
  ```

  Override `score()` for imperative control (dynamic names or weights,
  parse-and-fail — the bundled agentic-judge env, or the debate verdict above);
  `await super().score(task, traces)` keeps the decorated ones running.
- `setup()`/`teardown()` hooks bracket the serving lifetime for env-owned shared
  resources. Only `eval` runs multi-agent envs; `gepa` and `replay` refuse
  anything but `SingleAgentEnv`.

For the single-agent case none of this is machinery the user sees: `SingleAgentEnv`
declares one `agent` (`--env.agent.harness.id codex`, `--env.agent.max_turns 20`),
`rollout()` is `await agents.agent.run(task)`, and the episode carries exactly one
trace.

The run's `[env]` block is the whole run — the env is the encompassing entity,
composing three separately-chosen concerns:

- **`env.taskset`** — *what to solve*: the seed rows every rollout starts from,
  their data, their per-trace judgement (`--env.taskset.id`, or the positional
  `eval <taskset-id>`).
- **each agent's `harness`** — *how that LLM interfaces with the world*: the
  program driving model calls, tools, a runtime — pinned per agent, never a
  run-wide flag.
- **the env itself** — *the control flow between agents*: who runs, in what order,
  judged how across the finished set (`--env.id`).

## Reusable envs: `--env.id`

An interaction pattern that isn't specific to one dataset — n attempts, a judge, a
modeled user — is its own plugin, paired with any taskset from the CLI:

```bash
uv run eval gsm8k-v1 --env.id best-of-n --env.n 8
uv run eval my-task-v1 --env.id agentic-judge --env.judge.harness.runtime.type docker
```

The same pairing as TOML — `env.id` plus one `[env.<agent>]` block per agent — is
checked in as `configs/agentic_judge.toml` (`uv run eval @ configs/agentic_judge.toml`).

`--env.id` resolves like every plugin id — a bundled env (below), a local package
exporting an `Environment` subclass via `__all__`, or a Hub `org/name[@version]` —
and its `EnvConfig` surface typed on the CLI (`--env.<agent>.*`, `-h` renders
them). Empty (the default) keeps the taskset's own story: the env its package
ships (a *recipe* env like `code_golf_v1`, where the interaction is intrinsic to
the data), else `SingleAgentEnv`. An explicit id wins over a bundled recipe env.

Bundled envs (`verifiers/v1/envs/`):

| id | agents | what it does |
| --- | --- | --- |
| `best-of-n` | `agent` | `--env.n` independent attempts per rollout; `score()` marks the argmax-reward sibling (`best`) and whether any reached `--env.threshold` (`pass_at_n`) — rejection sampling and pass@k. A single-agent env keeps the single-agent name, so `--env.agent.*` flags compose unchanged. |
| `agentic-judge` | `solver`, `judge` | agent-as-judge: the solver plays the task; a code-executing judge agent verifies the finished attempt with real execution, always in its own sandbox, never on the host. The judge's task mirrors the solver task's world (same image, a fresh box in its original state) with the graded transcript uploaded (`/tmp/transcript.md`/`.json`). The verdict channel is a file: the judge writes `{"score": 0-10, "reasoning": ...}` to `/tmp/verdict.json` in its box, scraped onto its trace while the box is alive and validated STRICTLY onto the solver's trace as the `judge` reward — a missing, malformed, or off-scale verdict fails the rollout instead of clamping. The judge must land in a container: pin `--env.judge.harness.runtime.type docker\|prime`, or construction refuses. A judgement that needs no execution belongs on the plugged tier (`env.taskset.task.judges`), not on an agent. |
