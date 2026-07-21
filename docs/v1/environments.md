# Multi-agent environments

One eval rollout doesn't have to be one agent run. `Env` is abstract, and every
run gets a concrete subclass: plain tasksets resolve to the bundled
`SingleAgentEnv` (one `agent` playing the taskset), and a package can export its
own (via `__all__`, alongside its [`Taskset`](tasksets.md) — the same plugin
idiom as a bundled harness). An env declares its config as an `EnvConfig`
subclass — each agent an `AgentConfig` field, plus its own knobs — writes
`run()`, and optionally overrides `setup()` and `finalize()`:

```python
class DebateConfig(vf.EnvConfig):
    pro: vf.AgentConfig = vf.AgentConfig()
    con: vf.AgentConfig = vf.AgentConfig()
    judge: vf.AgentConfig = vf.AgentConfig(model="openai/gpt-5-mini")


class JudgeTask(vf.Task):
    @classmethod
    def from_traces(cls, task: vf.Task, pro: vf.Trace, con: vf.Trace) -> "JudgeTask":
        """Traces -> the judge's task: a plain minted row."""
        prompt = (
            f"Question: {task.data.prompt_text}\n\n"
            f"PRO argued:\n{pro.last_reply}\n\nCON argued:\n{con.last_reply}\n\n"
            "Who won? Reply with exactly 'pro' or 'con'."
        )
        return cls(vf.TaskData(idx=task.data.idx, prompt=prompt))


class DebateEnv(vf.Env[DebateConfig]):
    def setup(self, agents: vf.Agents):
        """Per-agent standing the env hardcodes: the judge grades the debate,
        so its tokens are never training data."""
        agents.judge.trainable = False

    async def run(self, task: vf.Task, agents: vf.Agents):
        """How the agents interact on one task: imperative Python over Agent
        values. A loop is rounds, asyncio.gather is fan-out, a Task classmethod
        from traces is chaining. Every finished run joins the episode's traces
        automatically, stamped with its agent's name and the shared EpisodeInfo."""
        pro, con = await asyncio.gather(agents.pro.run(task), agents.con.run(task))
        await agents.judge.run(JudgeTask.from_traces(task, pro, con))

    async def finalize(self, task: vf.Task, traces: list[vf.Trace]):
        """Cross-trace judgement over the finished traces (per-trace judgement
        already ran on each trace's own task); each trace's `agent.name` stamp
        names its agent. Record results in place."""
        by_agent = {t.agent.name: t for t in traces}
        winner = (by_agent["judge"].last_reply or "").strip().lower()
        by_agent["pro"].record_reward("won", float(winner == "pro"))
        by_agent["con"].record_reward("won", float(winner == "con"))
```

- **Agents are typed fields on the env's config** (`Env[DebateConfig]` binds
  it; `self.config` reads it), so the CLI addresses them for free:
  `--env.pro.model ...`, `--env.judge.client.base_url ...`, `--env.con.max_turns 4` —
  the framework narrows the run's `env` field to the selected env's config class by
  the env id (else the taskset id), and a partial override deep-merges with the
  declared default (`--env.judge.sampling.temperature 0` doesn't reset the
  judge's pinned model). An `AgentConfig`'s **model context** defaults to the run's own —
  `AgentConfig()` is "the policy under evaluation/training" (the serve protocol
  carries model/client/sampling per rollout request, which is what makes self-play
  trainable). Its **harness** does not: an unpinned agent runs the taskset's default
  harness (its bundled one, else `bash`) — there is no run-level harness. An agent
  pins only what makes it a different actor: its own harness or runtime
  (`--env.judge.harness.runtime.type docker`), a frozen model, an off-train
  endpoint, tighter limits — and a declared pin is the env author's per-agent
  default. Per-run caps (turns, tokens, the setup/rollout/finalize/scoring
  timeouts, whole-run retries) live only on agents — there is no env-level cap;
  the env keeps just its own hooks' bounds (`--env.timeout.episode` for one
  episode's `run()`, `--env.timeout.finalize` for `finalize()`).
- **The declared fields ARE the agents.** The base scrapes every `AgentConfig` off
  the config — recursively: a field is an `Agent` under its field name, a list of
  `AgentConfig`s an index-addressable list — into the `Agents` container that
  `run()` receives (`agents.pro`, `agents.solvers[0]`). The config is the only
  naming site, so there is no separate declaration to drift from what `run()`
  actually does.
- **Tasks mint from traces via classmethods.** An env that chains one agent's
  output into another's task gives the downstream `Task` subclass a
  `from_trace(...)` classmethod (`from_traces(...)` when it reads several) — the
  minted row is ordinary `TaskData`, and the task carries its own judgement and
  needs (`tools`, `NEEDS_CONTAINER`, `setup`/`finalize` hooks).
- **Task x agent fit validates on ground truth, per run.** Tasks require (declared
  `tools`, `NEEDS_CONTAINER`), harnesses support — and `Agent.run` checks the pair
  on every task it's actually given, before any work
  (`verifiers/v1/utils/compile.py`). An env-minted task carries its own needs,
  which is why a bare verdict task pairs a grader with *any* taskset; the taskset's
  shared tool servers ride only its own tasks (a run may pass `shared_tools=` to
  override). There is deliberately no upfront compilation: a misconfigured agent
  surfaces at runtime, on its own run.
- **`setup()` is env truth, not config.** Whether an agent trains is decided by the
  env's design — a grader that grades the policy must never be trainable, no matter
  what a run config says — so it is set in place on the constructed agents
  (default: everyone trains) rather than exposed as a per-agent knob. An env that
  legitimately wants the flip exposes its *own* switch: the proposer-solver
  example's `--env.train_solver false` is a config field its `setup()` consults.
- **The base builds the agents** — fresh for every env-rollout, riding the eval's
  serving resources (shared interception pool, shared tool servers, per-endpoint
  clients — all env-owned and borrowed, so an agent is a cheap per-rollout value
  and concurrent episodes share no agent state) — and hands them into `run()`. The
  hook never constructs agents.
- **One env-rollout is one episode — a flat `list[Trace]`.** Every completed run's
  trace joins it in completion order, and each trace is self-contained: its `agent`
  info carries the agent name and trainability, and its `episode` stamp
  (`EpisodeInfo`) the episode id and env id — so a flat bag of traces (a
  `traces.jsonl`, a serve response) reconstitutes its episodes with no nested
  schema, and the trace API ingests them natively. Episodes succeed and resume
  as a unit; retries are per agent run (`--env.<agent>.retries`), never of a
  whole episode. An agent failure is data on its trace (the hook decides what a
  failed participant means); an exception in `run()`/`finalize()` is the env-rollout
  failing — it lands on the shared `EpisodeInfo.errors`, mirrored on every trace
  that completed before it.
- **Cross-agent judgement is `finalize()`.** It sees the flat finished set and
  mutates traces in place — sibling comparisons (the bundled best-of-n's `best`/
  `pass_at_n`), verdict parsing (solver-grader), shared team signals. Bounded by
  `--env.timeout.finalize`.

The judge above is the pattern the bundled `solver-grader` env productionizes:
pair it with any taskset and a code-executing grader verifies the solver's attempt
with real execution in its own sandbox — reach for it before writing a `JudgeTask`
of your own (see the bundled envs below). A judgement that needs no execution
doesn't need an agent at all: plug an LLM judge in as an `env.taskset.task.judges`
entry (one bare call inside `Task.score`).

For the single-agent case none of this is machinery the user sees: `SingleAgentEnv`
declares one `agent` (`--env.agent.harness.id codex`, `--env.agent.max_turns 20`),
its `run()` is `await agents.agent.run(task)`, and the episode is exactly one trace.

The run's `[env]` block is the whole run — the env is the encompassing entity, composing three separately-chosen concerns:

- **`env.taskset`** — *what to solve*: the seed rows every rollout starts from, their
  data, their per-trace judgement (`--env.taskset.id`, or the positional
  `eval <taskset-id>`).
- **each agent's `harness`** — *how that LLM interfaces with the world*: the program
  driving model calls, tools, a runtime — pinned per agent, never a run-wide flag.
- **the env itself** — *the control flow between agents*: who runs, in what order,
  judged how across the finished set (`--env.id`).

## Reusable envs: `--env.id`

An interaction pattern that isn't specific to one dataset — n attempts, a grader, a
modeled user — is its own plugin, paired with any taskset from the CLI:

```bash
uv run eval gsm8k-v1 --env.id best-of-n --env.n 8
uv run eval my-task-v1 --env.id solver-grader --env.grader.harness.runtime.type docker
```

The same pairing as TOML — `env.id` plus one `[env.<agent>]` block per agent — is
checked in as `configs/solver_grader.toml` (`uv run eval @ configs/solver_grader.toml`).

`--env.id` resolves like every plugin id — a bundled env (below), a local package
exporting an `Env` subclass via `__all__`, or a Hub `org/name[@version]` —
and its `EnvConfig` surface typed on the CLI (`--env.<agent>.*`, `-h` renders them).
Empty (the default) keeps the taskset's own story: the env its package ships (a
*recipe* env like `code_golf_v1` or the bundled `proposer-solver`, where the
interaction is intrinsic to the data), else `SingleAgentEnv`. An explicit id wins
over a bundled recipe env.

Bundled envs (`verifiers/v1/envs/`):

| id | agents | what it does |
| --- | --- | --- |
| `best-of-n` | `agent` | `--env.n` independent attempts per rollout; `finalize()` marks the argmax-reward sibling (`best`) and whether any reached `--env.threshold` (`pass_at_n`) — rejection sampling and pass@k. A single-agent-shaped env keeps the `agent` name, so `--env.agent.*` flags compose unchanged. |
| `solver-grader` | `solver`, `grader` | agent-as-judge: the solver plays the task; a code-executing grader verifies the finished attempt with real execution, always in its own sandbox, never on the host. The grader's task (`GradeTask.from_trace`) mirrors the solver task's world (same image, a fresh box in its original state) with the graded transcript uploaded (`/tmp/transcript.md`/`.json`); the grader writes its verdict to `/tmp/verdict.json`, which `GradeTask.finalize` scrapes onto its trace and the env's `finalize()` records on the solver's trace as the `grader` reward. Pin a container: `--env.grader.harness.runtime.type docker\|prime`. |

The task-generation shape (a proposer mints verified problems, n solvers race
them, `learnability` rewards the proposer where half succeed) ships as the
`proposer-solver-v1` example package under `environments/`.
