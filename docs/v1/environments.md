# Multi-agent environments

One eval rollout doesn't have to be one agent run. `Environment` is a concrete class
whose defaults are the single-agent case; a package can export a subclass (via
`__all__`, alongside its [`Taskset`](tasksets.md) — the same plugin idiom as a bundled
harness) that declares its parameters as an `EnvParams` subclass and overrides up to
three methods:

```python
class DebateParams(vf.EnvParams):
    pro: vf.AgentConfig = vf.AgentConfig()
    con: vf.AgentConfig = vf.AgentConfig()
    judge: vf.AgentConfig = vf.AgentConfig(model="openai/gpt-5-mini", trainable=False)


def judge_task(task: vf.Task, pro: vf.Trace, con: vf.Trace) -> vf.Task:
    """Traces -> the judge's task: a plain minted row, lineage stamped."""
    prompt = (
        f"Question: {task.data.prompt_text}\n\n"
        f"PRO argued:\n{pro.last_reply}\n\nCON argued:\n{con.last_reply}\n\n"
        "Who won? Reply with exactly 'pro' or 'con'."
    )
    return vf.Task(
        vf.TaskData(
            idx=task.data.idx,
            prompt=prompt,
            sources=(pro.id, con.id),
            relation="judges",
        )
    )


class DebateEnv(vf.Environment[DebateParams]):
    def roles(self) -> dict[str, vf.Role]:
        """The topology: who plays which role, and what each needs. The debaters
        play the dataset; the judge grades an env-minted verdict task."""
        return {
            "pro": vf.Role(self.params.pro),
            "con": vf.Role(self.params.con),
            "judge": vf.Role(self.params.judge, mcp=False, container=False),
        }

    async def rollout(
        self, task: vf.Task, agents: Mapping[str, vf.Agent]
    ) -> list[vf.Trace]:
        """How the agents interact on one task: imperative Python over Agent values.
        A loop is rounds, asyncio.gather is fan-out, a function from traces to task
        data is chaining. The returned traces are the rollout's episode."""
        pro, con = await asyncio.gather(
            agents["pro"].run(task), agents["con"].run(task)
        )
        verdict = await agents["judge"].run(judge_task(task, pro, con))
        return [pro, con, verdict]

    async def score(self, task: vf.Task, traces: list[vf.Trace]) -> None:
        """Sibling-dependent judgement over the finished set (per-trace judgement
        already ran on each trace's own task). Attach via record_reward/record_metric."""
        pro, con, verdict = traces
        winner = (verdict.last_reply or "").strip().lower()
        pro.record_reward("won", float(winner == "pro"))
        con.record_reward("won", float(winner == "con"))
```

- **Roles are typed fields on the env's params block** (`Environment[DebateParams]`
  binds it; `self.params` reads it), so the CLI addresses them for free:
  `--env.pro.model ...`, `--env.judge.client.base_url ...`, `--env.con.max_turns 4` —
  the framework narrows the `env` field by taskset id exactly as it narrows
  `taskset`/`harness`, and a partial override deep-merges with the declared role
  default (`--env.judge.sampling.temperature 0` doesn't reset the judge's pinned
  model). An `AgentConfig`'s every field defaults to the run's own settings —
  `AgentConfig()` is "the policy under evaluation/training" (which is what makes
  self-play trainable); a role pins only what makes it a different actor (its own
  harness, a frozen model, an off-train endpoint, tighter limits, `trainable=False`).
- **The 1:1 mapping is the default.** With no `roles()` override, every declared
  `AgentConfig` field plays the dataset under its field name — an env whose roles
  all need exactly what the taskset provides (self-play, fan-out like the bundled
  best-of-n) never writes `roles()` at all. `DebateEnv` overrides it for one
  reason only: the judge's needs differ.
- **A role declares what it needs from the taskset's world.** `vf.Role(cfg)`
  plays the dataset: the taskset's needs apply (declared tools mean the role's
  harness must support MCP; `NEEDS_CONTAINER` means no subprocess runtime), and the
  role is handed the taskset's shared tool servers. A role whose tasks the env
  mints itself says so — `vf.Role(cfg, mcp=False, container=False)` for a bare
  model actor like a judge or a simulated user — and then pairs with *any* taskset.
  Needs can also be computed: the bundled judge env decides `container=` from
  whether its judge's harness executes code. Keeping the declaration honest with
  `rollout()` is the env author's job; `Agent.run` still validates every concrete
  task it's given, as the backstop.
- **The base builds the agents** — one per role, inside the eval's serving resources
  (shared interception pool, shared tool servers, per-endpoint clients) — and hands
  them into `rollout()`. The hook never constructs agents.
- **One env-rollout is one `Episode`** on the wire (`traces.jsonl`, the serve
  protocol): the task, a rollout-level `errors` list, and one trace per agent run,
  each stamped with its `role` and `trainable`. Episodes succeed, resume, and retry
  as a unit. An agent failure is data on its trace (the hook decides what a failed
  participant means); an exception in `rollout()`/`score()` is the env-rollout
  failing, and every trace that completed before it is still captured on the episode.
- **Cross-agent signals can be declarative.** The default `score()` runs the env's
  own decorated `@vf.reward`/`@vf.metric` methods: each is invoked once per target
  trace and records there, with the finished sibling set in reach (`trace` — the
  target, `traces` — all of them in `rollout()`'s order, `task`). `role=` narrows
  the targets to one role's traces; unset means every trace (a shared team signal).
  The bundled best-of-n's whole judgement is two such metrics:

  ```python
      @vf.metric
      async def pass_at_n(self, trace, traces):
          return float(max(t.reward for t in traces) >= self.params.threshold)
  ```

  Override `score()` for imperative control (dynamic names or weights,
  parse-and-fail — the bundled judge env, or the debate verdict above);
  `await super().score(task, traces)` keeps the decorated ones running.
- `score()` is bounded by `--timeout.score`; `setup()`/`teardown()` hooks bracket the
  serving lifetime for env-owned shared resources.

The judge seat above is the pattern the bundled judge env productionizes: pair
`--env.id judge` with any taskset and the same grading runs spec-driven (write
criteria once, as a plugin) and sandboxed when the judge executes code — reach for
it before writing a `judge_task` of your own (see the bundled envs below).

For the single-agent case none of this is visible: the base `roles()` is one `"solver"`
role driven by `--harness.*`, `rollout()` is `[await agents["solver"].run(task)]`, and
the episode wraps exactly one unstamped trace.

The three axes of a run are orthogonal:

- **taskset** — *what to solve*: the rows, their data, their per-trace judgement.
- **harness** — *how the LLM interfaces with the world*: the program driving model
  calls, tools, a runtime.
- **env** — *the control flow between agents*: who runs, in what order, judged how
  across the finished set.

## Reusable envs: `--env.id`

An interaction pattern that isn't specific to one dataset — n attempts, a judge, a
modeled user — is its own plugin, paired with any taskset from the CLI:

```bash
uv run eval --taskset.id gsm8k-v1 --env.id best-of-n --env.n 8
uv run eval --taskset.id my-task-v1 --env.id judge --env.judge.model openai/gpt-5-mini
```

`--env.id` resolves like every plugin id — a bundled env (below), a local package
exporting an `Environment` subclass via `__all__`, or a Hub `org/name[@version]` —
and its `EnvParams` surface typed on the CLI (`--env.<role>.*`, `-h` renders them).
Empty (the default) keeps the taskset's own story: the env its package ships (a
*recipe* env like `code_golf_v1`, where the interaction is
intrinsic to the data), else the single-agent base. An explicit id wins over a
bundled recipe env.

Bundled envs (`verifiers/v1/envs/`):

| id | roles | what it does |
| --- | --- | --- |
| `best-of-n` | `solver` | `--env.n` independent attempts per rollout; `score()` marks the argmax-reward sibling (`best`) and whether any reached `--env.threshold` (`pass_at_n`) — rejection sampling and pass@k. |
| `judge` | `solver`, `judge` | the solver plays the task; a judge agent (in-process `direct` harness, `trainable=False` by default) grades the finished attempt. The verdict spec is a **judge plugin** (`--env.spec.id score\|rubric\|reference`, same registry and format as `taskset.task.judges`) — write your grading criteria once, run them as a bare call or as an agent. Verdict + per-criterion metrics land on the solver's trace; point `--env.judge.harness.id` at a real harness and the judge investigates with tools, `--env.spec.view full_trace` shows it the whole transcript. A judge that executes code is always played in its own sandbox, never on the host: its verdict task mirrors the solver task's world (same image, a fresh box in its original state) with the graded transcript uploaded (`/tmp/transcript.md`/`.json`), riding the run's container runtime — a fully-subprocess run must pin one (`--env.judge.harness.runtime.type docker`). |
