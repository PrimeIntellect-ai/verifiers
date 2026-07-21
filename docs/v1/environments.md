# Multi-agent environments

One eval rollout doesn't have to be one agent run. `Env` is abstract, and
every run gets a concrete subclass: plain tasksets resolve to the bundled
`SingleAgentEnv` (one `agent` playing the taskset), and a package can export
its own (via `__all__`, alongside its [`Taskset`](tasksets.md) — the same plugin
idiom as a bundled harness). An env declares its config as an `EnvConfig` subclass —
each agent an `AgentConfig` field, plus its own knobs — writes `run()`, and
optionally overrides `setup()` and `finalize()`:

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


class DebateEnv(vf.Env[DebateConfig]):
    async def setup(self, agents: vf.Agents) -> None:
        """Per-agent standing the env hardcodes: the judge grades the debate,
        so its tokens are never training data."""
        agents.judge.trainable = False

    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        """How the agents interact on one task: imperative Python over Agent
        values. A loop is rounds, a TaskGroup is fan-out, a Task classmethod is
        chaining. Returns nothing — every finished run joins the episode
        automatically, stamped with its standing."""
        pro, con = await asyncio.gather(agents.pro.run(task), agents.con.run(task))
        await agents.judge.run(VerdictTask.from_traces(task, pro, con))

    async def finalize(self, task: vf.Task, episode: vf.Episode) -> None:
        """Sibling-dependent judgement over the finished episode (per-trace
        judgement already ran on each trace's own task); `episode.traces` is
        the flat episode, each trace's `agent_name` stamp naming its agent.
        Attach via record_reward/record_metric, in program order."""
        by_agent = {t.agent_name: t for t in episode.traces}
        winner = (by_agent["judge"].last_reply or "").strip().lower()
        by_agent["pro"].record_reward("won", float(winner == "pro"))
        by_agent["con"].record_reward("won", float(winner == "con"))
```

For the single-agent case none of this is machinery the user sees: `SingleAgentEnv`
declares one `agent` (`--env.agent.harness.id codex`, `--env.agent.max_turns 20`),
`run()` is `await agents.agent.run(task)`, and the episode carries exactly one
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
exporting an `Env` subclass via `__all__`, or a Hub `org/name[@version]` —
and its `EnvConfig` surface typed on the CLI (`--env.<agent>.*`, `-h` renders
them). Empty (the default) keeps the taskset's own story: the env its package
ships (a *recipe* env like `code_golf_v1`, where the interaction is intrinsic to
the data), else `SingleAgentEnv`. An explicit id wins over a bundled recipe env.

Bundled envs (`verifiers/v1/envs/`):

| id | agents | what it does |
| --- | --- | --- |
| `best-of-n` | `agent` | `--env.n` independent attempts per rollout; its metrics mark the argmax-reward sibling (`best`) and whether any reached `--env.threshold` (`pass_at_n`) — rejection sampling and pass@k. A single-agent env keeps the single-agent name, so `--env.agent.*` flags compose unchanged. |
| `agentic-judge` | `solver`, `judge` | agent-as-judge: the solver plays the task; a code-executing judge agent verifies the finished attempt with real execution, always in its own sandbox, never on the host. The judge's task mirrors the solver task's world (same image, a fresh box in its original state) with the graded transcript uploaded (`/tmp/transcript.md`/`.json`). The verdict channel is a file: the judge writes `{"score": 0-10, "reasoning": ...}` to `/tmp/verdict.json` in its box, scraped onto its trace while the box is alive and validated STRICTLY onto the solver's trace as the `judge` reward — a missing, malformed, or off-scale verdict fails the rollout instead of clamping. The judge must land in a container: pin `--env.judge.harness.runtime.type docker\|prime`, or construction refuses. A judgement that needs no execution belongs on the plugged tier (`env.taskset.task.judges`), not on an agent. |
