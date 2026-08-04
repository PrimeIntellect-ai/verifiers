# The Env

An `Env` defines the control flow between agents. In the simplest case, it is just a `SingleAgentEnv` where a single agent solves a task from a taskset. In more advanced settings, it can define grader-solver (agentic judges) or proposer-solver episodes which chains multiple agents.

The core method is `Env.run`, which builds up an `Episode` artifact implicitly: it returns nothing, and every finished agent run joins the episode automatically — one `Trace` per agent run. The `setup` and `finalize` hooks let you configure which agents should be trained in prime-rl or set cross-agent rewards.

This example illustrates two agents, `pro` and `con`, arguing for opposing positions on a question from a taskset, judged by a `judge` agent.

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
        agents.judge.trainable = False

    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        pro, con = await asyncio.gather(agents.pro.run(task), agents.con.run(task))
        await agents.judge.run(VerdictTask.from_traces(task, pro, con))

    async def finalize(self, task: vf.Task, episode: vf.Episode) -> None:
        by_agent = {t.agent.name: t for t in episode.traces}
        winner = (by_agent["judge"].last_reply or "").strip().lower()
        by_agent["pro"].record_reward("won", float(winner == "pro"))
        by_agent["con"].record_reward("won", float(winner == "con"))
```

## Episode

An `Episode` holds all agents' traces from a single invocation of `Env.run`. A good mental model is: the `Trace` is an agent's local view of a rollout, while the `Episode` is the global view.

## Pluggability

Just like tasksets and harnesses, an `Env` can be user-defined for full expressiveness over multi-agent interaction patterns — export an `Env` subclass via `__all__`. Otherwise, verifiers ships with a handful of built-ins.

| id | agents | what it does |
| --- | --- | --- |
| `single-agent` | `agent` | (default) one `agent` plays the taskset |
| `best-of-n` | `agent` | `n` independent attempts per episode; its metrics mark the argmax-reward sibling (`best`) and whether any reached `--env.threshold` (`pass_at_n`) — rejection sampling and pass@k. |
| `agentic-judge` | `solver`, `judge` | the solver plays the task; a code-executing judge verifies its collected artifacts in a fresh runtime with the same policy. |
| `shared-agentic-judge` | `solver`, `judge` | the solver plays the task; a code-executing judge explicitly verifies it in the same runtime. |

## Grading artifacts

Use artifacts to carry files between runtimes. Files written to
`/logs/artifacts/` are collected implicitly; declare other paths on the task data:

```python
class MyData(vf.TaskData):
    artifacts: list[vf.Artifact] = [
        vf.Artifact(source="/work/report", exclude=[".git"])
    ]


class MyTask(vf.Task[MyData]):
    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        trace.state.artifacts = await vf.collect(runtime, self.data.artifacts)
```

Declared paths must exist when collected. The implicit directory is optional.

## Concurrency

Write independent agents as independent (`asyncio.gather`, a `TaskGroup`) — how many actually run at once is the run's call, not the env's. Two knobs bound it, and the **episode is the unit** at the outer one:

| knob | bounds |
| --- | --- |
| `max_concurrent` / `-c` | episodes in flight (per worker when served) |
| `env.max_concurrent_agents` | agent runs inside one episode — **1** by default |

At the default, `-c 128` is 128 live agent runs whatever the env does internally. Set `max_concurrent_agents` higher (or `None` for no limit) when you want an episode's fan-out to run together — `-c` still caps the episodes carrying it:

```bash
uv run eval gsm8k-v1 --env.id best-of-n --env.n 16 --env.max-concurrent-agents None -c 128
```

Turn-taking envs are unaffected: an interaction holds its agent permit only around an active segment, never while awaiting its caller, so `user-sim` and games alternate at any setting.
