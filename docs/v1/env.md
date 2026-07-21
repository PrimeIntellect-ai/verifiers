# The Env

An `Env` programs interactions between agents. In the simplest case, it is just
a `SingleAgentEnv` where a single agent solves a task from a taskset. In more
advanced settings, it can define grader-solver (agentic judges) or
proposer-solver episodes which chains multiple agents.

The core method is `Env.run`, which builds up an `Episode` artifact implicitly:
it returns nothing, and every finished agent run joins the episode automatically
— one `Trace` per agent run. The `setup` and `finalize` hooks let you configure
which agents should be trained in prime-rl or set cross-agent rewards.

This example illustrates two agents, `pro` and `con`, arguing for opposing
positions on a question from a taskset, judged by a `judge` agent.

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
        by_agent = {t.agent_name: t for t in episode.traces}
        winner = (by_agent["judge"].last_reply or "").strip().lower()
        by_agent["pro"].record_reward("won", float(winner == "pro"))
        by_agent["con"].record_reward("won", float(winner == "con"))
```

## Episode

An `Episode` holds all agents' traces from a single invocation of `Env.run`.

## Pluggability

Just like tasksets and harnesses, an `Env` can be user-defined for full
expressiveness over multi-agent interaction patterns — export an `Env` subclass
via `__all__`. Otherwise, verifiers ships with a handful of built-ins.

| id | agents | what it does |
| --- | --- | --- |
| `single-agent` | `agent` | (default) one `agent` plays the taskset |
| `best-of-n` | `agent` | `n` independent attempts per episode; its metrics mark the argmax-reward sibling (`best`) and whether any reached `--env.threshold` (`pass_at_n`) — rejection sampling and pass@k. |
| `agentic-judge` | `solver`, `judge` | the solver plays the task; a code-executing judge agent verifies the finished attempt with real execution. |
