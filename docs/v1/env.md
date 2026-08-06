# The Env

An `Env` defines the control flow between `Agents`. In the simplest case, it is just a `SingleAgentEnv` where a single agent solves a task from a taskset.

Its core signature is `Env.run(task: Task, agents: Agents)` -> None — it is passed an initial task and pre-initialized agents and then programs the full multi-agent control flow; every finished agent run automatically joins the resulting `Episode`, which holds all the traces of all the agents.

```python
class Env(ABC):
    @abstractmethod
    async def run(self, task: Task, agents: Agents) -> None:
        """Run a single multi-agent episode."""
        ...
```

verifiers comes with different pre-built `Env`s to use:

- The `AgenticJudgeEnv` defines the sequential interaction between a solver and judge agent. The judge can re-use the same runtime after the solver (`SharedAgenticJudgeEnv`) or use its own, new runtime `IsolatedAgenticJudgeEnv`.
- The `UserSimEnv` models users as agents, and the episode is a turn-by-turn conversation between the user and assistant agents.
- The `BestOfNEnv` runs n independent attempts at the same task, then marks which attempt achieved the highest reward (best) and whether any attempt crossed a success threshold (pass_at_n), which is useful for rejection sampling and pass@k evaluation.
