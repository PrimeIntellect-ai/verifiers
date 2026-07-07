"""Demo 4: a third-party harness (mini-swe-agent) + runtime-verified scoring.

The agent runs the real mini-swe-agent CLI in a fresh prime sandbox; the attached
taskset's @reward re-executes the agent's artifact IN the live box — judging the world,
not the transcript.
"""

import asyncio

import verifiers.v1 as vf
from verifiers.v1.harnesses.mini_swe_agent import (
    MiniSWEAgentHarness,
    MiniSWEAgentHarnessConfig,
)

TASK = (
    "Write a Python script at /app/fizz.py that prints the FizzBuzz sequence for "
    "1..15, one item per line. Run it to verify it works, then finish."
)


class WorldTaskset(vf.Taskset[vf.Task, vf.TasksetConfig]):
    @vf.reward
    async def artifact_runs(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        result = await runtime.run(["python3", "/app/fizz.py"], {})
        if result.exit_code != 0:
            return 0.0
        lines = [line.strip() for line in result.stdout.strip().splitlines()]
        expected = [
            "FizzBuzz"
            if i % 15 == 0
            else "Fizz"
            if i % 3 == 0
            else "Buzz"
            if i % 5 == 0
            else str(i)
            for i in range(1, 16)
        ]
        return 1.0 if lines == expected else 0.5


async def main() -> None:
    agent = vf.Agent(
        MiniSWEAgentHarness(MiniSWEAgentHarnessConfig(id="mini-swe-agent")),
        vf.ModelContext(
            model="z-ai/glm-5.2", client=vf.resolve_client(vf.EvalClientConfig())
        ),
        vf.PrimeConfig(labels=["agent-programs-demo"]),
        timeout=vf.TimeoutConfig(rollout=420),
    )
    trace = await agent.run(
        vf.Task(idx=0, prompt=TASK), taskset=WorldTaskset(vf.TasksetConfig())
    )
    print("stop:", trace.stop_condition, "| error:", trace.error)
    print("turns:", trace.num_turns, "| usage:", trace.usage)
    print("rewards:", trace.rewards, "-> reward:", trace.reward)
    print("agent stamp:", trace.info["agent"])


if __name__ == "__main__":
    asyncio.run(main())
