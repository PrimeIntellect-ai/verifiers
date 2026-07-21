"""The standalone Agent: one agent, one task, one trace — subprocess runtime.

`PRIME_API_KEY` in the environment, then: uv run python examples/agent.py
Multi-agent interactions belong in an `Environment` (see docs/v1/environments.md);
this is the primitive underneath.
"""

import asyncio
import json

import verifiers.v1 as vf


async def main() -> None:
    # An agent is built from its config alone; pass a live Client to share a
    # connection pool across agents.
    solver = vf.make_agent(vf.AgentConfig(model="z-ai/glm-5.2"))
    task = vf.Task(
        vf.TaskData(idx=0, prompt="What is 2+2? Answer with just the number.")
    )
    trace = await solver.run(task)
    print("stop:", trace.stop_condition)
    print("error:", trace.error)
    print("turns:", trace.num_turns)
    print("usage:", trace.usage)
    last = trace.assistant_messages[-1].content if trace.assistant_messages else None
    print("answer:", last)
    print("agent stamp:", json.dumps(trace.info["agent"], indent=2))


if __name__ == "__main__":
    asyncio.run(main())
