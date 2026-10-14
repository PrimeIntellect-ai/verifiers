"""The standalone Agent: one agent, one task, one trace.

`PRIME_API_KEY` in the environment, then: uv run python examples/agent.py
"""

import asyncio

import verifiers.v1 as vf


async def main() -> None:
    # Everything declarative rides the config (harness None = the built-in bash);
    # live resources (a shared Client, an Interception) are injected, not configured.
    solver = vf.make_agent(vf.AgentConfig(model="z-ai/glm-5.2"))
    task = vf.Task(vf.TaskData(prompt="What is 2+2? Answer with just the number."))
    async with solver:
        trace = await solver.run(task)
    answer = trace.assistant_messages[-1].content if trace.assistant_messages else None
    print(f"stop:   {trace.stop_condition}")
    print(f"answer: {answer}")


if __name__ == "__main__":
    asyncio.run(main())
