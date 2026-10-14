"""Standalone agent example."""

import asyncio

import verifiers.v1 as vf


async def main() -> None:
    solver = vf.make_agent(vf.AgentConfig(model="z-ai/glm-5.3"))
    prompt = "What is 2+2? Answer with just the number."
    task = vf.Task(vf.TaskData(prompt=prompt))
    async with solver:
        trace = await solver.run(task)
    answer = trace.assistant_messages[-1].content if trace.assistant_messages else None
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
