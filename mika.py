import asyncio
import os
import time

from openai import AsyncOpenAI

from verifiers import load_environment


async def main():
    env = load_environment("math500")
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    semaphore = asyncio.Semaphore(1)
    inputs = env.get_eval_inputs(num_examples=8, rollouts_per_example=1)
    s = time.time()
    (
        await env.generate(
            inputs=inputs,
            client=client,
            model="gpt-4.1-mini",
            semaphore=semaphore,
        )
    )
    print(f"Single task: {time.time() - s:.2f} seconds")

    inputs = env.get_eval_inputs(num_examples=4, rollouts_per_example=1)
    semaphore = asyncio.Semaphore(1)
    s = time.time()
    await asyncio.gather(
        *[
            env.generate(
                inputs=inputs,
                client=client,
                model="gpt-4.1-mini",
                semaphore=semaphore,
            ),
            env.generate(
                inputs=inputs,
                client=client,
                model="gpt-4.1-mini",
                semaphore=semaphore,
            ),
        ]
    )
    print(f"Parallel tasks: {time.time() - s:.2f} seconds")


if __name__ == "__main__":
    asyncio.run(main())
