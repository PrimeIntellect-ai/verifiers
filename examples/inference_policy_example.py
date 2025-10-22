"""
Example usage of InferencePolicy abstractions.

Demonstrates how to use policies for deployment-ready inference.
"""

import asyncio
from openai import AsyncOpenAI

import verifiers as vf


async def main():
    # Example 1: Using APIPolicy with OpenAI
    # ========================================
    print("Example 1: APIPolicy with OpenAI API")

    # Create policy from client (backwards compatible)
    client = AsyncOpenAI(api_key="your-api-key")
    policy = vf.InferencePolicy.from_client(client, model="gpt-4")

    # Or create directly
    policy = vf.APIPolicy(client=client, model="gpt-4")

    # Generate response
    response = await policy.generate(
        prompt=[{"role": "user", "content": "Hello!"}],
        sampling_args={"temperature": 0.7, "max_tokens": 100}
    )
    print(f"Response: {response.choices[0].message.content}")

    # Example 2: Using with Environment (backwards compatible)
    # =========================================================
    print("\nExample 2: Using policy with Environment")

    # Load environment
    env = vf.load_environment("math-python")

    # Old way (still works)
    results_old = env.evaluate(
        client=client,
        model="gpt-4",
        num_examples=5
    )

    # New way (using policy) - planned for future
    # results_new = env.evaluate(
    #     policy=policy,
    #     num_examples=5
    # )

    # Example 3: VLLMPolicy for high-throughput serving
    # ==================================================
    print("\nExample 3: VLLMPolicy for production deployment")

    try:
        # Requires vLLM server running
        vllm_policy = vf.VLLMPolicy(
            host="localhost",
            port=8000,
            model="your-model"
        )

        # Use for inference
        response = await vllm_policy.generate(
            prompt=[{"role": "user", "content": "Solve: 2+2=?"}],
            sampling_args={"temperature": 0.0}
        )
        print(f"vLLM Response: {response.choices[0].message.content}")

        # Optional: Enable weight syncing for online learning
        # vllm_policy.enable_weight_sync()
        # vllm_policy.sync_weights(model)

    except Exception as e:
        print(f"VLLMPolicy example skipped: {e}")

    # Example 4: Custom deployment scenarios
    # =======================================
    print("\nExample 4: Flexible deployment patterns")

    # API-based evaluation (no training infrastructure needed)
    api_policy = vf.APIPolicy(
        client=AsyncOpenAI(api_key="test", base_url="https://api.openai.com/v1"),
        model="gpt-4"
    )

    # Can be used anywhere that accepts a policy
    # - Evaluation scripts
    # - Production serving
    # - A/B testing different models
    # - Local development


if __name__ == "__main__":
    asyncio.run(main())
