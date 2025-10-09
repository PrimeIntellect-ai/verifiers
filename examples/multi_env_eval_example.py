import asyncio
from openai import AsyncOpenAI
import os

from verifiers.scripts.eval import eval_environments_parallel, push_eval_to_env_hub


async def example_multi_env_eval():
    """Example: Evaluate multiple environments in parallel."""

    client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "your-api-key"),
        base_url="https://api.openai.com/v1",
    )

    envs = ["gsm8k", "wordle"]

    # Run parallel evaluation
    results = await eval_environments_parallel(
        envs=envs,
        env_args_dict={
            "gsm8k": {},
            "wordle": {},
        },
        client=client,
        model="gpt-4o-mini",
        num_examples=[10, 10],
        rollouts_per_example=[3, 3],
        max_concurrent=[32, 32],
        sampling_args={
            "temperature": 0.7,
            "max_tokens": 2048,
        },
    )

    # Process results
    for env_name, output in results.items():
        print(f"\n=== {env_name} ===")
        print(f"Number of samples: {len(output.reward)}")
        print(f"Average reward: {sum(output.reward) / len(output.reward):.3f}")
        print(f"Rewards: {output.reward[:5]}...")  # Show first 5

        # Show metrics if available
        if output.metrics:
            for metric_name, metric_values in output.metrics.items():
                avg = sum(metric_values) / len(metric_values)
                print(f"Average {metric_name}: {avg:.3f}")


async def example_per_env_sampling_args():
    """
    Example: Per-environment sampling arguments.

    This shows how to configure different sampling parameters for each environment.
    Useful when different tasks require different generation strategies.
    """

    client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "your-api-key"),
        base_url="https://api.openai.com/v1",
    )

    envs = ["gsm8k", "wordle"]

    # Global sampling args (fallback)
    global_sampling = {
        "temperature": 0.7,
        "max_tokens": 2048,
    }

    # Per-environment sampling args
    sampling_args_dict = {
        "gsm8k": {
            "temperature": 0.9,
            "max_tokens": 4096,
            "top_p": 0.95,
        },
        "wordle": {
            "temperature": 0.3,
            "max_tokens": 512,
        },
    }

    results = await eval_environments_parallel(
        envs=envs,
        env_args_dict={env: {} for env in envs},
        client=client,
        model="gpt-4o-mini",
        num_examples=[10, 10, 10],
        rollouts_per_example=[3, 3, 3],
        max_concurrent=[32, 32, 32],
        sampling_args=global_sampling,  # Fallback
        sampling_args_dict=sampling_args_dict,  # Per-env overrides
    )

    # Display results
    for env_name, output in results.items():
        print(f"\n=== {env_name} ===")
        print(f"Average reward: {sum(output.reward) / len(output.reward):.3f}")

        # Show which sampling args were used
        if env_name in sampling_args_dict:
            print(f"Used sampling args: {sampling_args_dict[env_name]}")
        else:
            print(f"Used global sampling args: {global_sampling}")


async def example_with_env_hub():
    """
    Example: Evaluate and save to Prime Hub.

    NOTE: Before pushing evaluations, you must first push the environments to Prime Hub:
        1. Using verifiers: env.push_to_env_hub(hub_name='owner/gsm8k')
        2. Using prime CLI: prime env push gsm8k
        3. Via web: https://app.primeintellect.ai/environments

    The system will check if the environment exists before pushing eval results.
    """

    client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "your-api-key"),
        base_url="https://api.openai.com/v1",
    )

    envs = ["gsm8k", "wordle"]
    model = "gpt-4o-mini"

    # Run evaluation
    results = await eval_environments_parallel(
        envs=envs,
        env_args_dict={"gsm8k": {}, "wordle": {}},
        client=client,
        model=model,
        num_examples=[10, 10],
        rollouts_per_example=[3, 3],
        max_concurrent=[32, 32],
        sampling_args={"temperature": 0.7, "max_tokens": 2048},
    )

    for env_name, output in results.items():
        # Calculate metrics
        avg_reward = sum(output.reward) / len(output.reward)

        metrics = {
            "avg_reward": float(avg_reward),
            "num_samples": len(output.reward),
        }

        # Add any additional metrics from the output
        for metric_name, metric_values in output.metrics.items():
            metrics[f"avg_{metric_name}"] = float(
                sum(metric_values) / len(metric_values)
            )

        # Prepare metadata
        metadata = {
            "environment": env_name,
            "model": model,
            "num_examples": 10,
            "rollouts_per_example": 3,
            "sampling_args": {"temperature": 0.7, "max_tokens": 2048},
        }

        # Save to hub (will check if environment exists first)
        push_eval_to_env_hub(
            eval_name=f"{model.replace('/', '-')}-{env_name}",
            model_name=model,
            environment_id=env_name,
            metrics=metrics,
            metadata=metadata,
        )


if __name__ == "__main__":
    print("Example 1: Basic multi-environment evaluation")
    asyncio.run(example_multi_env_eval())

    print("\n" + "=" * 80 + "\n")
    print("Example 2: Per-environment sampling arguments")
    asyncio.run(example_per_env_sampling_args())

    print("\n" + "=" * 80 + "\n")
    print("Example 3: With Prime Hub integration")
    asyncio.run(example_with_env_hub())
