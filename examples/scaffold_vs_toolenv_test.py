"""
Compare Scaffold approach vs existing ToolEnv approach.

This test verifies that ToolScaffold produces consistent results with ToolEnv.
"""

import argparse
import asyncio
import random

from datasets import Dataset
from openai import AsyncOpenAI

import verifiers as vf
from verifiers.scaffolds import ToolScaffold


# Same tools as tool_test environment
def tool_A(x: int) -> int:
    """Tool for adding 1 to an integer.

    Args:
        x: The integer to add 1 to.

    Returns:
        The integer plus 1.
    """
    return x + 1


def tool_B(x: str) -> str:
    """Tool for concatenating a string with "2".

    Args:
        x: The string to concatenate with "2".

    Returns:
        The string concatenated with "2".
    """
    return x + "2"


def tool_C(x: float) -> float:
    """Tool for adding 3.0 to a float.

    Args:
        x: The float to add 3.0 to.

    Returns:
        The float plus 3.0.
    """
    return x + 3.0


def tool_D(x: bool) -> bool:
    """Tool for negating a boolean.

    Args:
        x: The boolean to negate.

    Returns:
        The negated boolean.
    """
    return not x


tool_list = [tool_A, tool_B, tool_C, tool_D]
tool_name_list = [tool.__name__ for tool in tool_list]


def create_test_dataset(num_examples: int = 10, seed: int = 42) -> Dataset:
    """Create test dataset matching tool_test environment."""
    random.seed(seed)
    rows = []
    for i in range(num_examples):
        tool_names = random.sample(tool_name_list, random.randint(1, len(tool_name_list)))
        prompt = [
            {
                "role": "user",
                "content": f"Call the following tools with arguments of your choice: {tool_names}",
            }
        ]
        info = {"tool_names": tool_names}
        rows.append({"prompt": prompt, "info": info, "example_id": i})
    return Dataset.from_list(rows)


def check_tool_calls(completion, info) -> float:
    """Check if completion tool calls match expected tools."""
    if not completion:
        return 0.0
    last_msg = completion[-1] if isinstance(completion, list) else completion
    tool_calls = last_msg.get("tool_calls", [])
    called_tool_names = sorted(
        [call.get("function", {}).get("name", "") for call in tool_calls]
    )
    expected_tool_names = sorted(info["tool_names"])
    return 1.0 if called_tool_names == expected_tool_names else 0.0


async def test_with_toolenv(client: AsyncOpenAI, model: str, dataset: Dataset) -> dict:
    """Test using traditional ToolEnv approach."""
    print("\n" + "=" * 60)
    print("Testing with ToolEnv")
    print("=" * 60)

    rubric = vf.Rubric(funcs=[check_tool_calls])
    env = vf.ToolEnv(
        dataset=dataset,
        rubric=rubric,
        tools=tool_list,
        max_turns=1,  # Single turn for comparison
    )

    # Prepare inputs - dataset already has example_id from create_test_dataset
    inputs = dataset.to_list()

    # Generate
    results = await env.generate(
        inputs=inputs,
        client=client,
        model=model,
        sampling_args={"temperature": 0, "max_tokens": 200},
        max_concurrent=5,
    )

    rewards = results["reward"]
    accuracy = sum(rewards) / len(rewards) if rewards else 0
    print(f"Accuracy: {accuracy:.1%}")
    print(f"Total rollouts: {len(rewards)}")

    return {
        "accuracy": accuracy,
        "total_rollouts": len(rewards),
        "rewards": rewards,
    }


async def test_with_scaffold(client: AsyncOpenAI, model: str, dataset: Dataset) -> dict:
    """Test using ToolScaffold approach."""
    print("\n" + "=" * 60)
    print("Testing with ToolScaffold")
    print("=" * 60)

    scaffold = ToolScaffold(
        client=client,
        model=model,
        tools=tool_list,
        max_tool_turns=1,  # Match ToolEnv max_turns
        sampling_args={"temperature": 0, "max_tokens": 200},
    )

    results = []
    for item in dataset:
        prompt = item["prompt"]
        info = item["info"]

        result = await scaffold.generate(prompt, state={})

        # Extract the assistant message with tool calls
        # In scaffold, tool calls are in the first assistant message
        completion = result.messages[len(prompt):]

        # Find the assistant message with tool calls
        assistant_msg = None
        for msg in completion:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                assistant_msg = msg
                break
            elif msg.get("role") == "assistant":
                assistant_msg = msg

        if assistant_msg:
            score = check_tool_calls([assistant_msg], info)
        else:
            score = 0.0

        results.append(score)
        print(f"  Example {item['example_id']}: expected {info['tool_names']}, score={score}")

    accuracy = sum(results) / len(results) if results else 0
    print(f"\nAccuracy: {accuracy:.1%}")
    print(f"Total rollouts: {len(results)}")

    return {
        "accuracy": accuracy,
        "total_rollouts": len(results),
        "rewards": results,
    }


async def main():
    parser = argparse.ArgumentParser(description="Scaffold vs ToolEnv Comparison")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Model to use")
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="API base URL")
    parser.add_argument("--api-key", default="EMPTY", help="API key")
    parser.add_argument("--num-examples", type=int, default=10, help="Number of examples")
    args = parser.parse_args()

    client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)

    print("=" * 60)
    print("SCAFFOLD VS TOOLENV COMPARISON")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Examples: {args.num_examples}")

    # Create dataset
    dataset = create_test_dataset(num_examples=args.num_examples)

    # Run both approaches
    toolenv_results = await test_with_toolenv(client, args.model, dataset)
    scaffold_results = await test_with_scaffold(client, args.model, dataset)

    # Summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Approach':<20} {'Accuracy':<15}")
    print("-" * 35)
    print(f"{'ToolEnv':<20} {toolenv_results['accuracy']:.1%}")
    print(f"{'ToolScaffold':<20} {scaffold_results['accuracy']:.1%}")

    diff = abs(toolenv_results['accuracy'] - scaffold_results['accuracy'])
    if diff < 0.1:
        print(f"\nResults are consistent (diff={diff:.1%})")
    else:
        print(f"\nWARNING: Results differ significantly (diff={diff:.1%})")


if __name__ == "__main__":
    asyncio.run(main())
