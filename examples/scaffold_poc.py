"""
Scaffold PoC: Compare agent performance with and without tools.

This example demonstrates how scaffolds decouple tool management from environments.
The same environment (a simple QA task) can be run with:
1. A bare scaffold (no tools) - model must answer from knowledge
2. A tool scaffold - model can use tools to help answer

Usage:
    python examples/scaffold_poc.py --model <model_name> --base-url <api_url>

Example with local vLLM:
    python examples/scaffold_poc.py --model Qwen/Qwen2.5-7B-Instruct --base-url http://localhost:8000/v1
"""

import argparse
import asyncio

from datasets import Dataset
from openai import AsyncOpenAI

import verifiers as vf


# Simple calculator tool for demonstration
async def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A mathematical expression to evaluate (e.g. "2 + 2 * 3")

    Returns:
        The result of the evaluation.
    """
    try:
        # Safe eval for basic math
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return f"Error: Invalid characters in expression"
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


async def search(query: str) -> str:
    """Search for information (mock implementation).

    Args:
        query: The search query.

    Returns:
        Search results.
    """
    # Mock search results for demo
    mock_results = {
        "python": "Python is a high-level programming language created by Guido van Rossum in 1991.",
        "rust": "Rust is a systems programming language focused on safety and performance.",
        "capital of france": "The capital of France is Paris.",
        "speed of light": "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
    }

    query_lower = query.lower()
    for key, value in mock_results.items():
        if key in query_lower:
            return value

    return f"No results found for: {query}"


def create_dataset() -> Dataset:
    """Create a simple QA dataset with math and factual questions."""
    return Dataset.from_list([
        {
            "prompt": [{"role": "user", "content": "What is 15 * 23 + 42?"}],
            "answer": "387",
        },
        {
            "prompt": [{"role": "user", "content": "What is the capital of France?"}],
            "answer": "Paris",
        },
        {
            "prompt": [{"role": "user", "content": "Calculate: (100 - 37) * 2"}],
            "answer": "126",
        },
        {
            "prompt": [{"role": "user", "content": "What is the speed of light in meters per second?"}],
            "answer": "299792458",
        },
    ])


async def correct_answer(completion, answer) -> float:
    """Check if the answer appears in the completion."""
    if not completion:
        return 0.0
    response = completion[-1].get("content", "")
    return 1.0 if answer.lower() in response.lower() else 0.0


async def run_with_scaffold(
    env: vf.Environment,
    scaffold: vf.Scaffold,
    inputs: list,
    name: str,
) -> dict:
    """Run evaluation with a specific scaffold."""
    print(f"\n{'='*60}")
    print(f"Running with: {name}")
    print(f"{'='*60}")

    results = []
    for i, input_item in enumerate(inputs):
        prompt = input_item["prompt"]
        answer = input_item["answer"]

        # Use scaffold to generate
        scaffold_result = await scaffold.generate(prompt, state={})

        # Extract completion from scaffold result
        completion = scaffold_result.messages[len(prompt):]

        # Score
        score = await correct_answer(completion, answer)

        # Get response text
        response_text = ""
        if completion:
            response_text = completion[-1].get("content", "")[:100]

        results.append({
            "question": prompt[-1]["content"],
            "answer": answer,
            "correct": score > 0,
            "tool_calls": scaffold_result.tool_calls_made,
            "response_preview": response_text,
        })

        print(f"\nQ{i+1}: {prompt[-1]['content']}")
        print(f"Expected: {answer}")
        print(f"Correct: {'Yes' if score > 0 else 'No'}")
        print(f"Tool calls: {scaffold_result.tool_calls_made}")
        print(f"Response: {response_text}...")

    accuracy = sum(r["correct"] for r in results) / len(results)
    total_tool_calls = sum(r["tool_calls"] for r in results)

    print(f"\n{'-'*40}")
    print(f"Accuracy: {accuracy:.1%} ({sum(r['correct'] for r in results)}/{len(results)})")
    print(f"Total tool calls: {total_tool_calls}")

    return {
        "name": name,
        "accuracy": accuracy,
        "total_tool_calls": total_tool_calls,
        "results": results,
    }


async def main():
    parser = argparse.ArgumentParser(description="Scaffold PoC")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model to use")
    parser.add_argument("--base-url", default=None, help="API base URL (for local models)")
    parser.add_argument("--api-key", default="EMPTY", help="API key")
    args = parser.parse_args()

    # Create client
    client = AsyncOpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
    )

    # Create dataset and environment
    dataset = create_dataset()
    rubric = vf.Rubric(funcs=[correct_answer])
    env = vf.SingleTurnEnv(dataset=dataset, rubric=rubric)

    # Prepare inputs
    inputs = dataset.to_list()
    for i, item in enumerate(inputs):
        item["example_id"] = i

    # Create scaffolds
    sampling_args = {"temperature": 0.0, "max_tokens": 512}

    # 1. Bare scaffold (no tools)
    bare_scaffold = vf.Scaffold(
        client=client,
        model=args.model,
        sampling_args=sampling_args,
    )

    # 2. Tool scaffold (with calculator and search)
    tool_scaffold = vf.ToolScaffold(
        client=client,
        model=args.model,
        tools=[calculate, search],
        max_tool_turns=5,
        sampling_args=sampling_args,
    )

    # Run comparisons
    print("\n" + "="*60)
    print("SCAFFOLD POC: Comparing with/without tools")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Dataset: {len(inputs)} questions")

    bare_results = await run_with_scaffold(env, bare_scaffold, inputs, "Bare (no tools)")
    tool_results = await run_with_scaffold(env, tool_scaffold, inputs, "With tools (calculator + search)")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Scaffold':<30} {'Accuracy':<15} {'Tool Calls':<15}")
    print("-"*60)
    print(f"{'Bare (no tools)':<30} {bare_results['accuracy']:.1%}               {bare_results['total_tool_calls']}")
    print(f"{'With tools':<30} {tool_results['accuracy']:.1%}               {tool_results['total_tool_calls']}")

    # Cleanup
    await bare_scaffold.teardown()
    await tool_scaffold.teardown()


if __name__ == "__main__":
    asyncio.run(main())
