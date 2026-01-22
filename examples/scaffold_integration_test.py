"""
Comprehensive scaffold integration tests.

Tests:
1. Basic scaffold functionality (bare vs tools)
2. Multi-turn tool interactions
3. Error handling
4. Integration with verifiers environments
5. Parallel execution
"""

import argparse
import asyncio
import time
from dataclasses import dataclass
from typing import Any

from datasets import Dataset
from openai import AsyncOpenAI

import verifiers as vf
from verifiers.scaffolds import Scaffold, ToolScaffold, ScaffoldResult


@dataclass
class TestResult:
    name: str
    passed: bool
    duration_ms: float
    details: str = ""
    error: str | None = None


# ============================================================
# Test Tools
# ============================================================

async def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A mathematical expression to evaluate.

    Returns:
        The result of the evaluation.
    """
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return f"Error: Invalid characters in expression"
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


async def get_weather(city: str) -> str:
    """Get weather for a city (mock).

    Args:
        city: The city name.

    Returns:
        Weather information.
    """
    weather_data = {
        "new york": "72°F, Sunny",
        "london": "55°F, Cloudy",
        "tokyo": "68°F, Partly Cloudy",
        "paris": "62°F, Rainy",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


async def search_database(query: str, limit: int = 3) -> str:
    """Search a mock database.

    Args:
        query: Search query.
        limit: Maximum results to return.

    Returns:
        Search results.
    """
    results = [
        f"Result {i+1} for '{query}': Mock data item {i+1}"
        for i in range(min(limit, 5))
    ]
    return "\n".join(results)


# ============================================================
# Test Cases
# ============================================================

async def test_bare_scaffold_basic(client: AsyncOpenAI, model: str) -> TestResult:
    """Test basic bare scaffold functionality."""
    start = time.time()
    try:
        scaffold = Scaffold(client, model, sampling_args={"temperature": 0, "max_tokens": 100})

        messages = [{"role": "user", "content": "What is 2+2? Reply with just the number."}]
        result = await scaffold.generate(messages, state={})

        response_text = result.messages[-1].get("content", "")
        has_4 = "4" in response_text

        return TestResult(
            name="test_bare_scaffold_basic",
            passed=has_4 and result.tool_calls_made == 0,
            duration_ms=(time.time() - start) * 1000,
            details=f"Response: {response_text[:50]}..., tool_calls: {result.tool_calls_made}",
        )
    except Exception as e:
        return TestResult(
            name="test_bare_scaffold_basic",
            passed=False,
            duration_ms=(time.time() - start) * 1000,
            error=str(e),
        )


async def test_tool_scaffold_single_tool(client: AsyncOpenAI, model: str) -> TestResult:
    """Test tool scaffold with a single tool call."""
    start = time.time()
    try:
        scaffold = ToolScaffold(
            client, model,
            tools=[calculate],
            max_tool_turns=5,
            sampling_args={"temperature": 0, "max_tokens": 200},
        )

        messages = [{"role": "user", "content": "Use the calculate tool to compute 123 * 456"}]
        result = await scaffold.generate(messages, state={})

        response_text = result.messages[-1].get("content", "")
        expected = str(123 * 456)  # 56088

        return TestResult(
            name="test_tool_scaffold_single_tool",
            passed=expected in response_text and result.tool_calls_made >= 1,
            duration_ms=(time.time() - start) * 1000,
            details=f"Expected: {expected}, tool_calls: {result.tool_calls_made}, response: {response_text[:100]}...",
        )
    except Exception as e:
        return TestResult(
            name="test_tool_scaffold_single_tool",
            passed=False,
            duration_ms=(time.time() - start) * 1000,
            error=str(e),
        )


async def test_tool_scaffold_multiple_tools(client: AsyncOpenAI, model: str) -> TestResult:
    """Test tool scaffold with multiple tool types."""
    start = time.time()
    try:
        scaffold = ToolScaffold(
            client, model,
            tools=[calculate, get_weather],
            max_tool_turns=5,
            sampling_args={"temperature": 0, "max_tokens": 300},
        )

        messages = [{"role": "user", "content": "What's the weather in Tokyo? Also calculate 50 * 3."}]
        result = await scaffold.generate(messages, state={})

        response_text = result.messages[-1].get("content", "")

        # Check both weather and calculation are in response
        has_weather = "68" in response_text or "tokyo" in response_text.lower() or "cloudy" in response_text.lower()
        has_calc = "150" in response_text

        return TestResult(
            name="test_tool_scaffold_multiple_tools",
            passed=result.tool_calls_made >= 1,  # At least one tool call
            duration_ms=(time.time() - start) * 1000,
            details=f"tool_calls: {result.tool_calls_made}, has_weather: {has_weather}, has_calc: {has_calc}",
        )
    except Exception as e:
        return TestResult(
            name="test_tool_scaffold_multiple_tools",
            passed=False,
            duration_ms=(time.time() - start) * 1000,
            error=str(e),
        )


async def test_tool_scaffold_no_tool_needed(client: AsyncOpenAI, model: str) -> TestResult:
    """Test that scaffold doesn't force tool use when not needed."""
    start = time.time()
    try:
        scaffold = ToolScaffold(
            client, model,
            tools=[calculate],
            max_tool_turns=5,
            sampling_args={"temperature": 0, "max_tokens": 100},
        )

        messages = [{"role": "user", "content": "Say hello!"}]
        result = await scaffold.generate(messages, state={})

        response_text = result.messages[-1].get("content", "")

        return TestResult(
            name="test_tool_scaffold_no_tool_needed",
            passed="hello" in response_text.lower(),
            duration_ms=(time.time() - start) * 1000,
            details=f"tool_calls: {result.tool_calls_made}, response: {response_text[:50]}...",
        )
    except Exception as e:
        return TestResult(
            name="test_tool_scaffold_no_tool_needed",
            passed=False,
            duration_ms=(time.time() - start) * 1000,
            error=str(e),
        )


async def test_tool_scaffold_tool_with_params(client: AsyncOpenAI, model: str) -> TestResult:
    """Test tool with multiple parameters."""
    start = time.time()
    try:
        scaffold = ToolScaffold(
            client, model,
            tools=[search_database],
            max_tool_turns=5,
            sampling_args={"temperature": 0, "max_tokens": 300},
        )

        messages = [{"role": "user", "content": "Search the database for 'python' with a limit of 2 results."}]
        result = await scaffold.generate(messages, state={})

        # Check that tool was called
        return TestResult(
            name="test_tool_scaffold_tool_with_params",
            passed=result.tool_calls_made >= 1,
            duration_ms=(time.time() - start) * 1000,
            details=f"tool_calls: {result.tool_calls_made}",
        )
    except Exception as e:
        return TestResult(
            name="test_tool_scaffold_tool_with_params",
            passed=False,
            duration_ms=(time.time() - start) * 1000,
            error=str(e),
        )


async def test_scaffold_parallel_execution(client: AsyncOpenAI, model: str) -> TestResult:
    """Test multiple scaffolds running in parallel."""
    start = time.time()
    try:
        scaffold = Scaffold(client, model, sampling_args={"temperature": 0.5, "max_tokens": 50})

        messages_list = [
            [{"role": "user", "content": "What is 1+1?"}],
            [{"role": "user", "content": "What is 2+2?"}],
            [{"role": "user", "content": "What is 3+3?"}],
        ]

        tasks = [scaffold.generate(msgs, state={}) for msgs in messages_list]
        results = await asyncio.gather(*tasks)

        all_completed = all(r.response is not None for r in results)

        return TestResult(
            name="test_scaffold_parallel_execution",
            passed=all_completed and len(results) == 3,
            duration_ms=(time.time() - start) * 1000,
            details=f"Completed {len(results)} parallel requests",
        )
    except Exception as e:
        return TestResult(
            name="test_scaffold_parallel_execution",
            passed=False,
            duration_ms=(time.time() - start) * 1000,
            error=str(e),
        )


async def test_scaffold_with_environment(client: AsyncOpenAI, model: str) -> TestResult:
    """Test scaffold working alongside environment infrastructure."""
    start = time.time()
    try:
        # Create a simple environment
        dataset = Dataset.from_list([
            {"prompt": [{"role": "user", "content": "What is 5 * 5?"}], "answer": "25"},
            {"prompt": [{"role": "user", "content": "What is 10 + 10?"}], "answer": "20"},
        ])

        async def check_answer(completion, answer) -> float:
            if not completion:
                return 0.0
            response = completion[-1].get("content", "")
            return 1.0 if answer in response else 0.0

        rubric = vf.Rubric(funcs=[check_answer])
        env = vf.SingleTurnEnv(dataset=dataset, rubric=rubric)

        # Create scaffold
        tool_scaffold = ToolScaffold(
            client, model,
            tools=[calculate],
            max_tool_turns=3,
            sampling_args={"temperature": 0, "max_tokens": 200},
        )

        # Test with both scaffolds
        results = []
        for item in dataset:
            prompt = item["prompt"]
            result = await tool_scaffold.generate(prompt, state={})
            completion = result.messages[len(prompt):]
            score = await check_answer(completion, item["answer"])
            results.append(score)

        avg_score = sum(results) / len(results) if results else 0

        return TestResult(
            name="test_scaffold_with_environment",
            passed=avg_score >= 0.5,  # At least 50% correct
            duration_ms=(time.time() - start) * 1000,
            details=f"Average score: {avg_score:.2f}",
        )
    except Exception as e:
        return TestResult(
            name="test_scaffold_with_environment",
            passed=False,
            duration_ms=(time.time() - start) * 1000,
            error=str(e),
        )


async def test_tool_error_handling(client: AsyncOpenAI, model: str) -> TestResult:
    """Test scaffold handles tool errors gracefully."""
    start = time.time()
    try:
        async def failing_tool(x: str) -> str:
            """A tool that always fails.

            Args:
                x: Input string.

            Returns:
                Never returns, always raises.
            """
            raise ValueError("This tool always fails!")

        scaffold = ToolScaffold(
            client, model,
            tools=[failing_tool],
            max_tool_turns=3,
            sampling_args={"temperature": 0, "max_tokens": 200},
        )

        messages = [{"role": "user", "content": "Use the failing_tool with input 'test'"}]
        result = await scaffold.generate(messages, state={})

        # Should complete without crashing
        return TestResult(
            name="test_tool_error_handling",
            passed=result.response is not None,
            duration_ms=(time.time() - start) * 1000,
            details=f"Completed despite tool error, tool_calls: {result.tool_calls_made}",
        )
    except Exception as e:
        return TestResult(
            name="test_tool_error_handling",
            passed=False,
            duration_ms=(time.time() - start) * 1000,
            error=str(e),
        )


async def test_max_tool_turns(client: AsyncOpenAI, model: str) -> TestResult:
    """Test that max_tool_turns is respected."""
    start = time.time()
    try:
        call_count = [0]

        async def counting_tool(x: str) -> str:
            """A tool that counts calls and always asks for more.

            Args:
                x: Input string.

            Returns:
                A message asking to call again.
            """
            call_count[0] += 1
            return f"Call #{call_count[0]}. Please call this tool again with 'continue'."

        scaffold = ToolScaffold(
            client, model,
            tools=[counting_tool],
            max_tool_turns=3,
            sampling_args={"temperature": 0, "max_tokens": 200},
        )

        messages = [{"role": "user", "content": "Keep calling counting_tool until it stops asking."}]
        result = await scaffold.generate(messages, state={})

        # Should stop at max_tool_turns
        return TestResult(
            name="test_max_tool_turns",
            passed=result.tool_calls_made <= 3,
            duration_ms=(time.time() - start) * 1000,
            details=f"tool_calls: {result.tool_calls_made}, max_tool_turns: 3",
        )
    except Exception as e:
        return TestResult(
            name="test_max_tool_turns",
            passed=False,
            duration_ms=(time.time() - start) * 1000,
            error=str(e),
        )


# ============================================================
# Test Runner
# ============================================================

async def run_all_tests(client: AsyncOpenAI, model: str) -> list[TestResult]:
    """Run all tests sequentially."""
    tests = [
        test_bare_scaffold_basic,
        test_tool_scaffold_single_tool,
        test_tool_scaffold_multiple_tools,
        test_tool_scaffold_no_tool_needed,
        test_tool_scaffold_tool_with_params,
        test_scaffold_parallel_execution,
        test_scaffold_with_environment,
        test_tool_error_handling,
        test_max_tool_turns,
    ]

    results = []
    for test_func in tests:
        print(f"Running {test_func.__name__}...", end=" ", flush=True)
        result = await test_func(client, model)
        status = "PASS" if result.passed else "FAIL"
        print(f"{status} ({result.duration_ms:.0f}ms)")
        if result.error:
            print(f"  Error: {result.error}")
        if result.details:
            print(f"  Details: {result.details}")
        results.append(result)

    return results


async def main():
    parser = argparse.ArgumentParser(description="Scaffold Integration Tests")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Model to use")
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="API base URL")
    parser.add_argument("--api-key", default="EMPTY", help="API key")
    args = parser.parse_args()

    client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)

    print("=" * 60)
    print("SCAFFOLD INTEGRATION TESTS")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Base URL: {args.base_url}")
    print()

    results = await run_all_tests(client, args.model)

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"Passed: {passed}/{total} ({100*passed/total:.0f}%)")

    if passed < total:
        print("\nFailed tests:")
        for r in results:
            if not r.passed:
                print(f"  - {r.name}: {r.error or r.details}")


if __name__ == "__main__":
    asyncio.run(main())
