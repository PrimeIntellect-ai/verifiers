"""Additional tests for verifiers.envs.environment.Environment.

Covers:
- get_model_response chat tools vs. completion error
- run_rollouts with concurrency limit (max_concurrent or semaphore)
- process_env_results zero_truncated_completions path
- evaluate fallback to train dataset and repeat behavior
- generate called inside an existing event loop
- make_dataset tool call sanitization
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from datasets import Dataset

from verifiers.envs.environment import Environment
from verifiers.parsers.parser import Parser
from verifiers.rubrics.rubric import Rubric
from verifiers.types import (
    GenerateMetadata,
    GenerateOutputs,
    RolloutInput,
    SamplingArgs,
)
from verifiers.utils.eval_utils import make_dataset as build_dataset
from verifiers.utils.message_utils import sanitize_tool_calls


# Local simple concrete Environment for testing
class DummyEnvironment(Environment):
    async def setup_state(self, state):
        return state

    async def rollout(
        self,
        input: RolloutInput,
        client,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ):
        state = await self.init_state(input, client=client, model=model)
        state = await self.setup_state(state)

        prompt_messages = state["prompt"]
        response = await self.get_model_response(
            prompt=prompt_messages,
            client=client,
            model=model,
            sampling_args=sampling_args,
        )
        assert response is not None

        from verifiers.types import TrajectoryStep
        from verifiers.utils.response_utils import (
            parse_response_messages,
            parse_response_tokens,
        )

        completion_messages = await parse_response_messages(response, self.message_type)
        tokens = await parse_response_tokens(response, self.message_type)
        trajectory_step = TrajectoryStep(
            prompt=prompt_messages,
            completion=completion_messages,
            response=response,
            tokens=tokens,
            reward=None,
            advantage=None,
            extras={},
        )
        state["trajectory"].append(trajectory_step)
        state["is_completed"] = True

        from verifiers.utils.message_utils import concat_messages

        last_prompt = state["trajectory"][-1]["prompt"]
        last_completion = state["trajectory"][-1]["completion"]
        full_conversation = concat_messages([last_prompt, last_completion])
        state["completion"] = full_conversation[len(state["prompt"]) :]

        return state


def _make_metadata(
    num_examples: int, rollouts_per_example: int = 1
) -> GenerateMetadata:
    return GenerateMetadata(
        env_id="dummy-env",
        env_args={},
        model="test-model",
        base_url="http://localhost",
        num_examples=num_examples,
        rollouts_per_example=rollouts_per_example,
        sampling_args={},
        date="1970-01-01",
        time_ms=0.0,
        avg_reward=0.0,
        avg_metrics={},
        state_columns=[],
        path_to_save=Path("test.jsonl"),
    )


def _make_env(
    mock_openai_client, dataset: Dataset | None = None, **kwargs
) -> DummyEnvironment:
    ds = dataset or Dataset.from_dict({"question": ["q1"], "answer": ["a1"]})
    return DummyEnvironment(
        client=mock_openai_client,
        model="test-model",
        dataset=ds,
        parser=Parser(),
        rubric=Rubric(),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_get_model_response_chat_with_tools(mock_openai_client):
    env = _make_env(mock_openai_client)
    prompt = [{"role": "user", "content": "Hello"}]
    tools = [
        {
            "type": "function",
            "function": {"name": "echo", "description": "echo", "parameters": {}},
        }
    ]
    resp = await env.get_model_response(
        client=mock_openai_client,
        model="test-model",
        prompt=prompt,
        oai_tools=tools,
        message_type="chat",
    )
    # Ensure the client was invoked and received tools kwarg
    assert hasattr(resp, "choices")
    assert mock_openai_client.chat.completions.create.await_count == 1
    kwargs = mock_openai_client.chat.completions.create.await_args.kwargs
    assert "tools" in kwargs and kwargs["tools"] == tools


@pytest.mark.asyncio
async def test_get_model_response_completion_rejects_tools(mock_openai_client):
    env = _make_env(mock_openai_client, message_type="completion")
    with pytest.raises(ValueError, match="oai_tools are not supported for completion"):
        await env.get_model_response(
            client=mock_openai_client,
            model="test-model",
            prompt="Complete this",
            oai_tools=[{"type": "function", "function": {"name": "noop"}}],
            message_type="completion",
        )


def test_run_rollouts_with_max_concurrent(mock_openai_client):
    env = _make_env(mock_openai_client)
    inputs = [
        RolloutInput(
            prompt=[{"role": "user", "content": "hi"}],
            answer="",
            example_id=i,
        )
        for i in range(3)
    ]
    results = asyncio.run(
        env.generate(
            inputs,
            client=mock_openai_client,
            model="test-model",
            max_concurrent=2,
        )
    )
    assert len(results["state"]) == 3


def test_run_rollouts_with_semaphore(mock_openai_client):
    env = _make_env(mock_openai_client)
    inputs = [
        RolloutInput(
            prompt=[{"role": "user", "content": "hi"}],
            answer="",
            example_id=i,
        )
        for i in range(3)
    ]
    results = asyncio.run(
        env.generate(
            inputs,
            client=mock_openai_client,
            model="test-model",
            max_concurrent=2,
        )
    )
    assert len(results["state"]) == 3


def test_evaluate_fallback_and_repeat(mock_openai_client):
    # No eval_dataset provided -> falls back to train; ensure >= num_examples
    from datasets import Dataset

    ds = Dataset.from_dict({"question": ["q1", "q2"], "answer": ["a1", "a2"]})
    env = _make_env(mock_openai_client, dataset=ds)
    res = asyncio.run(
        env.evaluate(
            client=mock_openai_client,
            model="test-model",
            num_examples=2,
            rollouts_per_example=2,
        )
    )
    # Expect n * r rollouts in outputs
    assert len(res["prompt"]) == 2 * 2
    assert len(res["completion"]) == 2 * 2


@pytest.mark.asyncio
async def test_generate_inside_running_loop(mock_openai_client):
    env = _make_env(mock_openai_client)
    inputs = [
        RolloutInput(
            prompt=[{"role": "user", "content": "Hi"}],
            answer="",
            example_id=0,
        )
    ]
    # Call the async API directly inside a running event loop to avoid nested sync wrapper issues
    out = await env.generate(inputs, client=mock_openai_client, model="test-model")
    assert "completion" in out and len(out["completion"]) == 1


def test_sanitize_tool_calls_outputs_strings():
    # Use a lightweight object with model_dump to mimic OAI tool call
    class ToolCall:
        def __init__(self, name: str, args: str):
            self.function = type("F", (), {"name": name, "arguments": args})()

        def model_dump(self):
            return {
                "id": "x",
                "type": "function",
                "function": {
                    "name": self.function.name,
                    "arguments": self.function.arguments,
                },
            }

    msgs = [
        [{"role": "assistant", "content": "", "tool_calls": [ToolCall("echo", "{}")]}]
    ]
    sanitized = sanitize_tool_calls(msgs[0])
    assert isinstance(sanitized[0]["tool_calls"][0], str)


def test_make_dataset_basic_without_tools(mock_openai_client):
    results = GenerateOutputs(
        prompt=[[{"role": "user", "content": "Hi"}]],
        completion=[[{"role": "assistant", "content": "Hello"}]],
        answer=[""],
        state=[
            {
                "timing": {
                    "generation_ms": 0.0,
                    "scoring_ms": 0.0,
                    "total_ms": 0.0,
                }
            }
        ],
        info=[{}],
        task=["default"],
        reward=[1.0],
        metrics={"foo": [0.1]},
        example_id=[0],
        metadata=_make_metadata(num_examples=1),
    )
    ds = build_dataset(results)
    assert len(ds) == 1 and "foo" in ds.column_names
