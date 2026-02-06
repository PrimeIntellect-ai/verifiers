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
import json
from typing import Callable

import pytest
from datasets import Dataset
from openai import AsyncOpenAI

import verifiers as vf
from verifiers.envs.environment import Environment
from verifiers.parsers.parser import Parser
from verifiers.rubrics.rubric import Rubric
from verifiers.types import (
    GenerateOutputs,
    RolloutInput,
    SamplingArgs,
)
from verifiers.utils.message_utils import sanitize_tool_calls
from verifiers.utils.save_utils import make_dataset as build_dataset


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
        state = await self.init_state(
            input, client=client, model=model, sampling_args=sampling_args
        )
        state = await self.setup_state(state)

        prompt_messages = state["prompt"]
        response = await self.get_model_response(state=state, prompt=prompt_messages)
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
            is_truncated=False,
            trajectory_id=state["trajectory_id"],
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


@pytest.fixture
def make_dummy_env() -> Callable[[AsyncOpenAI, Dataset | None], DummyEnvironment]:
    def _make_dummy_env(
        mock_openai_client: AsyncOpenAI, dataset: Dataset | None = None, **kwargs
    ) -> DummyEnvironment:
        dataset = dataset or Dataset.from_dict({"question": ["q1"], "answer": ["a1"]})
        return DummyEnvironment(
            client=mock_openai_client,
            model="test-model",
            dataset=dataset,
            parser=Parser(),
            rubric=Rubric(),
            **kwargs,
        )

    return _make_dummy_env


@pytest.mark.asyncio
async def test_get_model_response_chat_with_tools(
    mock_openai_client, make_dummy_env, make_input
):
    env = make_dummy_env(mock_openai_client)
    prompt: vf.Messages = [{"role": "user", "content": "Hello"}]
    tools = [
        {
            "type": "function",
            "function": {"name": "echo", "description": "echo", "parameters": {}},
        }
    ]
    state = await env.init_state(
        input=make_input(prompt=prompt),
        client=mock_openai_client,
        model="test-model",
    )
    state["oai_tools"] = tools
    resp = await env.get_model_response(
        state=state,
        prompt=prompt,
    )
    # Ensure the client was invoked and received tools kwarg
    assert hasattr(resp, "choices")
    assert mock_openai_client.chat.completions.create.await_count == 1
    kwargs = mock_openai_client.chat.completions.create.await_args.kwargs
    assert "tools" in kwargs and kwargs["tools"] == tools


@pytest.mark.asyncio
async def test_get_model_response_completion_rejects_tools(
    mock_openai_client, make_dummy_env, make_input
):
    env = make_dummy_env(mock_openai_client, message_type="completion")
    with pytest.raises(vf.ModelError):
        state = await env.init_state(
            input=make_input(prompt=[{"role": "user", "content": "Complete this"}]),
            client=mock_openai_client,
            model="test-model",
        )
        state["oai_tools"] = [{"type": "function", "function": {"name": "noop"}}]
        await env.get_model_response(state=state, prompt="Complete this")


def test_run_rollouts_with_max_concurrent(
    mock_openai_client, make_dummy_env, make_input
):
    env = make_dummy_env(mock_openai_client)
    inputs = [make_input(example_id=i) for i in range(3)]
    outputs = asyncio.run(
        env.generate(
            inputs,
            client=mock_openai_client,
            model="test-model",
            max_concurrent=2,
        )
    )
    states = outputs["outputs"]
    assert len(states) == 3


def test_evaluate_fallback_and_repeat(mock_openai_client, make_dummy_env, make_input):
    # No eval_dataset provided -> falls back to train; ensure >= num_examples
    from datasets import Dataset

    ds = Dataset.from_dict({"question": ["q1", "q2"], "answer": ["a1", "a2"]})
    env = make_dummy_env(mock_openai_client, dataset=ds)
    outputs = asyncio.run(
        env.evaluate(
            client=mock_openai_client,
            model="test-model",
            num_examples=2,
            rollouts_per_example=2,
        )
    )
    # Expect n * r rollouts in outputs
    states = outputs["outputs"]
    assert len(states) == 2 * 2


@pytest.mark.asyncio
async def test_generate_inside_running_loop(
    mock_openai_client, make_dummy_env, make_input
):
    env = make_dummy_env(mock_openai_client)
    inputs = [make_input(example_id=0)]
    # Call the async API directly inside a running event loop to avoid nested sync wrapper issues
    outputs = await env.generate(inputs, client=mock_openai_client, model="test-model")
    states = outputs["outputs"]
    assert len(states) == 1
    assert states[0].get("completion") is not None


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


def test_make_dataset_basic_without_tools(make_metadata, make_output):
    results = GenerateOutputs(outputs=[make_output()], metadata=make_metadata())
    ds = build_dataset(results)
    assert len(ds) == 1 and "foo" in ds.column_names


@pytest.mark.asyncio
async def test_generate_resume_raises_on_metadata_mismatch(
    tmp_path, mock_openai_client, make_dummy_env, make_input
):
    env = make_dummy_env(mock_openai_client)

    results_path = tmp_path / "resume"
    results_path.mkdir()
    (results_path / "results.jsonl").write_text("", encoding="utf-8")
    (results_path / "metadata.json").write_text(
        json.dumps(
            {
                "env_id": env.env_id,
                "model": "test-model",
                "num_examples": 2,
                "rollouts_per_example": 1,
            }
        ),
        encoding="utf-8",
    )

    inputs = [make_input(example_id=0)]
    with pytest.raises(ValueError, match="metadata mismatch"):
        await env.generate(
            inputs=inputs,
            client=mock_openai_client,
            model="test-model",
            results_path=results_path,
        )
