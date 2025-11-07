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
from typing import List

import pytest
from datasets import Dataset

from verifiers.envs.environment import Environment
from verifiers.parsers.parser import Parser
from verifiers.rubrics.rubric import Rubric
from verifiers.types import (
    GenerateMetadata,
    GenerateOutputs,
    Info,
    Messages,
    SamplingArgs,
    State,
)
from verifiers.utils.eval_utils import make_dataset as build_dataset
from verifiers.utils.message_utils import sanitize_tool_calls


# Local simple concrete Environment for testing
class DummyEnvironment(Environment):
    async def rollout(
        self,
        client,
        model,
        prompt: Messages,
        completion: Messages | None = None,
        answer: str = "",
        state: State | None = None,
        task: str = "default",
        info: Info | None = {},
        example_id: int = 0,
        sampling_args: SamplingArgs | None = None,
        **kwargs,
    ):
        response = await self.get_model_response(
            prompt=prompt, client=client, model=model, sampling_args=sampling_args
        )
        assert response is not None
        info = info or {}
        if completion is None:
            completion = await self.init_completion()
        if state is None:
            state = await self.init_state(
                prompt=prompt,
                completion=completion,
                answer=answer,
                task=task,
                info=info,
                example_id=example_id,
            )
        if self.message_type == "chat":
            assert isinstance(completion, list)
            state.setdefault("responses", [])
            state["responses"].append(response)
            message = {
                "role": "assistant",
                "content": response.choices[0].message.content,
            }
            completion.append(message)
            state["completion"] = completion
        else:
            assert isinstance(completion, str)
            state.setdefault("responses", [])
            state["responses"].append(response)
            completion = completion + (response.choices[0].text or "")
            state["completion"] = completion
        return completion, state


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
    prompts = [[{"role": "user", "content": "hi"}] for _ in range(3)]
    answers = ["", "", ""]
    coro = env.run_rollouts(
        client=mock_openai_client,
        model="test-model",
        prompts=prompts,
        answers=answers,
        tasks=["default"] * 3,
        infos=[{}] * 3,
        max_concurrent=2,
        example_ids=list(range(len(prompts))),
    )
    results: List = asyncio.run(coro)
    assert len(results) == 3


def test_run_rollouts_with_semaphore(mock_openai_client):
    env = _make_env(mock_openai_client)
    prompts = [[{"role": "user", "content": "hi"}] for _ in range(3)]
    answers = ["", "", ""]
    coro = env.run_rollouts(
        client=mock_openai_client,
        model="test-model",
        prompts=prompts,
        answers=answers,
        tasks=["default"] * 3,
        infos=[{}] * 3,
        semaphore=asyncio.Semaphore(2),
        example_ids=list(range(len(prompts))),
    )
    results: List = asyncio.run(coro)
    assert len(results) == 3


def test_process_env_results_zero_truncated_reward_vllm(mock_openai_client):
    # Use pre-formatted dataset to avoid map/progress side effects in test
    ds = Dataset.from_dict(
        {
            "prompt": [[{"role": "user", "content": "q"}]],
            "answer": ["a"],
        }
    )
    env = _make_env(mock_openai_client, dataset=ds, message_type="completion")

    # Mock tokenizer: encode maps length to token list
    class Tok:
        def encode(self, text, **kwargs):
            return list(range(len(text)))

    prompts = ["Hello!"]  # 6 tokens
    completions = ["World!!!"]  # 8 tokens
    # Minimal vLLM-style completion response covering entire completion text
    mock_choice = type("C", (), {})()
    mock_choice.text = completions[0]
    mock_choice.logprobs = type("LP", (), {})()
    mock_choice.logprobs.tokens = ["token_id:1"] * len(completions[0])
    mock_choice.logprobs.token_logprobs = [-0.1] * len(completions[0])
    mock_completion = type("R", (), {})()
    mock_completion.choices = [mock_choice]
    states = [{"responses": [mock_completion], "responses_start_idx": [0]}]
    rewards = [1.0]

    out = env.process_env_results_vllm(
        prompts,
        completions,
        states,
        rewards,
        Tok(),  # type: ignore[arg-type]
        max_seq_len=10,  # force truncation (6 + 8 > 10)
        mask_truncated_completions=True,
        zero_truncated_completions=True,
    )

    assert out.rewards == [0.0]
    assert len(out.prompt_ids[0]) + len(out.completion_ids[0]) <= 10
    print("end_zero_truncated")


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
            score_rollouts=False,
            interleave_scoring=False,
        )
    )
    # Expect n * r rollouts in outputs
    assert len(res.prompt) == 2 * 2
    assert len(res.completion) == 2 * 2


@pytest.mark.asyncio
async def test_generate_inside_running_loop(mock_openai_client):
    env = _make_env(mock_openai_client)
    inputs = {
        "prompt": [[{"role": "user", "content": "Hi"}]],
        "answer": [""],
        "example_id": [0],
    }
    # Call the async API directly inside a running event loop to avoid nested sync wrapper issues
    out = await env.a_generate(
        inputs, client=mock_openai_client, model="test-model", interleave_scoring=False
    )
    assert hasattr(out, "completion") and len(out.completion) == 1


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


def test_truncation_masks_completion_format_vllm(mock_openai_client):
    # Duplicate of zero_truncated test under a different name to avoid any runner quirk
    ds = Dataset.from_dict(
        {
            "prompt": [[{"role": "user", "content": "q"}]],
            "answer": ["a"],
        }
    )
    env = _make_env(mock_openai_client, dataset=ds, message_type="completion")

    class Tok:
        def encode(self, text, **kwargs):
            return list(range(len(text)))

    prompts = ["Hello!"]
    completions = ["World!!!"]
    # Minimal vLLM-style completion response covering entire completion text
    mock_choice2 = type("C2", (), {})()
    mock_choice2.text = completions[0]
    mock_choice2.logprobs = type("LP2", (), {})()
    mock_choice2.logprobs.tokens = ["token_id:1"] * len(completions[0])
    mock_choice2.logprobs.token_logprobs = [-0.1] * len(completions[0])
    mock_completion2 = type("R2", (), {})()
    mock_completion2.choices = [mock_choice2]
    out = env.process_env_results_vllm(
        prompts,
        completions,
        [{"responses": [mock_completion2], "responses_start_idx": [0]}],
        [1.0],
        Tok(),  # type: ignore[arg-type]
        max_seq_len=10,
        mask_truncated_completions=True,
        zero_truncated_completions=True,
    )
    assert out.rewards == [0.0]
    assert len(out.prompt_ids[0]) + len(out.completion_ids[0]) <= 10
