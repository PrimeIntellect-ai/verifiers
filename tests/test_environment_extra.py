"""Additional tests for verifiers.envs.environment.Environment.

Covers:
- get_model_response chat tools vs. completion error
- run_rollouts with semaphore
- process_env_results zero_truncated_completions path
- evaluate fallback to train dataset and repeat behavior
- generate called inside an existing event loop
- make_dataset tool call sanitization
"""

from __future__ import annotations

import asyncio
from typing import List

import pytest
from datasets import Dataset

from verifiers.envs.environment import Environment
from verifiers.parsers.parser import Parser
from verifiers.rubrics.rubric import Rubric
from verifiers.types import GenerateOutputs


# Local simple concrete Environment for testing
class DummyEnvironment(Environment):
    async def rollout(
        self,
        client,
        model,
        prompt,
        answer: str = "",
        task: str = "default",
        info: dict = {},
        sampling_args: dict = {},
        **kwargs,
    ):
        response = await self.get_model_response(
            prompt=prompt, client=client, model=model, sampling_args=sampling_args
        )
        if self.message_type == "chat":
            completion = [
                {"role": "assistant", "content": response.choices[0].message.content}
            ]
            state = {"responses": [response]}
        else:
            completion = response.choices[0].text
            state = {}
        return completion, state


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
        max_concurrent=2,
    )
    results: List = asyncio.run(coro)
    assert len(results) == 3


def test_process_env_results_zero_truncated_reward_vllm(mock_openai_client):
    print("begin_zero_truncated")
    # Use pre-formatted dataset to avoid map/progress side effects in test
    ds = Dataset.from_dict(
        {
            "prompt": [[{"role": "user", "content": "q"}]],
            "answer": ["a"],
        }
    )
    env = _make_env(mock_openai_client, dataset=ds)

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
        Tok(),
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
    res = env.evaluate(
        client=mock_openai_client,
        model="test-model",
        num_examples=2,
        rollouts_per_example=2,
        score_rollouts=False,
    )
    # Expect n * r rollouts in outputs
    assert len(res.prompt) == 2 * 2
    assert len(res.completion) == 2 * 2


@pytest.mark.asyncio
async def test_generate_inside_running_loop(mock_openai_client):
    env = _make_env(mock_openai_client)
    inputs = {"prompt": [[{"role": "user", "content": "Hi"}]], "answer": [""]}
    # Call the async API directly inside a running event loop to avoid nested sync wrapper issues
    out = await env.a_generate(inputs, client=env.client)
    assert hasattr(out, "completion") and len(out.completion) == 1


def test_sanitize_tool_calls_outputs_strings(mock_openai_client):
    env = _make_env(mock_openai_client)

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
    sanitized = env._sanitize_tool_calls(msgs[0])
    assert isinstance(sanitized[0]["tool_calls"][0], str)


def test_make_dataset_basic_without_tools(mock_openai_client):
    env = _make_env(mock_openai_client)
    results = GenerateOutputs(
        prompt=[[{"role": "user", "content": "Hi"}]],
        completion=[[{"role": "assistant", "content": "Hello"}]],
        answer=[""],
        state=[{}],
        info=[{}],
        task=["default"],
        reward=[1.0],
        metrics={"foo": [0.1]},
    )
    ds = env.make_dataset(results)
    assert len(ds) == 1 and "foo" in ds.column_names


def test_truncation_masks_completion_format_vllm(mock_openai_client):
    # Duplicate of zero_truncated test under a different name to avoid any runner quirk
    ds = Dataset.from_dict(
        {
            "prompt": [[{"role": "user", "content": "q"}]],
            "answer": ["a"],
        }
    )
    env = _make_env(mock_openai_client, dataset=ds)

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
        Tok(),
        max_seq_len=10,
        mask_truncated_completions=True,
        zero_truncated_completions=True,
    )
    assert out.rewards == [0.0]
    assert len(out.prompt_ids[0]) + len(out.completion_ids[0]) <= 10
