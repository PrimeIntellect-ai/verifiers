from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import verifiers.v1 as vf
from verifiers.clients.anthropic_messages_client import AnthropicMessagesClient
from verifiers.clients.client import Client
from verifiers.clients.openai_chat_completions_client import OpenAIChatCompletionsClient
from verifiers.types import ClientConfig, Response, ResponseMessage, Usage
from verifiers.utils.prompt_cache_utils import (
    EndpointIdentity,
    apply_prompt_cache_to_request,
    resolve_prompt_cache_policy,
    should_prefire_prompt_cache_group,
)
from verifiers.utils.save_utils import state_to_output
from verifiers.utils.usage_utils import extract_usage_token_details


class RecordingClient(Client):
    def __init__(self, config: ClientConfig):
        super().__init__(config)
        self.request = {}

    def setup_client(self, config):
        return object()

    async def to_native_tool(self, tool):
        return tool

    async def to_native_prompt(self, messages):
        return messages, {}

    async def get_native_response(self, prompt, model, sampling_args, tools=None, **kwargs):
        self.request = {
            "prompt": prompt,
            "model": model,
            "sampling_args": sampling_args,
            "tools": tools,
            "kwargs": kwargs,
        }
        return object()

    async def raise_from_native_response(self, response):
        _ = response

    async def from_native_response(self, response):
        _ = response
        return Response(
            id="resp",
            created=0,
            model="model",
            usage=None,
            message=ResponseMessage(
                content="ok",
                finish_reason="stop",
                is_truncated=False,
            ),
        )

    async def close(self) -> None:
        pass


class GroupOrderClient(Client):
    def __init__(self, config: ClientConfig):
        super().__init__(config)
        self.first_active = False
        self.started_during_first: list[int] = []

    def setup_client(self, config):
        return object()

    async def get_response(self, prompt, model, sampling_args, tools=None, **kwargs):
        _ = prompt, model, sampling_args, tools
        state = kwargs["state"]
        rollout_index = int(state["rollout_index"])
        if rollout_index == 0:
            self.first_active = True
            await asyncio.sleep(0.01)
            self.first_active = False
        elif self.first_active:
            self.started_during_first.append(rollout_index)
        return Response(
            id=f"resp-{rollout_index}",
            created=0,
            model="model",
            usage=Usage(
                prompt_tokens=1,
                reasoning_tokens=0,
                completion_tokens=1,
                total_tokens=2,
            ),
            message=ResponseMessage(
                content="ok",
                finish_reason="stop",
                is_truncated=False,
            ),
        )

    async def to_native_tool(self, tool):
        return tool

    async def to_native_prompt(self, messages):
        return messages, {}

    async def get_native_response(self, prompt, model, sampling_args, tools=None, **kwargs):
        raise AssertionError("get_response is implemented directly")

    async def raise_from_native_response(self, response):
        _ = response

    async def from_native_response(self, response):
        _ = response

    async def close(self) -> None:
        pass


class IndexedTaskset(vf.Taskset):
    async def init_group(self, task, num_rollouts):
        tasks, states = await super().init_group(task, num_rollouts)
        for index, state in enumerate(states):
            state["rollout_index"] = index
        return tasks, states


def test_endpoint_identity_normalizes_official_origins():
    identity = EndpointIdentity.from_url(
        "https://api.openai.com/v1", "openai_chat_completions"
    )

    assert identity is not None
    assert identity.origin == "https://api.openai.com"
    assert identity.host == "api.openai.com"
    assert identity.path == "/v1"


def test_prompt_cache_policy_is_inferred_from_url_and_type():
    assert (
        resolve_prompt_cache_policy(
            ClientConfig(
                client_type="openai_responses",
                api_base_url="https://api.openai.com/v1",
            ),
            "gpt-5.4-mini",
        ).mode
        == "implicit"
    )
    assert (
        resolve_prompt_cache_policy(
            ClientConfig(
                client_type="anthropic_messages",
                api_base_url="https://api.anthropic.com",
            ),
            "claude-sonnet-4-5",
        ).mode
        == "anthropic_top_level"
    )
    assert (
        resolve_prompt_cache_policy(
            ClientConfig(
                client_type="openai_chat_completions",
                api_base_url="https://openrouter.ai/api/v1",
            ),
            "anthropic/claude-sonnet-4.5",
        ).mode
        == "openrouter_anthropic_top_level"
    )
    assert (
        resolve_prompt_cache_policy(
            ClientConfig(
                client_type="openai_chat_completions",
                api_base_url="https://api.example.com/v1",
            ),
            "model",
        ).mode
        == "disabled"
    )


def test_prompt_cache_false_disables_inferred_provider_policy():
    policy = resolve_prompt_cache_policy(
        ClientConfig(
            client_type="openai_chat_completions",
            api_base_url="https://api.openai.com/v1",
            prompt_cache=False,
        ),
        "gpt-5.4-mini",
    )

    assert policy.mode == "disabled"
    assert not policy.prefire_groups


def test_anthropic_request_policy_adds_top_level_cache_control():
    native_prompt, native_tools, sampling_args, extra_kwargs = (
        apply_prompt_cache_to_request(
            config=ClientConfig(
                client_type="anthropic_messages",
                api_base_url="https://api.anthropic.com",
            ),
            model="claude-sonnet-4-5",
            native_prompt=[{"role": "user", "content": "question"}],
            native_tools=None,
            sampling_args={"max_tokens": 16},
            extra_kwargs={},
        )
    )

    assert native_prompt == [{"role": "user", "content": "question"}]
    assert native_tools is None
    assert sampling_args == {"max_tokens": 16}
    assert extra_kwargs["cache_control"] == {"type": "ephemeral"}


def test_openrouter_anthropic_policy_uses_extra_body_cache_control():
    native_prompt, native_tools, sampling_args, extra_kwargs = (
        apply_prompt_cache_to_request(
            config=ClientConfig(
                client_type="openai_chat_completions",
                api_base_url="https://openrouter.ai/api/v1",
            ),
            model="anthropic/claude-sonnet-4.5",
            native_prompt=[{"role": "user", "content": "question"}],
            native_tools=[],
            sampling_args={"max_tokens": 16, "extra_body": {"foo": "bar"}},
            extra_kwargs={},
        )
    )

    assert native_prompt == [{"role": "user", "content": "question"}]
    assert native_tools == []
    assert extra_kwargs == {}
    assert sampling_args["extra_body"] == {
        "foo": "bar",
        "cache_control": {"type": "ephemeral"},
    }


def test_openai_policy_does_not_mutate_request():
    native_prompt, native_tools, sampling_args, extra_kwargs = (
        apply_prompt_cache_to_request(
            config=ClientConfig(
                client_type="openai_chat_completions",
                api_base_url="https://api.openai.com/v1",
            ),
            model="gpt-5.4-mini",
            native_prompt=[{"role": "user", "content": "question"}],
            native_tools=None,
            sampling_args={"max_tokens": 16},
            extra_kwargs={},
        )
    )

    assert native_prompt == [{"role": "user", "content": "question"}]
    assert native_tools is None
    assert sampling_args == {"max_tokens": 16}
    assert extra_kwargs == {}


def test_group_prefire_is_tied_to_cache_policy():
    assert should_prefire_prompt_cache_group(
        ClientConfig(
            client_type="openai_chat_completions",
            api_base_url="https://api.openai.com/v1",
        ),
        "gpt-5.4-mini",
        2,
    )
    assert not should_prefire_prompt_cache_group(
        ClientConfig(
            client_type="openai_chat_completions",
            api_base_url="https://api.openai.com/v1",
            prompt_cache=False,
        ),
        "gpt-5.4-mini",
        2,
    )
    assert not should_prefire_prompt_cache_group(
        ClientConfig(
            client_type="openai_chat_completions",
            api_base_url="https://api.example.com/v1",
        ),
        "model",
        2,
    )
    assert not should_prefire_prompt_cache_group(
        ClientConfig(
            client_type="openai_chat_completions",
            api_base_url="https://api.openai.com/v1",
        ),
        "gpt-5.4-mini",
        1,
    )


@pytest.mark.asyncio
async def test_client_request_hook_applies_prompt_cache_policy():
    client = RecordingClient(
        ClientConfig(
            client_type="anthropic_messages",
            api_base_url="https://api.anthropic.com",
        )
    )

    await client.get_response(
        prompt=[],
        model="claude-sonnet-4-5",
        sampling_args={"max_tokens": 16},
    )

    assert client.request["kwargs"]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.asyncio
async def test_v1_group_prefire_serializes_first_rollout_for_cached_provider():
    client = GroupOrderClient(
        ClientConfig(
            client_type="openai_chat_completions",
            api_base_url="https://api.openai.com/v1",
        )
    )
    env = vf.Env(
        taskset=IndexedTaskset(source=[{"question": "q"}]),
        harness=vf.Harness(max_turns=1),
    )

    await env._run_group_states(
        [
            {"prompt": [{"role": "user", "content": "q"}], "example_id": 0},
            {"prompt": [{"role": "user", "content": "q"}], "example_id": 0},
            {"prompt": [{"role": "user", "content": "q"}], "example_id": 0},
        ],
        client,
        "gpt-5.4-mini",
        {},
    )

    assert client.started_during_first == []


@pytest.mark.asyncio
async def test_v1_group_prefire_is_skipped_for_generic_provider():
    client = GroupOrderClient(
        ClientConfig(
            client_type="openai_chat_completions",
            api_base_url="https://api.example.com/v1",
        )
    )
    env = vf.Env(
        taskset=IndexedTaskset(source=[{"question": "q"}]),
        harness=vf.Harness(max_turns=1),
    )

    await env._run_group_states(
        [
            {"prompt": [{"role": "user", "content": "q"}], "example_id": 0},
            {"prompt": [{"role": "user", "content": "q"}], "example_id": 0},
            {"prompt": [{"role": "user", "content": "q"}], "example_id": 0},
        ],
        client,
        "model",
        {},
    )

    assert client.started_during_first


@pytest.mark.asyncio
async def test_openai_usage_splits_cached_input_tokens():
    client = OpenAIChatCompletionsClient(object())
    message = SimpleNamespace(
        content="ok",
        tool_calls=None,
        model_dump=lambda: {},
    )
    native_response = SimpleNamespace(
        id="resp",
        created=0,
        model="gpt-5.4-mini",
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=5,
            total_tokens=105,
            prompt_tokens_details=SimpleNamespace(
                cached_tokens=80,
                cache_write_tokens=10,
            ),
        ),
        choices=[
            SimpleNamespace(
                message=message,
                finish_reason="stop",
            )
        ],
    )

    response = await client.from_native_response(native_response)

    assert response.usage is not None
    assert response.usage.prompt_tokens == 20
    assert response.usage.cached_input_tokens == 80
    assert response.usage.cache_write_input_tokens == 10
    assert response.usage.total_tokens == 25


@pytest.mark.asyncio
async def test_anthropic_usage_splits_cache_read_and_write_tokens():
    client = AnthropicMessagesClient(object())
    native_response = SimpleNamespace(
        id="resp",
        model="claude-sonnet-4-5",
        stop_reason="end_turn",
        content=[SimpleNamespace(type="text", text="ok")],
        usage=SimpleNamespace(
            input_tokens=5,
            output_tokens=7,
            cache_read_input_tokens=80,
            cache_creation_input_tokens=10,
        ),
    )

    response = await client.from_native_response(native_response)

    assert response.usage is not None
    assert response.usage.prompt_tokens == 15
    assert response.usage.cached_input_tokens == 80
    assert response.usage.cache_write_input_tokens == 10
    assert response.usage.total_tokens == 22


def test_native_anthropic_usage_counts_cache_writes_as_uncached_input():
    response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=5,
            output_tokens=7,
            cache_read_input_tokens=80,
            cache_creation_input_tokens=10,
        )
    )

    assert extract_usage_token_details(response) == {
        "input_tokens": 15,
        "output_tokens": 7,
        "cached_input_tokens": 80,
        "cache_write_input_tokens": 10,
    }


def test_serialized_response_usage_counts_cache_details():
    response = {
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 7,
            "prompt_tokens_details": {
                "cached_tokens": 80,
                "cache_write_tokens": 10,
            },
        }
    }

    assert extract_usage_token_details(response) == {
        "input_tokens": 20,
        "output_tokens": 7,
        "cached_input_tokens": 80,
        "cache_write_input_tokens": 10,
    }


def test_state_output_fallback_reads_serialized_trajectory_usage():
    task = vf.Task(
        {
            "example_id": 0,
            "prompt": [{"role": "user", "content": "q"}],
        }
    ).freeze()
    state = vf.State.for_task(task)
    state["trajectory"] = [
        {
            "response": {
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 7,
                    "prompt_tokens_details": {"cached_tokens": 80},
                }
            }
        }
    ]

    output = state_to_output(state)

    assert output["token_usage"]["input_tokens"] == 20.0
    assert output["token_usage"]["cached_input_tokens"] == 80.0
    assert output["token_usage"]["output_tokens"] == 7.0
