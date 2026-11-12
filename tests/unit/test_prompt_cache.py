"""Gateway prompt-cache markers for Anthropic models on evaluator relays."""

from copy import deepcopy

import pytest

from verifiers.v1.clients.prompt_cache import maybe_add_prompt_cache_control
from verifiers.v1.dialects.anthropic import AnthropicDialect
from verifiers.v1.dialects.chat import ChatDialect
from verifiers.v1.dialects.responses import ResponsesDialect

CACHE = {"type": "ephemeral"}


def _anthropic_body(history: bool = False) -> dict:
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "dynamic rollout turn"}],
        }
    ]
    if history:
        messages += [
            {"role": "assistant", "content": "working"},
            {"role": "user", "content": "continue"},
        ]
    return {
        "model": "anthropic/claude-opus-4.8",
        "system": [{"type": "text", "text": "stable instructions"}],
        "tools": [
            {
                "name": "read_file",
                "description": "Read a file",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
        "messages": messages,
    }


def _chat_body(history: bool = False) -> dict:
    messages = [
        {"role": "system", "content": "stable instructions"},
        {"role": "user", "content": "dynamic rollout turn"},
    ]
    if history:
        messages += [
            {"role": "assistant", "content": "working"},
            {"role": "user", "content": "continue"},
        ]
    return {
        "model": "anthropic/claude-opus-4.8",
        "messages": messages,
        "tools": [
            {
                "type": "function",
                "function": {"name": "read_file", "parameters": {}},
            }
        ],
    }


@pytest.mark.parametrize(
    "base_url",
    ["https://api.pinference.ai/api/v1", "https://openrouter.ai/api/v1"],
)
def test_anthropic_one_shot_marks_only_static_prefix(base_url: str) -> None:
    body = _anthropic_body()
    original = deepcopy(body)

    updated = maybe_add_prompt_cache_control(AnthropicDialect(), body, base_url)

    assert updated is not body
    assert body == original
    assert updated["tools"][-1]["cache_control"] == CACHE
    assert updated["system"][-1]["cache_control"] == CACHE
    assert "cache_control" not in updated["messages"][-1]["content"][0]
    assert "ttl" not in str(updated)


def test_long_retention_emits_one_hour_ttl() -> None:
    body = _anthropic_body()

    updated = maybe_add_prompt_cache_control(
        AnthropicDialect(),
        body,
        "https://api.pinference.ai/api/v1",
        cache_retention="long",
    )

    assert updated["tools"][-1]["cache_control"] == {
        "type": "ephemeral",
        "ttl": "1h",
    }
    assert updated["system"][-1]["cache_control"] == {
        "type": "ephemeral",
        "ttl": "1h",
    }


def test_none_retention_disables_markers() -> None:
    body = _anthropic_body()
    original = deepcopy(body)

    updated = maybe_add_prompt_cache_control(
        AnthropicDialect(),
        body,
        "https://api.pinference.ai/api/v1",
        cache_retention="none",
    )

    assert updated is body
    assert body == original


def test_env_long_retention_promotes_but_explicit_short_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PI_CACHE_RETENTION", "long")
    body = _anthropic_body()

    promoted = maybe_add_prompt_cache_control(
        AnthropicDialect(), body, "https://api.pinference.ai/api/v1"
    )
    assert promoted["tools"][-1]["cache_control"]["ttl"] == "1h"

    explicit = maybe_add_prompt_cache_control(
        AnthropicDialect(),
        body,
        "https://api.pinference.ai/api/v1",
        cache_retention="short",
    )
    assert explicit["tools"][-1]["cache_control"] == CACHE


def test_anthropic_multi_turn_marks_latest_message() -> None:
    body = _anthropic_body(history=True)

    updated = maybe_add_prompt_cache_control(
        AnthropicDialect(), body, "https://api.pinference.ai/api/v1"
    )

    assert updated["messages"][-1]["content"][0]["cache_control"] == CACHE
    assert "ttl" not in str(updated)


def test_chat_marks_static_prefix_and_latest_turn() -> None:
    body = _chat_body(history=True)
    original = deepcopy(body)

    updated = maybe_add_prompt_cache_control(
        ChatDialect(), body, "https://api.pinference.ai/api/v1"
    )

    assert body == original
    assert updated["tools"][-1]["cache_control"] == CACHE
    assert updated["messages"][0]["content"][0]["cache_control"] == CACHE
    assert updated["messages"][-1]["content"][0]["cache_control"] == CACHE
    assert "ttl" not in str(updated)


def test_anthropic_server_tool_falls_back_to_custom_tool() -> None:
    body = _anthropic_body()
    body["tools"] = [
        {
            "name": "read_file",
            "description": "Read a file",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 5,
        },
    ]

    updated = maybe_add_prompt_cache_control(
        AnthropicDialect(), body, "https://api.pinference.ai/api/v1"
    )

    assert updated["tools"][0]["cache_control"] == CACHE
    assert "cache_control" not in updated["tools"][1]
    assert updated["system"][-1]["cache_control"] == CACHE


def test_malformed_chat_roles_do_not_raise() -> None:
    body = _chat_body(history=True)
    body["messages"][0]["role"] = []
    body["messages"][-1]["role"] = []

    updated = maybe_add_prompt_cache_control(
        ChatDialect(), body, "https://api.pinference.ai/api/v1"
    )

    assert isinstance(updated, dict)


def test_malformed_anthropic_block_types_do_not_raise() -> None:
    body = _anthropic_body(history=True)
    body["messages"][-1]["content"] = [{"type": [], "text": "bad"}]

    updated = maybe_add_prompt_cache_control(
        AnthropicDialect(), body, "https://api.pinference.ai/api/v1"
    )

    assert isinstance(updated, dict)


def test_nested_tool_result_marker_suppresses_injection() -> None:
    body = _anthropic_body(history=True)
    body["messages"][-1]["content"] = [
        {
            "type": "tool_result",
            "content": [
                {
                    "type": "text",
                    "text": "nested",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
    ]
    original = deepcopy(body)

    updated = maybe_add_prompt_cache_control(
        AnthropicDialect(), body, "https://api.pinference.ai/api/v1"
    )

    assert updated is body
    assert body == original


def test_input_schema_property_named_cache_control_is_not_a_marker() -> None:
    body = _anthropic_body()
    body["tools"] = [
        {
            "name": "lookup",
            "input_schema": {
                "type": "object",
                "properties": {"cache_control": {"type": "string"}},
            },
        }
    ]

    updated = maybe_add_prompt_cache_control(
        AnthropicDialect(), body, "https://api.pinference.ai/api/v1"
    )

    assert updated["tools"][-1]["cache_control"] == CACHE


def test_claude_substring_is_not_an_anthropic_model() -> None:
    body = _chat_body()
    body["model"] = "org/qwen-claude-distill"
    original = deepcopy(body)

    updated = maybe_add_prompt_cache_control(
        ChatDialect(), body, "https://api.pinference.ai/api/v1"
    )

    assert updated is body
    assert body == original


def test_caller_marker_and_ttl_are_left_alone() -> None:
    body = _anthropic_body()
    body["system"] = [
        {
            "type": "text",
            "text": "stable instructions",
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        }
    ]
    original = deepcopy(body)

    updated = maybe_add_prompt_cache_control(
        AnthropicDialect(), body, "https://api.pinference.ai/api/v1"
    )

    assert updated is body
    assert body == original
    assert "cache_control" not in body["tools"][-1]


def test_non_gateway_host_is_untouched() -> None:
    body = _anthropic_body()
    original = deepcopy(body)

    updated = maybe_add_prompt_cache_control(
        AnthropicDialect(), body, "https://api.anthropic.com/v1"
    )

    assert updated is body
    assert body == original


def test_non_anthropic_model_is_untouched() -> None:
    body = _chat_body()
    body["model"] = "openai/gpt-5-mini"
    original = deepcopy(body)

    updated = maybe_add_prompt_cache_control(
        ChatDialect(), body, "https://api.pinference.ai/api/v1"
    )

    assert updated is body
    assert body == original


def test_responses_requests_are_not_rewritten() -> None:
    body = {"model": "anthropic/claude-opus-4.8", "input": "Find the bug."}
    original = deepcopy(body)

    updated = maybe_add_prompt_cache_control(
        ResponsesDialect(), body, "https://api.pinference.ai/api/v1"
    )

    assert updated is body
    assert body == original
