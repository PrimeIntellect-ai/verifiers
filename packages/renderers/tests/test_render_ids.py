"""Barrage test: renderer.render_ids() must match tokenizer.apply_chat_template().

Every test case runs against every (model, renderer) pair from conftest.
If a test passes, the renderer is token-for-token correct for that case.

GPT-OSS is auto-skipped here by ``conftest._skip_gpt_oss_for_hf_parity_tests``
since our GptOssRenderer matches openai-harmony / vLLM, not the HF Jinja
template. See ``test_gpt_oss_harmony_parity.py`` for harmony parity coverage.
"""

from functools import lru_cache

from renderers import create_renderer
from transformers import AutoTokenizer


def _expected(tokenizer, messages, **kwargs):
    # Match the Renderer Protocol's default for add_generation_prompt (False)
    # — some tokenizers (e.g. Kimi's) default it to True in their config,
    # which would otherwise make this parity check fail on the flag alone.
    # Callers that explicitly want the gen prompt still pass it through.
    kwargs.setdefault("add_generation_prompt", False)
    result = tokenizer.apply_chat_template(
        messages, tokenize=True, return_dict=False, **kwargs
    )
    if isinstance(result, dict):
        return list(result["input_ids"])
    if isinstance(result, str):
        # Some tokenizers return str even with tokenize=True; force encode
        return list(tokenizer.encode(result, add_special_tokens=False))
    return list(result)


# ── Basic messages ───────────────────────────────────────────────────


def test_system_and_user(model_name, tokenizer, renderer):
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"},
    ]
    assert renderer.render_ids(msgs) == _expected(tokenizer, msgs)


def test_single_turn(model_name, tokenizer, renderer):
    msgs = [
        {"role": "system", "content": "You are a math tutor."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ]
    assert renderer.render_ids(msgs) == _expected(tokenizer, msgs)


def test_no_system_message(model_name, tokenizer, renderer):
    msgs = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    assert renderer.render_ids(msgs) == _expected(tokenizer, msgs)


def test_multi_turn(model_name, tokenizer, renderer):
    msgs = [
        {"role": "user", "content": "A"},
        {"role": "assistant", "content": "B"},
        {"role": "user", "content": "C"},
        {"role": "assistant", "content": "D"},
    ]
    assert renderer.render_ids(msgs) == _expected(tokenizer, msgs)


def test_multi_turn_many_rounds(model_name, tokenizer, renderer):
    msgs = [
        {"role": "user", "content": "A"},
        {"role": "assistant", "content": "B"},
        {"role": "user", "content": "C"},
        {"role": "assistant", "content": "D"},
        {"role": "user", "content": "E"},
        {"role": "assistant", "content": "F"},
    ]
    assert renderer.render_ids(msgs) == _expected(tokenizer, msgs)


def test_empty_assistant_content(model_name, tokenizer, renderer):
    msgs = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": ""},
    ]
    assert renderer.render_ids(msgs) == _expected(tokenizer, msgs)


# ── Thinking / reasoning ────────────────────────────────────────────


def test_reasoning_content_field(model_name, tokenizer, renderer):
    msgs = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "reasoning_content": "Simple arithmetic", "content": "4"},
    ]
    assert renderer.render_ids(msgs) == _expected(tokenizer, msgs)


def test_thinking_multi_turn(model_name, tokenizer, renderer):
    msgs = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "reasoning_content": "greeting", "content": "Hi!"},
        {"role": "user", "content": "Bye"},
        {"role": "assistant", "reasoning_content": "farewell", "content": "Goodbye!"},
    ]
    assert renderer.render_ids(msgs) == _expected(tokenizer, msgs)


# ── Tool definitions ─────────────────────────────────────────────────


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city name"},
                },
                "required": ["city"],
            },
        },
    }
]


def test_tools_with_system(model_name, tokenizer, renderer):
    msgs = [
        {"role": "system", "content": "You are a weather assistant."},
        {"role": "user", "content": "Weather?"},
    ]
    assert renderer.render_ids(msgs, tools=TOOLS) == _expected(
        tokenizer, msgs, tools=TOOLS
    )


def test_tools_without_system(model_name, tokenizer, renderer):
    msgs = [{"role": "user", "content": "Weather?"}]
    assert renderer.render_ids(msgs, tools=TOOLS) == _expected(
        tokenizer, msgs, tools=TOOLS
    )


# ── Tool calls ───────────────────────────────────────────────────────


def test_tool_call_with_content(model_name, tokenizer, renderer):
    msgs = [
        {"role": "user", "content": "Weather in Paris?"},
        {
            "role": "assistant",
            "content": "Let me check.",
            "tool_calls": [
                {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
            ],
        },
    ]
    assert renderer.render_ids(msgs, tools=TOOLS) == _expected(
        tokenizer, msgs, tools=TOOLS
    )


def test_tool_call_no_content(model_name, tokenizer, renderer):
    msgs = [
        {"role": "user", "content": "Weather in Paris?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
            ],
        },
    ]
    try:
        expected = _expected(tokenizer, msgs, tools=TOOLS)
    except TypeError as exc:
        # Qwen3-VL upstream chat-template bug. The assistant branch only
        # checks `is string` and then unconditionally iterates the else,
        # so `content=None` raises TypeError on `for x in None`:
        #
        #   {%- if message.content is string %}
        #       {{- message.content }}
        #   {%- else %}
        #       {%- for content_item in message.content %}   ← crash
        #           {%- if 'text' in content_item %}
        #               {{- content_item.text }}
        #           ...
        #
        # Same pattern in the user-role branch. The fix upstream would be
        # `{%- elif message.content %}` to skip iteration on falsy content.
        # Until Qwen ships a fixed template this is xfail; the imperative
        # xfail auto-promotes to xpass when upstream is fixed.
        import pytest

        pytest.xfail(f"{model_name}: apply_chat_template raised {exc!r}")
    assert renderer.render_ids(msgs, tools=TOOLS) == expected


def test_multiple_tool_calls(model_name, tokenizer, renderer):
    msgs = [
        {"role": "user", "content": "Weather in Paris and London?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}},
                {"function": {"name": "get_weather", "arguments": {"city": "London"}}},
            ],
        },
    ]
    assert renderer.render_ids(msgs, tools=TOOLS) == _expected(
        tokenizer, msgs, tools=TOOLS
    )


# ── Tool responses ───────────────────────────────────────────────────


def test_single_tool_response(model_name, tokenizer, renderer):
    msgs = [
        {"role": "user", "content": "Weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
            ],
        },
        {"role": "tool", "content": '{"temp": 20}'},
        {"role": "assistant", "content": "It's 20 degrees."},
    ]
    assert renderer.render_ids(msgs, tools=TOOLS) == _expected(
        tokenizer, msgs, tools=TOOLS
    )


def test_tool_response_with_name(model_name, tokenizer, renderer):
    """Tool message carries `name` (function name). Pins the renderer contract
    that if `name` is supplied it gets rendered (e.g., GPT-OSS Harmony emits
    `<|start|>functions.{name} to=assistant`). Catches regressions where a
    renderer silently drops the field."""
    msgs = [
        {"role": "user", "content": "Weather in Paris?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
            ],
        },
        {"role": "tool", "name": "get_weather", "content": '{"temp": 20}'},
        {"role": "assistant", "content": "It's 20 degrees."},
    ]
    assert renderer.render_ids(msgs, tools=TOOLS) == _expected(
        tokenizer, msgs, tools=TOOLS
    )


def test_consecutive_tool_responses(model_name, tokenizer, renderer):
    msgs = [
        {"role": "user", "content": "Weather in Paris and London?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}},
                {"function": {"name": "get_weather", "arguments": {"city": "London"}}},
            ],
        },
        {"role": "tool", "content": '{"temp": 20}'},
        {"role": "tool", "content": '{"temp": 15}'},
        {"role": "assistant", "content": "Paris: 20, London: 15."},
    ]
    assert renderer.render_ids(msgs, tools=TOOLS) == _expected(
        tokenizer, msgs, tools=TOOLS
    )


# ── Full tool cycle ──────────────────────────────────────────────────


def test_full_tool_cycle(model_name, tokenizer, renderer):
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the weather in Paris?"},
        {
            "role": "assistant",
            "content": "Let me check.",
            "tool_calls": [
                {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
            ],
        },
        {"role": "tool", "content": '{"temp": 20, "condition": "sunny"}'},
        {"role": "assistant", "content": "It is 20 degrees and sunny."},
    ]
    assert renderer.render_ids(msgs, tools=TOOLS) == _expected(
        tokenizer, msgs, tools=TOOLS
    )


def test_multi_step_tool_cycle(model_name, tokenizer, renderer):
    """Two rounds of tool calling."""
    msgs = [
        {"role": "user", "content": "Compare weather in Paris and London"},
        {
            "role": "assistant",
            "content": "Let me check Paris.",
            "tool_calls": [
                {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
            ],
        },
        {"role": "tool", "content": '{"temp": 20}'},
        {
            "role": "assistant",
            "content": "Now London.",
            "tool_calls": [
                {"function": {"name": "get_weather", "arguments": {"city": "London"}}}
            ],
        },
        {"role": "tool", "content": '{"temp": 15}'},
        {"role": "assistant", "content": "Paris: 20, London: 15."},
    ]
    assert renderer.render_ids(msgs, tools=TOOLS) == _expected(
        tokenizer, msgs, tools=TOOLS
    )


# ── Qwen3-VL routing ────────────────────────────────────────────────────


@lru_cache
def _qwen3_vl():
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-VL-4B-Instruct", trust_remote_code=True
    )
    renderer = create_renderer(tokenizer, renderer="auto")
    return tokenizer, renderer


def test_qwen3_vl_auto_renderer():
    _, renderer = _qwen3_vl()
    assert type(renderer).__name__ == "Qwen3VLRenderer"
