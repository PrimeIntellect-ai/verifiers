import asyncio
import base64
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import verifiers as vf
import verifiers.clients.renderer_client as renderer_client_module
from renderers import RendererPool
from renderers import config_from_name
from renderers.base import ParsedResponse, RenderedTokens, create_renderer
from verifiers.clients.renderer_client import (
    RendererClient,
    _attach_tool_call_names,
    _get_incremental_prompt_ids,
    _is_valid_incremental_tail,
    _offload_image_parts_inplace,
    _offload_prompt_and_trajectory_images,
    _step_token_ids,
    _to_renderer_message,
)
from verifiers.errors import EmptyModelResponseError
from verifiers.types import (
    AssistantMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)


def test_renderer_client_honors_configured_renderer_config():
    from renderers import Qwen3VLRendererConfig

    RendererClient._shared_pools.clear()

    cfg = Qwen3VLRendererConfig()
    client = object.__new__(RendererClient)
    client._renderer = None
    client._pool_size = 1
    client._config = vf.ClientConfig(client_type="renderer", renderer_config=cfg)

    sentinel_pool = RendererPool.__new__(RendererPool)
    with patch(
        "verifiers.clients.renderer_client.create_renderer_pool",
        return_value=sentinel_pool,
    ) as create_pool_mock:
        pool = client._get_renderer_or_pool("Qwen/Qwen3-VL-4B-Instruct")

    assert pool is sentinel_pool
    create_pool_mock.assert_called_once_with(
        "Qwen/Qwen3-VL-4B-Instruct",
        cfg,
        size=1,
    )


def test_renderer_client_uses_renderer_model_name_override():
    from renderers import Qwen3VLRendererConfig

    RendererClient._shared_pools.clear()

    cfg = Qwen3VLRendererConfig()
    client = object.__new__(RendererClient)
    client._renderer = None
    client._pool_size = 1
    client._config = vf.ClientConfig(
        client_type="renderer",
        renderer_config=cfg,
        renderer_model_name="Qwen/Qwen3-VL-4B-Instruct",
    )

    sentinel_pool = RendererPool.__new__(RendererPool)
    with patch(
        "verifiers.clients.renderer_client.create_renderer_pool",
        return_value=sentinel_pool,
    ) as create_pool_mock:
        pool = client._get_renderer_or_pool("r8-smoke")

    assert pool is sentinel_pool
    create_pool_mock.assert_called_once_with(
        "Qwen/Qwen3-VL-4B-Instruct",
        cfg,
        size=1,
    )


def test_renderer_client_threads_chat_template_kwargs_into_pool():
    """``sampling_args.extra_body.chat_template_kwargs`` land on the
    concrete RendererConfig (resolving Auto/None against
    ``MODEL_RENDERER_MAP`` when needed) and are stripped from the params
    forwarded to /generate. Covers explicit / Auto / None bases."""
    from renderers import AutoRendererConfig, Qwen3RendererConfig

    bases = [
        Qwen3RendererConfig(enable_thinking=True),
        AutoRendererConfig(preserve_all_thinking=True),
        None,
    ]
    for base in bases:
        RendererClient._shared_pools.clear()

        client = object.__new__(RendererClient)
        client._renderer = None
        client._pool_size = 1
        client._config = vf.ClientConfig(client_type="renderer", renderer_config=base)
        client._client = object()  # type: ignore[attr-defined]

        sentinel_pool = RendererPool.__new__(RendererPool)
        captured: dict = {}

        async def _fake_generate(**kwargs):
            captured.update(kwargs)
            return {"content": "ok"}

        with (
            patch(
                "verifiers.clients.renderer_client.create_renderer_pool",
                return_value=sentinel_pool,
            ) as create_pool_mock,
            patch(
                "verifiers.clients.renderer_client.generate",
                side_effect=_fake_generate,
            ),
        ):
            asyncio.run(
                client.get_native_response(
                    prompt=[{"role": "user", "content": "hi"}],
                    model="Qwen/Qwen3-8B",
                    sampling_args={
                        "extra_body": {
                            "chat_template_kwargs": {"enable_thinking": False},
                            "top_k": 20,
                        }
                    },
                    tools=None,
                )
            )

        expected_preserve_all = (
            base.preserve_all_thinking
            if isinstance(base, AutoRendererConfig)
            else False
        )
        create_pool_mock.assert_called_once_with(
            "Qwen/Qwen3-8B",
            Qwen3RendererConfig(
                enable_thinking=False,
                preserve_all_thinking=expected_preserve_all,
            ),
            size=1,
        )
        assert captured["sampling_params"] == {"top_k": 20}


def test_renderer_client_auto_resolves_against_renderer_model_name_override():
    """When ``ClientConfig.renderer_model_name`` overrides the API request
    model, auto-resolution looks up the OVERRIDE in ``MODEL_RENDERER_MAP``
    (it's what loads the tokenizer the renderer holds) — not the request
    model, which may not be in the map at all."""
    from renderers import Qwen3RendererConfig

    RendererClient._shared_pools.clear()

    client = object.__new__(RendererClient)
    client._renderer = None
    client._pool_size = 1
    client._config = vf.ClientConfig(
        client_type="renderer",
        renderer_model_name="Qwen/Qwen3-8B",  # override
    )
    client._client = object()  # type: ignore[attr-defined]

    sentinel_pool = RendererPool.__new__(RendererPool)

    async def _fake_generate(**kwargs):
        return {"content": "ok"}

    with (
        patch(
            "verifiers.clients.renderer_client.create_renderer_pool",
            return_value=sentinel_pool,
        ) as create_pool_mock,
        patch("verifiers.clients.renderer_client.generate", side_effect=_fake_generate),
    ):
        asyncio.run(
            client.get_native_response(
                prompt=[{"role": "user", "content": "hi"}],
                model="r8-smoke",  # not in MODEL_RENDERER_MAP
                sampling_args={
                    "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
                },
                tools=None,
            )
        )

    # Resolves against renderer_model_name (Qwen/Qwen3-8B → "qwen3") rather
    # than the request "r8-smoke" (which would fall through to default).
    create_pool_mock.assert_called_once_with(
        "Qwen/Qwen3-8B",
        Qwen3RendererConfig(enable_thinking=False),
        size=1,
    )


def test_renderer_client_rejects_invalid_chat_template_kwargs():
    """Unknown / mistyped chat_template_kwargs surface as a pydantic
    ``ValidationError`` (``extra="forbid"`` on the typed RendererConfig)."""
    from pydantic import ValidationError
    from renderers import Qwen3RendererConfig

    RendererClient._shared_pools.clear()

    client = object.__new__(RendererClient)
    client._renderer = None
    client._pool_size = 1
    client._config = vf.ClientConfig(
        client_type="renderer", renderer_config=Qwen3RendererConfig()
    )
    client._client = object()  # type: ignore[attr-defined]

    with pytest.raises(ValidationError, match="enable_thinkng"):
        asyncio.run(
            client.get_native_response(
                prompt=[{"role": "user", "content": "hi"}],
                model="Qwen/Qwen3-8B",
                sampling_args={
                    "extra_body": {
                        "chat_template_kwargs": {"enable_thinkng": False},  # typo
                    }
                },
                tools=None,
            )
        )


# Provenance: Eli's review on PR #1068, comment 3150580768.
#   "RendererClient parses the GPT-OSS assistant tool call into ToolCall(name=...),
#   but ToolEnv returns ToolMessage with only content/tool_call_id, and
#   _to_renderer_message forwards only role/content/tool_call_id. As a result every
#   GPT-OSS tool result rendered through this path becomes `functions.unknown
#   to=assistant`."
# The verifiers ToolMessage schema has no `name` field, so per-message conversion
# can't recover it — we have to walk the list and look up the prior assistant
# tool_call by tool_call_id. This test pins the contract of that lookup.
def test_attach_tool_call_names_recovers_function_name_from_prior_call():
    messages = [
        _to_renderer_message(UserMessage(content="2+2?")),
        _to_renderer_message(
            AssistantMessage(
                content=None,
                tool_calls=[
                    ToolCall(id="c1", name="calculator", arguments="{}"),
                ],
            )
        ),
        _to_renderer_message(ToolMessage(content="4", tool_call_id="c1")),
    ]
    out = _attach_tool_call_names(messages)
    assert out[2]["role"] == "tool"
    assert out[2]["name"] == "calculator"


def test_attach_tool_call_names_preserves_existing_name():
    """Caller-supplied `name` (already on a Mapping input) wins over recovery."""
    messages: list = [
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "c1", "function": {"name": "calculator", "arguments": "{}"}}
            ],
        },
        {
            "role": "tool",
            "name": "explicit",
            "content": "4",
            "tool_call_id": "c1",
        },
    ]
    out = _attach_tool_call_names(messages)
    assert out[1]["name"] == "explicit"


def test_attach_tool_call_names_unknown_tool_call_id_left_unset():
    """A tool message with no matching prior call gets no `name` (renderer
    falls back to its own default, e.g. GPT-OSS uses 'unknown')."""
    messages = [
        _to_renderer_message(UserMessage(content="hi")),
        _to_renderer_message(ToolMessage(content="orphan", tool_call_id="nope")),
    ]
    out = _attach_tool_call_names(messages)
    assert "name" not in out[1]


@pytest.mark.asyncio
async def test_to_native_tool_returns_openai_envelope():
    """``RendererClient.to_native_tool`` must wrap each ``Tool`` in the
    OpenAI envelope (``{"type": "function", "function": {...}}``) — the
    same shape ``OpenAIChatCompletionsClient`` sends server-side under
    TITO/MITO. Modern function-calling models (Qwen3 family, GLM, Kimi)
    saw the envelope at training time, so the renderer client's prompt
    must match. Regression for the bare-form bug where rollout-mode
    tool envs produced uniformly zero rewards because the model never
    emitted ``<tool_call>`` blocks under an out-of-distribution prompt.
    """
    from verifiers.types import Tool

    client = object.__new__(RendererClient)
    tool = Tool(
        name="get_weather",
        description="Get the weather for a city",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )

    native = await client.to_native_tool(tool)
    assert native["type"] == "function"
    assert native["function"]["name"] == "get_weather"
    assert native["function"]["description"] == "Get the weather for a city"
    assert native["function"]["parameters"]["required"] == ["city"]
    assert "strict" not in native["function"]


@pytest.mark.asyncio
async def test_to_native_tool_propagates_strict_flag():
    """When ``Tool.strict`` is set the envelope must carry it through —
    OpenAI's strict-schema enforcement only kicks in on the inner function
    object, never the envelope itself."""
    from verifiers.types import Tool

    client = object.__new__(RendererClient)
    tool = Tool(
        name="get_weather",
        description="Get the weather for a city",
        parameters={"type": "object", "properties": {}},
        strict=True,
    )

    native = await client.to_native_tool(tool)
    assert native["function"]["strict"] is True


@pytest.mark.asyncio
async def test_renderer_client_accepts_dict_native_response_with_content():
    client = object.__new__(RendererClient)

    await client.raise_from_native_response({"content": "done"})


@pytest.mark.asyncio
async def test_renderer_client_rejects_empty_dict_native_response():
    client = object.__new__(RendererClient)

    with pytest.raises(EmptyModelResponseError):
        await client.raise_from_native_response({})


@pytest.mark.asyncio
async def test_from_native_response_uses_request_id_and_token_lengths():
    """vLLM's /inference/v1/generate returns ``request_id`` (not ``id``) and
    no ``usage``/``model``/``created``. ``Response.id`` should pick up
    ``request_id``; usage is reconstructed from token-list lengths;
    model/created are unused metadata.
    """
    client = object.__new__(RendererClient)
    response_dict = {
        "request_id": "resp-42",
        "content": "ok",
        "reasoning_content": None,
        "tool_calls": None,
        "finish_reason": "stop",
        "prompt_ids": [1, 2, 3],
        "completion_ids": [4, 5],
        "completion_logprobs": [-0.1, -0.2],
        "routed_experts": None,
    }

    response = await client.from_native_response(response_dict)
    assert response.id == "resp-42"
    assert response.usage.prompt_tokens == 3
    assert response.usage.completion_tokens == 2
    assert response.usage.total_tokens == 5


@pytest.mark.asyncio
async def test_get_native_response_forwards_extra_headers_to_generate():
    captured: dict = {}
    client = object.__new__(RendererClient)
    client._renderer = object()
    client._pool_size = 1
    client._config = vf.ClientConfig(client_type="renderer")
    client._client = object()  # type: ignore[attr-defined]

    async def _fake_generate(**kwargs):
        captured.update(kwargs)
        return {"content": "ok"}

    with (
        patch.object(RendererClient, "_get_renderer_or_pool", return_value=object()),
        patch("verifiers.clients.renderer_client.generate", side_effect=_fake_generate),
    ):
        response = await client.get_native_response(
            prompt=[{"role": "user", "content": "hi"}],
            model="test-model",
            sampling_args={"extra_headers": {"X-Static": "static"}},
            tools=None,
            extra_headers={"X-Session-ID": "trajectory-123", "X-Static": "state"},
        )

    assert response == {"content": "ok"}
    assert captured["extra_headers"] == {
        "X-Static": "state",
        "X-Session-ID": "trajectory-123",
    }


@pytest.mark.asyncio
async def test_mm_fallback_retries_hash_only_failure_with_full_payloads():
    calls = 0

    async def fake_generate(*, force_full_pixels=False, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            # First attempt sends prior-turn images hash-only.
            assert force_full_pixels is False
            raise RuntimeError("Expected a cached item for mm_hash=abc")
        # Fallback re-materializes every image and re-sends in full.
        assert force_full_pixels is True
        return {"content": "ok"}

    # A descriptor-only image (no pixel_values) is what gets sent hash-only, so
    # a cache-miss on it is retryable. The retry decision is made locally from
    # this sidecar — no ContextVar crosses the renderer's thread boundary.
    mm_data = SimpleNamespace(
        mm_items={"image": [{"pixel_values": None, "image_grid_thw": [1, 2, 2]}]}
    )

    with patch.object(renderer_client_module, "generate", side_effect=fake_generate):
        result = await renderer_client_module._generate_with_mm_fallback(
            client=object(),
            renderer=object(),
            messages=[],
            model="test-model",
            multi_modal_data=mm_data,
        )

    assert calls == 2
    assert result == {"content": "ok"}


@pytest.mark.asyncio
async def test_mm_fallback_does_not_retry_without_hash_only_payloads():
    calls = 0

    async def fake_generate(*, force_full_pixels=False, **_kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("Expected a cached item for mm_hash=abc")

    # No descriptor-only images means nothing was sent hash-only, so a
    # cache-miss-shaped error is not attributable to our hashing — don't retry.
    with patch.object(renderer_client_module, "generate", side_effect=fake_generate):
        with pytest.raises(RuntimeError, match="Expected a cached item"):
            await renderer_client_module._generate_with_mm_fallback(
                client=object(),
                renderer=object(),
                messages=[],
                model="test-model",
                multi_modal_data=None,
            )

    assert calls == 1


class _BridgeRenderer:
    supports_tools = True

    # Token ID 99 plays the role of the stop / end-of-turn marker in the
    # token streams these tests construct.
    _STOP_TOKEN_ID = 99

    def __init__(self, bridge_base=None, bridge_full=None):
        self.bridge_base = bridge_base or [10, 99, 30]
        self.bridge_full = bridge_full or [10, 99, 30, 40, 50]
        self.calls = []
        self.bridge_calls = 0

    def render_ids(self, messages, *, tools=None, add_generation_prompt=False):
        self.calls.append((messages, tools, add_generation_prompt))
        if len(messages) == 1 and add_generation_prompt is False:
            return list(self.bridge_base)
        if len(messages) > 1 and add_generation_prompt is True:
            return list(self.bridge_full)
        raise AssertionError((messages, tools, add_generation_prompt))

    def bridge_to_next_turn(
        self,
        previous_prompt_ids,
        previous_completion_ids,
        new_messages,
        *,
        tools=None,
    ):
        """Mimic the Renderer Protocol's bridge contract enough for the
        ``_get_incremental_prompt_ids`` tests to exercise the post-bridge
        plumbing.

        Returns ``prev_prompt + prev_completion + trailing + extension``,
        where ``trailing`` is whatever ``bridge_base`` emits AFTER the stop
        token (the "turn boundary" tokens our renderers emit between turns)
        and ``extension`` is the suffix ``bridge_full`` adds on top of
        ``bridge_base``.
        """
        self.bridge_calls += 1
        # Find the stop token in bridge_base and split into close + trailing.
        try:
            stop_idx = self.bridge_base.index(self._STOP_TOKEN_ID)
        except ValueError:
            stop_idx = len(self.bridge_base) - 1
        trailing = list(self.bridge_base[stop_idx + 1 :])
        extension = list(self.bridge_full[len(self.bridge_base) :])
        return RenderedTokens(
            token_ids=(
                list(previous_prompt_ids)
                + list(previous_completion_ids)
                + trailing
                + extension
            )
        )

    def parse_response(self, token_ids, *, tools=None):
        return ParsedResponse(content="")

    def get_stop_token_ids(self):
        return [self._STOP_TOKEN_ID]


@pytest.mark.parametrize(
    ("tail", "expected"),
    [
        ([{"role": "tool", "content": "a"}], True),
        ([{"role": "tool", "content": "a"}, {"role": "tool", "content": "b"}], True),
        ([{"role": "user", "content": "next"}], True),
        ([{"role": "tool", "content": "a"}, {"role": "user", "content": "next"}], True),
        ([{"role": "assistant", "content": "no"}], False),
        (
            [{"role": "user", "content": "next"}, {"role": "tool", "content": "late"}],
            False,
        ),
    ],
)
def test_incremental_tail_accepts_tool_and_user_followups(tail, expected):
    assert _is_valid_incremental_tail(tail) is expected


@pytest.mark.asyncio
async def test_get_incremental_prompt_ids_matches_tool_tail_without_rerendering_completion():
    renderer = _BridgeRenderer(bridge_base=[10, 99, 30], bridge_full=[10, 99, 30, 40])
    prompt_messages = [SystemMessage(content="s"), UserMessage(content="u")]
    completion_messages = [
        AssistantMessage(
            content=None,
            tool_calls=[ToolCall(id="call_0", name="lookup", arguments="{}")],
        )
    ]
    prompt = [
        *[_to_renderer_message(m) for m in prompt_messages + completion_messages],
        _to_renderer_message(ToolMessage(content="result", tool_call_id="call_0")),
    ]
    state = {
        "trajectory": [
            {
                "prompt": prompt_messages,
                "completion": completion_messages,
                "tokens": {
                    "prompt_ids": [1, 2],
                    "completion_ids": [3, 99],
                    "is_truncated": False,
                },
                "is_truncated": False,
            }
        ]
    }

    result = await _get_incremental_prompt_ids(
        renderer=renderer, prompt=prompt, state=state, tools=None
    )

    assert result is not None
    assert result.token_ids == [1, 2, 3, 99, 30, 40]
    # The bridge stitches over the completion without re-rendering it —
    # one bridge call, zero render_ids calls (older diff-based bridges
    # called render_ids twice).
    assert renderer.bridge_calls == 1
    assert renderer.calls == []


@pytest.mark.asyncio
async def test_get_incremental_prompt_ids_accepts_tool_then_user_tail():
    renderer = _BridgeRenderer(bridge_base=[10, 99], bridge_full=[10, 99, 40, 50])
    prompt_messages = [SystemMessage(content="s"), UserMessage(content="u")]
    completion_messages = [
        AssistantMessage(
            content=None,
            tool_calls=[ToolCall(id="call_0", name="lookup", arguments="{}")],
        )
    ]
    prompt = [
        *[_to_renderer_message(m) for m in prompt_messages + completion_messages],
        _to_renderer_message(ToolMessage(content="result", tool_call_id="call_0")),
        _to_renderer_message(UserMessage(content="continue")),
    ]
    state = {
        "trajectory": [
            {
                "prompt": prompt_messages,
                "completion": completion_messages,
                "tokens": {
                    "prompt_ids": [1, 2],
                    "completion_ids": [3, 99],
                    "is_truncated": False,
                },
                "is_truncated": False,
            }
        ]
    }

    result = await _get_incremental_prompt_ids(
        renderer=renderer, prompt=prompt, state=state, tools=None
    )

    assert result is not None
    assert result.token_ids == [1, 2, 3, 99, 40, 50]


@pytest.mark.asyncio
async def test_get_incremental_prompt_ids_accepts_multimodal_tool_user_tail():
    renderer = _BridgeRenderer(bridge_base=[10, 99], bridge_full=[10, 99, 40, 50])
    prompt_messages = [
        SystemMessage(content="s"),
        UserMessage(
            content=[
                {"type": "text", "text": "inspect"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,abc"},
                },
            ]
        ),
    ]
    completion_messages = [
        AssistantMessage(
            content=None,
            tool_calls=[ToolCall(id="call_0", name="lookup", arguments="{}")],
        )
    ]
    prompt = [
        *[_to_renderer_message(m) for m in prompt_messages + completion_messages],
        _to_renderer_message(
            ToolMessage(
                content=[
                    {"type": "text", "text": "result"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,def"},
                    },
                ],
                tool_call_id="call_0",
            )
        ),
        _to_renderer_message(UserMessage(content="continue")),
    ]
    state = {
        "trajectory": [
            {
                "prompt": prompt_messages,
                "completion": completion_messages,
                "tokens": {
                    "prompt_ids": [1, 2],
                    "completion_ids": [3, 99],
                    "is_truncated": False,
                },
                "is_truncated": False,
            }
        ]
    }

    result = await _get_incremental_prompt_ids(
        renderer=renderer, prompt=prompt, state=state, tools=None
    )

    assert result is not None
    assert result.token_ids == [1, 2, 3, 99, 40, 50]


# ── Live image offload + bridge-matching validation ───────────────────


def _data_uri(payload: bytes, media: str = "jpeg") -> str:
    return f"data:image/{media};base64," + base64.b64encode(payload).decode("ascii")


def _image_urls(value):
    """Collect every image url reachable from dicts/lists/Pydantic messages."""
    urls = []
    if isinstance(value, dict):
        if value.get("type") == "image_url" and isinstance(value.get("image_url"), dict):
            urls.append(value["image_url"].get("url"))
        for v in value.values():
            urls.extend(_image_urls(v))
    elif isinstance(value, (list, tuple)):
        for v in value:
            urls.extend(_image_urls(v))
    elif getattr(value, "type", None) == "image_url":
        src = getattr(value, "image_url", None)
        urls.append(getattr(src, "url", None))
    else:
        content = getattr(value, "content", None)
        if isinstance(content, (list, tuple)):
            urls.extend(_image_urls(content))
    return [u for u in urls if u is not None]


def test_offload_rewrites_prompt_base64_to_file(tmp_path, monkeypatch):
    monkeypatch.setenv("VF_RENDERER_IMAGE_OFFLOAD_DIR", str(tmp_path))
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hi"},
                {"type": "image_url", "image_url": {"url": _data_uri(b"img-A", "jpeg")}},
            ],
        }
    ]
    stats = _offload_prompt_and_trajectory_images(prompt, state=None)

    url = prompt[0]["content"][1]["image_url"]["url"]
    assert stats["prompt_rewritten"] == 1
    assert url.startswith("file://") and url.endswith(".jpg")
    # File holds the decoded bytes (not the base64), content-addressed.
    assert Path(url[len("file://") :]).read_bytes() == b"img-A"


def test_offload_relative_dir_yields_absolute_loadable_file_uri(tmp_path, monkeypatch):
    # Regression: a RELATIVE VF_RENDERER_IMAGE_OFFLOAD_DIR must still yield an
    # ABSOLUTE file:/// URL. A relative path produced "file://rel/seg/..." which
    # urlparse mis-reads (first segment → host, rest → a bogus absolute path) →
    # the renderer FileNotFoundError'd on every rollout. (Caught by the local
    # smoke when --output-dir was relative.)
    from urllib.parse import urlparse

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VF_RENDERER_IMAGE_OFFLOAD_DIR", "rel_sub/assets/images")  # relative!
    prompt = [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": _data_uri(b"img-Z", "jpeg")}}]}
    ]
    _offload_prompt_and_trajectory_images(prompt, state=None)

    url = prompt[0]["content"][0]["image_url"]["url"]
    parsed = urlparse(url)
    assert url.startswith("file:///")  # absolute + well-formed
    assert parsed.netloc == ""  # no path segment swallowed as the URL host
    assert Path(parsed.path).read_bytes() == b"img-Z"  # loadable at the urlparse path


@pytest.mark.parametrize(
    "make_prompt",
    [
        # plain-dict message (the native/wire representation)
        lambda uri: [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": uri}}]}],
        # Pydantic message (how MultiTurnEnv stores trajectory prompts)
        lambda uri: [UserMessage(content=[{"type": "image_url", "image_url": {"url": uri}}])],
    ],
    ids=["dict", "pydantic"],
)
def test_offload_image_parts_handles_dict_and_pydantic(tmp_path, make_prompt):
    uri = _data_uri(b"traj-img", "png")
    messages = make_prompt(uri)

    count, _ = _offload_image_parts_inplace(messages, tmp_path)

    assert count == 1
    assert _image_urls(messages)[0].startswith("file://")
    # Idempotent: a second pass finds nothing to rewrite.
    assert _offload_image_parts_inplace(messages, tmp_path)[0] == 0


def test_offload_only_rewrites_newest_trajectory_step(tmp_path, monkeypatch):
    # Append-only: prior steps were offloaded when they were newest, so the
    # helper only needs to touch trajectory[-1]. (The old step here stays as-is.)
    monkeypatch.setenv("VF_RENDERER_IMAGE_OFFLOAD_DIR", str(tmp_path))
    old_uri = _data_uri(b"old", "png")
    new_uri = _data_uri(b"new", "png")
    state = {
        "trajectory": [
            {"prompt": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": old_uri}}]}]},
            {"prompt": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": new_uri}}]}]},
        ]
    }
    stats = _offload_prompt_and_trajectory_images([], state)
    assert stats["trajectory_rewritten"] == 1
    assert _image_urls(state["trajectory"][-1]["prompt"])[0].startswith("file://")


def _multimodal_bridge_scenario():
    """Current prompt extends a stored trajectory step; both reference the same
    'abc' screenshot in the shared prefix (the bridge anchor)."""
    abc = _data_uri(b"screenshot-abc", "png")
    deff = _data_uri(b"screenshot-def", "png")
    renderer = _BridgeRenderer(bridge_base=[10, 99], bridge_full=[10, 99, 40, 50])
    prompt_messages = [
        SystemMessage(content="s"),
        UserMessage(
            content=[
                {"type": "text", "text": "inspect"},
                {"type": "image_url", "image_url": {"url": abc}},
            ]
        ),
    ]
    completion_messages = [
        AssistantMessage(
            content=None,
            tool_calls=[ToolCall(id="call_0", name="lookup", arguments="{}")],
        )
    ]
    prompt = [
        *[_to_renderer_message(m) for m in prompt_messages + completion_messages],
        _to_renderer_message(
            ToolMessage(
                content=[
                    {"type": "text", "text": "result"},
                    {"type": "image_url", "image_url": {"url": deff}},
                ],
                tool_call_id="call_0",
            )
        ),
        _to_renderer_message(UserMessage(content="continue")),
    ]
    state = {
        "trajectory": [
            {
                "prompt": prompt_messages,
                "completion": completion_messages,
                "tokens": {"prompt_ids": [1, 2], "completion_ids": [3, 99], "is_truncated": False},
                "is_truncated": False,
            }
        ]
    }
    return renderer, prompt, state


@pytest.mark.asyncio
async def test_bridge_prefix_matching_survives_image_offload(tmp_path, monkeypatch):
    """THE validation: after immediate base64->file:// offload of BOTH the
    current prompt and the stored trajectory, the bridge prefix match must still
    succeed (it compares the shared 'abc' screenshot by file path now)."""
    monkeypatch.setenv("VF_RENDERER_IMAGE_OFFLOAD_DIR", str(tmp_path))
    renderer, prompt, state = _multimodal_bridge_scenario()

    # Offload exactly as get_native_response does, before bridge matching.
    _offload_prompt_and_trajectory_images(prompt, state)

    # The anchor screenshot resolved to the SAME file:// url on both sides.
    prompt_prefix_url = _image_urls(prompt[1])[0]  # current prompt's prior-turn user image
    traj_url = _image_urls(state["trajectory"][0]["prompt"])[0]  # stored trajectory image
    assert prompt_prefix_url.startswith("file://")
    assert prompt_prefix_url == traj_url

    result = await _get_incremental_prompt_ids(
        renderer=renderer, prompt=prompt, state=state, tools=None
    )
    # Bridge still matched and extended — prefix logic intact under offload.
    assert result is not None
    assert result.token_ids == [1, 2, 3, 99, 40, 50]


@pytest.mark.asyncio
async def test_bridge_misses_if_trajectory_not_offloaded(tmp_path):
    """Negative control proving the test has teeth: if only the current prompt is
    offloaded (file://) while the trajectory keeps base64, the prefix no longer
    matches and the bridge bails. This is why get_native_response must offload
    BOTH sides before matching."""
    renderer, prompt, state = _multimodal_bridge_scenario()
    # Offload ONLY the current prompt; leave the trajectory as base64.
    _offload_image_parts_inplace(prompt, tmp_path)

    result = await _get_incremental_prompt_ids(
        renderer=renderer, prompt=prompt, state=state, tools=None
    )
    assert result is None


# ── Parity across real renderers: truncated most-recent step ──────────
#
# When vLLM hits max_tokens mid-completion, the previous step carries
# is_truncated=True and completion_ids without an end-of-turn stop token.
# The anchor loop in _get_incremental_prompt_ids used to skip every
# truncated step, so the bridge never ran and the caller fell back to a
# full re-render. The extension property then broke whenever BPE
# round-trip diverged and the rollout fragmented.
#
# These tests run across every hand-coded renderer in the parity matrix
# to make sure that regression stays fixed: the bridge anchors on the
# truncated step and returns prefix-preserving ids.

# Mirror of packages/renderers/tests/conftest.py::RENDERER_MODELS,
# restricted to hand-coded renderers (DefaultRenderer never bridges by
# design — covered separately in the bails-on-default test below).
_TRUNCATED_ANCHOR_MODELS = [
    pytest.param("Qwen/Qwen3-8B", "auto", id="Qwen/Qwen3-8B"),
    pytest.param("Qwen/Qwen3.5-9B", "auto", id="Qwen/Qwen3.5-9B"),
    pytest.param("Qwen/Qwen3-VL-4B-Instruct", "auto", id="Qwen/Qwen3-VL-4B-Instruct"),
    pytest.param("zai-org/GLM-5", "auto", id="zai-org/GLM-5"),
    pytest.param("zai-org/GLM-4.7-Flash", "auto", id="zai-org/GLM-4.7-Flash"),
    pytest.param("THUDM/GLM-4.5-Air", "auto", id="THUDM/GLM-4.5-Air"),
    pytest.param("MiniMaxAI/MiniMax-M2.5", "auto", id="MiniMaxAI/MiniMax-M2.5"),
    pytest.param(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "auto",
        id="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    ),
    pytest.param("openai/gpt-oss-20b", "gpt-oss", id="openai/gpt-oss-20b"),
]


@lru_cache(maxsize=None)
def _load_tokenizer_and_renderer(model_name: str, renderer_name: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    renderer = create_renderer(tokenizer, config_from_name(renderer_name))
    return tokenizer, renderer


def _build_truncated_state(tokenizer, renderer):
    """Construct a single-step trajectory whose most-recent step is
    truncated. prev_prompt_ids / prev_completion_ids use the renderer's
    own tokens so the assertion reflects what the real orchestrator
    hands the bridge — the exact tokens vLLM produced for the partial
    assistant turn, with no end-of-turn marker at the end.
    """
    step_prompt = [{"role": "user", "content": "Guess a 5-letter word."}]
    prev_prompt_ids = renderer.render_ids(step_prompt, add_generation_prompt=True)
    truncated_text = (
        "I'll start with a common word. Let me think about this — "
        "the most frequent letters are E, A, R, I, O, T, N, S, L"
    )
    prev_completion_ids = tokenizer.encode(truncated_text, add_special_tokens=False)

    step_completion = [{"role": "assistant", "content": truncated_text}]
    state = {
        "trajectory": [
            {
                "prompt": step_prompt,
                "completion": step_completion,
                "tokens": {
                    "prompt_ids": list(prev_prompt_ids),
                    "completion_ids": list(prev_completion_ids),
                    "is_truncated": True,
                },
                "is_truncated": True,
            }
        ]
    }
    next_turn_prompt = (
        step_prompt
        + step_completion
        + [{"role": "user", "content": "Your guess was invalid. Give a 5-letter word."}]
    )
    return prev_prompt_ids, prev_completion_ids, state, next_turn_prompt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_id,renderer_id",
    _TRUNCATED_ANCHOR_MODELS,
)
async def test_get_incremental_prompt_ids_bridges_over_truncated_step(
    model_id, renderer_id
):
    """The bridge anchors on the truncated step and returns new prompt_ids
    that start with prev_prompt + prev_completion byte-identically (the
    extension invariant). This is what keeps interleave_rollout from
    fragmenting the rollout into two samples."""
    tokenizer, renderer = _load_tokenizer_and_renderer(model_id, renderer_id)
    prev_prompt_ids, prev_completion_ids, state, next_turn_prompt = (
        _build_truncated_state(tokenizer, renderer)
    )

    result = await _get_incremental_prompt_ids(
        renderer=renderer, prompt=next_turn_prompt, state=state, tools=None
    )

    prefix = list(prev_prompt_ids) + list(prev_completion_ids)
    assert result is not None, f"{model_id}: bridge returned None on truncated anchor"
    result_ids = result.token_ids
    assert result_ids[: len(prefix)] == prefix, (
        f"{model_id}: bridge result does not prefix-preserve "
        f"prev_prompt + prev_completion"
    )
    assert len(result_ids) > len(prefix), (
        f"{model_id}: bridge produced no tail tokens for the new user turn"
    )


@pytest.mark.asyncio
async def test_get_incremental_prompt_ids_bails_for_default_renderer():
    """DefaultRenderer always bails to None so the caller falls back to a
    full apply_chat_template re-render — preserving main's
    TITO-on-truncation behavior for unknown templates."""
    tokenizer, renderer = _load_tokenizer_and_renderer(
        "Qwen/Qwen2.5-0.5B-Instruct", "default"
    )
    _, _, state, next_turn_prompt = _build_truncated_state(tokenizer, renderer)

    result = await _get_incremental_prompt_ids(
        renderer=renderer, prompt=next_turn_prompt, state=state, tools=None
    )

    assert result is None


# ── _step_token_ids: fallback to raw response tokens when step.tokens was
# truncated by parse_response_tokens (empty completion_ids / prompt_ids on
# overlong turns). The bridge needs un-truncated anchor tokens to chain
# across turns; training budget is still enforced at sample-assembly time.


class _RawTokens:
    def __init__(self, prompt_ids, completion_ids):
        self.prompt_ids = prompt_ids
        self.completion_ids = completion_ids


class _ResponseMessage:
    def __init__(self, tokens):
        self.tokens = tokens


class _Response:
    def __init__(self, message):
        self.message = message


def test_step_token_ids_happy_path():
    """Populated step.tokens → returns those directly without inspecting response."""
    step = {
        "tokens": {"prompt_ids": [1, 2, 3], "completion_ids": [4, 5]},
        "response": None,
    }
    assert _step_token_ids(step) == ([1, 2, 3], [4, 5])


def test_step_token_ids_falls_back_on_empty_completion():
    """Overlong prompt: parse_response_tokens zeros completion_ids in step.tokens
    but raw tokens on step.response.message.tokens are still intact. Fallback
    must return the raw tokens so bridge can extend past the overlong turn."""
    step = {
        "tokens": {"prompt_ids": [1, 2, 3], "completion_ids": []},
        "response": _Response(
            _ResponseMessage(_RawTokens([10, 11, 12, 13], [14, 15, 16]))
        ),
    }
    assert _step_token_ids(step) == ([10, 11, 12, 13], [14, 15, 16])


def test_step_token_ids_falls_back_when_tokens_is_none():
    """step.tokens==None (e.g. agent-completed sentinel or uninitialized step)
    should still fall back to raw response tokens when available."""
    step = {
        "tokens": None,
        "response": _Response(_ResponseMessage(_RawTokens([20, 21], [22, 23]))),
    }
    assert _step_token_ids(step) == ([20, 21], [22, 23])


def test_step_token_ids_returns_none_when_both_sources_empty():
    """If both step.tokens and response.message.tokens are empty/absent, must
    return None (caller falls back to full re-render)."""
    step = {
        "tokens": {"prompt_ids": [], "completion_ids": []},
        "response": _Response(_ResponseMessage(_RawTokens([], []))),
    }
    assert _step_token_ids(step) is None


def test_step_token_ids_returns_none_when_truncated_and_no_response():
    """Guard: empty step.tokens + no response object → None, not an AttributeError."""
    step = {
        "tokens": {"prompt_ids": [1, 2], "completion_ids": []},
        "response": None,
    }
    assert _step_token_ids(step) is None


# ---------------------------------------------------------------------------
# renderers.OverlongPromptError → vf.OverlongPromptError translation.
# ---------------------------------------------------------------------------


def test_get_native_response_translates_renderer_overlong_to_vf_overlong():
    """A pre-flight overflow surfaced by ``renderers.client.generate`` as
    ``renderers.OverlongPromptError`` must be rebadged into
    ``verifiers.errors.OverlongPromptError`` so the
    ``MultiTurnEnv.prompt_too_long`` ``@vf.stop`` condition (which catches
    via ``vf.Error``) picks it up. The decorator-driven path
    (BadRequestError → OverlongPromptError) is exercised separately by the
    live integration test in the matching renderers PR."""
    import asyncio

    from renderers import OverlongPromptError as RendererOverlongPromptError

    from verifiers.clients.renderer_client import RendererClient
    from verifiers.errors import OverlongPromptError

    client = object.__new__(RendererClient)
    client._renderer = object()
    client._pool_size = 1
    client._config = vf.ClientConfig(client_type="renderer")
    client._client = object()  # type: ignore[attr-defined]

    async def _fake_generate(**kwargs):
        raise RendererOverlongPromptError(prompt_len=99, max_prompt_len=8)

    with (
        patch.object(RendererClient, "_get_renderer_or_pool", return_value=object()),
        patch("verifiers.clients.renderer_client.generate", side_effect=_fake_generate),
    ):
        with pytest.raises(OverlongPromptError):
            asyncio.run(
                client.get_native_response(
                    prompt=[{"role": "user", "content": "hi"}],
                    model="test-model",
                    sampling_args={},
                    tools=None,
                )
            )


# ---------------------------------------------------------------------------
# Renderer prompt_attribution pass-through.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_native_response_threads_prompt_attribution_into_generate():
    """Bridge result's :class:`RenderedTokens` is passed through as
    ``generate(prompt_attribution=...)`` and surfaces unchanged on the
    result dict."""
    bridged = RenderedTokens(
        token_ids=[1, 2, 3, 4],
        message_indices=[-1, -1, 0, 0],
        sampled_mask=[False, False, False, False],
        is_content=[False, False, False, True],
        message_roles=["tool"],
    )
    captured: dict = {}

    async def _fake_get_incremental(**kwargs):
        return bridged

    async def _fake_generate(**kwargs):
        captured.update(kwargs)
        return {
            "request_id": "r-1",
            "prompt_ids": list(bridged.token_ids),
            "completion_ids": [5, 6],
            "completion_logprobs": [-0.1, -0.2],
            "content": "ok",
            "reasoning_content": None,
            "tool_calls": [],
            "finish_reason": "stop",
            "routed_experts": None,
            "multi_modal_data": None,
            "prompt_attribution": bridged,
        }

    client = object.__new__(RendererClient)
    client._renderer = object()
    client._pool_size = 1
    client._config = vf.ClientConfig(client_type="renderer")
    client._client = object()  # type: ignore[attr-defined]

    with (
        patch.object(RendererClient, "_get_renderer_or_pool", return_value=object()),
        patch(
            "verifiers.clients.renderer_client._get_incremental_prompt_ids",
            side_effect=_fake_get_incremental,
        ),
        patch("verifiers.clients.renderer_client.generate", side_effect=_fake_generate),
    ):
        result = await client.get_native_response(
            prompt=[
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "second"},
            ],
            model="test-model",
            sampling_args={},
            tools=None,
            state={
                "trajectory": [
                    {
                        "prompt": [{"role": "user", "content": "first"}],
                        "completion": [{"role": "assistant", "content": "ok"}],
                    }
                ]
            },
        )

    assert captured.get("prompt_attribution") is bridged
    assert captured.get("prompt_ids") == list(bridged.token_ids)
    assert result["prompt_attribution"] is bridged


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "attribution",
    [
        pytest.param(
            RenderedTokens(
                token_ids=[1, 2, 3],
                message_indices=[0, 0, 0],
                sampled_mask=[False, False, False],
                is_content=[False, True, True],
                message_roles=["user"],
            ),
            id="present",
        ),
        pytest.param(None, id="missing"),
    ],
)
async def test_from_native_response_carries_prompt_attribution(attribution):
    """``from_native_response`` lifts ``prompt_attribution`` from the raw
    ``generate`` result dict onto :class:`ResponseTokens`. Missing key
    resolves to ``None`` rather than ``KeyError``."""
    client = object.__new__(RendererClient)
    response_dict = {
        "request_id": "r-2",
        "content": "ok",
        "reasoning_content": None,
        "tool_calls": [],
        "finish_reason": "stop",
        "prompt_ids": [1, 2, 3],
        "completion_ids": [4, 5],
        "completion_logprobs": [-0.1, -0.2],
        "routed_experts": None,
        "multi_modal_data": None,
    }
    if attribution is not None:
        response_dict["prompt_attribution"] = attribution

    response = await client.from_native_response(response_dict)

    assert response.message.tokens is not None
    assert response.message.tokens.prompt_attribution is attribution
