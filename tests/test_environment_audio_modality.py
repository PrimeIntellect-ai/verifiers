# tests/test_environment_audio_modalities.py
import logging
import types

import pytest

from verifiers.envs.environment import Environment

DUMMY_B64 = "ZHVtbXk="


def _get_client_and_sink():
    """
    Prefer repo mock client if available; otherwise use a stub that captures kwargs.
    Returns (client, get_kwargs_fn).
    """
    try:
        from tests.mock_openai_client import MockOpenAIClient

        mock = MockOpenAIClient()
        calls = {"kwargs": None}

        async def _wrap_create(**kwargs):
            calls["kwargs"] = kwargs
            return {"ok": True}

        mock.chat.completions.create = _wrap_create

        def _get():
            return calls["kwargs"]

        return mock, _get
    except Exception:

        class _DummyCompletions:
            def __init__(self):
                self.kwargs = None

            async def create(self, **kwargs):
                self.kwargs = kwargs
                return {"ok": True}

        class _DummyChat:
            def __init__(self):
                self.completions = _DummyCompletions()

        class _DummyClient:
            def __init__(self):
                self.chat = _DummyChat()

        dummy = _DummyClient()
        return dummy, lambda: dummy.chat.completions.kwargs


@pytest.mark.asyncio
async def test_sets_modalities_text_when_audio_and_missing():
    client, get_kwargs = _get_client_and_sink()
    prompt = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {"data": DUMMY_B64, "format": "wav"},
                },
                {"type": "text", "text": "Describe this audio"},
            ],
        }
    ]
    fake_self = types.SimpleNamespace(
        message_type="chat", logger=logging.getLogger("test")
    )

    await Environment.get_model_response(
        fake_self,
        client=client,
        model="gpt-4o-audio-preview",
        prompt=prompt,
        oai_tools=None,
        sampling_args=None,
        message_type=None,
    )

    kwargs = get_kwargs()
    assert kwargs is not None
    assert kwargs.get("modalities") == ["text"]
    assert kwargs.get("messages") == prompt


@pytest.mark.asyncio
async def test_does_not_override_existing_modalities():
    client, get_kwargs = _get_client_and_sink()
    prompt = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {"data": DUMMY_B64, "format": "wav"},
                }
            ],
        }
    ]
    fake_self = types.SimpleNamespace(
        message_type="chat", logger=logging.getLogger("test")
    )

    await Environment.get_model_response(
        fake_self,
        client=client,
        model="gpt-4o-audio-preview",
        prompt=prompt,
        sampling_args={"modalities": ["text", "audio"]},
        oai_tools=None,
        message_type=None,
    )

    kwargs = get_kwargs()
    assert kwargs is not None
    assert kwargs.get("modalities") == ["text", "audio"]


@pytest.mark.asyncio
async def test_does_not_add_modalities_when_no_audio():
    client, get_kwargs = _get_client_and_sink()
    prompt = [{"role": "user", "content": "hello"}]
    fake_self = types.SimpleNamespace(
        message_type="chat", logger=logging.getLogger("test")
    )

    await Environment.get_model_response(
        fake_self,
        client=client,
        model="gpt-4.1-mini",
        prompt=prompt,
        sampling_args=None,
        oai_tools=None,
        message_type=None,
    )

    kwargs = get_kwargs()
    assert kwargs is not None
    assert "modalities" not in kwargs
