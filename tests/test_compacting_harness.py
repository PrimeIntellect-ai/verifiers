"""The compacting harness: the pure compaction logic in the default program (summary request
shape, in-place message rebuild, threshold/usage reading) and the harness-side config/argv
wiring. No network, no runtimes — the model client is a stub."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from verifiers.v1.harnesses.compacting import (
    CompactingHarness,
    CompactingHarnessConfig,
)
from verifiers.v1.loaders import harness_class, harness_config_type

PROGRAM_PATH = Path(__file__).resolve().parents[1] / "verifiers" / "v1" / "harnesses" / "default" / "program.py"


def load_program():
    spec = importlib.util.spec_from_file_location("default_program", PROGRAM_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


program = load_program()


class StubClient:
    """Captures chat.completions.create kwargs and returns a canned summary completion."""

    def __init__(self, content="the summary", prompt_tokens=123):
        self.calls = []

        async def create(**kwargs):
            # Snapshot the messages: `compact` mutates the list in place after the call,
            # and the assertions need what the request actually contained.
            kwargs["messages"] = [dict(m) for m in kwargs["messages"]]
            self.calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=content,
                            tool_calls=None,
                            model_dump=lambda exclude_none=False: {
                                "role": "assistant",
                                "content": content,
                            },
                        )
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=prompt_tokens),
            )

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


TOOLS = [{"type": "function", "function": {"name": "bash", "parameters": {}}}]


@pytest.mark.asyncio
async def test_compact_rebuilds_messages_in_place():
    client = StubClient(content="progress so far")
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "step 1"},
        {"role": "tool", "tool_call_id": "1", "content": "output"},
    ]
    original = messages  # identity must be preserved (the loop keeps its reference)
    await program.compact(client, "m", messages, TOOLS, "checkpoint!", "framing:")

    assert messages is original
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0] == {"role": "system", "content": "sys"}
    assert messages[1]["content"] == "framing:\n\nprogress so far"

    # The summary request saw the FULL conversation (incl. the checkpoint prompt appended
    # last) and advertised the tools without allowing calls.
    (call,) = client.calls
    assert call["messages"][-1] == {"role": "user", "content": "checkpoint!"}
    assert [m["role"] for m in call["messages"][:-1]] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert call["tools"] == TOOLS
    assert call["tool_choice"] == "none"


@pytest.mark.asyncio
async def test_compact_without_system_or_tools():
    client = StubClient(content="notes")
    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "work"},
    ]
    await program.compact(client, "m", messages, [], "checkpoint", "framing:")

    assert [m["role"] for m in messages] == ["user"]
    assert messages[0]["content"].endswith("notes")
    # No tools -> no tools/tool_choice on the summary request.
    (call,) = client.calls
    assert call["tools"] is None
    assert "tool_choice" not in call


def test_prompt_tokens_reads_usage():
    completion = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=77))
    assert program.prompt_tokens(completion) == 77
    assert program.prompt_tokens(SimpleNamespace(usage=None)) == 0
    assert program.prompt_tokens(SimpleNamespace(usage=SimpleNamespace(prompt_tokens=None))) == 0


def test_config_requires_positive_threshold():
    with pytest.raises(ValueError, match="positive"):
        CompactingHarnessConfig(id="compacting", compact_at_tokens=0)
    with pytest.raises(Exception):  # missing threshold
        CompactingHarnessConfig(id="compacting")


def test_extra_program_args():
    harness = CompactingHarness(CompactingHarnessConfig(id="compacting", compact_at_tokens=8192))
    assert harness.extra_program_args() == ["--compact-at-tokens=8192"]

    harness = CompactingHarness(
        CompactingHarnessConfig(
            id="compacting",
            compact_at_tokens=4096,
            checkpoint_prompt="summarize",
            compaction_framing="context:",
        )
    )
    assert harness.extra_program_args() == [
        "--compact-at-tokens=4096",
        "--checkpoint-prompt=summarize",
        "--compaction-framing=context:",
    ]


def test_loader_resolves_compacting():
    assert harness_class("compacting") is CompactingHarness
    assert harness_config_type("compacting") is CompactingHarnessConfig


def test_default_harness_appends_nothing():
    from verifiers.v1.harnesses.default import DefaultHarness, DefaultHarnessConfig

    assert DefaultHarness(DefaultHarnessConfig(id="default")).extra_program_args() == []
