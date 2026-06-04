import json
import sys
import types
from pathlib import Path

import pytest

import verifiers as vf
from harnesses import ReplayHarness
from tasksets import ReplayTaskset, ReplayTasksetConfig
from tasksets.replay import replay_task_record


class NoModelClient:
    def __init__(self) -> None:
        self.requests = 0

    async def get_response(self, **kwargs: object) -> object:
        _ = kwargs
        self.requests += 1
        raise AssertionError("ReplayHarness must not request model completions.")


class InlineReplayTaskset(ReplayTaskset):
    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        if split == "eval":
            return []
        return [
            {
                "messages": [
                    {"role": "user", "content": "Reverse abc."},
                    {"role": "assistant", "content": "cba"},
                    {"role": "user", "content": "Now uppercase it."},
                    {
                        "role": "assistant",
                        "content": "CBA",
                        "reasoning_content": "uppercased the prior answer",
                    },
                    {"role": "user", "content": "Thanks."},
                ],
            }
        ]


@pytest.mark.asyncio
async def test_replay_harness_prints_assistant_messages_into_trajectory() -> None:
    env = vf.Env(
        taskset=InlineReplayTaskset(),
        harness=ReplayHarness(config=vf.HarnessConfig()),
    )
    client = NoModelClient()

    state = await env.rollout(
        dict(env.get_dataset()[0]),
        client=client,
        model="mock-model",
    )

    assert client.requests == 0
    assert state["stop_condition"] == "replayed_messages"
    assert state["num_model_requests"] == 2
    assert state["prompt"] == [{"role": "user", "content": "Reverse abc."}]
    assert state["completion"] == [
        {"role": "assistant", "content": "cba"},
        {"role": "user", "content": "Now uppercase it."},
        {
            "role": "assistant",
            "content": "CBA",
            "reasoning_content": "uppercased the prior answer",
        },
    ]
    assert state["completion"][-1]["role"] == "assistant"

    first, second = state["trajectory"]
    assert first["prompt"] == [{"role": "user", "content": "Reverse abc."}]
    assert first["completion"] == [{"role": "assistant", "content": "cba"}]
    assert first["tokens"] is None
    assert "tokens" not in first["response"]["message"]

    assert second["prompt"] == [
        {"role": "user", "content": "Reverse abc."},
        {"role": "assistant", "content": "cba"},
        {"role": "user", "content": "Now uppercase it."},
    ]
    assert second["completion"] == [
        {
            "role": "assistant",
            "content": "CBA",
            "reasoning_content": "uppercased the prior answer",
        }
    ]
    assert second["tokens"] is None
    assert "tokens" not in second["response"]["message"]


@pytest.mark.asyncio
async def test_replay_harness_marks_partial_replay_as_truncated() -> None:
    env = vf.Env(
        taskset=InlineReplayTaskset(),
        harness=ReplayHarness(config=vf.HarnessConfig(max_turns=1)),
    )

    state = await env.rollout(
        dict(env.get_dataset()[0]),
        client=NoModelClient(),
        model="mock-model",
    )

    assert state["stop_condition"] == "max_turns_reached"
    assert state["is_truncated"] is True
    assert state["num_model_requests"] == 1
    assert state["completion"] == [{"role": "assistant", "content": "cba"}]
    step = state["trajectory"][0]
    assert step["is_truncated"] is True
    assert step["response"]["message"]["is_truncated"] is True


def test_replay_taskset_loads_env_local_json_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_dir = tmp_path / "local_replay_env"
    data_dir = env_dir / "data"
    data_dir.mkdir(parents=True)
    (env_dir / "local_replay_env.py").write_text("", encoding="utf-8")
    (data_dir / "example.json").write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Say ok."},
                    {"role": "assistant", "content": "ok"},
                ]
            }
        ),
        encoding="utf-8",
    )

    module = types.ModuleType("local_replay_env")
    module.__file__ = str(env_dir / "local_replay_env.py")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    local_taskset_type = type(
        "LocalReplayTaskset",
        (ReplayTaskset,),
        {"__module__": module.__name__},
    )

    taskset = local_taskset_type(config=ReplayTasksetConfig())

    assert taskset.load_tasks() == [
        {
            "messages": [
                {"role": "user", "content": "Say ok."},
                {"role": "assistant", "content": "ok"},
            ]
        }
    ]


def test_replay_taskset_canonicalizes_messages() -> None:
    task = replay_task_record(
        {
            "messages": [
                {"role": "user", "content": "Use the tool."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "search",
                                "arguments": {"query": "abc"},
                            },
                        }
                    ],
                },
            ]
        }
    )

    assert task["messages"] == [
        {"role": "user", "content": "Use the tool."},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "search",
                    "arguments": '{"query": "abc"}',
                }
            ],
        },
    ]


def test_replay_taskset_rejects_invalid_messages() -> None:
    with pytest.raises(TypeError, match="messages must be a list"):
        replay_task_record({"messages": "not a transcript"})

    with pytest.raises(ValueError, match="Unknown role"):
        replay_task_record({"messages": [{"role": "assistantish", "content": "no"}]})
