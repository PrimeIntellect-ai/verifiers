import json
from pathlib import Path

import pytest

import verifiers as vf
from harnesses import ReplayHarness, ReplayHarnessConfig
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


class ManyTurnReplayTaskset(ReplayTaskset):
    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        if split == "eval":
            return []
        messages = []
        for index in range(11):
            messages.append({"role": "user", "content": f"Turn {index}?"})
            messages.append({"role": "assistant", "content": f"reply {index}"})
        return [{"messages": messages}]


@pytest.mark.asyncio
async def test_replay_harness_prints_assistant_messages_into_trajectory() -> None:
    env = vf.Env(
        taskset=InlineReplayTaskset(),
        harness=ReplayHarness(config=ReplayHarnessConfig()),
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
        harness=ReplayHarness(config=ReplayHarnessConfig(max_turns=1)),
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


@pytest.mark.asyncio
async def test_replay_harness_defaults_to_all_assistant_messages() -> None:
    env = vf.Env(
        taskset=ManyTurnReplayTaskset(),
        harness=ReplayHarness(config=ReplayHarnessConfig()),
    )

    state = await env.rollout(
        dict(env.get_dataset()[0]),
        client=NoModelClient(),
        model="mock-model",
    )

    assert state["stop_condition"] == "replayed_messages"
    assert state["is_truncated"] is False
    assert state["num_model_requests"] == 11
    assert len(state["trajectory"]) == 11
    assert state["completion"][-1] == {"role": "assistant", "content": "reply 10"}


def test_replay_taskset_loads_configured_local_jsonl_data(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "examples.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "Say ok."},
                            {"role": "assistant", "content": "ok"},
                        ]
                    }
                ),
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "Say yes."},
                            {"role": "assistant", "content": "yes"},
                        ]
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    taskset = ReplayTaskset(config=ReplayTasksetConfig(data_dir=str(data_dir)))

    assert taskset.load_tasks() == [
        {
            "messages": [
                {"role": "user", "content": "Say ok."},
                {"role": "assistant", "content": "ok"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Say yes."},
                {"role": "assistant", "content": "yes"},
            ]
        },
    ]


def test_replay_taskset_loads_subclass_local_jsonl_data(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "example.jsonl").write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Say ok."},
                    {"role": "assistant", "content": "ok"},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    local_taskset_type = type(
        "LocalReplayTaskset",
        (ReplayTaskset,),
        {"data_dir": str(data_dir)},
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


def test_replay_taskset_rejects_missing_local_source() -> None:
    taskset = ReplayTaskset(config=ReplayTasksetConfig())

    with pytest.raises(FileNotFoundError, match="requires dataset or data_dir"):
        taskset.load_tasks()


def test_replay_taskset_rejects_conflicting_sources(tmp_path: Path) -> None:
    taskset = ReplayTaskset(
        config=ReplayTasksetConfig(dataset="owner/dataset", data_dir=str(tmp_path))
    )

    with pytest.raises(ValueError, match="cannot set both dataset and data_dir"):
        taskset.load_tasks()


def test_replay_taskset_rejects_empty_local_data_dir(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    taskset = ReplayTaskset(config=ReplayTasksetConfig(data_dir=str(data_dir)))

    with pytest.raises(FileNotFoundError, match="must contain at least one JSONL"):
        taskset.load_tasks()


def test_replay_taskset_rejects_json_files(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
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

    taskset = ReplayTaskset(config=ReplayTasksetConfig(data_dir=str(data_dir)))

    with pytest.raises(ValueError, match=r"accepts only \.jsonl files"):
        taskset.load_tasks()


def test_replay_taskset_rejects_non_object_jsonl_rows(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "example.jsonl").write_text("[]\n", encoding="utf-8")

    taskset = ReplayTaskset(config=ReplayTasksetConfig(data_dir=str(data_dir)))

    with pytest.raises(TypeError, match="example.jsonl:1 must contain one JSON object"):
        taskset.load_tasks()


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
