import json
import sys
import types
from pathlib import Path

import pytest

import verifiers as vf
from harnesses import ReplayHarness
from tasksets import ReplayTaskset, ReplayTasksetConfig


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
