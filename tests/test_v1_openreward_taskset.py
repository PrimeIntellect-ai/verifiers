from collections.abc import Awaitable, Callable
from typing import cast

import pytest

import verifiers as vf

pytest.importorskip("openreward")

from openreward.api.environments.types import (
    Task as OpenRewardTask,
    TextBlock as OpenRewardTextBlock,
    ToolOutput as OpenRewardToolOutput,
)
from tasksets import openreward


class FakeOpenRewardSession:
    def __init__(self, task: OpenRewardTask):
        self.task = task
        self.entered = False
        self.exited = False
        self.calls: list[tuple[str, dict[str, object]]] = []

    def __enter__(self) -> "FakeOpenRewardSession":
        self.entered = True
        return self

    def __exit__(self, *exc: object) -> None:
        self.exited = True

    def get_prompt(self) -> list[OpenRewardTextBlock]:
        return [OpenRewardTextBlock(text="Solve the task.")]

    def list_tools(self, format: str | None = None) -> list[dict[str, object]]:
        assert format == "openai"
        return [
            {
                "type": "function",
                "name": "answer",
                "description": "Submit an answer",
                "parameters": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
            }
        ]

    def call_tool(
        self, tool_name: str, input: dict[str, object]
    ) -> OpenRewardToolOutput:
        self.calls.append((tool_name, input))
        return OpenRewardToolOutput(
            blocks=[OpenRewardTextBlock(text="Correct.")],
            reward=1.0,
            finished=True,
            metadata={"status": "ok"},
        )


class FakeOpenRewardSplit:
    def __init__(self, name: str, type: str):
        self.name = name
        self.type = type


class FakeOpenRewardEnvironment:
    def __init__(self):
        self.sessions: list[FakeOpenRewardSession] = []
        self.task_range_calls: list[tuple[str, int | None, int | None]] = []
        self.splits = [
            FakeOpenRewardSplit(name="train", type="train"),
            FakeOpenRewardSplit(name="official-test", type="test"),
        ]

    def list_splits(self) -> list[FakeOpenRewardSplit]:
        return self.splits

    def list_tasks(self, split: str) -> list[OpenRewardTask]:
        return [
            OpenRewardTask(
                server_name="owner/env",
                environment_name="env",
                namespace="owner",
                task_spec={"id": f"{split}-0"},
            )
        ]

    def get_task_range(
        self, split: str, start: int | None = None, stop: int | None = None
    ) -> list[OpenRewardTask]:
        self.task_range_calls.append((split, start, stop))
        return [
            OpenRewardTask(
                server_name="owner/env",
                environment_name="env",
                namespace="owner",
                task_spec={"id": f"{split}-{index}"},
            )
            for index in range(start or 0, stop or 0)
        ]

    def session(self, task: OpenRewardTask) -> FakeOpenRewardSession:
        session = FakeOpenRewardSession(task)
        self.sessions.append(session)
        return session


class FakeOpenRewardEnvironmentsAPI:
    def __init__(self, environment: FakeOpenRewardEnvironment):
        self.environment = environment
        self.get_calls: list[dict[str, object]] = []

    def get(
        self,
        name: str,
        variant: str | None = None,
        base_url: str | None = None,
    ) -> FakeOpenRewardEnvironment:
        self.get_calls.append({"name": name, "variant": variant, "base_url": base_url})
        return self.environment


class FakeOpenRewardClient:
    instances: list["FakeOpenRewardClient"] = []

    def __init__(self, environment: FakeOpenRewardEnvironment):
        self.environments = FakeOpenRewardEnvironmentsAPI(environment)
        self.closed = False
        FakeOpenRewardClient.instances.append(self)

    def __enter__(self) -> "FakeOpenRewardClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_openreward_client(monkeypatch):
    FakeOpenRewardClient.instances.clear()
    environment = FakeOpenRewardEnvironment()

    def client_factory():
        return FakeOpenRewardClient(environment)

    monkeypatch.setattr(openreward, "OpenReward", client_factory)
    return environment


def test_openreward_taskset_loads_serializable_tasks(fake_openreward_client):
    taskset = openreward.OpenRewardTaskset(
        config=openreward.OpenRewardTasksetConfig(
            environment="owner/env",
            split="train",
            num_train_examples=2,
        )
    )

    tasks = list(taskset.get_dataset())
    task = taskset.to_task(tasks[0])

    assert fake_openreward_client.task_range_calls == [("train", 0, 2)]
    assert task["openreward"]["environment"] == "owner/env"
    assert task["openreward"]["task"] == {
        "server_name": "owner/env",
        "environment_name": "env",
        "namespace": "owner",
        "task_spec": {"id": "train-0"},
    }
    assert set(taskset.named_toolsets) == {"openreward"}


def test_openreward_taskset_eval_tasks_use_test_split(fake_openreward_client):
    taskset = openreward.OpenRewardTaskset(
        config=openreward.OpenRewardTasksetConfig(
            environment="owner/env",
            split="train",
            num_eval_examples=2,
        )
    )

    tasks = list(taskset.eval_tasks())

    assert fake_openreward_client.task_range_calls == [("official-test", 0, 2)]
    assert tasks[0]["openreward"]["split"] == "official-test"
    assert tasks[0]["openreward"]["task"]["task_spec"] == {"id": "official-test-0"}


def test_openreward_taskset_eval_tasks_empty_without_test_split(
    fake_openreward_client,
):
    fake_openreward_client.splits = [FakeOpenRewardSplit(name="train", type="train")]
    taskset = openreward.OpenRewardTaskset(
        config=openreward.OpenRewardTasksetConfig(
            environment="owner/env",
            num_eval_examples=2,
        )
    )

    assert len(taskset.get_eval_dataset()) == 0
    assert fake_openreward_client.task_range_calls == []


@pytest.mark.asyncio
async def test_openreward_taskset_setup_and_tool_call(fake_openreward_client):
    taskset = openreward.OpenRewardTaskset(
        config=openreward.OpenRewardTasksetConfig(
            environment="owner/env",
            split="train",
            num_train_examples=1,
        )
    )
    env = vf.Env(taskset=taskset, harness=vf.Harness())
    task = next(iter(taskset))
    state = vf.State.for_task(task)

    await env.harness.setup_state(task, state)
    await env.harness.runtime.setup_rollout(task, state)

    assert state["prompt"] == [vf.UserMessage(content="Solve the task.")]
    assert state["tools"] == ["answer"]
    state["trajectory"].append({"reward": None})
    tool = cast(
        Callable[..., Awaitable[object]],
        env.harness.runtime.tool_calls(task, state)["answer"],
    )
    result = await tool(answer="4")

    session = fake_openreward_client.sessions[0]
    assert session.entered is True
    assert session.calls == [("answer", {"answer": "4"})]
    assert result == "Correct."
    assert state["trajectory"][-1]["reward"] == 1.0
    assert state["openreward_finished"] is True

    await env.harness.runtime.cleanup_rollout(task, state)
    assert session.exited is True
    assert FakeOpenRewardClient.instances[-1].closed is True
