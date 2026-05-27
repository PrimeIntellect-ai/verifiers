from collections.abc import Awaitable, Callable
from typing import cast

import pytest

import verifiers as vf
from tasksets import openreward


class FakeTextBlock:
    type = "text"

    def __init__(self, text: str):
        self.text = text


class FakeToolOutput:
    def __init__(self, text: str, reward: float, finished: bool):
        self.blocks = [FakeTextBlock(text)]
        self.reward = reward
        self.finished = finished
        self.metadata = {"status": "ok"}


class FakeOpenRewardSession:
    def __init__(self, task: object):
        self.task = task
        self.entered = False
        self.exited = False
        self.calls: list[tuple[str, dict[str, object]]] = []

    def __enter__(self) -> "FakeOpenRewardSession":
        self.entered = True
        return self

    def __exit__(self, *exc: object) -> None:
        self.exited = True

    def get_prompt(self) -> list[FakeTextBlock]:
        return [FakeTextBlock("Solve the task.")]

    def call_tool(self, tool_name: str, input: dict[str, object]) -> FakeToolOutput:
        self.calls.append((tool_name, input))
        return FakeToolOutput("Correct.", 1.0, True)


class FakeOpenRewardEnvironment:
    def __init__(self):
        self.sessions: list[FakeOpenRewardSession] = []
        self.task_range_calls: list[tuple[str, int | None, int | None]] = []

    def list_tasks(self, split: str) -> list[dict[str, object]]:
        return [{"id": f"{split}-0"}]

    def get_task_range(
        self, split: str, start: int | None = None, stop: int | None = None
    ) -> list[dict[str, object]]:
        self.task_range_calls.append((split, start, stop))
        return [{"id": f"{split}-{index}"} for index in range(start or 0, stop or 0)]

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

    def session(self, task: object) -> FakeOpenRewardSession:
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

    monkeypatch.setattr(openreward, "openreward_client", client_factory)
    return environment


def test_openreward_taskset_loads_serializable_rows(fake_openreward_client):
    taskset = openreward.OpenRewardTaskset(
        config=openreward.OpenRewardTasksetConfig(
            environment="owner/env",
            split="train",
            num_train_examples=2,
        )
    )

    rows = list(taskset.get_dataset())
    task = taskset.to_task(rows[0])

    assert fake_openreward_client.task_range_calls == [("train", 0, 2)]
    assert task["openreward"]["environment"] == "owner/env"
    assert task["openreward"]["task"] == {"task_spec": {"id": "train-0"}}
    assert task["openreward"]["tools"][0]["name"] == "answer"
    assert task["toolsets"]["openreward"]["fn"] == (
        "tasksets.openreward:openreward_toolset"
    )


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
