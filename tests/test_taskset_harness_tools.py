from typing import Any, cast

import pytest

import verifiers as vf


def test_tool_runtime_classes_are_not_top_level_api():
    for name in ("ToolRegistry", "ToolHandle", "ToolInjector"):
        assert name not in vf.__all__
        with pytest.raises(AttributeError):
            getattr(vf, name)


def test_schema_only_tools_are_rejected():
    taskset = vf.Taskset(
        source=[{"prompt": [vf.UserMessage(content="hello")]}],
        tools=[
            vf.Tool(
                name="echo",
                description="Echo input.",
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            )
        ],
    )
    with pytest.raises(TypeError, match="Schema-only tools"):
        vf.Env(taskset=taskset, harness=vf.Harness())


@pytest.mark.asyncio
async def test_setup_state_does_not_cache_resource_derived_values(mock_client):
    taskset = vf.Taskset(source=[{"prompt": [vf.UserMessage(content="hello")]}])
    env = vf.Env(taskset=taskset, harness=vf.Harness())
    task = taskset.to_task(
        {"prompt": [vf.UserMessage(content="hello")], "example_id": 0}
    )
    harness = env.resources.harness
    assert harness is not None

    async with env.resources.rollout(
        task,
        client=mock_client,
        model="test-model",
        sampling_args={"temperature": 0},
    ):
        state = await harness.setup_state(task, env.resources)

    assert "model" not in state
    assert "sampling_args" not in state
    assert "tool_defs" not in state


@pytest.mark.asyncio
async def test_setup_state_copies_serialized_task_payload(mock_client):
    class DoneHarness(vf.Harness):
        async def setup_state(self, task, resources):
            state = await super().setup_state(task, resources)
            state["done"] = True
            return state

    async def project_reward(project, legacy, task):
        assert project == "explicit"
        assert legacy == "fallback"
        assert task == {
            "project": "explicit",
            "legacy": "fallback",
            "example_id": 0,
            "info": {},
        }
        return 1.0

    taskset = vf.Taskset(
        source=[{"task": {"project": "explicit"}, "legacy": "fallback"}],
        rubric=vf.Rubric(funcs=[project_reward]),
    )
    harness = DoneHarness()
    env = vf.Env(taskset=taskset, harness=harness)
    task = taskset.to_task(
        cast(
            dict[str, Any],
            {"task": {"project": "explicit"}, "legacy": "fallback", "example_id": 0},
        )
    )

    async with env.resources.rollout(task, mock_client, "test-model"):
        state = await harness.run(task, env.resources)

    assert state["task"] == {
        "project": "explicit",
        "legacy": "fallback",
        "example_id": 0,
        "info": {},
    }
    assert "task" not in state["input"]
    assert state["input"]["project"] == "explicit"
    assert state["input"]["legacy"] == "fallback"
    assert state["reward"] == 1.0
    assert state["metrics"]["project_reward"] == 1.0


def test_task_payload_falls_back_to_legacy_extra_inputs():
    taskset = vf.Taskset(source=[{"project": "legacy"}])
    task = taskset.to_task(cast(dict[str, Any], {"project": "legacy", "example_id": 0}))

    assert dict(task) == {"project": "legacy", "example_id": 0, "info": {}}
    assert "task" not in task


def test_task_is_top_level_immutable():
    task = vf.Task(project="frozen")

    with pytest.raises(TypeError, match="immutable"):
        task["project"] = "mutated"
    with pytest.raises(TypeError, match="immutable"):
        task.update({"new": "value"})


def test_task_rows_must_be_json_serializable():
    with pytest.raises(TypeError, match="JSON-serializable"):
        vf.Task(resource=object())
