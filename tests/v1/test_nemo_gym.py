import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from verifiers.v1.env import EnvConfig, Environment
from verifiers.v1.loaders import default_harness_id, taskset_config_type
from verifiers.v1.tasksets.nemo_gym import (
    NeMoGymConfig,
    NeMoGymTask,
    NeMoGymTaskset,
)
from verifiers.v1.types import SystemMessage, UserMessage


def test_nemo_gym_loads_packaged_example(tmp_path, monkeypatch) -> None:
    data = (
        tmp_path
        / "resources_servers"
        / "example_single_tool_call"
        / "data"
        / "example.jsonl"
    )
    data.parent.mkdir(parents=True)
    row = {
        "responses_create_params": {
            "input": [
                {"role": "developer", "content": "Be helpful."},
                {"role": "user", "content": "What's it like in SF?"},
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
        }
    }
    data.write_text(json.dumps(row) + "\n")
    monkeypatch.setitem(sys.modules, "nemo_gym", SimpleNamespace(PARENT_DIR=tmp_path))

    [task] = NeMoGymTaskset(NeMoGymConfig()).load_tasks()

    assert task.name == "example_single_tool_call:0"
    assert task.prompt == [
        SystemMessage(content="Be helpful."),
        UserMessage(content="What's it like in SF?"),
    ]
    assert "nemo_gym_call" in task.system_prompt
    assert "get_weather" in task.system_prompt
    assert task.nemo_gym_row == row


def test_nemo_gym_uses_the_standard_verifiers_harness() -> None:
    env = Environment(
        EnvConfig.model_validate(
            {
                "taskset": {"id": "nemo_gym"},
                "harness": {},
            }
        )
    )
    task = NeMoGymTask(
        idx=0,
        prompt="hello",
        nemo_gym_row={"responses_create_params": {"tools": [{"name": "tool"}]}},
    )

    [toolset] = env.taskset.tools(task)

    assert env.harness.config.id == "default"
    assert toolset.server_name == "nemo_gym"
    assert toolset.config.nemo_env == "example_single_tool_call"


@pytest.mark.asyncio
async def test_nemo_gym_forwards_tools_to_the_resource_server() -> None:
    taskset = NeMoGymTaskset(NeMoGymConfig())
    task = NeMoGymTask(
        idx=0,
        prompt="hello",
        nemo_gym_row={"responses_create_params": {"tools": [{"name": "get_weather"}]}},
    )
    [toolset] = taskset.tools(task)
    toolset.tool_names = {"get_weather"}
    response = Mock()
    response.json.return_value = {"weather_description": "cold"}
    toolset.client = AsyncMock()
    toolset.client.post.return_value = response

    result = await toolset.call("get_weather", {"city": "sf"})

    toolset.client.post.assert_awaited_once_with("/get_weather", json={"city": "sf"})
    response.raise_for_status.assert_called_once()
    assert result == {"weather_description": "cold"}


def test_nemo_gym_is_only_a_taskset() -> None:
    assert default_harness_id("nemo_gym") == "default"
    assert taskset_config_type("nemo_gym") is NeMoGymConfig
