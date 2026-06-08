from pathlib import Path
import importlib.util

import pytest


@pytest.mark.asyncio
async def test_openenv_echo_async_step_sets_tool_reward() -> None:
    pytest.importorskip("openenv")
    from openenv.core.env_server.mcp_types import CallToolAction

    path = (
        Path(__file__).parents[1]
        / "environments/openenv_echo_v1/openenv_echo_v1/proj/server/echo_environment.py"
    )
    spec = importlib.util.spec_from_file_location("openenv_echo_environment", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    observation = await module.EchoEnvironment().step_async(
        CallToolAction(
            tool_name="echo_message",
            arguments={"message": "hello from openenv"},
        )
    )

    assert observation.reward == pytest.approx(1.8)
