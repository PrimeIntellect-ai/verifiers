import json
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest

from verifiers.v1.clients import ModelContext
from verifiers.v1.harnesses.cline.harness import ClineHarness, ClineHarnessConfig
from verifiers.v1.runtimes import ProgramResult
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace


@pytest.mark.asyncio
async def test_prepare_acp_allows_user_to_open_conversation():
    harness = ClineHarness(ClineHarnessConfig(id="cline"))
    runtime = AsyncMock()
    runtime.run.return_value = ProgramResult(exit_code=0, stdout="", stderr="")

    config = await harness.prepare_acp(
        cast(ModelContext, SimpleNamespace(model="test/model")),
        cast(Trace, SimpleNamespace(id="trace-id")),
        runtime,
        "http://model.invalid/v1",
        "interception-secret",
        {"resume": "http://tool.invalid/mcp"},
        TaskData(prompt=None, system_prompt="system"),
    )

    assert config.prompt is None
    assert config.system_prompt == "system"
    assert "--acp" in config.command
    assert config.mcp_urls == {}
    mcp_payload = next(
        call.args[1]
        for call in runtime.write.await_args_list
        if call.args[0].endswith("cline_mcp_settings.json")
    )
    assert json.loads(mcp_payload) == {
        "mcpServers": {
            "resume": {
                "transport": {
                    "type": "streamableHttp",
                    "url": "http://tool.invalid/mcp",
                }
            }
        }
    }
