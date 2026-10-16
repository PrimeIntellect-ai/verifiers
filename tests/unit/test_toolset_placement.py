import asyncio
from types import SimpleNamespace

import pytest

from verifiers.v1.errors import ToolsetError
from verifiers.v1.mcp.launch import _serve
from verifiers.v1.mcp.toolset import ToolsetConfig


def test_service_placement_excludes_other_placements():
    assert ToolsetConfig(service="tools").service == "tools"
    with pytest.raises(ValueError):
        ToolsetConfig(service="tools", colocated=True)
    with pytest.raises(ValueError):
        ToolsetConfig(service="tools", url="http://example.com/mcp")


@pytest.mark.parametrize(
    ("services", "message"),
    [
        ({}, "whose Compose services define it"),
        (
            {"other": SimpleNamespace(is_local=False)},
            "whose Compose services define it",
        ),
        ({"tools": SimpleNamespace(is_local=True)}, "Prime or Modal VM"),
    ],
)
def test_service_placement_needs_a_remote_compose_service(services, message):
    server = SimpleNamespace(config=ToolsetConfig(service="tools"), server_name="gym")

    async def serve():
        async with _serve(server, services=services):
            pass

    with pytest.raises(ToolsetError, match=message):
        asyncio.run(serve())
