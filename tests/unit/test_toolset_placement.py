import asyncio
from types import SimpleNamespace

import pytest

import verifiers.v1 as vf
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
    "runtime", [vf.PrimeConfig(allow=[]), vf.DockerConfig(), vf.PrimeConfig()]
)
def test_service_placement_takes_no_runtime_of_its_own(runtime):
    with pytest.raises(ValueError, match="no runtime of its own"):
        ToolsetConfig(service="tools", runtime=runtime)


def test_service_placement_survives_a_config_round_trip():
    config = ToolsetConfig(service="tools")
    assert ToolsetConfig.model_validate(config.model_dump()).service == "tools"


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
