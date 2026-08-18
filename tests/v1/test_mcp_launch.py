from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest

from verifiers.v1.mcp import launch
from verifiers.v1.mcp.server import ServerBase
from verifiers.v1.runtimes.base import ProgramResult, Runtime


class _Config:
    def model_dump_json(self) -> str:
        return "{}"


class _Server:
    config = _Config()
    server_name = "test"


class _PrebuiltServer(_Server):
    RUNTIME_PYTHON = "/opt/server/.venv/bin/python"


class _Runtime:
    def __init__(self, runtime_type: str) -> None:
        self.type = runtime_type
        self.info = SimpleNamespace(id="/tmp/runtime")
        self.published_port = 8000
        self.background_command: list[str] | None = None

    async def run_background(
        self, command: list[str], env: dict[str, str], log: str
    ) -> None:
        self.background_command = command

    async def run(self, command: list[str], env: dict[str, str]) -> ProgramResult:
        return ProgramResult(exit_code=0, stdout="", stderr="")


@pytest.mark.parametrize(
    ("server", "runtime_type", "installed_python", "expected_python", "install_calls"),
    [
        (
            _PrebuiltServer(),
            "docker",
            "/installed/python",
            _PrebuiltServer.RUNTIME_PYTHON,
            0,
        ),
        (_Server(), "docker", "/installed/python", "/installed/python", 1),
        (_PrebuiltServer(), "subprocess", "/installed/python", sys.executable, 0),
    ],
)
async def test_serve_in_runtime_selects_python(
    monkeypatch: pytest.MonkeyPatch,
    server: _Server,
    runtime_type: str,
    installed_python: str,
    expected_python: str,
    install_calls: int,
) -> None:
    install = AsyncMock(return_value=installed_python)
    monkeypatch.setattr(launch, "_install_in_sandbox", install)
    runtime = _Runtime(runtime_type)

    await launch.serve_in_runtime(
        cast(ServerBase, server), cast(Runtime, runtime), exposed=True
    )

    assert runtime.background_command is not None
    if runtime_type == "subprocess":
        assert runtime.background_command[0] == expected_python
    else:
        assert expected_python in runtime.background_command[-1]
    assert install.await_count == install_calls
