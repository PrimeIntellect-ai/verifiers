from types import SimpleNamespace

import pytest

from verifiers.v1.harnesses.claude_code.harness import (
    ClaudeCodeHarness,
    ClaudeCodeHarnessConfig,
)
from verifiers.v1.harnesses.kimi_code.harness import (
    KimiCodeHarness,
    KimiCodeHarnessConfig,
)
from verifiers.v1.runtimes.base import ProgramResult, Runtime


class FakeRuntime(Runtime):
    def __init__(self, *, program_error=False, cleanup_error=False):
        super().__init__("test")
        self.config = SimpleNamespace(type="fake")
        self.info = SimpleNamespace(id="test")
        self.program_error = program_error
        self.cleanup_error = cleanup_error
        self.writes = []
        self.commands = []

    async def start(self):
        return None

    async def run(self, argv, env):
        self.commands.append((argv, env))
        if self.cleanup_error:
            raise RuntimeError("cleanup failed")
        return ProgramResult(0, "", "")

    async def run_program(self, argv, env):
        if self.program_error:
            raise RuntimeError("program failed")
        self.program = (argv, env)
        return ProgramResult(0, "", "")

    async def read(self, path):
        return b""

    async def write(self, path, data):
        self.writes.append((path, data))


def trace():
    return SimpleNamespace(
        task=SimpleNamespace(
            data=SimpleNamespace(prompt="hello", system_prompt=None),
        )
    )


def context():
    return SimpleNamespace(
        model="model", client=SimpleNamespace(base_url="https://api.example")
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "harness_cls",
    [
        (ClaudeCodeHarness, ClaudeCodeHarnessConfig),
        (KimiCodeHarness, KimiCodeHarnessConfig),
    ],
)
async def test_launch_uses_unique_home_and_cleans_it(harness_cls):
    harness_type, config_type = harness_cls
    harness = harness_type(config_type())
    runtime = FakeRuntime()
    await harness.launch(context(), trace(), runtime, "https://endpoint", "secret", {})
    assert runtime.writes
    home = runtime.writes[0][0].rsplit("/", 1)[0]
    assert home.startswith("/tmp/vf-")
    assert runtime.commands[-1][0] == ["rm", "-rf", home]


@pytest.mark.asyncio
async def test_program_failure_still_cleans_kimi_home():
    harness = KimiCodeHarness(KimiCodeHarnessConfig())
    runtime = FakeRuntime(program_error=True)
    with pytest.raises(RuntimeError, match="program failed"):
        await harness.launch(
            context(), trace(), runtime, "https://endpoint", "secret", {}
        )
    assert runtime.commands[-1][0][0:2] == ["rm", "-rf"]
