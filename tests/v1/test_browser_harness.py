from types import SimpleNamespace
from typing import cast

import pytest

from verifiers.v1.clients import ModelContext
from verifiers.v1.harnesses.browser.harness import (
    BROWSER_SYSTEM_PROMPT,
    BrowserHarness,
    BrowserHarnessConfig,
)
from verifiers.v1.loaders import harness_class, harness_config_type
from verifiers.v1.runtimes import (
    DockerConfig,
    ProgramResult,
    Runtime,
    SubprocessConfig,
)
from verifiers.v1.task import Task, TaskData
from verifiers.v1.trace import Trace
from verifiers.v1.types import SystemMessage, UserMessage
from verifiers.v1.utils.compile import validate_pairing


class FakeRuntime:
    def __init__(self) -> None:
        self.program: list[str] = []

    async def prepare_uv_script(self, source: str, env: dict) -> list[str]:
        return ["python", "browser-program.py"]

    async def run_program(self, argv: list[str], env: dict) -> ProgramResult:
        self.program = argv
        return ProgramResult(exit_code=0, stdout="", stderr="")

    async def write(self, path: str, data: bytes) -> None:
        pass


def harness(**config) -> BrowserHarness:
    return BrowserHarness(BrowserHarnessConfig(id="browser", **config))


def test_browser_harness_resolves_from_loader():
    assert harness_class("browser") is BrowserHarness
    assert harness_config_type("browser") is BrowserHarnessConfig


def test_browser_config_requires_exactly_the_cdp_pair():
    with pytest.raises(ValueError, match="needs cdp_url"):
        BrowserHarnessConfig(id="browser", browser="cdp")
    with pytest.raises(ValueError, match="only valid"):
        BrowserHarnessConfig(
            id="browser",
            browser="chromium",
            cdp_url="http://127.0.0.1:9222",
        )

    config = BrowserHarnessConfig(
        id="browser",
        browser="cdp",
        cdp_url="wss://browser.example/devtools",
    )
    assert config.cdp_url == "wss://browser.example/devtools"


def test_browser_harness_requires_a_container_runtime():
    browser = harness()

    with pytest.raises(ValueError, match="NEEDS_CONTAINER"):
        validate_pairing(browser, Task, SubprocessConfig())

    validate_pairing(browser, Task, DockerConfig())


@pytest.mark.asyncio
async def test_resume_does_not_repeat_the_browser_system_prompt():
    runtime = FakeRuntime()
    trace = SimpleNamespace(
        id="trace",
        branches=[
            SimpleNamespace(messages=[SystemMessage(content=BROWSER_SYSTEM_PROMPT)])
        ],
    )

    await harness().resume(
        cast(ModelContext, SimpleNamespace(model="model")),
        cast(Trace, trace),
        cast(Runtime, runtime),
        "http://model.example/v1",
        "secret",
        {},
        TaskData(),
        [UserMessage(content="Continue.")],
    )

    assert not any(arg.startswith("--system-prompt=") for arg in runtime.program)


@pytest.mark.asyncio
async def test_cdp_endpoint_is_forwarded_to_the_program():
    endpoint = "wss://browser.example/devtools"
    runtime = FakeRuntime()

    await harness(browser="cdp", cdp_url=endpoint).launch(
        cast(ModelContext, SimpleNamespace(model="model")),
        cast(Trace, SimpleNamespace(id="trace")),
        cast(Runtime, runtime),
        "http://model.example/v1",
        "secret",
        {},
        TaskData(prompt="Use the browser."),
    )

    assert f"--cdp-url={endpoint}" in runtime.program
