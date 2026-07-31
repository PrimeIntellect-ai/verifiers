import os
import subprocess
from types import SimpleNamespace
from typing import cast

import pytest

from verifiers.v1.clients import ModelContext
from verifiers.v1.harnesses.browser import program
from verifiers.v1.harnesses.browser.harness import (
    BROWSER_SYSTEM_PROMPT,
    BrowserHarness,
    BrowserHarnessConfig,
    teardown_argv,
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
        self.writes: list[str] = []

    async def prepare_uv_script(self, source: str, env: dict) -> list[str]:
        return ["python", "browser-program.py"]

    async def run_program(self, argv: list[str], env: dict) -> ProgramResult:
        self.program = argv
        return ProgramResult(exit_code=0, stdout="", stderr="")

    async def write(self, path: str, data: bytes) -> None:
        self.writes.append(path)


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


def test_cdp_credentials_are_not_inherited_by_model_code(tmp_path):
    endpoint = "wss://browser.example/devtools?token=secret"

    daemon_env, tool_env = program.browser_environments(endpoint, tmp_path)

    assert daemon_env["BU_CDP_WS"] == endpoint
    assert "BU_CDP_URL" not in tool_env
    assert "BU_CDP_WS" not in tool_env


def test_stale_owned_browser_is_stopped_before_relaunch(tmp_path, monkeypatch):
    browser = tmp_path / "fake-chromium"
    browser.write_text(
        "#!/usr/bin/env python3\n"
        "import signal\n"
        "import sys\n"
        "print('DevTools listening on ws://127.0.0.1:43210/devtools/browser/id', "
        "file=sys.stderr, flush=True)\n"
        "signal.pause()\n"
    )
    browser.chmod(0o755)
    monkeypatch.setenv("BH_CHROME_PATH", str(browser))

    stale = subprocess.Popen([str(browser)], stderr=subprocess.DEVNULL)
    (tmp_path / "browser.pid").write_text(str(stale.pid))
    (tmp_path / "cdp-endpoint").write_text("http://127.0.0.1:1")

    try:
        endpoint = program.ensure_chromium(tmp_path)
        replacement = int((tmp_path / "browser.pid").read_text())

        assert endpoint == "http://127.0.0.1:43210"
        assert replacement != stale.pid
        stale.wait(timeout=2)
    finally:
        program._stop_recorded_browser(tmp_path)

    os.waitpid(replacement, 0)
    with pytest.raises(ProcessLookupError):
        os.kill(replacement, 0)


def test_teardown_stops_only_recorded_processes(tmp_path):
    browser = subprocess.Popen(["sleep", "30"])
    daemon = subprocess.Popen(["sleep", "30"])
    external = subprocess.Popen(["sleep", "30"])
    runtime = tmp_path / "bh-home" / "runtime"
    runtime.mkdir(parents=True)
    (tmp_path / "browser.pid").write_text(str(browser.pid))
    (runtime / "bu-default.pid").write_text(str(daemon.pid))

    try:
        subprocess.run(teardown_argv(str(tmp_path)), check=True)

        browser.wait(timeout=2)
        daemon.wait(timeout=2)
        assert external.poll() is None
        assert not tmp_path.exists()
    finally:
        if external.poll() is None:
            external.terminate()
        external.wait(timeout=2)
