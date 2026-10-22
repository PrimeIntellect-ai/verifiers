"""The host's static uv for a box that cannot install one (a hardened image with no
pip, curl, wget, apt-get or apk), against a fake box. No sandbox is used."""

import pytest

from verifiers.v1.runtimes import host_uv
from verifiers.v1.runtimes.base import ProgramResult, RuntimeCache
from verifiers.v1.runtimes.host_uv import UV_UNAVAILABLE, needs_host_uv, uv_target
from verifiers.v1.runtimes.runtime import Runtime


@pytest.mark.parametrize(
    ("exit_code", "uploaded", "expected"),
    [
        (UV_UNAVAILABLE, False, True),  # no uv and no way to install one
        (UV_UNAVAILABLE, True, False),  # the host's uv is there and still fails
        (1, False, False),  # uv ran; the script's sync or discovery failed
        (0, False, False),
    ],
)
def test_only_a_box_that_cannot_get_uv_gets_the_hosts(exit_code, uploaded, expected):
    assert needs_host_uv(exit_code, uploaded) is expected


@pytest.mark.parametrize(
    ("machine", "target"),
    [
        ("x86_64\n", "x86_64-unknown-linux-musl"),
        ("aarch64", "aarch64-unknown-linux-musl"),
        ("arm64", "aarch64-unknown-linux-musl"),
    ],
)
def test_the_static_build_matches_the_box_architecture(machine, target):
    assert uv_target(machine) == target
    assert target in host_uv.UV_SHA256


def test_an_unknown_architecture_is_refused():
    with pytest.raises(RuntimeError, match="riscv64"):
        uv_target("riscv64")


class HardenedBox:
    """Answers preparation with UV_UNAVAILABLE until ~/.local/bin/uv is installed."""

    scripts_dir = "/box/scripts"

    def __init__(self):
        self.cache = RuntimeCache()
        self.files: dict[str, bytes] = {}
        self.uv: bytes | None = None
        self.accepts_uv = True
        self.prepares = 0

    async def write(self, runtime, path, data):
        self.files[path] = data

    async def run(self, runtime, argv, env):
        if argv == ["uname", "-m"]:
            return ProgramResult(0, "x86_64\n", "")
        script = argv[2]
        if "/.local/bin/uv" in script:
            self.uv = self.files.pop(argv[-1])
            return ProgramResult(0, "", "")
        self.prepares += 1
        assert env["UV_OFFLINE"] == "false"
        if self.uv is None or not self.accepts_uv:
            return ProgramResult(
                UV_UNAVAILABLE, "", "sh: 1: apt-get: not found\nsh: 1: apk: not found"
            )
        return ProgramResult(0, "/box/env/bin/python\n", "")


async def test_a_hardened_box_gets_the_hosts_uv_once(monkeypatch):
    async def binary(machine):
        assert machine == "x86_64\n"
        return b"static uv"

    monkeypatch.setattr("verifiers.v1.runtimes.runtime.host_uv_binary", binary)
    box = HardenedBox()
    runtime = Runtime(box)  # type: ignore[arg-type]

    argv = await runtime.prepare_uv_script("print(1)\n", activate=False)
    assert argv[0] == "/box/env/bin/python"
    assert box.uv == b"static uv"
    assert box.prepares == 2  # failed for want of uv, then succeeded with it

    await runtime.prepare_uv_script("print(2)\n", activate=False)
    assert box.prepares == 3  # the next script finds uv in place


async def test_a_box_that_still_fails_with_the_hosts_uv_reports_it(monkeypatch):
    async def binary(machine):
        return b"unusable uv"

    monkeypatch.setattr("verifiers.v1.runtimes.runtime.host_uv_binary", binary)
    box = HardenedBox()
    box.accepts_uv = False  # e.g. a noexec home: the uploaded uv cannot run
    with pytest.raises(RuntimeError, match="apt-get: not found"):
        await Runtime(box).prepare_uv_script("print(1)\n")  # type: ignore[arg-type]
    assert box.prepares == 2  # one retry, no loop
