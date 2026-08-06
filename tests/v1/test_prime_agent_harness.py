import asyncio
import hashlib
import os
import shlex
import subprocess
import sys
from pathlib import PurePosixPath
from types import ModuleType, SimpleNamespace

import pytest

from verifiers.v1.acp import ACP_SOURCE, ACPHarnessSession, _new_segment, _packet
from verifiers.v1.harnesses.prime_agent.harness import (
    HARNESS_STATE_FILENAME,
    INSTALL,
    REFINEMENT_HISTORY_FILENAME,
    STATE_ROOT,
    PrimeAgentHarness,
    PrimeAgentHarnessConfig,
    _guarded_install,
)
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace


def test_trace_root_encodes_untrusted_trace_id() -> None:
    trace = Trace.model_construct(id="../../etc")

    root = PurePosixPath(PrimeAgentHarness.trace_root(trace))

    assert root.parent == PurePosixPath(STATE_ROOT)
    assert root.name == hashlib.sha256(trace.id.encode()).hexdigest()[:32]


@pytest.mark.asyncio
async def test_harness_state_directory_is_staged(tmp_path) -> None:
    state = b'{"schema":1,"entries":{},"refinements":[]}\n'
    history = b'{"id":"ref-1","appliedEdits":[]}\n'
    (tmp_path / HARNESS_STATE_FILENAME).write_bytes(state)
    (tmp_path / REFINEMENT_HISTORY_FILENAME).write_bytes(history)

    class Runtime:
        def __init__(self) -> None:
            self.writes: dict[str, bytes] = {}

        async def run(self, argv: list[str], env: dict[str, str]):
            del env
            if argv[:2] == ["sh", "-c"] and argv[3] == "prime-agent-harness-state":
                target = argv[4]
                code = 0 if target in self.writes else 1
                return SimpleNamespace(exit_code=code, stdout="", stderr="")
            return SimpleNamespace(exit_code=0, stdout="", stderr="")

        async def write(self, path: str, data: bytes) -> None:
            self.writes[path] = data

    trace = Trace.model_construct(id="trace")
    runtime = Runtime()
    harness = PrimeAgentHarness(PrimeAgentHarnessConfig(harness_state_dir=tmp_path))

    await harness.install_harness_state(runtime, trace)

    root = PrimeAgentHarness.runtime_harness_state_dir(trace)
    assert runtime.writes == {
        f"{root}/{HARNESS_STATE_FILENAME}": state,
        f"{root}/{REFINEMENT_HISTORY_FILENAME}": history,
    }

    (tmp_path / HARNESS_STATE_FILENAME).write_bytes(b"replacement")
    await harness.install_harness_state(runtime, trace)

    assert runtime.writes[f"{root}/{HARNESS_STATE_FILENAME}"] == state


@pytest.mark.asyncio
async def test_harness_state_directory_rejects_symlink(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    link = tmp_path / "link"
    link.symlink_to(source, target_is_directory=True)
    harness = PrimeAgentHarness(PrimeAgentHarnessConfig(harness_state_dir=link))

    with pytest.raises(ValueError, match="is not a folder"):
        await harness.install_harness_state(
            SimpleNamespace(), Trace.model_construct(id="trace")
        )


@pytest.mark.asyncio
async def test_acp_eof_reports_process_exit_and_stderr() -> None:
    async def stream(*chunks: bytes, delay: float = 0):
        await asyncio.sleep(delay)
        for chunk in chunks:
            yield chunk

    class Process:
        stdout = stream()
        # Reproduce the remote ordering where stdout and process exit arrive
        # before the final stderr frame.
        stderr = stream(b"daemon worker received SIGTERM\n", delay=0.01)

        async def write(self, data: bytes) -> None:
            del data
            await asyncio.sleep(0)

        async def wait(self) -> int:
            return 143

        async def terminate(self) -> None:
            raise AssertionError("the exited process must not be terminated")

        async def kill(self) -> None:
            raise AssertionError("the exited process must not be killed")

    class Runtime:
        async def prepare_uv_script(self, source: str, env: dict[str, str]):
            del source, env
            return ["acp-runner"]

        async def open_process(self, argv: list[str], env: dict[str, str]):
            del argv, env
            return Process()

    session = ACPHarnessSession(
        SimpleNamespace(config=SimpleNamespace(id="prime-agent")),
        SimpleNamespace(),
        Trace.model_construct(id="trace"),
        Runtime(),
        "http://intercept",
        "secret",
        {},
        TaskData(prompt="hello"),
        env={},
        command=["prime-agent", "--mode", "acp"],
        prompt="hello",
        system_prompt=None,
    )

    with pytest.raises(EOFError) as caught:
        await session._run(None)

    assert "exit 143" in str(caught.value)
    assert "daemon worker received SIGTERM" in str(caught.value)


@pytest.mark.asyncio
async def test_acp_reader_ignores_process_keepalives() -> None:
    async def stream():
        yield _packet({"type": "keepalive"}) + _packet(
            {"ok": True, "reply": "finished"}
        )

    class Process:
        stdout = stream()
        stderr = stream()

        async def write(self, data: bytes) -> None:
            del data

    class Runtime:
        async def prepare_uv_script(self, source: str, env: dict[str, str]):
            del source, env
            return ["acp-runner"]

        async def open_process(self, argv: list[str], env: dict[str, str]):
            del argv, env
            return Process()

    session = ACPHarnessSession(
        SimpleNamespace(config=SimpleNamespace(id="prime-agent")),
        SimpleNamespace(),
        Trace.model_construct(id="trace"),
        Runtime(),
        "http://intercept",
        "secret",
        {},
        TaskData(prompt="hello"),
        env={},
        command=["prime-agent", "--mode", "acp"],
        prompt="hello",
        system_prompt=None,
    )

    result = await session._run(None)

    assert result.stdout == "finished"


@pytest.mark.asyncio
async def test_acp_runner_emits_keepalives_during_silent_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acp = ModuleType("acp")
    acp.PROTOCOL_VERSION = 1
    acp.Client = object
    acp.RequestError = Exception
    acp.image_block = lambda *args: args
    acp.spawn_agent_process = lambda *args, **kwargs: None
    acp.text_block = lambda value: value
    schema = ModuleType("acp.schema")
    for name in (
        "AgentMessageChunk",
        "AllowedOutcome",
        "ClientCapabilities",
        "DeniedOutcome",
        "HttpMcpServer",
        "PermissionOption",
        "RequestPermissionResponse",
        "TextContentBlock",
        "ToolCall",
        "ToolCallUpdate",
    ):
        setattr(schema, name, type(name, (), {}))
    monkeypatch.setitem(sys.modules, "acp", acp)
    monkeypatch.setitem(sys.modules, "acp.schema", schema)
    namespace = {"__name__": "acp_runner_test"}
    exec(compile(ACP_SOURCE, "acp_runner.py", "exec"), namespace)  # noqa: S102
    with_keepalives = namespace["with_keepalives"]

    class Stream:
        def __init__(self) -> None:
            self.data = bytearray()

        def write(self, data: bytes) -> None:
            self.data.extend(data)

        def flush(self) -> None:
            pass

    async def delayed_reply() -> str:
        await asyncio.sleep(0.03)
        return "finished"

    stream = Stream()
    reply = await with_keepalives(delayed_reply(), stream, interval=0.005)

    assert reply == "finished"
    keepalive = _packet({"type": "keepalive"})
    assert len(stream.data) >= len(keepalive) * 3
    assert len(stream.data) % len(keepalive) == 0
    assert bytes(stream.data) == keepalive * (len(stream.data) // len(keepalive))


def test_acp_followup_sends_only_new_message_segment() -> None:
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "follow-up"},
    ]

    assert _new_segment(messages) == [{"role": "user", "content": "follow-up"}]


def test_install_lock_uses_process_lifetime_lock(tmp_path) -> None:
    lock = tmp_path / "install.lock"
    marker = tmp_path / "installed"
    lock.write_text("a leftover advisory lock file is not held")
    command = f": > {shlex.quote(str(marker))}"
    script = _guarded_install(str(tmp_path), str(lock), command)

    result = subprocess.run(
        ["sh", "-c", script],
        capture_output=True,
        text=True,
        timeout=2,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert marker.exists()
    assert "/proc/" not in script
    assert 'rm -f "$lock"' not in script


def test_install_lock_serializes_concurrent_installers(tmp_path) -> None:
    lock = tmp_path / "install.lock"
    critical = tmp_path / "critical"
    visits = tmp_path / "visits"
    command = (
        f"mkdir {shlex.quote(str(critical))}; "
        f"printf x >> {shlex.quote(str(visits))}; "
        "sleep 0.2; "
        f"rmdir {shlex.quote(str(critical))}"
    )
    script = _guarded_install(str(tmp_path), str(lock), command)

    first = subprocess.Popen(["sh", "-c", script], stderr=subprocess.PIPE, text=True)
    second = subprocess.Popen(["sh", "-c", script], stderr=subprocess.PIPE, text=True)
    _, first_error = first.communicate(timeout=3)
    _, second_error = second.communicate(timeout=3)

    assert first.returncode == 0, first_error
    assert second.returncode == 0, second_error
    assert visits.read_text() == "xx"


@pytest.mark.parametrize(
    "version",
    ["1.2.3", "v1.2.3", "1.2.3-rc.1", "1.2.3-rc.1+build.7"],
)
def test_prime_agent_version_accepts_semver(version: str) -> None:
    config = PrimeAgentHarnessConfig(
        version=version,
        tarball_sha256="a" * 64,
    )

    assert config.version == version.removeprefix("v")


@pytest.mark.parametrize(
    "version",
    ["01.2.3", "1.02.3", "1.2.03", "1.2.3-01", "1.2.3-.."],
)
def test_prime_agent_version_rejects_invalid_semver(version: str) -> None:
    with pytest.raises(ValueError, match="semantic version"):
        PrimeAgentHarnessConfig(version=version, tarball_sha256="a" * 64)


def test_node_install_is_scoped_to_prime_agent_identity() -> None:
    first = PrimeAgentHarness(PrimeAgentHarnessConfig(tarball_sha256="a" * 64))
    second = PrimeAgentHarness(PrimeAgentHarnessConfig(tarball_sha256="b" * 64))

    assert first.node_root() != second.node_root()
    assert first.node_bin_dir().startswith(first.node_root())


@pytest.mark.parametrize("failed_move", [0, 1, 2])
def test_install_activation_is_transactional(tmp_path, failed_move: int) -> None:
    root = tmp_path / "install"
    old_bin = root / "node_modules" / ".bin" / "prime-agent"
    old_bin.parent.mkdir(parents=True)
    old_bin.write_text("old install")
    old_bin.chmod(0o700)
    (root / ".installed").write_text("old digest")

    tarball = tmp_path / "prime-agent.tgz"
    tarball.write_bytes(b"new package")
    digest = hashlib.sha256(tarball.read_bytes()).hexdigest()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()

    curl = fake_bin / "curl"
    curl.write_text('#!/bin/sh\ncp "$TEST_TARBALL" "$4"\n')
    npm = fake_bin / "npm"
    npm.write_text(
        "#!/bin/sh\n"
        'while [ "$#" -gt 0 ]; do\n'
        '  if [ "$1" = --prefix ]; then prefix=$2; shift 2; else shift; fi\n'
        "done\n"
        'mkdir -p "$prefix/node_modules/.bin"\n'
        'printf %s new-install > "$prefix/node_modules/.bin/prime-agent"\n'
        'chmod 700 "$prefix/node_modules/.bin/prime-agent"\n'
    )
    mv = fake_bin / "mv"
    mv.write_text(
        "#!/bin/sh\n"
        'count=$(cat "$TEST_MV_COUNT" 2>/dev/null || printf 0)\n'
        "count=$((count + 1))\n"
        'printf %s "$count" > "$TEST_MV_COUNT"\n'
        'if [ "$count" -eq "$TEST_FAILED_MOVE" ]; then\n'
        '  /bin/mv "$@"\n'
        "  exit 1\n"
        "fi\n"
        'exec /bin/mv "$@"\n'
    )
    node = fake_bin / "node"
    node.write_text('#!/bin/sh\n[ "$1" = --version ] && printf v22.19.0\nexit 0\n')
    for executable in (curl, npm, mv, node):
        executable.chmod(0o700)

    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "TEST_FAILED_MOVE": str(failed_move),
        "TEST_MV_COUNT": str(tmp_path / "mv-count"),
        "TEST_TARBALL": str(tarball),
        "VF_NODE_BIN_DIR": str(fake_bin),
        "VF_NODE_ROOT": str(tmp_path / "node"),
        "VF_NODE_VERSION": "22.19.0",
        "VF_PRIME_AGENT_BIN": str(old_bin),
        "VF_PRIME_AGENT_INSTALL_DIR": str(root),
        "VF_PRIME_AGENT_KERNEL_VENV": str(tmp_path / "kernel"),
        "VF_PRIME_AGENT_TARBALL": "https://example.invalid/prime-agent.tgz",
        "VF_PRIME_AGENT_TARBALL_SHA256": digest,
    }
    result = subprocess.run(
        ["sh", "-c", INSTALL],
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    if failed_move:
        assert result.returncode != 0
        assert old_bin.read_text() == "old install"
        assert (root / ".installed").read_text() == "old digest"
    else:
        assert result.returncode == 0, result.stderr
        assert old_bin.read_text() == "new-install"
        assert (root / ".installed").read_text() == digest
    assert not list(tmp_path.glob("install.previous.*"))
    assert not list(tmp_path.glob("install.staging.*"))
