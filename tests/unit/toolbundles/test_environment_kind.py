"""An environment-kind bundle's server is confined to its declared endpoints: it
connects only to declared sockets, writes only under declared paths (and /tmp), and
runs only declared commands, and loads no native library to get around that."""

import json
import socket
import subprocess
import sys
import textwrap
import time

import pytest

from verifiers.v1.toolbundles import BundleManifest, vf_tools

TOOLS = textwrap.dedent(
    """
    import socket
    import subprocess
    import sys

    from vf_tools import tool


    @tool
    def connect(state, port: int) -> bool:
        with socket.create_connection(("127.0.0.1", port), timeout=5):
            return True


    @tool
    def write(state, path: str, text: str) -> int:
        with open(path, "w") as handle:
            return handle.write(text)


    @tool
    def run(state, command: str) -> str:
        return subprocess.run([command, "hi"], capture_output=True, text=True).stdout


    @tool
    def load(state) -> bool:
        import ctypes

        ctypes.CDLL(None)
        return True
    """
)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def served(tmp_path):
    bundle = tmp_path / "env"
    bundle.mkdir()
    (bundle / "tools.py").write_text(TOOLS)
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    open_port = listener.getsockname()[1]
    port = free_port()
    server = subprocess.Popen(
        [
            sys.executable,
            vf_tools.__file__,
            "serve",
            "--bundle",
            str(bundle),
            "--port",
            str(port),
            "--endpoints",
            json.dumps([f"127.0.0.1:{open_port}", str(allowed)]),
            "--commands",
            json.dumps(["echo"]),
        ]
    )
    url = f"http://127.0.0.1:{port}/mcp"
    for _ in range(100):
        try:
            vf_tools.rpc(url, "tools/list", timeout=1)
            break
        except OSError:
            time.sleep(0.1)
    yield url, open_port, allowed, tmp_path
    server.kill()
    listener.close()


def call(url: str, name: str, arguments: dict) -> bool:
    result = vf_tools.rpc(url, "tools/call", {"name": name, "arguments": arguments})
    return not result["isError"]


def test_declared_endpoints_are_reachable(served):
    url, open_port, allowed, _ = served
    assert call(url, "connect", {"port": open_port})
    assert call(url, "write", {"path": str(allowed / "x.txt"), "text": "ok"})
    assert (allowed / "x.txt").read_text() == "ok"
    assert call(url, "run", {"command": "echo"})


def test_undeclared_endpoints_are_refused(served):
    url, _, _, tmp_path = served
    assert not call(url, "connect", {"port": free_port()})
    assert not call(url, "write", {"path": str(tmp_path / "outside.txt"), "text": "no"})
    assert not (tmp_path / "outside.txt").exists()
    assert not call(url, "run", {"command": "ls"})
    assert not call(url, "load", {})


def test_an_environment_bundle_declares_its_observation(tmp_path):
    (tmp_path / "tools.py").write_text("")
    (tmp_path / "bundle.toml").write_text(
        'name = "env"\nversion = 1\nkind = "environment"\n'
    )
    with pytest.raises(ValueError):
        BundleManifest.read(tmp_path)
    (tmp_path / "bundle.toml").write_text(
        'name = "env"\nversion = 1\nendpoints = ["127.0.0.1:5432"]\n'
    )
    with pytest.raises(ValueError):
        BundleManifest.read(tmp_path)
