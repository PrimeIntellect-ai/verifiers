"""The rlm harness maps its session's stop reason onto the trace's stop condition,
and installs nano-rlm on images whose apt cannot install git."""

import asyncio
import shutil
import subprocess
import tarfile
from types import SimpleNamespace

import pytest

import verifiers.v1 as vf
from verifiers.v1.harnesses.rlm.harness import (
    RLM_SESSION_METADATA_KEY,
    RLMHarness,
    RLMHarnessConfig,
)


@pytest.mark.parametrize(
    "reason,condition",
    [
        ("max_total_tokens", "max_total_tokens"),
        ("max_total_turns", "max_turns"),
        ("token_budget", "max_output_tokens"),
        ("compaction_failed", "compaction_failed"),
        ("done", "agent_completed"),
        (None, "agent_completed"),
    ],
)
def test_rlm_stop_reason_survives_completion_and_serialization(reason, condition):
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="q")),
    )
    harness = RLMHarness(RLMHarnessConfig(id="rlm"))
    metadata = {
        RLM_SESSION_METADATA_KEY: {
            "session_id": trace.id,
            "metrics": {"turns": 1},
            "last_stop_reason": reason,
        }
    }
    harness.acp_turn_result(
        trace, vf.ACPTurn(reply="partial", response_metadata=metadata)
    )
    trace.stop("agent_completed")
    harness.acp_close_result(trace, metadata)

    restored = vf.WireTrace.model_validate_json(trace.model_dump_json())
    assert restored.stop_condition == condition
    assert restored.is_truncated is (condition != "agent_completed")


STUB_TOOLS = ("sh", "bash", "mkdir", "rm", "tar", "gzip", "touch", "cat", "chmod")


async def _install_script(monkeypatch, tmp_path) -> str:
    from verifiers.v1.harnesses.rlm import harness

    scripts = []

    async def installed(runtime, **kwargs):
        scripts.append(kwargs["install"])

    async def acp_setup(self, runtime):
        pass

    monkeypatch.setattr(harness, "RLM_CACHE_DIR", str(tmp_path / "rlm"))
    monkeypatch.setattr(harness, "ensure_installed", installed)
    monkeypatch.setattr(harness.ACPHarness, "setup", acp_setup)
    await RLMHarness(RLMHarnessConfig(version="abc123")).setup(
        SimpleNamespace(user=None)
    )  # type: ignore[arg-type]
    return scripts[0]


def _box(tmp_path, tools: dict[str, str]) -> dict[str, str]:
    """A PATH holding only basic tools plus `tools` (name -> script body)."""
    bin_dir = tmp_path / "box-bin"
    bin_dir.mkdir()
    for name in STUB_TOOLS:
        (bin_dir / name).symlink_to(shutil.which(name))
    for name, body in tools.items():
        (bin_dir / name).write_text(f"#!/bin/sh\n{body}\n")
        (bin_dir / name).chmod(0o755)
    return {"PATH": str(bin_dir)}


def test_an_image_without_git_or_working_apt_fetches_the_rlm_tarball(
    monkeypatch, tmp_path
):
    """Images that lock apt down (emptied sources, pinned snapshots) cannot apt-get
    git; with curl the source tarball goes in instead and apt is never touched."""
    source = tmp_path / "nano-rlm-abc123"
    source.mkdir()
    (source / "install.sh").write_text(
        'mkdir -p "$UV_INSTALL_DIR" && printf "#!/bin/sh\\n" > "$UV_INSTALL_DIR/rlm" '
        '&& chmod +x "$UV_INSTALL_DIR/rlm"\n'
    )
    tarball = tmp_path / "rlm.tar.gz"
    with tarfile.open(tarball, "w:gz") as tar:
        tar.add(source, arcname=source.name)
    fetched = tmp_path / "fetched"
    env = _box(
        tmp_path,
        {
            "curl": f'echo "$@" > {fetched}; cat {tarball}',
            "apt-get": "echo 'E: Unable to locate package git' >&2; exit 100",
        },
    )
    script = asyncio.run(_install_script(monkeypatch, tmp_path))

    done = subprocess.run(
        ["sh", "-c", script], env=env, capture_output=True, text=True, check=False
    )

    assert done.returncode == 0, done.stderr
    assert fetched.read_text().split()[-1] == (
        "https://codeload.github.com/PrimeIntellect-ai/nano-rlm/tar.gz/abc123"
    )
    [directory] = (tmp_path / "rlm").parent.glob("rlm-*")
    assert (directory / ".ready").exists() and (directory / "bin/rlm").exists()


def test_a_failed_tarball_fetch_fails_the_install(monkeypatch, tmp_path):
    env = _box(tmp_path, {"curl": "exit 22", "apt-get": "exit 100"})
    script = asyncio.run(_install_script(monkeypatch, tmp_path))

    done = subprocess.run(
        ["sh", "-c", script], env=env, capture_output=True, text=True, check=False
    )

    assert done.returncode != 0
    assert not list((tmp_path / "rlm").parent.glob("rlm-*/.ready"))


def test_an_image_with_only_python_fetches_through_the_urllib_curl(
    monkeypatch, tmp_path
):
    """No git, no curl, apt locked down (python-slim task images): python3's urllib
    stands in for curl during the install, and is gone afterwards."""
    source = tmp_path / "nano-rlm-abc123"
    source.mkdir()
    (source / "install.sh").write_text(
        # install.sh downloads through whatever `curl` is on its PATH
        'curl -LsSf "$CURL_PROBE" -o "$UV_INSTALL_DIR.probe" && mkdir -p "$UV_INSTALL_DIR" '
        '&& printf "#!/bin/sh\\n" > "$UV_INSTALL_DIR/rlm" && chmod +x "$UV_INSTALL_DIR/rlm"\n'
    )
    tarball = tmp_path / "rlm.tar.gz"
    with tarfile.open(tarball, "w:gz") as tar:
        tar.add(source, arcname=source.name)
    (tmp_path / "probe").write_text("probed")
    fetched = tmp_path / "fetched"
    real_python = shutil.which("python3")
    # The stand-in runs `python3 curl.py ARGS`: the codeload URL comes from the
    # tarball, anything else through the real urllib script.
    env = _box(
        tmp_path,
        {
            "python3": (
                f'case "$*" in *codeload*) echo "$@" > {fetched}; cat {tarball};; '
                f'*) exec {real_python} "$@";; esac'
            ),
            "apt-get": "exit 100",
        },
    )
    env["CURL_PROBE"] = (tmp_path / "probe").as_uri()
    script = asyncio.run(_install_script(monkeypatch, tmp_path))

    done = subprocess.run(
        ["sh", "-c", script], env=env, capture_output=True, text=True, check=False
    )

    assert done.returncode == 0, done.stderr
    assert fetched.read_text().split()[-1].endswith("nano-rlm/tar.gz/abc123")
    [directory] = (tmp_path / "rlm").parent.glob("rlm-*")
    assert (directory / ".ready").exists() and (directory / "bin/rlm").exists()
    assert (directory / "bin.probe").read_text() == "probed"
    assert not (directory / "curl-shim").exists()


def test_the_urllib_curl_writes_to_a_file_or_stdout(tmp_path):
    from verifiers.v1.harnesses.rlm.harness import URLLIB_CURL

    page = tmp_path / "page"
    page.write_text("body")
    run = [shutil.which("python3"), "-c", URLLIB_CURL, "-sSfL", "--retry", "3"]

    out = tmp_path / "out"
    subprocess.run([*run, page.as_uri(), "-o", str(out)], check=True)
    printed = subprocess.run(
        [*run, page.as_uri()], check=True, capture_output=True, text=True
    )
    missing = subprocess.run(
        [*run, (tmp_path / "nope").as_uri()], capture_output=True, check=False
    )

    assert out.read_text() == "body" and printed.stdout == "body"
    assert missing.returncode != 0
