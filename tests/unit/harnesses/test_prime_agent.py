import hashlib
import os
import subprocess
from types import SimpleNamespace
from typing import Literal

import pytest

from verifiers.v1.harnesses.prime_agent import harness as prime_agent


class PackagePrimeAgentConfig(prime_agent.PrimeAgentHarnessConfig):
    commit: Literal["package-commit"] = "package-commit"
    release_version: str = "0.10.2"


@pytest.mark.asyncio
async def test_setup_uses_package_release_config(monkeypatch):
    captured = {}

    async def ensure_installed(runtime, **kwargs):
        captured.update(kwargs)

    async def prepare_uv_script(script, env=None, *, activate=True):
        return []

    monkeypatch.setattr(prime_agent, "ensure_installed", ensure_installed)
    runtime = SimpleNamespace(prepare_uv_script=prepare_uv_script)
    harness = prime_agent.PrimeAgentHarness(PackagePrimeAgentConfig())

    await harness.setup(runtime)

    assert captured["env"]["PRIME_AGENT_COMMIT"] == "package-commit"
    assert captured["env"]["PRIME_AGENT_RELEASE_VERSION"] == "0.10.2"


def run_install(tmp_path, version, monkeypatch, *, checksum=None, missing_hash=False):
    payload = b"release tarball payload"
    expected = checksum or hashlib.sha256(payload).hexdigest()
    asset = f"prime-agent-{version}-linux-x64.tar.gz"
    hashes = {} if missing_hash else {asset: expected}
    bindir = tmp_path / "mock-bin"
    bindir.mkdir()
    url_log = tmp_path / "urls"
    (bindir / "uname").write_text(
        '#!/bin/sh\ncase "$1" in -s) echo Linux;; -m) echo x86_64;; esac\n'
    )
    (bindir / "curl").write_text(
        "#!/bin/sh\n"
        "output=; url=\n"
        'while [ "$#" -gt 0 ]; do\n'
        '  if [ "$1" = -o ]; then output=$2; shift 2; else url=$1; shift; fi\n'
        "done\n"
        f"printf '%s\\n' \"$url\" >> {url_log}\n"
        f"printf '%s' '{payload.decode()}' > \"$output\"\n"
    )
    (bindir / "tar").write_text(
        "#!/bin/sh\n"
        'while [ "$#" -gt 0 ]; do if [ "$1" = -C ]; then dest=$2; shift 2; else shift; fi; done\n'
        "cat > \"$dest/prime-agent\" <<'SCRIPT'\n"
        "#!/bin/sh\n"
        'if [ "$1" = --version ]; then printf \'%s\\n\' "$PRIME_AGENT_RELEASE_VERSION"; fi\n'
        'SCRIPT\nchmod +x "$dest/prime-agent"\n'
    )
    (bindir / "uv").write_text(
        '#!/bin/sh\ncase "$*" in *--help*) echo --script;; esac\nexit 0\n'
    )
    for executable in bindir.iterdir():
        executable.chmod(0o755)

    captured = {}

    async def ensure_installed(runtime, **kwargs):
        captured.update(kwargs)

    async def prepare_uv_script(script, env=None, *, activate=True):
        return []

    # setup provides the real install script and version-specific environment without network access.
    monkeypatch.setattr(prime_agent, "ensure_installed", ensure_installed)

    async def configure():
        harness = prime_agent.PrimeAgentHarness(
            prime_agent.PrimeAgentHarnessConfig(release_version=version, sha256=hashes)
        )
        await harness.setup(SimpleNamespace(prepare_uv_script=prepare_uv_script))

    import asyncio

    asyncio.run(configure())
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    captured["env"]["AGENT_PRIME_AGENT_DIR"] = str(agent_dir)
    environment = {
        **os.environ,
        **captured["env"],
        "PATH": f"{bindir}:/usr/bin:/bin",
        "HOME": str(tmp_path),
    }
    result = subprocess.run(
        ["bash", "-c", captured["install"]],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, url_log.read_text().splitlines() if url_log.exists() else []


@pytest.mark.parametrize("version", ["0.10.5", "0.10.2"])
def test_install_verifies_checksum_from_selected_release(
    tmp_path, monkeypatch, version
):
    result, urls = run_install(tmp_path, version, monkeypatch)

    assert result.returncode == 0, result.stderr
    assert urls == [
        f"https://github.com/PrimeIntellect-ai/prime-agent/releases/download/v{version}/prime-agent-{version}-linux-x64.tar.gz"
    ]


def test_default_release_hashes_are_configured():
    config = prime_agent.PrimeAgentHarnessConfig()

    assert config.sha256 == prime_agent.PRIME_AGENT_SHA256
    assert "b9f653f4c32e83a2270f0b52edea6a72c7bdbd1f7b5c3e64c55e75157094f3b3" in (
        prime_agent._install_script(config.sha256)
    )


def test_install_fails_closed_when_pinned_asset_hash_is_missing(tmp_path, monkeypatch):
    result, _ = run_install(tmp_path, "0.10.2", monkeypatch, missing_hash=True)

    assert result.returncode != 0
    assert (
        "no SHA256 configured for Prime Agent release asset prime-agent-0.10.2-linux-x64.tar.gz"
        in result.stderr
    )


def test_install_fails_closed_on_checksum_mismatch(tmp_path, monkeypatch):
    result, _ = run_install(tmp_path, "0.10.2", monkeypatch, checksum="0" * 64)

    assert result.returncode != 0
    assert "SHA256 verification failed for Prime Agent release 0.10.2" in result.stderr
