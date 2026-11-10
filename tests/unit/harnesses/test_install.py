"""Installing a harness's program into a runtime: `ensure_installed` reports whether the
install ran, and says so in the log, loudly when a completed install had to run again."""

import logging
import shlex
from pathlib import Path

import pytest

from verifiers.v1.configs.runtime import SubprocessConfig
from verifiers.v1.harnesses.rlm import harness as rlm
from verifiers.v1.harnesses.utils.install import HarnessNotInstalled, ensure_installed
from verifiers.v1.runtimes import Runtime, provision_runtime


async def test_ensure_installed_reports_and_logs_whether_it_installed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    directory = str(tmp_path / "tool")
    ready = f"{directory}/.ready"

    async def install(runtime: Runtime) -> bool:
        return await ensure_installed(
            runtime,
            directory=directory,
            ready=f"test -f {shlex.quote(ready)}",
            install=f"echo installing && touch {shlex.quote(ready)}",
            env={},
            label="tool",
        )

    caplog.set_level(logging.INFO)
    async with provision_runtime(SubprocessConfig()) as runtime:
        assert await install(runtime) is True
        assert "Installed tool in" in caplog.text
        assert "Reinstalled" not in caplog.text

        caplog.clear()
        assert await install(runtime) is False
        assert caplog.text == ""

        # Something removed the install this runtime had completed: it runs again,
        # and that is a warning.
        Path(ready).unlink()
        assert await install(runtime) is True
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1 and "Reinstalled tool" in warnings[0].getMessage()


async def test_ensure_installed_without_a_check_always_installs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    async with provision_runtime(SubprocessConfig()) as runtime:
        for _ in range(2):
            assert await ensure_installed(
                runtime, directory=str(tmp_path), install="true", env={}, label="t"
            )
    assert caplog.text == ""


async def test_ensure_installed_raises_when_the_install_fails(tmp_path: Path) -> None:
    async with provision_runtime(SubprocessConfig()) as runtime:
        with pytest.raises(RuntimeError, match="tool install failed: boom"):
            await ensure_installed(
                runtime,
                directory=str(tmp_path),
                ready="false",
                install="echo boom >&2; exit 1",
                env={},
                label="tool",
            )


async def test_rlm_installs_under_var_tmp_and_reuses_an_install_from_tmp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = rlm.RLMHarness(rlm.RLMHarnessConfig())
    key = harness._cache_key()
    assert harness._install_dir() == f"/var/tmp/vf-rlm-{key}"

    # An image that installed this version under the old /tmp path.
    monkeypatch.setattr(rlm, "RLM_CACHE_DIR", str(tmp_path / "var" / "vf-rlm"))
    monkeypatch.setattr(rlm, "RLM_LEGACY_CACHE_DIR", str(tmp_path / "tmp" / "vf-rlm"))
    legacy = tmp_path / "tmp" / f"vf-rlm-{key}"
    (legacy / "bin").mkdir(parents=True)
    (legacy / "bin" / "rlm").write_text("#!/bin/sh\necho legacy rlm\n")
    (legacy / "bin" / "rlm").chmod(0o755)
    (legacy / ".ready").touch()

    async def no_acp_setup(self, runtime) -> None:
        pass

    monkeypatch.setattr(rlm.ACPHarness, "setup", no_acp_setup)
    async with provision_runtime(SubprocessConfig()) as runtime:
        await harness.setup(runtime)
        binary = f"{harness._install_dir()}/bin/rlm"
        result = await runtime.run([binary], {})
        assert result.stdout.strip() == "legacy rlm", result
        assert harness._install_dir().startswith(str(tmp_path / "var"))


async def test_never_checks_instead_of_installing(tmp_path: Path) -> None:
    directory = str(tmp_path / "tool")
    marker = tmp_path / "installed"

    async def never(runtime: Runtime, ready: str | None, check: str | None = None):
        return await ensure_installed(
            runtime,
            directory=directory,
            ready=ready,
            check=check,
            install=f"touch {shlex.quote(str(marker))}",
            env={},
            label="tool",
            never=True,
        )

    async with provision_runtime(SubprocessConfig()) as runtime:
        assert await never(runtime, ready="true") is False
        with pytest.raises(
            HarnessNotInstalled, match=f"tool is not installed in {directory}"
        ):
            await never(runtime, ready="false")
        # An install that checks for itself names its check for `never`.
        assert await never(runtime, ready=None, check="true") is False
        with pytest.raises(ValueError, match="no install check"):
            await never(runtime, ready=None)
    assert not marker.exists()


def test_rlm_with_skills_cannot_use_never(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="install='never'"):
        rlm.RLMHarnessConfig(skills=[tmp_path], install="never")
