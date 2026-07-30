"""Tests for the Prime CLI plugin and v1 eval bridge.

Covers:
- v1 TOML dispatch: v1 config detected and routed to v1 eval
- Prime-injected legacy flags stripped safely
- Legacy (non-v1) targets still route to existing v0 eval
- Dry-run route creates no model calls/sandboxes
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import verifiers.cli.plugins.prime as prime_plugin
from verifiers.cli.commands import eval_v1_prime

# ---------------------------------------------------------------------------
# Workspace helpers (shared with existing tests)
# ---------------------------------------------------------------------------


def _make_workspace(tmp_path: Path) -> tuple[Path, Path]:
    workspace = tmp_path / "workspace"
    env_dir = workspace / "environments" / "my_env"
    env_dir.mkdir(parents=True)
    (workspace / "verifiers").mkdir()
    (workspace / "pyproject.toml").write_text(
        '[project]\nname = "workspace"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )
    return workspace, env_dir


def _touch_python(venv_root: Path) -> Path:
    python_bin = prime_plugin._venv_python(venv_root)
    python_bin.parent.mkdir(parents=True, exist_ok=True)
    python_bin.write_text("", encoding="utf-8")
    return python_bin


def _write_v1_toml(path: Path) -> Path:
    path.write_text(
        textwrap.dedent("""\
            model = "test/model"
            [env.taskset]
            id = "test-taskset"
            """),
        encoding="utf-8",
    )
    return path


def _write_non_v1_toml(path: Path) -> Path:
    path.write_text(
        textwrap.dedent("""\
            [env]
            id = "legacy-env"
            """),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Existing plugin-structure tests (updated for eval_v1_prime module)
# ---------------------------------------------------------------------------


def test_find_workspace_root_from_nested_environment_dir(tmp_path: Path):
    workspace, env_dir = _make_workspace(tmp_path)
    assert prime_plugin._find_workspace_root(env_dir) == workspace


def test_resolve_workspace_python_prefers_workspace_venv_over_uv_env(
    tmp_path: Path, monkeypatch
):
    workspace, env_dir = _make_workspace(tmp_path)
    workspace_python = _touch_python(workspace / ".venv")
    _touch_python(env_dir / ".venv")

    monkeypatch.setattr(prime_plugin, "_python_can_import_module", lambda *_: True)
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", str(env_dir / ".venv"))
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)

    assert prime_plugin._resolve_workspace_python(env_dir) == str(workspace_python)


def test_eval_module_points_to_v1_bridge():
    plugin = prime_plugin.PrimeCLIPlugin()
    assert plugin.eval_module == "verifiers.cli.commands.eval_v1_prime"


def test_build_module_command_eval_rewrites_relative_env_dir_path(
    tmp_path: Path, monkeypatch
):
    workspace, env_dir = _make_workspace(tmp_path)
    plugin = prime_plugin.PrimeCLIPlugin()

    monkeypatch.chdir(env_dir)
    monkeypatch.setattr(prime_plugin, "_resolve_workspace_python", lambda *_: "python")

    command = plugin.build_module_command(
        plugin.eval_module,
        ["my-env", "--env-dir-path", "./environments"],
    )

    assert command == [
        "python",
        "-m",
        plugin.eval_module,
        "my-env",
        "--env-dir-path",
        str((workspace / "environments").resolve()),
    ]


def test_build_module_command_install_adds_workspace_env_path(
    tmp_path: Path, monkeypatch
):
    workspace, env_dir = _make_workspace(tmp_path)
    plugin = prime_plugin.PrimeCLIPlugin()

    monkeypatch.chdir(env_dir)
    monkeypatch.setattr(prime_plugin, "_resolve_workspace_python", lambda *_: "python")

    command = plugin.build_module_command(plugin.install_module, ["my-env"])

    assert command == [
        "python",
        "-m",
        plugin.install_module,
        "my-env",
        "--path",
        str((workspace / "environments").resolve()),
    ]


# ---------------------------------------------------------------------------
# v1 TOML dispatch
# ---------------------------------------------------------------------------


class TestV1Dispatch:
    """Prove v1 TOML configs route to the v1 eval entrypoint."""

    def test_is_v1_config_detects_taskset_toml(self, tmp_path: Path):
        toml = _write_v1_toml(tmp_path / "config.toml")
        assert eval_v1_prime._is_v1_config(str(toml)) is True

    def test_is_v1_config_rejects_non_v1_toml(self, tmp_path: Path):
        toml = _write_non_v1_toml(tmp_path / "legacy.toml")
        assert eval_v1_prime._is_v1_config(str(toml)) is False

    def test_is_v1_config_rejects_missing_file(self, tmp_path: Path):
        assert eval_v1_prime._is_v1_config(str(tmp_path / "nope.toml")) is False

    def test_is_v1_config_rejects_non_toml(self, tmp_path: Path):
        (tmp_path / "config.yaml").write_text("env: {}", encoding="utf-8")
        assert eval_v1_prime._is_v1_config(str(tmp_path / "config.yaml")) is False

    def test_main_routes_v1_toml_to_v1_eval(self, tmp_path: Path):
        """When argv[0] is a v1 TOML, main() calls v1 cli eval.main."""
        toml = _write_v1_toml(tmp_path / "v1.toml")
        captured: list[list[str]] = []

        def fake_v1_main(argv):
            captured.append(argv)

        with patch("verifiers.v1.cli.eval.main.main", side_effect=fake_v1_main):
            eval_v1_prime.main([str(toml), "--model", "test/model"])

        assert len(captured) == 1
        args = captured[0]
        assert args[0] == "@"
        assert args[1] == str(toml.resolve())
        assert "--model" in args

    def test_translated_args_preserves_model_and_output_dir(self, tmp_path: Path):
        toml = _write_v1_toml(tmp_path / "v1.toml")
        result = eval_v1_prime._translated_v1_args(
            [
                str(toml),
                "--model",
                "test/model",
                "--output-dir",
                "/tmp/out",
            ]
        )
        assert result[0] == "@"
        assert result[1] == str(toml.resolve())
        assert "--model" in result
        assert "test/model" in result
        assert "--output-dir" in result
        assert "/tmp/out" in result

    def test_translated_args_short_flags_normalized(self, tmp_path: Path):
        toml = _write_v1_toml(tmp_path / "v1.toml")
        result = eval_v1_prime._translated_v1_args(
            [
                str(toml),
                "-m",
                "test/model",
                "-o",
                "/tmp/out",
            ]
        )
        assert "--model" in result
        assert "test/model" in result
        assert "--output-dir" in result
        assert "/tmp/out" in result
        # short flags should not appear
        assert "-m" not in result
        assert "-o" not in result


# ---------------------------------------------------------------------------
# Prime-injected legacy flags stripped safely
# ---------------------------------------------------------------------------


class TestLegacyFlagStripping:
    """Prove Prime-injected v0-only flags are dropped, not forwarded to v1."""

    def test_provider_stripped(self, tmp_path: Path):
        toml = _write_v1_toml(tmp_path / "v1.toml")
        result = eval_v1_prime._translated_v1_args(
            [
                str(toml),
                "--provider",
                "local",
                "--model",
                "test/model",
            ]
        )
        assert "local" not in result
        assert "--provider" not in result
        assert "--model" in result

    def test_short_provider_stripped(self, tmp_path: Path):
        toml = _write_v1_toml(tmp_path / "v1.toml")
        result = eval_v1_prime._translated_v1_args(
            [
                str(toml),
                "-p",
                "local",
                "--model",
                "test/model",
            ]
        )
        assert "local" not in result
        assert "-p" not in result

    def test_api_base_url_stripped(self, tmp_path: Path):
        toml = _write_v1_toml(tmp_path / "v1.toml")
        result = eval_v1_prime._translated_v1_args(
            [
                str(toml),
                "--api-base-url",
                "http://localhost:8000/v1",
                "--model",
                "test/model",
            ]
        )
        assert "http://localhost:8000/v1" not in result
        assert "--api-base-url" not in result

    def test_api_key_var_stripped(self, tmp_path: Path):
        toml = _write_v1_toml(tmp_path / "v1.toml")
        result = eval_v1_prime._translated_v1_args(
            [
                str(toml),
                "--api-key-var",
                "MY_KEY",
                "--model",
                "test/model",
            ]
        )
        assert "MY_KEY" not in result
        assert "--api-key-var" not in result

    def test_endpoints_path_stripped(self, tmp_path: Path):
        toml = _write_v1_toml(tmp_path / "v1.toml")
        result = eval_v1_prime._translated_v1_args(
            [
                str(toml),
                "--endpoints-path",
                "./configs/endpoints.toml",
                "--model",
                "test/model",
            ]
        )
        assert "--endpoints-path" not in result

    def test_env_dir_path_stripped(self, tmp_path: Path):
        toml = _write_v1_toml(tmp_path / "v1.toml")
        result = eval_v1_prime._translated_v1_args(
            [
                str(toml),
                "--env-dir-path",
                "./environments",
                "--model",
                "test/model",
            ]
        )
        assert "--env-dir-path" not in result

    def test_skip_upload_translates_to_no_push(self, tmp_path: Path):
        toml = _write_v1_toml(tmp_path / "v1.toml")
        result = eval_v1_prime._translated_v1_args(
            [
                str(toml),
                "--skip-upload",
                "--model",
                "test/model",
            ]
        )
        assert "--skip-upload" not in result
        assert "--no-push" in result

    def test_no_push_is_preserved(self, tmp_path: Path):
        toml = _write_v1_toml(tmp_path / "v1.toml")
        result = eval_v1_prime._translated_v1_args([str(toml), "--no-push"])
        assert "--no-push" in result

    def test_save_results_stripped(self, tmp_path: Path):
        toml = _write_v1_toml(tmp_path / "v1.toml")
        result = eval_v1_prime._translated_v1_args(
            [
                str(toml),
                "--save-results",
                "-s",
                "--model",
                "test/model",
            ]
        )
        assert "--save-results" not in result
        assert "-s" not in result

    def test_full_prime_injection_stripped(self, tmp_path: Path):
        """Simulate the full set of Prime-injected v0 flags."""
        toml = _write_v1_toml(tmp_path / "v1.toml")
        result = eval_v1_prime._translated_v1_args(
            [
                str(toml),
                "--skip-upload",
                "--provider",
                "local",
                "--model",
                "internal/glm-5.2-fast",
                "--output-dir",
                "/tmp/out",
                "--env-dir-path",
                "./environments",
                "--api-base-url",
                "http://localhost:8000/v1",
                "--api-key-var",
                "VLLM_API_KEY",
                "--endpoints-path",
                "./endpoints.toml",
                "--header",
                "x-custom=value",
                "--header-from-state",
                "state",
                "--save-results",
                "-s",
                "--dry-run",
                "--verbose",
            ]
        )
        # Only safe flags remain
        assert "@" in result
        assert str(toml.resolve()) in result
        assert "--model" in result
        assert "internal/glm-5.2-fast" in result
        assert "--output-dir" in result
        assert "/tmp/out" in result
        assert "--dry-run" in result
        assert "--verbose" in result
        assert "--no-push" in result
        # All v0-only content stripped
        for stripped in [
            "--provider",
            "-p",
            "local",
            "--api-base-url",
            "--api-key-var",
            "--endpoints-path",
            "--env-dir-path",
            "--header",
            "--header-from-state",
            "--skip-upload",
            "--save-results",
            "-s",
        ]:
            assert stripped not in result, f"{stripped} should have been stripped"


# ---------------------------------------------------------------------------
# Legacy (non-v1) targets still route to existing v0 eval
# ---------------------------------------------------------------------------


class TestLegacyRouting:
    """Prove non-v1 targets fall through to the existing v0 eval script."""

    def test_main_routes_non_v1_to_legacy_eval(self, tmp_path: Path):
        """A non-TOML positional arg goes to legacy eval."""
        captured: list[list[str]] = []

        def fake_legacy_main():
            captured.append(list(sys.argv))

        with patch("verifiers.scripts.eval.main", side_effect=fake_legacy_main):
            eval_v1_prime.main(["my-env"])

        assert len(captured) == 1
        assert captured[0][1:] == ["my-env"]

    def test_main_routes_non_v1_toml_to_legacy_eval(self, tmp_path: Path):
        """A TOML without [env.taskset] goes to legacy eval."""
        toml = _write_non_v1_toml(tmp_path / "legacy.toml")
        called = False

        def fake_legacy_main():
            nonlocal called
            called = True

        with patch("verifiers.scripts.eval.main", side_effect=fake_legacy_main):
            eval_v1_prime.main([str(toml)])

        assert called is True

    def test_main_no_args_routes_to_legacy(self):
        called = False

        def fake_legacy_main():
            nonlocal called
            called = True

        with patch("verifiers.scripts.eval.main", side_effect=fake_legacy_main):
            eval_v1_prime.main([])

        assert called is True

    def test_main_restores_sys_argv_after_legacy(self, tmp_path: Path):
        """sys.argv must be restored even if legacy eval raises."""
        original = list(sys.argv)
        with patch("verifiers.scripts.eval.main", side_effect=RuntimeError("boom")):
            try:
                eval_v1_prime.main(["my-env"])
            except RuntimeError:
                pass
        assert sys.argv == original


# ---------------------------------------------------------------------------
# Dry-run route creates no model calls/sandboxes
# ---------------------------------------------------------------------------


class TestDryRunNoSideEffects:
    """Prove --dry-run routes through v1 eval's dry-run path without
    making model calls or creating sandboxes."""

    def test_dry_run_translated_to_v1(self, tmp_path: Path):
        toml = _write_v1_toml(tmp_path / "v1.toml")
        result = eval_v1_prime._translated_v1_args(
            [
                str(toml),
                "--dry-run",
                "--model",
                "test/model",
            ]
        )
        assert "--dry-run" in result
        assert "@" == result[0]

    def test_dry_run_calls_v1_main_not_legacy(self, tmp_path: Path):
        toml = _write_v1_toml(tmp_path / "v1.toml")
        v1_called = False
        legacy_called = False

        def fake_v1_main(argv):
            nonlocal v1_called
            v1_called = True
            assert "--dry-run" in argv

        def fake_legacy_main():
            nonlocal legacy_called
            legacy_called = True

        with (
            patch("verifiers.v1.cli.eval.main.main", side_effect=fake_v1_main),
            patch("verifiers.scripts.eval.main", side_effect=fake_legacy_main),
        ):
            eval_v1_prime.main([str(toml), "--dry-run", "--model", "test/model"])

        assert v1_called is True
        assert legacy_called is False
