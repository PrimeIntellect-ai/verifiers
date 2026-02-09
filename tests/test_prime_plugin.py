from pathlib import Path

import verifiers.cli.plugins.prime as prime_plugin


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


def test_build_module_command_install_adds_workspace_env_path(
    tmp_path: Path, monkeypatch
):
    workspace, env_dir = _make_workspace(tmp_path)
    plugin = prime_plugin.PrimeCLIPlugin()

    monkeypatch.setattr(prime_plugin, "_current_cwd", lambda: env_dir)
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


def test_build_module_command_eval_rewrites_relative_env_dir_path(
    tmp_path: Path, monkeypatch
):
    workspace, env_dir = _make_workspace(tmp_path)
    plugin = prime_plugin.PrimeCLIPlugin()

    monkeypatch.setattr(prime_plugin, "_current_cwd", lambda: env_dir)
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


def test_build_module_command_gepa_adds_workspace_env_dir_path(
    tmp_path: Path, monkeypatch
):
    workspace, env_dir = _make_workspace(tmp_path)
    plugin = prime_plugin.PrimeCLIPlugin()

    monkeypatch.setattr(prime_plugin, "_current_cwd", lambda: env_dir)
    monkeypatch.setattr(prime_plugin, "_resolve_workspace_python", lambda *_: "python")

    command = plugin.build_module_command(plugin.gepa_module, ["my-env"])

    assert command == [
        "python",
        "-m",
        plugin.gepa_module,
        "my-env",
        "--env-dir-path",
        str((workspace / "environments").resolve()),
    ]


def test_build_module_command_build_adds_workspace_env_path(
    tmp_path: Path, monkeypatch
):
    workspace, env_dir = _make_workspace(tmp_path)
    plugin = prime_plugin.PrimeCLIPlugin()

    monkeypatch.setattr(prime_plugin, "_current_cwd", lambda: env_dir)
    monkeypatch.setattr(prime_plugin, "_resolve_workspace_python", lambda *_: "python")

    command = plugin.build_module_command(plugin.build_module, ["my-env"])

    assert command == [
        "python",
        "-m",
        plugin.build_module,
        "my-env",
        "--path",
        str((workspace / "environments").resolve()),
    ]


def test_build_module_command_init_adds_workspace_env_path(tmp_path: Path, monkeypatch):
    workspace, env_dir = _make_workspace(tmp_path)
    plugin = prime_plugin.PrimeCLIPlugin()

    monkeypatch.setattr(prime_plugin, "_current_cwd", lambda: env_dir)
    monkeypatch.setattr(prime_plugin, "_resolve_workspace_python", lambda *_: "python")

    command = plugin.build_module_command(plugin.init_module, ["my-env"])

    assert command == [
        "python",
        "-m",
        plugin.init_module,
        "my-env",
        "--path",
        str((workspace / "environments").resolve()),
    ]
