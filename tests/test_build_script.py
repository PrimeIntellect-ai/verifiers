from pathlib import Path

from verifiers.scripts import build


def test_resolve_env_push_target_defaults_to_environments_dir(tmp_path: Path):
    base_dir = tmp_path / "workspace" / "environments"
    env_name, env_path = build._resolve_env_push_target("my-env", str(base_dir))

    assert env_name == "my-env"
    assert env_path == (base_dir / "my_env").resolve()


def test_resolve_env_push_target_appends_env_id_to_custom_base_path(tmp_path: Path):
    base_dir = tmp_path / "workspace" / "custom_envs"
    env_name, env_path = build._resolve_env_push_target("env-name", str(base_dir))

    assert env_name == "env-name"
    assert env_path == (base_dir / "env_name").resolve()


def test_resolve_env_push_target_uses_explicit_environment_path_when_env_id_missing(
    tmp_path: Path,
):
    explicit_env_path = tmp_path / "workspace" / "environments" / "already_normalized"
    env_name, env_path = build._resolve_env_push_target(None, str(explicit_env_path))

    assert env_name == "already-normalized"
    assert env_path == explicit_env_path.resolve()
