from pathlib import Path

from verifiers.scripts.init import init_environment


def read_env_file(root: Path, env_id: str) -> str:
    module_name = env_id.replace("-", "_")
    return (root / module_name / f"{module_name}.py").read_text()


def test_init_default_writes_v0_stub(tmp_path: Path) -> None:
    root = init_environment("foo", path=str(tmp_path))
    content = read_env_file(tmp_path, "foo")

    assert root == tmp_path / "foo"
    assert "def load_environment(**kwargs) -> vf.Environment:" in content
    assert "NotImplementedError" in content
    assert "load_taskset" not in content
    assert "EnvTaskset" not in content
