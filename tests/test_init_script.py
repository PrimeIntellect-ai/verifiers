from pathlib import Path
import verifiers as vf
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














def test_init_openenv_multifile_exports_taskset_loader(tmp_path: Path) -> None:
    init_environment(
        "openenv-pkg",
        path=str(tmp_path),
        openenv=True,
        multi_file=True,
    )
    init_content = (
        tmp_path / "openenv_pkg" / "openenv_pkg" / "__init__.py"
    ).read_text()

    assert "from .openenv_pkg import load_environment, load_taskset" in init_content
    assert "__all__ = ['load_environment', 'load_taskset']" in init_content
