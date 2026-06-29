import subprocess
import zipfile
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


def test_init_v0_build_contains_importable_module(tmp_path: Path) -> None:
    root = init_environment("audit-env", path=str(tmp_path))
    (root / "proj").mkdir()
    (root / "proj" / ".build.json").write_text('{"image":"example"}\n')
    dist = tmp_path / "dist"

    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(dist)],
        cwd=root,
        check=True,
        capture_output=True,
    )

    with zipfile.ZipFile(next(dist.glob("*.whl"))) as wheel:
        assert "audit_env.py" in wheel.namelist()
        assert "proj/.build.json" in wheel.namelist()
