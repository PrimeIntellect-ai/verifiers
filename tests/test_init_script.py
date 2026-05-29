from pathlib import Path

import pytest
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


def test_init_v1_writes_thin_taskset_template(tmp_path: Path) -> None:
    init_environment("bar", path=str(tmp_path), v1=True)
    content = read_env_file(tmp_path, "bar")

    assert "class BarTasksetConfig(vf.TasksetConfig):" in content
    assert "class BarTaskset(vf.Taskset[BarTasksetConfig]):" in content
    assert 'def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:' in content
    assert (
        "def load_system_prompt(self, config: BarTasksetConfig) -> vf.SystemPrompt:"
        in content
    )
    assert "async def correct_answer(self, task: vf.Task, state: vf.State)" in content
    assert "def load_taskset(config: BarTasksetConfig) -> BarTaskset:" in content
    assert "return BarTaskset(config=config)" in content
    assert "assert isinstance(taskset_config, BarTasksetConfig)" in content
    assert "return vf.Env(taskset=load_taskset(taskset_config))" in content
    assert "class EnvTaskset(" not in content
    assert "_default_" not in content
    assert 'tasks: str = "load_tasks"' not in content
    assert 'rewards: list[str] = ["correct_answer"]' not in content


def test_init_v1_template_loads_with_vf_load_environment(
    tmp_path: Path, monkeypatch
) -> None:
    init_environment("loadable-v1", path=str(tmp_path), v1=True)
    monkeypatch.syspath_prepend(str(tmp_path / "loadable_v1"))

    with pytest.raises(RuntimeError, match="Load the system prompt"):
        vf.load_environment("loadable-v1")


def test_init_v1_with_harness_writes_harness_stub(tmp_path: Path) -> None:
    init_environment("baz", path=str(tmp_path), v1=True, with_harness=True)
    content = read_env_file(tmp_path, "baz")

    assert "class BazTaskset(vf.Taskset[BazTasksetConfig]):" in content
    assert "class BazHarnessConfig(vf.HarnessConfig):" in content
    assert "class BazHarness(vf.Harness[BazHarnessConfig]):" in content
    assert "def load_harness(config: BazHarnessConfig) -> BazHarness:" in content
    assert "assert isinstance(taskset_config, BazTasksetConfig)" in content
    assert "assert isinstance(harness_config, BazHarnessConfig)" in content
    assert "harness=load_harness(harness_config)" in content


def test_init_with_harness_without_v1_warns_and_uses_v0(tmp_path: Path, capsys) -> None:
    init_environment("plain", path=str(tmp_path), with_harness=True)
    content = read_env_file(tmp_path, "plain")
    captured = capsys.readouterr()

    assert "--with-harness only applies with --v1; ignoring." in captured.out
    assert "def load_environment(**kwargs) -> vf.Environment:" in content
    assert "load_harness" not in content


def test_init_v1_multifile_exports_component_loaders(tmp_path: Path) -> None:
    init_environment("pkg-env", path=str(tmp_path), v1=True, multi_file=True)
    package_dir = tmp_path / "pkg_env" / "pkg_env"
    init_content = (package_dir / "__init__.py").read_text()
    env_content = (package_dir / "pkg_env.py").read_text()

    assert "from .pkg_env import load_environment, load_taskset" in init_content
    assert "__all__ = ['load_environment', 'load_taskset']" in init_content
    assert "class PkgEnvTaskset(vf.Taskset[PkgEnvTasksetConfig]):" in env_content
    assert "return PkgEnvTaskset(config=config)" in env_content


def test_init_openenv_writes_v1_taskset_template(tmp_path: Path) -> None:
    init_environment("openenv-sample", path=str(tmp_path), openenv=True)
    content = read_env_file(tmp_path, "openenv-sample")
    pyproject = (tmp_path / "openenv_sample" / "pyproject.toml").read_text()

    assert (
        "from tasksets.openenv import OpenEnvTaskset, OpenEnvTasksetConfig" in content
    )
    assert (
        "def load_taskset(config: OpenEnvTasksetConfig) -> OpenEnvTaskset:" in content
    )
    assert "assert isinstance(taskset_config, OpenEnvTasksetConfig)" in content
    assert "return vf.Env(taskset=load_taskset(taskset_config))" in content
    assert "vf.OpenEnvEnv" not in content
    assert '"tasksets>=0.1.0.post0"' in pyproject


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
