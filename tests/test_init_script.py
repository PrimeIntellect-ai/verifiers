from pathlib import Path
import verifiers as vf
import verifiers.v1 as vf1
from verifiers.scripts.init import init_environment


def read_env_file(root: Path, env_id: str) -> str:
    module_name = env_id.replace("-", "_")
    taskset_file = root / module_name / module_name / "taskset.py"
    if taskset_file.exists():
        return taskset_file.read_text()
    package_file = root / module_name / module_name / f"{module_name}.py"
    if package_file.exists():
        return package_file.read_text()
    return (root / module_name / f"{module_name}.py").read_text()


def package_dir(root: Path, env_id: str) -> Path:
    module_name = env_id.replace("-", "_")
    return root / module_name / module_name


def test_init_default_writes_v0_stub(tmp_path: Path) -> None:
    root = init_environment("foo", path=str(tmp_path))
    content = read_env_file(tmp_path, "foo")

    assert root == tmp_path / "foo"
    assert "def load_environment(**kwargs) -> vf.Environment:" in content
    assert "NotImplementedError" in content
    assert "load_taskset" not in content
    assert "EnvTaskset" not in content


def test_init_v1_writes_taskset_template(tmp_path: Path) -> None:
    init_environment("bar", path=str(tmp_path), v1=True)
    content = read_env_file(tmp_path, "bar")
    package = package_dir(tmp_path, "bar")
    init_content = (package / "__init__.py").read_text()
    pyproject = (tmp_path / "bar" / "pyproject.toml").read_text()
    user_server = (package / "servers" / "user.py").read_text()
    tools_server = (package / "servers" / "toolset.py").read_text()

    assert "class BarTasksetConfig(vf.TasksetConfig):" in content
    assert "class BarTask(vf.Task):" in content
    assert "answer: str" in content
    assert "class BarTaskset(vf.Taskset[BarTasksetConfig]):" in content
    assert "task_type = BarTask" in content
    assert 'system_prompt: vf.SystemPrompt = "Answer exactly."' in content
    assert '"""Taskset implementation for bar.' in content
    assert 'def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:' in content
    assert (
        '"""Return serializable task records as a list, generator, or Dataset."""'
        in content
    )
    assert "def load_system_prompt" not in content
    assert "async def correct_answer(self, task: BarTask, state: vf.State)" in content
    assert "task.answer" in content
    assert "task.data" not in content
    assert "def load_taskset(config: BarTasksetConfig) -> BarTaskset:" in content
    assert '"""Typed taskset loader used by vf.load_taskset."""' in content
    assert "return BarTaskset(config=config)" in content
    assert 'loader="bar.servers.user:ExampleUser"' in content
    assert 'loader="bar.servers.toolset:ExampleToolset"' in content
    assert "def load_user(" not in content
    assert "def load_toolsets(" not in content
    assert "def load_environment" not in content
    assert '"""bar environment package."""' in init_content
    assert "load_environment" not in init_content
    assert 'include = ["bar/**/*", "pyproject.toml", "README.md"]' in pyproject
    assert "class ExampleUser(vf.User):" in user_server
    assert "@vf.tool(" in user_server
    assert (
        "def respond(self, task: dict, state: dict, transcript: list[dict]) -> dict:"
        in (user_server)
    )
    assert "class ExampleToolset(vf.Toolset):" in tools_server
    assert "@vf.tool" in tools_server
    assert "def reverse_text(self, text: str) -> str:" in tools_server
    assert "class EnvTaskset(" not in content
    assert "_default_" not in content
    assert 'tasks: str = "load_tasks"' not in content
    assert 'rewards: list[str] = ["correct_answer"]' not in content


def test_init_v1_template_loads_with_vf_load_environment(
    tmp_path: Path, monkeypatch
) -> None:
    init_environment("loadable-v1", path=str(tmp_path), v1=True)
    monkeypatch.syspath_prepend(str(tmp_path / "loadable_v1"))

    env = vf.load_environment("loadable-v1")
    taskset = vf1.load_taskset("loadable-v1")

    dataset = env.get_dataset()

    assert len(dataset) == 1
    assert dataset[0]["answer"] == "cba"
    assert taskset.config.system_prompt == "Answer exactly."


def test_init_v1_with_harness_writes_harness_stub(tmp_path: Path) -> None:
    init_environment("baz", path=str(tmp_path), v1=True, with_harness=True)
    taskset_content = read_env_file(tmp_path, "baz")
    harness_content = (package_dir(tmp_path, "baz") / "harness.py").read_text()

    assert "class BazTaskset(vf.Taskset[BazTasksetConfig]):" in taskset_content
    assert "class BazHarnessConfig(vf.HarnessConfig):" in harness_content
    assert "class BazHarness(vf.Harness[BazHarnessConfig]):" in harness_content
    assert "def load_harness(config: BazHarnessConfig) -> BazHarness:" in (
        harness_content
    )
    assert "def load_environment" not in taskset_content
    assert "def load_environment" not in harness_content


def test_init_with_harness_without_v1_warns_and_uses_v0(tmp_path: Path, capsys) -> None:
    init_environment("plain", path=str(tmp_path), with_harness=True)
    content = read_env_file(tmp_path, "plain")
    captured = capsys.readouterr()

    assert "--with-harness only applies with --v1; ignoring." in captured.out
    assert "def load_environment(**kwargs) -> vf.Environment:" in content
    assert "load_harness" not in content


def test_init_v1_multifile_exports_component_loaders(tmp_path: Path) -> None:
    init_environment("pkg-env", path=str(tmp_path), v1=True, multi_file=True)
    package = package_dir(tmp_path, "pkg-env")
    init_content = (package / "__init__.py").read_text()
    taskset_content = (package / "taskset.py").read_text()

    assert '"""pkg-env environment package."""' in init_content
    assert "load_environment" not in init_content
    assert "class PkgEnvTaskset(vf.Taskset[PkgEnvTasksetConfig]):" in taskset_content
    assert "return PkgEnvTaskset(config=config)" in taskset_content
    assert (package / "servers" / "user.py").exists()
    assert (package / "servers" / "toolset.py").exists()


def test_init_openenv_writes_v1_taskset_template(tmp_path: Path) -> None:
    init_environment("openenv-sample", path=str(tmp_path), openenv=True)
    content = read_env_file(tmp_path, "openenv-sample")
    package = package_dir(tmp_path, "openenv-sample")
    pyproject = (tmp_path / "openenv_sample" / "pyproject.toml").read_text()

    assert "from tasksets import OpenEnvTaskset, OpenEnvTasksetConfig" in content
    assert (
        "def load_taskset(config: OpenEnvTasksetConfig) -> OpenEnvTaskset:" in content
    )
    assert "def load_environment" not in content
    assert "vf.OpenEnvEnv" not in content
    assert '"tasksets[openenv]>=0.1.5"' in pyproject
    assert 'include = ["openenv_sample/**/*", "pyproject.toml", "README.md"]' in (
        pyproject
    )
    assert (package / "proj" / "openenv.yaml").exists()


def test_init_openenv_multifile_uses_component_package(tmp_path: Path) -> None:
    init_environment(
        "openenv-pkg",
        path=str(tmp_path),
        openenv=True,
        multi_file=True,
    )
    init_content = (
        tmp_path / "openenv_pkg" / "openenv_pkg" / "__init__.py"
    ).read_text()

    assert '"""openenv-pkg environment package."""' in init_content
    assert "load_environment" not in init_content
    assert (tmp_path / "openenv_pkg" / "openenv_pkg" / "taskset.py").exists()
    assert (tmp_path / "openenv_pkg" / "openenv_pkg" / "proj").is_dir()
