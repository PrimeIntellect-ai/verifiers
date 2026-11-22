"""`vf-describe`: a package's descriptor, read by importing it and never loading it."""

import json
import sys
import textwrap
import uuid

import pytest

from verifiers.v1.cli.describe import SCHEMA_VERSION, DescribeError, describe, main

PACKAGE = '''
import verifiers.v1 as vf
from verifiers.v1.harnesses.bash import BashHarness


class LookupToolset(vf.Toolset):
    @vf.tool
    async def lookup(self, term: str) -> str:
        """Look a term up."""
        return term


class QuizData(vf.TaskData):
    answer: str = ""


class QuizTaskConfig(vf.TaskConfig):
    tools: vf.ToolsetConfig = vf.ToolsetConfig()


class QuizTask(vf.Task[QuizData, vf.State, QuizTaskConfig]):
    NEEDS_CONTAINER = True

    @vf.reward(weight=0.5, priority=1)
    async def correct(self, trace) -> float:
        """Whether the answer is right."""
        return 1.0

    @vf.reward
    async def formatted(self, trace) -> float:
        return 1.0

    @vf.metric
    async def length(self, trace) -> float:
        return 0.0

    @classmethod
    def toolsets(cls, config):
        return [LookupToolset(config.tools)]


class QuizTasksetConfig(vf.TasksetConfig):
    split: str = "train"
    task: QuizTaskConfig = QuizTaskConfig()


class QuizTaskset(vf.Taskset[QuizTask, QuizTasksetConfig]):
    """A quiz."""

    INFINITE = True

    def load(self):
        raise AssertionError("describing a package never loads its tasks")


class QuizHarness(BashHarness):
    SUPPORTS_SKILLS = True


class QuizEnvConfig(vf.EnvConfig):
    solver: vf.AgentConfig = vf.AgentConfig()
    critic: vf.AgentConfig = vf.AgentConfig(harness=vf.HarnessConfig(id="null"))


class QuizEnv(vf.Env[QuizEnvConfig]):
    async def run(self, task, agents):
        raise AssertionError("describing a package never runs an episode")


__all__ = [{exports}]
'''


@pytest.fixture
def package(tmp_path, monkeypatch):
    """Write a package exporting `exports` and return its id."""
    monkeypatch.syspath_prepend(str(tmp_path))

    def write(*exports: str, source: str = PACKAGE, name: str = "") -> str:
        name = name or f"quiz_{uuid.uuid4().hex[:8]}"
        (tmp_path / name).mkdir()
        body = source.replace("{exports}", ", ".join(repr(e) for e in exports))
        (tmp_path / name / "__init__.py").write_text(textwrap.dedent(body))
        return name.replace("_", "-")

    return write


def test_a_taskset_package_is_described_without_loading_its_tasks(package) -> None:
    pid = package("QuizTaskset")
    descriptor = describe(pid)

    assert descriptor["schema_version"] == SCHEMA_VERSION
    assert descriptor["errors"] == []
    module = descriptor["modules"][0]
    assert descriptor["exports"] == {
        "taskset": f"{module}.QuizTaskset",
        "harness": None,
        "env": None,
        "judge": None,
    }
    taskset = descriptor["taskset"]
    assert taskset["infinite"] is True
    assert taskset["doc"] == "A quiz."
    assert "split" in taskset["config_schema"]["properties"]
    task = taskset["task"]
    assert task["needs_container"] is True
    assert "answer" in task["data_schema"]["properties"]
    # Sorted by descending priority, then name, as the task scores them.
    assert [(r["name"], r["weight"], r["priority"]) for r in task["rewards"]] == [
        ("correct", 0.5, 1),
        ("formatted", 1.0, 0),
    ]
    assert task["rewards"][0]["doc"] == "Whether the answer is right."
    assert task["rewards"][0]["source"]["file"] == f"{module}/__init__.py"
    assert [m["name"] for m in task["metrics"]] == ["length"]
    assert [
        (t["class"].rsplit(".", 1)[1], [tool["name"] for tool in t["tools"]])
        for t in task["toolsets"]
    ] == [("LookupToolset", ["lookup"])]
    # A plain taskset runs in the single-agent env, with the built-in harness.
    assert descriptor["env"]["exported"] is False
    assert descriptor["env"]["roles"] == [
        {"name": "agent", "doc": None, "harness": "bash", "harness_pinned": False}
    ]
    assert descriptor["harness"] is None
    assert descriptor["default_harness"] == "bash"


def test_a_bundled_harness_and_env_are_described_with_the_roles(package) -> None:
    pid = package("QuizTaskset", "QuizHarness", "QuizEnv")
    descriptor = describe(pid)

    assert descriptor["errors"] == []
    # The bundled harness is the default one; a pinned role keeps its own.
    assert descriptor["default_harness"] == pid
    assert descriptor["env"]["exported"] is True
    assert [
        (role["name"], role["harness"], role["harness_pinned"])
        for role in descriptor["env"]["roles"]
    ] == [("solver", pid, False), ("critic", "null", True)]
    harness = descriptor["harness"]
    assert harness["capabilities"]["SUPPORTS_SKILLS"] is True
    assert harness["capabilities"]["NEEDS_CONTAINER"] is False
    assert harness["dialects"] == ["chat_completions"]
    assert harness["config_schema"]["properties"]["id"]


def test_rewards_plugged_in_the_default_task_config_are_merged_unimported(
    package,
) -> None:
    source = PACKAGE.replace(
        "    tools: vf.ToolsetConfig = vf.ToolsetConfig()",
        "    tools: vf.ToolsetConfig = vf.ToolsetConfig()\n"
        "    rewards: dict[str, vf.RewardFunctionConfig] = {\n"
        '        "formatted": vf.RewardFunctionConfig(weight=0.25),\n'
        '        "graded": vf.RewardFunctionConfig(fn="a_module_nobody_has:grade"),\n'
        "    }",
    )
    descriptor = describe(package("QuizTaskset", source=source))
    rewards = descriptor["taskset"]["task"]["rewards"]

    # The plugged function is named, never imported: what only it would set is unknown.
    assert descriptor["errors"] == []
    assert [(r["name"], r["weight"], r["fn"]) for r in rewards] == [
        ("correct", 0.5, None),
        ("formatted", 0.25, None),
        ("graded", None, "a_module_nobody_has:grade"),
    ]


def test_a_taskset_named_like_a_built_in_harness_takes_none_of_its_exports(
    package, monkeypatch
) -> None:
    monkeypatch.delitem(sys.modules, "bash", raising=False)
    descriptor = describe(package("QuizTaskset", name="bash"))

    assert descriptor["exports"]["taskset"] == "bash.QuizTaskset"
    assert descriptor["exports"]["harness"] is None
    assert descriptor["default_harness"] == "bash"


def test_a_package_without_plugin_exports_is_refused(package) -> None:
    source = "def load_environment():\n    return None\n"
    with pytest.raises(
        DescribeError, match="exports no Taskset, Harness, Env or Judge"
    ):
        describe(package(source=source))


def test_a_missing_package_exits_with_a_json_error(capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["no-such-package-anywhere"])

    assert exit_info.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    error = json.loads(captured.err)["error"]
    assert "not found" in error["message"]


def test_the_descriptor_is_json_on_stdout_alone(package, capsys) -> None:
    source = 'print("hello from the import")\n' + PACKAGE
    main([package("QuizTaskset", source=source)])

    captured = capsys.readouterr()
    assert json.loads(captured.out)["taskset"]["infinite"] is True
    assert "hello from the import" in captured.err


def test_a_dependency_missing_inside_the_package_is_named(package) -> None:
    source = "import a_dependency_nobody_installed\n" + PACKAGE
    with pytest.raises(DescribeError, match="a_dependency_nobody_installed"):
        describe(package("QuizTaskset", source=source))


def test_a_built_in_tasksets_bundled_env_is_found_in_its_module(
    tmp_path, monkeypatch
) -> None:
    import verifiers.v1.tasksets as builtin_tasksets

    name = f"quiz_{uuid.uuid4().hex[:8]}"
    (tmp_path / name).mkdir()
    body = PACKAGE.replace("{exports}", "'QuizTaskset', 'QuizEnv'")
    (tmp_path / name / "__init__.py").write_text(textwrap.dedent(body))
    monkeypatch.setattr(
        builtin_tasksets, "__path__", [*builtin_tasksets.__path__, str(tmp_path)]
    )

    descriptor = describe(name.replace("_", "-"))

    assert descriptor["modules"] == [f"verifiers.v1.tasksets.{name}"]
    assert descriptor["env"]["exported"] is True
    assert [role["name"] for role in descriptor["env"]["roles"]] == [
        "solver",
        "critic",
    ]


def test_a_harness_package_named_like_a_built_in_resolves_to_the_built_in(
    package, monkeypatch
) -> None:
    monkeypatch.delitem(sys.modules, "bash", raising=False)
    descriptor = describe(package("QuizHarness", name="bash"))

    # As `harness_class("bash")` resolves it: the built-in shadows the package.
    assert descriptor["exports"]["harness"] == (
        "verifiers.v1.harnesses.bash.harness.BashHarness"
    )
    assert descriptor["exports"]["taskset"] is None


def test_a_judge_is_described_as_its_id_loads_it(package, monkeypatch) -> None:
    source = PACKAGE + (
        "\n\nclass QuizJudge(vf.Judge):\n"
        "    pass\n\n\n"
        "__all__ = [*__all__, 'QuizJudge']\n"
    )
    monkeypatch.delitem(sys.modules, "rubric", raising=False)
    descriptor = describe(package("QuizTaskset", source=source, name="rubric"))

    # `judge_class("rubric")` loads the built-in judge, whatever the package exports.
    assert descriptor["exports"]["taskset"] == "rubric.QuizTaskset"
    assert descriptor["exports"]["judge"] == "verifiers.v1.judges.rubric.RubricJudge"


def test_describing_never_runs_a_configs_validators(package) -> None:
    source = PACKAGE.replace(
        '    split: str = "train"\n',
        '    split: str = "train"\n\n'
        '    @pydantic.model_validator(mode="after")\n'
        "    def _check(self):\n"
        '        raise AssertionError("describing a package never validates a config")\n',
    ).replace(
        "import verifiers.v1 as vf\n", "import pydantic\nimport verifiers.v1 as vf\n"
    )
    descriptor = describe(package("QuizTaskset", source=source))

    assert descriptor["errors"] == []
    assert [
        t["class"].rsplit(".", 1)[1] for t in descriptor["taskset"]["task"]["toolsets"]
    ] == ["LookupToolset"]


def test_a_broken_library_named_like_a_built_in_taskset_is_no_judge(
    tmp_path, monkeypatch
) -> None:
    import verifiers.v1.tasksets as builtin_tasksets

    name = f"quiz_{uuid.uuid4().hex[:8]}"
    builtin = tmp_path / "builtin"
    (builtin / name).mkdir(parents=True)
    body = PACKAGE.replace("{exports}", "'QuizTaskset'")
    (builtin / name / "__init__.py").write_text(textwrap.dedent(body))
    monkeypatch.setattr(
        builtin_tasksets, "__path__", [*builtin_tasksets.__path__, str(builtin)]
    )
    # A third-party library with the taskset's name, which can't import.
    library = tmp_path / "library"
    (library / name).mkdir(parents=True)
    (library / name / "__init__.py").write_text("import a_dependency_nobody_has\n")
    monkeypatch.syspath_prepend(str(library))

    descriptor = describe(name.replace("_", "-"))

    assert (
        descriptor["exports"]["taskset"] == f"verifiers.v1.tasksets.{name}.QuizTaskset"
    )
    assert descriptor["exports"]["judge"] is None


def test_a_taskset_package_exporting_two_judges_is_refused(package) -> None:
    source = PACKAGE + (
        "\n\nclass FirstJudge(vf.Judge):\n    pass\n\n\n"
        "class SecondJudge(vf.Judge):\n    pass\n\n\n"
        "__all__ = [*__all__, 'FirstJudge', 'SecondJudge']\n"
    )
    with pytest.raises(DescribeError, match="exports 2 Judge subclasses"):
        describe(package("QuizTaskset", source=source))


def test_a_broken_package_named_like_a_built_in_harness_leaves_the_built_in(
    package, monkeypatch
) -> None:
    monkeypatch.delitem(sys.modules, "bash", raising=False)
    package(source="import a_dependency_nobody_has\n", name="bash")

    # `harness_class("bash")` never imports the broken package: the built-in wins.
    descriptor = describe("bash")

    assert descriptor["exports"]["harness"] == (
        "verifiers.v1.harnesses.bash.harness.BashHarness"
    )
