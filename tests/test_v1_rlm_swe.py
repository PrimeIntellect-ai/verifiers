import sys
import types
from collections.abc import Mapping
from pathlib import Path

import pytest
from datasets import Dataset

import verifiers.v1 as vf
from environments.rlm_swe_v1 import rlm_swe_v1
from verifiers.envs.experimental.composable.task import SandboxSpec
from verifiers.v1.packages.tasksets import swe as v1_swe
from verifiers.v1.utils.program_utils import merge_task_program, merge_task_sandbox


def as_mapping(value: object) -> Mapping[str, object]:
    assert isinstance(value, Mapping)
    return value


def test_rlm_harness_builds_sandbox_program_without_eager_checkout():
    harness = vf.RLM(
        config=vf.RLMConfig(local_checkout="/tmp/does-not-need-to-exist-yet")
    )
    program = as_mapping(harness.program)
    program_env = as_mapping(program["env"])
    artifacts = as_mapping(program["artifacts"])
    setup = program["setup"]

    assert isinstance(harness, vf.Harness)
    assert program["sandbox"] is not False
    assert isinstance(setup, list)
    assert "apt-get -o Acquire::Retries=3 update" in setup[0]
    assert "apt-get -o Acquire::Retries=3 install" in setup[0]
    assert "RLM_MODEL" in program_env
    assert "rlm_metrics" in artifacts


def test_rlm_harness_accepts_typed_config_surface():
    harness = vf.RLM(
        config=vf.RLMConfig(
            local_checkout="/tmp/checkout",
            rlm_tools=["bash", "edit"],
            rlm_max_turns=7,
            rlm_exec_timeout=11,
            env_vars={"CUSTOM": "1"},
        )
    )
    program = as_mapping(harness.program)
    program_env = as_mapping(program["env"])

    assert harness.config.rlm_tools == ["bash", "edit"]
    assert program_env["RLM_TOOLS"] == "bash,edit"
    assert program_env["RLM_MAX_TURNS"] == "7"
    assert program_env["RLM_EXEC_TIMEOUT"] == "11"
    assert program_env["CUSTOM"] == "1"


def test_rlm_harness_can_upload_skills(tmp_path: Path):
    skills = tmp_path / "skills"
    (skills / "edit").mkdir(parents=True)
    (skills / "edit" / "SKILL.md").write_text("---\nname: edit\n---\n")

    harness = vf.RLM(
        config=vf.RLMConfig(local_checkout="/tmp/checkout", skills=str(skills))
    )
    program = as_mapping(harness.program)
    dirs = as_mapping(program["dirs"])

    assert dirs["/rlm/skills"] == skills


def test_rlm_harness_uploads_taskset_skills_by_default(tmp_path: Path):
    skills = tmp_path / "taskset-skills"
    skills.mkdir()
    (skills / "SKILL.md").write_text("---\nname: taskset\n---\n")

    class SkillTaskset(vf.Taskset):
        def get_upload_dirs(self):
            return {"skills": skills}

    env = vf.Env(
        taskset=SkillTaskset(config=vf.TasksetConfig(source=[])),
        harness=vf.RLM(config=vf.RLMConfig(local_checkout="/tmp/checkout")),
    )
    program = as_mapping(env.harness.program)
    dirs = as_mapping(program["dirs"])

    assert dirs["/rlm/skills"] == skills


def test_taskset_discovers_sibling_skills_dir_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_name = "skill_taskset_module"
    module_file = tmp_path / f"{module_name}.py"
    skills = tmp_path / "skills"
    module_file.write_text("")
    skills.mkdir()
    (skills / "SKILL.md").write_text("---\nname: sibling\n---\n")
    module = types.ModuleType(module_name)
    module.__file__ = str(module_file)
    module.__package__ = ""
    monkeypatch.setitem(sys.modules, module_name, module)
    skill_taskset_type = type(
        "SkillTaskset", (vf.Taskset,), {"__module__": module_name}
    )

    taskset = skill_taskset_type(config=vf.TasksetConfig(source=[]))

    assert taskset.get_upload_dirs() == {"skills": skills}


def test_rlm_harness_explicit_skills_override_taskset_skills(tmp_path: Path):
    taskset_skills = tmp_path / "taskset-skills"
    explicit_skills = tmp_path / "explicit-skills"
    taskset_skills.mkdir()
    explicit_skills.mkdir()

    class SkillTaskset(vf.Taskset):
        def get_upload_dirs(self):
            return {"skills": taskset_skills}

    env = vf.Env(
        taskset=SkillTaskset(config=vf.TasksetConfig(source=[])),
        harness=vf.RLM(
            config=vf.RLMConfig(
                local_checkout="/tmp/checkout",
                skills=str(explicit_skills),
            )
        ),
    )
    program = as_mapping(env.harness.program)
    dirs = as_mapping(program["dirs"])

    assert dirs["/rlm/skills"] == explicit_skills


def test_swe_taskset_builds_v1_rows_from_legacy_taskset(monkeypatch):
    calls = patch_fake_swe_factory(monkeypatch)

    taskset = vf.SWETaskset(
        config=vf.SWETasksetConfig(
            task_type="r2e",
            dataset_name="fake-r2e",
            repo_path="/workspace/repo",
            timeout_minutes=30,
            env={"CUSTOM": "1"},
        )
    )
    task = next(iter(taskset))

    assert calls["backend"] == "r2e"
    assert calls["kwargs"]["dataset_name"] == "fake-r2e"
    assert calls["kwargs"]["repo_path"] == "/workspace/repo"
    assert calls["kwargs"]["timeout_minutes"] == 30
    assert task["taskset_id"] == "swe/fake"
    assert task["instruction"] == "Fix repo-0."
    assert task["sandbox"]["image"] == "fake/image:latest"
    assert task["sandbox"]["workdir"] == "/workspace/repo"
    assert task["sandbox"]["timeout_minutes"] == 30
    program_env = as_mapping(as_mapping(task["program"])["env"])
    assert program_env["AGENT_WORKDIR"] == "/workspace/repo"
    assert "/workspace/repo/.venv/bin" in program_env["AGENT_PATH"]
    assert program_env["PAGER"] == "cat"
    assert program_env["CUSTOM"] == "1"
    assert "PATH" not in program_env
    assert "skills" in taskset.get_upload_dirs()


def test_rlm_swe_environment_uses_v1_swe_taskset(monkeypatch):
    calls = patch_fake_swe_factory(monkeypatch)

    env = rlm_swe_v1.load_environment(
        config=rlm_swe_v1.RlmSweEnvConfig(
            taskset=rlm_swe_v1.RlmSweTasksetConfig(
                task_type="r2e",
                dataset_name="fake-r2e",
                repo_path="/workspace/repo",
                timeout_minutes=30,
                env={"CUSTOM": "1"},
            ),
            harness=vf.RLMConfig(
                local_checkout="/tmp/checkout",
                env_vars={"CALLER": "1"},
            ),
        ),
    )
    task = next(iter(env.taskset))
    program = as_mapping(env.harness.program)
    program_env = as_mapping(program["env"])
    merged_program = merge_task_program(program, task, kind="command")
    merged_env = as_mapping(merged_program["env"])
    merged_sandbox = merge_task_sandbox(as_mapping(env.harness.sandbox), task)

    assert isinstance(env, vf.Env)
    assert isinstance(env.taskset, rlm_swe_v1.R2ESWETaskset)
    assert isinstance(env.harness, vf.RLM)
    assert calls["backend"] == "r2e"
    assert task["taskset_id"] == "swe/fake"
    assert task["instruction"] == "Fix repo-0."
    assert task["sandbox"]["image"] == "fake/image:latest"
    assert task["sandbox"]["workdir"] == "/workspace/repo"
    assert task["sandbox"]["timeout_minutes"] == 30
    assert "CUSTOM" not in program_env
    assert program_env["CALLER"] == "1"
    assert program_env["RLM_TOOLS"] == "bash,edit"
    assert merged_sandbox["workdir"] == "/workspace/repo"
    assert merged_env["AGENT_WORKDIR"] == "/workspace/repo"
    assert "/workspace/repo/.venv/bin" in merged_env["AGENT_PATH"]
    assert merged_env["PAGER"] == "cat"
    assert merged_env["CUSTOM"] == "1"
    assert merged_env["CALLER"] == "1"


def test_swe_taskset_hooks_are_registered_with_runtime(monkeypatch):
    patch_fake_swe_factory(monkeypatch)
    taskset = vf.SWETaskset(config=vf.SWETasksetConfig())
    env = vf.Env(taskset=taskset)

    setup_names = [handler.__name__ for handler in env.harness.runtime.rollout_setup]
    cleanup_names = [
        handler.__name__ for handler in env.harness.runtime.rollout_cleanup
    ]
    signal_names = {signal["name"] for signal in env.harness.runtime.rollout_signals}

    assert setup_names.count("setup_swe_sandbox") == 1
    assert cleanup_names.count("cleanup_swe_state") == 1
    assert "solved" in signal_names


@pytest.mark.asyncio
async def test_swe_taskset_setup_reward_and_cleanup(monkeypatch):
    patch_fake_swe_factory(monkeypatch)
    taskset = vf.SWETaskset(config=vf.SWETasksetConfig(timeout_minutes=30))
    task = next(iter(taskset))
    state = vf.State.for_task(task)
    sandbox = FakeSandbox()

    await taskset.setup_swe_sandbox(task, state, sandbox=sandbox)
    reward = await next(
        handler for handler in taskset.rewards if handler.__name__ == "solved"
    )(task, state)
    await taskset.cleanup_swe_state(task, state)

    assert state["legacy_setup"] is True
    assert state["sandbox_id"] == "sandbox-1"
    assert state["test_timeout"] == 1800
    assert state["test_output"] == "fake test output"
    assert reward == 1.0
    assert "sandbox_client" not in state


def test_swe_taskset_rejects_split_for_non_split_backend(monkeypatch):
    patch_fake_swe_factory(monkeypatch)

    with pytest.raises(ValueError, match="does not accept split"):
        vf.SWETaskset(config=vf.SWETasksetConfig(task_type="r2e", split="test"))


def patch_fake_swe_factory(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    calls: dict[str, object] = {}

    def fake_make_swe_taskset(backend: str, **kwargs: object) -> FakeLegacyTaskset:
        taskset = FakeLegacyTaskset(kwargs)
        calls["backend"] = backend
        calls["kwargs"] = kwargs
        calls["taskset"] = taskset
        return taskset

    monkeypatch.setattr(v1_swe, "make_swe_taskset", fake_make_swe_taskset)
    return calls


class FakeLegacyTaskset:
    name = "swe/fake"
    default_workdir = "/workspace/repo"

    def __init__(self, kwargs: Mapping[str, object]):
        self.kwargs = dict(kwargs)
        self.setup_states: list[object] = []

    def get_dataset(self) -> Dataset:
        return Dataset.from_list(
            [
                {
                    "question": f"Fix repo-{index}.",
                    "answer": "",
                    "info": {
                        "instance_id": f"instance-{index}",
                        "repo": "example/repo",
                        "problem_statement": f"Fix repo-{index}.",
                        "docker_image": "fake/image:latest",
                        "timeout_minutes": self.kwargs.get("timeout_minutes"),
                        "passes": True,
                    },
                }
                for index in range(2)
            ]
        )

    def get_instruction(self, info: Mapping[str, object]) -> str:
        return str(info["problem_statement"])

    def get_sandbox_spec(self, info: Mapping[str, object]) -> SandboxSpec:
        return SandboxSpec(
            image=str(info["docker_image"]),
            cpu_cores=4,
            memory_gb=4,
            disk_size_gb=10,
            timeout_minutes=info.get("timeout_minutes"),
        )

    def get_workdir(self, info: Mapping[str, object]) -> str:
        _ = info
        return "/workspace/repo"

    def get_env_vars(self) -> dict[str, str]:
        return {
            "PATH": (
                "/opt/miniconda3/bin:/workspace/repo/.venv/bin:/root/.local/bin:"
                "/usr/local/bin:/usr/bin:/bin"
            ),
            "PAGER": "cat",
        }

    async def setup(self, state: vf.ConfigMap) -> None:
        self.setup_states.append(state)
        assert state["sandbox_client"] is FakeSandbox.client
        state["legacy_setup"] = True

    def get_rubric(self) -> "FakeLegacyRubric":
        return FakeLegacyRubric()

    async def validate_instance(self, state: vf.ConfigMap) -> bool:
        return bool(state.get("valid", True))


class FakeLegacyRubric:
    def _get_group_reward_funcs(self) -> list[object]:
        return []

    def _get_individual_reward_funcs(self) -> list[object]:
        return [fake_solved]

    def _get_individual_reward_weights(self) -> list[float]:
        return [1.0]

    async def _call_individual_reward_func(
        self, func: object, state: vf.ConfigMap
    ) -> float:
        return float(await func(state=state, info=state["info"]))


async def fake_solved(state: vf.ConfigMap, info: Mapping[str, object]) -> float:
    state["test_output"] = "fake test output"
    return float(bool(info.get("passes")))


fake_solved.__name__ = "solved"


class FakeSandbox:
    client = object()

    def __init__(self):
        self.id = "sandbox-1"
        self.lease = FakeLease()


class FakeLease:
    client = FakeSandbox.client
