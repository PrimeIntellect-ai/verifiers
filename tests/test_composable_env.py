"""Tests for the composable architecture: Task, TaskSet, SandboxTaskSet, SandboxSpec."""

import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, call

import pytest

import verifiers as vf
from verifiers.envs.experimental.composable import (
    ComposableEnv,
    Harness,
    SandboxSpec,
    SandboxTaskSet,
    Task,
    TaskSet,
    discover_sibling_dir,
)


# ── Mock Rubrics ──────────────────────────────────────────────────────


class MockSandboxRubric(vf.Rubric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_reward_func(self.solved)

    async def solved(self, state, **kwargs) -> float:
        return 1.0 if state.get("test_output") == "PASS" else 0.0


class MockMathRubric(vf.Rubric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_reward_func(self.correct)

    async def correct(self, state, **kwargs) -> float:
        return 1.0 if state.get("info", {}).get("id") == 0 else 0.0


# ── Mock TaskSets ───────────────────────────────────────────────────────


class MockSandboxTaskSet(SandboxTaskSet):
    """SandboxTaskSet for testing."""

    def get_instruction(self, info):
        return f"Fix bug #{info.get('id', 0)}"

    def get_sandbox_spec(self, info):
        return SandboxSpec(image="python:3.11-slim", cpu_cores=2, memory_gb=2)

    def get_rubric(self):
        return MockSandboxRubric()

    def get_workdir(self, info):
        return "/testbed"

    def get_env_vars(self):
        return {"FOO": "bar"}


class MockTaskSet(TaskSet):
    """Plain TaskSet (no sandbox) for testing."""

    def get_instruction(self, info):
        return info.get("question", "")

    def get_rubric(self):
        return MockMathRubric()


def _make_dataset(n=3):
    from datasets import Dataset

    return Dataset.from_dict(
        {
            "info": [{"id": i, "question": f"q{i}"} for i in range(n)],
            "answer": ["" for _ in range(n)],
        }
    )


# ── SandboxSpec ─────────────────────────────────────────────────────────


def test_sandbox_spec_defaults():
    spec = SandboxSpec()
    assert spec.image == "python:3.11-slim"
    assert spec.cpu_cores == 4


def test_sandbox_spec_custom():
    spec = SandboxSpec(image="lean-tactic:v4.27", gpu_count=1)
    assert spec.image == "lean-tactic:v4.27"
    assert spec.gpu_count == 1


# ── Task from SandboxTaskSet ───────────────────────────────────────────


def test_task_sandbox_spec():
    ts = MockSandboxTaskSet(dataset=_make_dataset(), name="test")
    task = ts[0]
    assert isinstance(task, Task)
    assert task.sandbox_spec is not None
    assert task.sandbox_spec.image == "python:3.11-slim"
    assert task.sandbox_spec.cpu_cores == 2


def test_task_image():
    ts = MockSandboxTaskSet(dataset=_make_dataset(), name="test")
    task = ts[0]
    assert task.image == "python:3.11-slim"


def test_task_workdir():
    ts = MockSandboxTaskSet(dataset=_make_dataset(), name="test")
    task = ts[0]
    assert task.workdir == "/testbed"


def test_task_repr_sandbox():
    ts = MockSandboxTaskSet(dataset=_make_dataset(), name="test")
    task = ts[0]
    assert "python:3.11-slim" in repr(task)


# ── Task from plain TaskSet ────────────────────────────────────────────


def test_task_no_sandbox():
    ts = MockTaskSet(dataset=_make_dataset(), name="math")
    task = ts[0]
    assert task.sandbox_spec is None
    assert task.image is None


def test_task_repr_no_sandbox():
    ts = MockTaskSet(dataset=_make_dataset(), name="math")
    task = ts[0]
    assert "no sandbox" in repr(task)


# ── TaskSet ─────────────────────────────────────────────────────────────


def test_taskset_isinstance():
    ts = MockTaskSet(dataset=_make_dataset(), name="math")
    assert not isinstance(ts, SandboxTaskSet)

    ts2 = MockSandboxTaskSet(dataset=_make_dataset(), name="swe")
    assert isinstance(ts2, SandboxTaskSet)


def test_taskset_len():
    ts = MockTaskSet(dataset=_make_dataset(5), name="test")
    assert len(ts) == 5


def test_taskset_iter():
    ts = MockTaskSet(dataset=_make_dataset(3), name="test")
    tasks = list(ts)
    assert len(tasks) == 3
    assert all(isinstance(t, Task) for t in tasks)


def test_taskset_filter():
    ts = MockSandboxTaskSet(dataset=_make_dataset(5), name="test")
    filtered = ts.filter(lambda ex: ex["info"]["id"] < 3)
    assert len(filtered) == 3
    assert isinstance(filtered, MockSandboxTaskSet)


def test_taskset_take():
    ts = MockSandboxTaskSet(dataset=_make_dataset(5), name="test")
    taken = ts.take(2)
    assert len(taken) == 2
    assert isinstance(taken, MockSandboxTaskSet)


def test_taskset_repr():
    ts = MockTaskSet(dataset=_make_dataset(), name="mytest")
    assert "mytest" in repr(ts)
    assert "3" in repr(ts)


@pytest.mark.asyncio
async def test_composable_env_exports_task_workdir():
    taskset = MockSandboxTaskSet(dataset=_make_dataset(), name="test")
    env = ComposableEnv(
        taskset=taskset,
        harness=Harness(run_command="true"),
    )

    env_vars = await env.build_env_vars(
        {
            "info": {"id": 0},
            "interception_base_url": "https://test.trycloudflare.com/v1",
        }
    )

    assert env_vars["AGENT_WORKDIR"] == "/testbed"
    assert env_vars["FOO"] == "bar"


@pytest.mark.asyncio
async def test_composable_env_quotes_paths_in_mkdir_command():
    taskset = MockSandboxTaskSet(dataset=_make_dataset(), name="test")
    env = ComposableEnv(
        taskset=taskset,
        harness=Harness(
            run_command="true",
            instruction_path="/tmp/with space/prompt.txt",
            system_prompt="system",
            system_prompt_path="/tmp/other path/system.txt",
        ),
    )
    env.sandbox_client = SimpleNamespace(
        execute_command=AsyncMock(),
        teardown=lambda: None,
    )
    env.taskset.setup = AsyncMock()
    env.upload_content = AsyncMock()

    await env.post_sandbox_setup({"sandbox_id": "sbx", "info": {"id": 0}})

    env.sandbox_client.execute_command.assert_awaited_once_with(
        "sbx",
        "mkdir -p '/tmp/other path' '/tmp/with space'",
        timeout=10,
    )


@pytest.mark.asyncio
async def test_composable_env_quotes_log_path_when_collecting_logs():
    taskset = MockSandboxTaskSet(dataset=_make_dataset(), name="test")
    env = ComposableEnv(
        taskset=taskset,
        harness=Harness(
            run_command="true",
            log_path="/tmp/log dir/agent.log",
        ),
    )
    env.sandbox_client = SimpleNamespace(
        execute_command=AsyncMock(
            return_value=SimpleNamespace(stdout="agent log\n", stderr="", exit_code=0)
        ),
        teardown=lambda: None,
    )

    state = {"sandbox_id": "sbx", "timing": {"total_ms": 0}}

    await env.post_rollout(state)

    env.sandbox_client.execute_command.assert_awaited_once_with(
        "sbx",
        "cat '/tmp/log dir/agent.log' 2>/dev/null || echo '<no logs>'",
        working_dir=None,
    )
    assert state["agent_logs"] == "agent log"


# ── install_env ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_composable_env_install_env_passes_to_execute():
    taskset = MockSandboxTaskSet(dataset=_make_dataset(), name="test")
    env = ComposableEnv(
        taskset=taskset,
        harness=Harness(
            run_command="true",
            install_script="install-agent",
            instruction_path="/tmp/prompt.txt",
        ),
        install_env={"GH_TOKEN": "secret"},
    )
    env.sandbox_client = SimpleNamespace(
        execute_command=AsyncMock(
            return_value=SimpleNamespace(stdout="", stderr="", exit_code=0)
        ),
        teardown=lambda: None,
    )
    env.taskset.setup = AsyncMock()
    env.upload_content = AsyncMock()

    await env.post_sandbox_setup({"sandbox_id": "sbx", "info": {"id": 0}})

    install_call = env.sandbox_client.execute_command.await_args_list[-1]
    assert install_call == call(
        "sbx", "install-agent", timeout=300, env={"GH_TOKEN": "secret"}
    )


@pytest.mark.asyncio
async def test_composable_env_install_env_none_by_default():
    taskset = MockSandboxTaskSet(dataset=_make_dataset(), name="test")
    env = ComposableEnv(
        taskset=taskset,
        harness=Harness(
            run_command="true",
            install_script="install-agent",
            instruction_path="/tmp/prompt.txt",
        ),
    )
    env.sandbox_client = SimpleNamespace(
        execute_command=AsyncMock(
            return_value=SimpleNamespace(stdout="", stderr="", exit_code=0)
        ),
        teardown=lambda: None,
    )
    env.taskset.setup = AsyncMock()
    env.upload_content = AsyncMock()

    await env.post_sandbox_setup({"sandbox_id": "sbx", "info": {"id": 0}})

    install_call = env.sandbox_client.execute_command.await_args_list[-1]
    assert install_call == call("sbx", "install-agent", timeout=300)


# ── get_upload_dirs ──────────────────────────────────────────────────────


def _make_temp_taskset_package(tmp_path, monkeypatch, *, with_skills: bool):
    package_name = f"fixture_{tmp_path.name.replace('-', '_')}"
    pkg_dir = tmp_path / package_name
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "taskset_mod.py").write_text("MARKER = 1\n")

    if with_skills:
        skill_dir = pkg_dir / "skills" / "demo"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: demo\n---\n")
        (skill_dir / "pyproject.toml").write_text(
            "[project]\nname = 'skill-demo'\nversion = '0.0.0'\n"
        )

    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    mod = importlib.import_module(f"{package_name}.taskset_mod")
    return mod, package_name


class MockSandboxTaskSetWithSkills(SandboxTaskSet):
    """SandboxTaskSet — skills auto-discovered via get_skills_dir()."""

    def get_instruction(self, info):
        return f"Fix bug #{info.get('id', 0)}"

    def get_sandbox_spec(self, info):
        return SandboxSpec(image="python:3.11-slim", cpu_cores=2, memory_gb=2)

    def get_rubric(self):
        return MockSandboxRubric()

    def get_workdir(self, info):
        return "/testbed"


@pytest.mark.asyncio
async def test_composable_env_uploads_task_dirs(tmp_path, monkeypatch):
    mod, _ = _make_temp_taskset_package(tmp_path, monkeypatch, with_skills=True)
    monkeypatch.setattr(MockSandboxTaskSetWithSkills, "__module__", mod.__name__)
    taskset = MockSandboxTaskSetWithSkills(dataset=_make_dataset(), name="test")
    env = ComposableEnv(
        taskset=taskset,
        harness=Harness(
            run_command="true",
            install_script="install-agent",
            skills_path="/task/skills",
        ),
    )
    env.sandbox_client = SimpleNamespace(
        execute_command=AsyncMock(
            return_value=SimpleNamespace(stdout="", stderr="", exit_code=0)
        ),
        teardown=lambda: None,
    )
    env.taskset.setup = AsyncMock()
    env.upload_content = AsyncMock()
    env.upload_file = AsyncMock()

    await env.post_sandbox_setup({"sandbox_id": "sbx", "info": {"id": 0}})

    env.upload_file.assert_awaited_once()
    upload_call = env.upload_file.await_args
    assert upload_call.args[0] == "sbx"
    assert upload_call.args[1] == "/tmp/_upload_task_skills.tar.gz"

    install_call = env.sandbox_client.execute_command.await_args_list[-1]
    assert install_call == call("sbx", "install-agent", timeout=300)
    extract_call = env.sandbox_client.execute_command.await_args_list[1]
    assert extract_call == call(
        "sbx",
        "mkdir -p /task && tar -xzf /tmp/_upload_task_skills.tar.gz -C / && rm -f /tmp/_upload_task_skills.tar.gz",
        timeout=60,
    )


@pytest.mark.asyncio
async def test_composable_env_no_upload_when_no_dirs(tmp_path, monkeypatch):
    mod, _ = _make_temp_taskset_package(tmp_path, monkeypatch, with_skills=False)
    monkeypatch.setattr(MockSandboxTaskSetWithSkills, "__module__", mod.__name__)
    taskset = MockSandboxTaskSetWithSkills(dataset=_make_dataset(), name="test")
    env = ComposableEnv(
        taskset=taskset,
        harness=Harness(
            run_command="true",
            install_script="install-agent",
            skills_path="/task/skills",
        ),
    )
    env.sandbox_client = SimpleNamespace(
        execute_command=AsyncMock(
            return_value=SimpleNamespace(stdout="", stderr="", exit_code=0)
        ),
        teardown=lambda: None,
    )
    env.taskset.setup = AsyncMock()
    env.upload_content = AsyncMock()
    env.upload_file = AsyncMock()

    await env.post_sandbox_setup({"sandbox_id": "sbx", "info": {"id": 0}})

    assert env.upload_file.await_count == 0


# ── discover_sibling_dir ─────────────────────────────────────────────────


def test_discover_sibling_dir_finds_skills(tmp_path, monkeypatch):
    mod, _ = _make_temp_taskset_package(tmp_path, monkeypatch, with_skills=True)
    monkeypatch.setattr(MockSandboxTaskSetWithSkills, "__module__", mod.__name__)
    result = discover_sibling_dir(MockSandboxTaskSetWithSkills, "skills")
    assert result is not None


def test_discover_sibling_dir_returns_none_without_skills(tmp_path, monkeypatch):
    mod, _ = _make_temp_taskset_package(tmp_path, monkeypatch, with_skills=False)
    monkeypatch.setattr(MockSandboxTaskSetWithSkills, "__module__", mod.__name__)
    result = discover_sibling_dir(MockSandboxTaskSetWithSkills, "skills")
    assert result is None


# ── get_skills_dir / auto-discovery ──────────────────────────────────────


def test_get_skills_dir_auto_discovers(tmp_path, monkeypatch):
    mod, _ = _make_temp_taskset_package(tmp_path, monkeypatch, with_skills=True)
    monkeypatch.setattr(MockSandboxTaskSetWithSkills, "__module__", mod.__name__)
    taskset = MockSandboxTaskSetWithSkills(dataset=_make_dataset(), name="test")
    assert taskset.get_skills_dir() is not None


def test_get_skills_dir_returns_none_without_skills(tmp_path, monkeypatch):
    mod, _ = _make_temp_taskset_package(tmp_path, monkeypatch, with_skills=False)
    monkeypatch.setattr(MockSandboxTaskSetWithSkills, "__module__", mod.__name__)
    taskset = MockSandboxTaskSetWithSkills(dataset=_make_dataset(), name="test")
    assert taskset.get_skills_dir() is None


def test_get_upload_dirs_includes_skills_automatically(tmp_path, monkeypatch):
    mod, _ = _make_temp_taskset_package(tmp_path, monkeypatch, with_skills=True)
    monkeypatch.setattr(MockSandboxTaskSetWithSkills, "__module__", mod.__name__)
    taskset = MockSandboxTaskSetWithSkills(dataset=_make_dataset(), name="test")
    upload_dirs = taskset.get_upload_dirs()
    assert "skills" in upload_dirs


def test_get_upload_dirs_empty_without_skills(tmp_path, monkeypatch):
    mod, _ = _make_temp_taskset_package(tmp_path, monkeypatch, with_skills=False)
    monkeypatch.setattr(MockSandboxTaskSetWithSkills, "__module__", mod.__name__)
    taskset = MockSandboxTaskSetWithSkills(dataset=_make_dataset(), name="test")
    assert taskset.get_upload_dirs() == {}


# ── Harness metrics collection ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_composable_env_collects_harness_metrics():
    taskset = MockSandboxTaskSet(dataset=_make_dataset(), name="test")
    metrics_data = {
        "turns": 3,
        "stop_reason": "done",
        "prompt_tokens": 100,
        "completion_tokens": 25,
    }
    env = ComposableEnv(
        taskset=taskset,
        harness=Harness(
            run_command="true",
            log_path="/tmp/log dir/agent.log",
            metrics_path="{workdir}/.rlm/sessions/*/meta.json",
            metrics_key="metrics",
            metrics_prefix="rlm_",
        ),
    )
    env.sandbox_client = SimpleNamespace(
        execute_command=AsyncMock(
            side_effect=[
                SimpleNamespace(stdout="agent log\n", stderr="", exit_code=0),
                SimpleNamespace(
                    stdout=json.dumps({"metrics": metrics_data}),
                    stderr="",
                    exit_code=0,
                ),
            ]
        ),
        teardown=lambda: None,
    )

    state = {
        "sandbox_id": "sbx",
        "info": {"id": 0},
        "timing": {"total_ms": 0},
        "trajectory": [],
    }

    await env.post_rollout(state)

    assert state["agent_logs"] == "agent log"
    assert state["rlm_turns"] == 3
    assert state["rlm_stop_reason"] == "done"
    assert state["rlm_prompt_tokens"] == 100
    assert state["rlm_completion_tokens"] == 25


@pytest.mark.asyncio
async def test_composable_env_metrics_with_key_whitelist():
    taskset = MockSandboxTaskSet(dataset=_make_dataset(), name="test")
    env = ComposableEnv(
        taskset=taskset,
        harness=Harness(
            run_command="true",
            metrics_path="{workdir}/metrics.json",
            metrics_prefix="agent_",
            metrics_keys=["turns", "tokens"],
        ),
    )
    env.sandbox_client = SimpleNamespace(
        execute_command=AsyncMock(
            return_value=SimpleNamespace(
                stdout=json.dumps({"turns": 5, "tokens": 200, "secret": "hidden"}),
                stderr="",
                exit_code=0,
            )
        ),
        teardown=lambda: None,
    )

    state = {
        "sandbox_id": "sbx",
        "info": {"id": 0},
        "timing": {"total_ms": 0},
        "trajectory": [],
    }

    await env.post_rollout(state)

    assert state["agent_turns"] == 5
    assert state["agent_tokens"] == 200
    assert "agent_secret" not in state


@pytest.mark.asyncio
async def test_composable_env_no_metrics_when_path_not_set():
    taskset = MockSandboxTaskSet(dataset=_make_dataset(), name="test")
    env = ComposableEnv(
        taskset=taskset,
        harness=Harness(run_command="true"),
    )
    env.sandbox_client = SimpleNamespace(
        execute_command=AsyncMock(),
        teardown=lambda: None,
    )

    state = {
        "sandbox_id": "sbx",
        "info": {"id": 0},
        "timing": {"total_ms": 0},
        "trajectory": [],
    }

    await env.post_rollout(state)

    # No execute_command calls since no log_path and no metrics_path
    env.sandbox_client.execute_command.assert_not_awaited()
