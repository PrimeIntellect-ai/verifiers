"""Tests for RLM harness integration with ComposableEnv.

Validates that rlm_harness() produces a Harness with the correct metrics
fields and that the install script is generated correctly.
"""

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
)
from verifiers.envs.experimental.composable.harnesses.rlm import (
    build_install_script,
    rlm_harness,
)


class MockSandboxRubric(vf.Rubric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_reward_func(self.solved)

    async def solved(self, state, **kwargs) -> float:
        return 1.0 if state.get("test_output") == "PASS" else 0.0


class MockSandboxTaskSet(SandboxTaskSet):
    def get_instruction(self, info):
        return f"Fix bug #{info.get('id', 0)}"

    def get_sandbox_spec(self, info):
        return SandboxSpec(image="python:3.11-slim", cpu_cores=2, memory_gb=2)

    def get_rubric(self):
        return MockSandboxRubric()

    def get_workdir(self, info):
        return "/testbed"


class MockSandboxTaskSetWithSkills(MockSandboxTaskSet):
    """Skills auto-discovered via get_skills_dir() — module monkeypatched in tests."""

    pass


def _make_dataset(n=3):
    from datasets import Dataset

    return Dataset.from_dict(
        {
            "info": [{"id": i, "question": f"q{i}"} for i in range(n)],
            "answer": ["" for _ in range(n)],
        }
    )


def _make_temp_taskset_package(tmp_path, monkeypatch, *, with_skills: bool):
    package_name = f"rlm_fixture_{tmp_path.name.replace('-', '_')}"
    pkg_dir = tmp_path / package_name
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "taskset_mod.py").write_text("MARKER = 1\n")

    if with_skills:
        skill_dir = pkg_dir / "skills" / "demo"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: demo\n---\n")
        (skill_dir / "pyproject.toml").write_text(
            "[project]\nname = 'rlm-skill-demo'\nversion = '0.0.0'\n"
        )

    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    mod = importlib.import_module(f"{package_name}.taskset_mod")
    monkeypatch.setattr(MockSandboxTaskSetWithSkills, "__module__", mod.__name__)
    return mod


# ── RLM harness ──────────────────────────────────────────────────────────


def test_rlm_harness_install_script_downloads_repo_install_sh():
    script = build_install_script()
    assert "git clone --depth 1 --branch main" in script
    assert "github.com/PrimeIntellect-ai/rlm.git" in script
    assert "bash /tmp/rlm-checkout/install.sh" in script


def test_rlm_harness_sets_metrics_fields():
    harness = rlm_harness()
    assert harness.metrics_path == "{workdir}/.rlm/sessions/*/meta.json"
    assert harness.metrics_key == "metrics"
    assert harness.metrics_prefix == "rlm_"


def test_rlm_harness_sets_skills_path():
    harness = rlm_harness()
    assert harness.skills_path == "/task/rlm-skills"


# ── install_env ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rlm_install_runs_without_skills(tmp_path, monkeypatch):
    _make_temp_taskset_package(tmp_path, monkeypatch, with_skills=False)
    taskset = MockSandboxTaskSetWithSkills(dataset=_make_dataset(), name="test")
    env = ComposableEnv(
        taskset=taskset,
        harness=Harness(
            run_command="true",
            install_script="install-rlm",
            instruction_path="/tmp/with space/prompt.txt",
            system_prompt="system",
            system_prompt_path="/tmp/other path/system.txt",
            skills_path="/task/rlm-skills",
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
    env.upload_file = AsyncMock()

    await env.post_sandbox_setup({"sandbox_id": "sbx", "info": {"id": 0}})

    assert env.upload_file.await_count == 0
    assert env.sandbox_client.execute_command.await_args_list == [
        call(
            "sbx",
            "mkdir -p '/tmp/other path' '/tmp/with space'",
            timeout=10,
        ),
        call(
            "sbx",
            "install-rlm",
            timeout=300,
            env={"GH_TOKEN": "secret"},
        ),
    ]


# ── Skills upload ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rlm_uploads_skills_before_install(tmp_path, monkeypatch):
    _make_temp_taskset_package(tmp_path, monkeypatch, with_skills=True)
    taskset = MockSandboxTaskSetWithSkills(dataset=_make_dataset(), name="test")
    env = ComposableEnv(
        taskset=taskset,
        harness=Harness(
            run_command="true",
            install_script="install-rlm",
            skills_path="/task/rlm-skills",
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
    assert upload_call.args[1] == "/tmp/_upload_task_rlm-skills.tar.gz"

    install_call = env.sandbox_client.execute_command.await_args_list[-1]
    assert install_call == call("sbx", "install-rlm", timeout=300)
    extract_call = env.sandbox_client.execute_command.await_args_list[1]
    assert extract_call == call(
        "sbx",
        "mkdir -p /task && tar -xzf /tmp/_upload_task_rlm-skills.tar.gz -C / && rm -f /tmp/_upload_task_rlm-skills.tar.gz",
        timeout=60,
    )


# ── RLM metrics via harness fields ──────────────────────────────────────


@pytest.mark.asyncio
async def test_rlm_collects_logs_and_metrics():
    taskset = MockSandboxTaskSet(dataset=_make_dataset(), name="test")
    metrics = {
        "turns": 3,
        "stop_reason": "done",
        "prompt_tokens": 100,
        "completion_tokens": 25,
    }
    harness = rlm_harness()
    env = ComposableEnv(
        taskset=taskset,
        harness=Harness(
            run_command=harness.run_command,
            log_path="/tmp/log dir/agent.log",
            metrics_path=harness.metrics_path,
            metrics_key=harness.metrics_key,
            metrics_prefix=harness.metrics_prefix,
        ),
    )
    env.sandbox_client = SimpleNamespace(
        execute_command=AsyncMock(
            side_effect=[
                SimpleNamespace(stdout="agent log\n", stderr="", exit_code=0),
                SimpleNamespace(
                    stdout=json.dumps({"metrics": metrics}),
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

    assert env.sandbox_client.execute_command.await_args_list == [
        call(
            "sbx",
            "cat '/tmp/log dir/agent.log' 2>/dev/null || echo '<no logs>'",
            working_dir=None,
        ),
        call(
            "sbx",
            'f=$(ls /testbed/.rlm/sessions/*/meta.json 2>/dev/null | head -1) && cat "$f" || echo "{}"',
            working_dir=None,
        ),
    ]
    assert state["agent_logs"] == "agent log"
    assert state["rlm_turns"] == 3
    assert state["rlm_stop_reason"] == "done"
    assert state["rlm_prompt_tokens"] == 100
    assert state["rlm_completion_tokens"] == 25
