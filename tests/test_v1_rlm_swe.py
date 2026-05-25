import base64
import io
import inspect
import sys
import tarfile
import types
from collections.abc import Mapping
from pathlib import Path

import pytest
from datasets import Dataset
from verifiers.types import Tool

import verifiers.v1 as vf
from environments.rlm_swe_v1 import rlm_swe_v1
from verifiers.v1.packages.harnesses.rlm import (
    DEFAULT_RLM_TOOL_SKILL_MARKER,
    DEFAULT_RLM_TOOL_SKILLS_ARCHIVE_PATH,
    DEFAULT_RLM_TOOL_SKILLS_MANIFEST_NAME,
)
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


def test_rlm_harness_preserves_program_setup_timeout_override():
    harness = vf.RLM(
        config=vf.RLMConfig(
            local_checkout="/tmp/checkout",
            program={"setup_timeout": 123},
        )
    )
    program = as_mapping(harness.program)

    assert program["setup_timeout"] == 123


def test_rlm_harness_uses_sandbox_setup_timeout_default():
    harness = vf.RLM(
        config=vf.RLMConfig(
            local_checkout="/tmp/checkout",
            sandbox={"setup_timeout": "777"},
        )
    )
    program = as_mapping(harness.program)

    assert program["setup_timeout"] == 777


def test_rlm_harness_keeps_minimum_setup_timeout_for_default_sandbox_config():
    harness = vf.RLM(
        config=vf.RLMConfig(
            local_checkout="/tmp/checkout",
            sandbox=vf.SandboxConfig(),
        )
    )
    program = as_mapping(harness.program)
    sandbox = as_mapping(harness.sandbox)

    assert program["setup_timeout"] == 600
    assert sandbox["setup_timeout"] == 600


def test_rlm_harness_can_upload_skills(tmp_path: Path):
    skills = tmp_path / "skills"
    (skills / "edit").mkdir(parents=True)
    (skills / "edit" / "SKILL.md").write_text("---\nname: edit\n---\n")

    harness = vf.RLM(
        config=vf.RLMConfig(local_checkout="/tmp/checkout", skills=str(skills))
    )
    program = as_mapping(harness.program)
    dirs = as_mapping(program["dirs"])
    files = as_mapping(program["files"])
    setup = program["setup"]

    assert dirs["/task/rlm-skills"] == skills
    assert files[DEFAULT_RLM_TOOL_SKILLS_ARCHIVE_PATH] == harness.vf_tool_skills_archive
    assert isinstance(setup, list)
    assert DEFAULT_RLM_TOOL_SKILLS_ARCHIVE_PATH in setup[1]
    assert DEFAULT_RLM_TOOL_SKILLS_MANIFEST_NAME in setup[1]
    assert DEFAULT_RLM_TOOL_SKILL_MARKER in setup[1]
    assert "rm -rf" in setup[1]
    assert "tar -tzf" in setup[1]


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

    assert dirs["/task/rlm-skills"] == skills


def test_rlm_harness_recomputes_taskset_skills(tmp_path: Path):
    first_skills = tmp_path / "first-skills"
    second_skills = tmp_path / "second-skills"
    first_skills.mkdir()
    second_skills.mkdir()

    class SkillTaskset(vf.Taskset):
        def __init__(self, skills: Path):
            super().__init__(config=vf.TasksetConfig(source=[]))
            self.skills = skills

        def get_upload_dirs(self):
            return {"skills": self.skills}

    class NoSkillTaskset(vf.Taskset):
        def get_upload_dirs(self):
            return {}

    harness = vf.RLM(config=vf.RLMConfig(local_checkout="/tmp/checkout"))
    vf.Env(taskset=SkillTaskset(first_skills), harness=harness)
    vf.Env(taskset=SkillTaskset(second_skills), harness=harness)
    program = as_mapping(harness.program)
    dirs = as_mapping(program["dirs"])

    assert dirs["/task/rlm-skills"] == second_skills

    vf.Env(taskset=NoSkillTaskset(config=vf.TasksetConfig(source=[])), harness=harness)
    program = as_mapping(harness.program)
    dirs = as_mapping(program["dirs"])

    assert "/task/rlm-skills" not in dirs


@pytest.mark.asyncio
async def test_rlm_harness_generates_skills_for_v1_tools():
    async def lookup_order(order_id: str) -> str:
        """Look up an order by ID."""
        return f"order:{order_id}"

    taskset = vf.Taskset(
        config=vf.TasksetConfig(
            source=[{"prompt": [{"role": "user", "content": "Find order A-1."}]}]
        )
    )
    taskset.add_toolset(vf.Toolset(tools=[lookup_order]))
    env = vf.Env(
        taskset=taskset,
        harness=vf.RLM(config=vf.RLMConfig(local_checkout="/tmp/checkout")),
    )
    task = next(iter(env.taskset))
    state = vf.State({"task": dict(task), "runtime": {}, "prompt": task["prompt"]})

    await env.harness.runtime.ensure_rollout_toolsets(task, state)
    env.harness.runtime.prepare_state(task, state)
    archive = base64.b64decode(env.harness.vf_tool_skills_archive(state))

    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        source = (
            tar.extractfile("lookup_order/src/lookup_order/lookup_order.py")
            .read()
            .decode()
        )
        skill_markdown = tar.extractfile("lookup_order/SKILL.md").read().decode()
        marker = (
            tar.extractfile(f"lookup_order/{DEFAULT_RLM_TOOL_SKILL_MARKER}")
            .read()
            .decode()
        )
    assert "async def run(order_id: str, **kwargs) -> object" in source
    assert "/vf/tools/" not in source
    assert "dill.loads" in source
    assert "Look up an order by ID." in skill_markdown
    assert "result = await lookup_order" in skill_markdown
    assert marker == "1\n"

    module = types.ModuleType("lookup_order")
    exec(source, module.__dict__)
    assert await module.run(order_id="A-1") == "order:A-1"


@pytest.mark.asyncio
async def test_vf_tool_skill_falls_back_for_runtime_bound_tools():
    async def stateful_lookup(order_id: str, state: vf.State) -> str:
        """Look up an order with rollout state."""
        return f"{state['tenant']}:{order_id}"

    taskset = vf.Taskset(
        config=vf.TasksetConfig(
            source=[{"prompt": [{"role": "user", "content": "Find order A-1."}]}]
        )
    )
    taskset.add_toolset(vf.Toolset(tools=[stateful_lookup]))
    env = vf.Env(
        taskset=taskset,
        harness=vf.RLM(config=vf.RLMConfig(local_checkout="/tmp/checkout")),
    )
    task = next(iter(env.taskset))
    state = vf.State({"task": dict(task), "runtime": {}, "prompt": task["prompt"]})

    await env.harness.runtime.ensure_rollout_toolsets(task, state)
    env.harness.runtime.prepare_state(task, state)
    archive = base64.b64decode(env.harness.vf_tool_skills_archive(state))

    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        source = (
            tar.extractfile("stateful_lookup/src/stateful_lookup/stateful_lookup.py")
            .read()
            .decode()
        )

    assert "/vf/tools/" in source
    assert "dill.loads" not in source


def test_vf_tool_skills_archive_avoids_base_skill_name_collisions(tmp_path: Path):
    skills = tmp_path / "skills"
    (skills / "lookup_order").mkdir(parents=True)
    harness = vf.RLM(
        config=vf.RLMConfig(local_checkout="/tmp/checkout", skills=str(skills))
    )
    tool_def = Tool(
        name="lookup_order",
        description="Look up an order.",
        parameters={"type": "object", "properties": {}},
    )
    setattr(harness.runtime, "tool_defs", lambda state: [tool_def])

    archive = base64.b64decode(harness.vf_tool_skills_archive(vf.State({})))

    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        names = tar.getnames()
        source = (
            tar.extractfile("lookup_order_2/src/lookup_order_2/lookup_order_2.py")
            .read()
            .decode()
        )

    assert "lookup_order_2/SKILL.md" in names
    assert "/vf/tools/" in source
    assert "'lookup_order'" in source


def test_vf_tool_skill_uses_arguments_dict_for_tool_parameters():
    harness = vf.RLM(config=vf.RLMConfig(local_checkout="/tmp/checkout"))
    tool_def = Tool(
        name="reserved_param",
        description="Reserved parameter.",
        parameters={
            "type": "object",
            "properties": {
                "_call_vf_tool": {"type": "string"},
                "limit": {"type": "integer", "default": 10},
            },
            "required": ["_call_vf_tool"],
        },
    )
    setattr(harness.runtime, "tool_defs", lambda state: [tool_def])
    archive = base64.b64decode(harness.vf_tool_skills_archive(vf.State({})))

    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        source = (
            tar.extractfile("reserved_param/src/reserved_param/reserved_param.py")
            .read()
            .decode()
        )

    assert "async def run(arguments: dict | None = None, **kwargs) -> object" in source
    assert "arguments = {**(arguments or {}), **kwargs}" in source
    assert 'json={"arguments": arguments}' in source
    assert "def _tool_arguments" not in source
    assert "limit=None" not in source


@pytest.mark.asyncio
async def test_vf_tool_skill_filters_extra_kwargs_for_closed_schemas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = vf.RLM(config=vf.RLMConfig(local_checkout="/tmp/checkout"))
    tool_def = Tool(
        name="list_events",
        description="List calendar events.",
        parameters={
            "type": "object",
            "properties": {"date": {"type": "string"}},
            "required": ["date"],
            "additionalProperties": False,
        },
    )
    setattr(harness.runtime, "tool_defs", lambda state: [tool_def])
    archive = base64.b64decode(harness.vf_tool_skills_archive(vf.State({})))

    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        source = (
            tar.extractfile("list_events/src/list_events/list_events.py")
            .read()
            .decode()
        )

    module = types.ModuleType("list_events")
    exec(source, module.__dict__)
    calls: list[dict[str, object]] = []

    class Response:
        content = b"{}"

        def json(self):
            return {"result": "ok"}

        def raise_for_status(self):
            return None

    class Requests:
        @staticmethod
        def post(url, json, headers, timeout):
            calls.append(json)
            return Response()

    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    module.requests = Requests

    assert list(inspect.signature(module.run).parameters) == ["date", "kwargs"]
    result = await module.run(user="Alice", date="2025-07-14")

    assert result == "ok"
    assert calls == [{"arguments": {"date": "2025-07-14"}}]


@pytest.mark.asyncio
async def test_vf_tool_skill_omits_unset_optional_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = vf.RLM(config=vf.RLMConfig(local_checkout="/tmp/checkout"))
    tool_def = Tool(
        name="search",
        description="Search documents.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        },
    )
    setattr(harness.runtime, "tool_defs", lambda state: [tool_def])
    archive = base64.b64decode(harness.vf_tool_skills_archive(vf.State({})))

    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        source = tar.extractfile("search/src/search/search.py").read().decode()

    module = types.ModuleType("search")
    exec(source, module.__dict__)
    calls: list[dict[str, object]] = []

    class Response:
        content = b"{}"

        def json(self):
            return {"result": "ok"}

        def raise_for_status(self):
            return None

    class Requests:
        @staticmethod
        def post(url, json, headers, timeout):
            calls.append(json)
            return Response()

    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    module.requests = Requests

    assert (
        "async def run(query: str, limit: int | None = None, **kwargs) -> object"
        in source
    )
    assert "if limit is not None:" in source
    assert "limit=None" not in source
    result = await module.run(query="docs")

    assert result == "ok"
    assert calls == [{"arguments": {"query": "docs"}}]


@pytest.mark.asyncio
async def test_vf_tool_skill_surfaces_verifier_tool_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = vf.RLM(config=vf.RLMConfig(local_checkout="/tmp/checkout"))
    tool_def = Tool(
        name="list_events",
        description="List calendar events.",
        parameters={
            "type": "object",
            "properties": {"date": {"type": "string"}},
            "required": ["date"],
            "additionalProperties": False,
        },
    )
    setattr(harness.runtime, "tool_defs", lambda state: [tool_def])
    archive = base64.b64decode(harness.vf_tool_skills_archive(vf.State({})))

    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        source = (
            tar.extractfile("list_events/src/list_events/list_events.py")
            .read()
            .decode()
        )

    module = types.ModuleType("list_events")
    exec(source, module.__dict__)

    class Response:
        content = b"{}"

        def json(self):
            return {"error": "unexpected keyword argument 'user'"}

        def raise_for_status(self):
            raise AssertionError("JSON tool errors should be raised first")

    class Requests:
        @staticmethod
        def post(url, json, headers, timeout):
            return Response()

    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    module.requests = Requests

    with pytest.raises(RuntimeError, match="unexpected keyword argument"):
        await module.run(date="2025-07-14")


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

    assert dirs["/task/rlm-skills"] == explicit_skills


def test_rlm_swe_environment_uses_v1_r2e_taskset(monkeypatch):
    calls: dict[str, object] = {}

    def fake_load_dataset(dataset_name: str, **kwargs: object) -> Dataset:
        calls["dataset_name"] = dataset_name
        calls["kwargs"] = kwargs
        return fake_r2e_dataset()

    monkeypatch.setattr(rlm_swe_v1, "load_dataset", fake_load_dataset)

    env = rlm_swe_v1.load_environment(
        config=rlm_swe_v1.RlmSweEnvConfig(
            taskset=rlm_swe_v1.RlmSweTasksetConfig(
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
    assert calls["dataset_name"] == "fake-r2e"
    assert task["taskset_id"] == "swe/r2e"
    assert task["instruction"] == "Fix repo-0."
    assert task["sandbox"]["image"] == (
        f"{rlm_swe_v1.REGISTRY_PREFIX}/r2e/image:latest"
    )
    assert task["sandbox"]["workdir"] == "/workspace/repo"
    assert task["sandbox"]["timeout_minutes"] == 30
    task_program_env = as_mapping(as_mapping(task["program"])["env"])
    assert task_program_env["AGENT_WORKDIR"] == "/workspace/repo"
    assert "/workspace/repo/.venv/bin" in task_program_env["AGENT_PATH"]
    assert task_program_env["PAGER"] == "cat"
    assert task_program_env["CUSTOM"] == "1"
    assert "CUSTOM" not in program_env
    assert program_env["CALLER"] == "1"
    assert program_env["RLM_TOOLS"] == "bash,edit"
    assert merged_sandbox["workdir"] == "/workspace/repo"
    assert merged_env["AGENT_WORKDIR"] == "/workspace/repo"
    assert "/workspace/repo/.venv/bin" in merged_env["AGENT_PATH"]
    assert merged_env["PAGER"] == "cat"
    assert merged_env["CUSTOM"] == "1"
    assert merged_env["CALLER"] == "1"


def test_rlm_swe_taskset_hooks_are_registered_with_runtime():
    taskset = rlm_swe_v1.load_taskset(config=rlm_swe_v1.RlmSweTasksetConfig())
    env = vf.Env(taskset=taskset)

    setup_names = [handler.__name__ for handler in env.harness.runtime.rollout_setup]
    cleanup_names = [
        handler.__name__ for handler in env.harness.runtime.rollout_cleanup
    ]
    signal_names = {signal["name"] for signal in env.harness.runtime.rollout_signals}

    assert setup_names.count("setup_r2e_sandbox") == 1
    assert cleanup_names.count("cleanup_r2e_state") == 1
    assert "solved" in signal_names


@pytest.mark.asyncio
async def test_rlm_swe_taskset_setup_and_reward(monkeypatch):
    monkeypatch.setattr(
        rlm_swe_v1, "load_dataset", lambda *args, **kwargs: fake_r2e_dataset()
    )
    taskset = rlm_swe_v1.load_taskset(
        config=rlm_swe_v1.RlmSweTasksetConfig(timeout_minutes=30)
    )
    task = next(iter(taskset))
    state = vf.State.for_task(task)
    sandbox = FakeSandbox()
    calls: dict[str, object] = {}

    async def fake_setup_sandbox(sandbox_arg: object, state_arg: vf.State) -> None:
        calls["setup_sandbox"] = sandbox_arg
        calls["setup_state"] = state_arg

    async def fake_run_tests(
        sandbox_arg: object,
        state_arg: vf.State,
        test_timeout: int,
    ) -> str:
        calls["run_tests"] = (sandbox_arg, state_arg, test_timeout)
        return """
=========================== short test summary info ============================
PASSED tests/test_example.py::test_fix
"""

    monkeypatch.setattr(taskset, "setup_sandbox", fake_setup_sandbox)
    monkeypatch.setattr(taskset, "run_tests", fake_run_tests)

    await taskset.setup_r2e_sandbox(task, state, sandbox=sandbox)
    reward = await taskset.solved(task, state)
    await taskset.cleanup_r2e_state(task, state)

    assert calls["setup_sandbox"] is sandbox
    assert calls["setup_state"] is state
    assert calls["run_tests"] == (sandbox, state, 1800)
    assert state["sandbox_id"] == "sandbox-1"
    assert state["test_timeout"] == 1800
    assert reward == 1.0
    assert "sandbox_client" not in state
    assert "_rlm_swe_sandbox" not in state


@pytest.mark.asyncio
async def test_rlm_swe_run_tests_quotes_env_values():
    taskset = rlm_swe_v1.load_taskset(
        config=rlm_swe_v1.RlmSweTasksetConfig(
            hide_tests_from_agent=False,
            env={"SAFE": "two words; $(echo nope)", "QUOTE": "it's ok"},
        )
    )
    sandbox = RecordingSandbox()

    output = await taskset.run_tests(sandbox, {}, 123)

    assert output == "test output"
    assert len(sandbox.background_jobs) == 1
    command = sandbox.background_jobs[0]["command"]
    assert "SAFE='two words; $(echo nope)'" in command
    assert "QUOTE='it'\"'\"'s ok'" in command
    assert command.endswith("/bin/bash run_tests.sh > test_output.txt 2>&1")


def test_rlm_swe_get_env_vars_uses_configured_repo_path():
    taskset = rlm_swe_v1.load_taskset(
        config=rlm_swe_v1.RlmSweTasksetConfig(repo_path="/workspace/repo")
    )

    path = taskset.get_env_vars()["PATH"]

    assert "/workspace/repo/.venv/bin" in path
    assert "/testbed/.venv/bin" not in path


def test_rlm_swe_reward_rejects_pytest_summary_without_nodeid():
    taskset = rlm_swe_v1.load_taskset(config=rlm_swe_v1.RlmSweTasksetConfig())
    test_output = """
=========================== short test summary info ============================
PASSED tests/test_example.py
"""

    reward = taskset.calculate_reward(
        test_output,
        {"expected_output_json": '{"test_fix": "PASSED"}'},
    )
    parsed = rlm_swe_v1.parse_log_pytest(test_output)

    assert reward == 0.0
    assert "" not in parsed


def test_rlm_swe_parse_log_pytest_uses_leading_status_token():
    test_output = """
=========================== short test summary info ============================
FAILED tests/test_PASSED_handler.py::test_fix - AssertionError
PASSED tests/test_failed_handler.py::test_other
ERROR tests/test_failed_handler.py::test_error - setup failed
"""

    parsed = rlm_swe_v1.parse_log_pytest(test_output)

    assert parsed == {
        "test_fix": "FAILED",
        "test_other": "PASSED",
        "test_error": "ERROR",
    }


def fake_r2e_dataset() -> Dataset:
    return Dataset.from_list(
        [
            {
                "commit_hash": f"commit-{index}",
                "repo_name": "example/repo",
                "problem_statement": f"Fix repo-{index}.",
                "docker_image": "r2e/image:latest",
                "expected_output_json": '{"test_fix": "PASSED"}',
                "parsed_commit_content": '{"file_diffs": []}',
            }
            for index in range(12)
        ]
    )


class FakeLease:
    client = object()


class FakeSandbox:
    id = "sandbox-1"
    lease = FakeLease()


class FakeCommandResult:
    def __init__(
        self,
        stdout: str = "",
        stderr: str = "",
        exit_code: int = 0,
    ):
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code


class RecordingSandbox:
    def __init__(self):
        self.background_jobs: list[dict[str, object]] = []
        self.commands: list[dict[str, object]] = []

    async def run_background_job(
        self,
        command: str,
        timeout: int | None = None,
        working_dir: str | None = None,
    ) -> FakeCommandResult:
        self.background_jobs.append(
            {
                "command": command,
                "timeout": timeout,
                "working_dir": working_dir,
            }
        )
        return FakeCommandResult()

    async def execute(
        self,
        command: str,
        timeout: int | None = None,
        working_dir: str | None = None,
    ) -> FakeCommandResult:
        self.commands.append(
            {
                "command": command,
                "timeout": timeout,
                "working_dir": working_dir,
            }
        )
        return FakeCommandResult(stdout="test output")
