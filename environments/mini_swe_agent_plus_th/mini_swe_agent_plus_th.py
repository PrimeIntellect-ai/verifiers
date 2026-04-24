from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import re
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, Literal, cast

from datasets import Dataset, load_dataset
from jinja2 import StrictUndefined, Template
from prime_sandboxes import CommandTimeoutError, SandboxOOMError, SandboxTimeoutError

import verifiers as vf
from verifiers.envs.experimental.channels import SandboxResources

logger = logging.getLogger(__name__)

LoadedSource = Dataset | Iterable[Mapping[str, Any]] | None
Source = LoadedSource | Callable[[], LoadedSource]

REPO_PATH = "/testbed"
ALT_PATH = "/root"
PATH_ENV = (
    "PATH=/opt/miniconda3/bin:/testbed/.venv/bin:/root/.local/bin:"
    "/root/.cargo/bin:/go/bin:/usr/local/go/bin:/usr/local/cargo:"
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)
ENV_VARS = f"export {PATH_ENV} PAGER=cat MANPAGER=cat LESS=-R PIP_PROGRESS_BAR=off TQDM_DISABLE=1;"

PROMPT_TEMPLATE = """<pr_description>

Consider the following PR description:

{problem_statement}

</pr_description>

<instructions>

# Task Instructions

You're a software engineer interacting continuously with a computer by submitting tool calls.

Make the minimal non-test source-code changes needed to fix the issue. Each assistant response must include exactly one tool call. When you are finished, call:

echo MINI_SWE_AGENT_FINAL_OUTPUT

</instructions>"""

SYSTEM_PROMPT = """You are a helpful assistant that can interact multiple times with tools to solve programming tasks.

Your response must contain exactly ONE tool call with the tool name and arguments."""

ACTION_OBSERVATION_TEMPLATE = """<returncode>{{ exit_code }}</returncode>
{% if output | length < 10000 -%}
<output>
{{ output -}}
</output>
{%- else -%}
<warning>
The output of your last command was too long.
Use a more selective command.
</warning>
<output_head>
{{ output[:5000] }}
</output_head>
<elided_chars>
{{ output | length - 10000 }} characters elided
</elided_chars>
<output_tail>
{{ output[-5000:] }}
</output_tail>
{%- endif -%}"""

FORMAT_ERROR_TEMPLATE = (
    """Please always provide exactly one tool call. Found {{ actions }} tool calls."""
)


def render_template(template: str, **kwargs: object) -> str:
    return Template(template, undefined=StrictUndefined).render(**kwargs)


def decolor_dict_keys(value: dict[str, str]) -> dict[str, str]:
    return {re.sub(r"\u001b\[\d+m", "", key): item for key, item in value.items()}


def parse_log_pytest(log: str | None) -> dict[str, str]:
    if log is None or "short test summary info" not in log:
        return {}
    status_map: dict[str, str] = {}
    for line in log.split("short test summary info", 1)[1].strip().splitlines():
        if "PASSED" in line:
            status_map[".".join(line.split("::")[1:])] = "PASSED"
        elif "FAILED" in line:
            status_map[".".join(line.split("::")[1:]).split(" - ")[0]] = "FAILED"
        elif "ERROR" in line:
            status_map[".".join(line.split("::")[1:]).split(" - ")[0] or line] = "ERROR"
    return status_map


def parse_log_fn(repo_name: str):
    return parse_log_pytest


def swebench_constants():
    return importlib.import_module("swebench.harness.constants")


def swebench_grading():
    return importlib.import_module("swebench.harness.grading")


def swebench_log_parsers():
    return importlib.import_module("swebench.harness.log_parsers")


def make_swebench_test_spec(info: Mapping[str, Any]):
    module = importlib.import_module("swebench.harness.test_spec.test_spec")
    return module.make_test_spec(info, namespace="swebench")


def get_logs_eval(test_spec: object, content: str) -> tuple[dict[str, str], bool]:
    constants = swebench_constants()

    repo = str(getattr(test_spec, "repo"))
    version = str(getattr(test_spec, "version"))
    test_cmd = constants.MAP_REPO_VERSION_TO_SPECS[repo][version]["test_cmd"]
    if isinstance(test_cmd, list):
        test_cmd = test_cmd[-1]
    bad_codes = [
        constants.APPLY_PATCH_FAIL,
        constants.RESET_FAILED,
        constants.TESTS_ERROR,
        constants.TESTS_TIMEOUT,
    ]
    if any(code in content for code in bad_codes):
        return {}, False
    parser = swebench_log_parsers().MAP_REPO_TO_PARSER[repo]
    return parser(content.split(test_cmd)[-1], test_spec), True


def process_example(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "question": PROMPT_TEMPLATE.format(problem_statement=row["problem_statement"]),
        "info": dict(row),
        "answer": "",
    }


def get_harness(dataset_name: str) -> str:
    if dataset_name.lower().startswith("r2e-gym/"):
        return "r2e"
    return "swebench"


class MiniSweTaskset(vf.Taskset):
    def __init__(
        self,
        source: Source,
        harness: Literal["r2e", "swebench"],
        labels: list[str],
        rubric: vf.Rubric,
    ):
        self.harness = harness
        self.labels = labels
        super().__init__(source=source, rubric=rubric, name="mini-swe-agent-plus-th")

    def channels(self, task: vf.Task | None = None) -> vf.ChannelMap:
        channels = super().channels(task)
        if task is None:
            return channels
        info = dict(task.info)
        image = str(info.get("docker_image") or "")
        if self.harness == "swebench":
            image = make_swebench_test_spec(info).instance_image_key
        channels["sandbox"] = {
            "spec": vf.SandboxSeed(
                image=f"us-central1-docker.pkg.dev/prime-intellect-platform/prod-sandbox/{image}",
                labels=[*self.labels, f"mini-swe-task:{task.example_id}"],
            )
        }
        return channels


class MiniSweMonitorRubric(vf.Rubric):
    def __init__(self):
        super().__init__()
        self.add_metric(self.command_timeout_count)
        self.add_metric(self.rollout_duration_seconds)
        self.add_metric(self.sandbox_oom)
        self.add_metric(self.sandbox_timeout)

    async def command_timeout_count(self, state: vf.State) -> int:
        return int(state.get("command_timeout_count", 0))

    async def rollout_duration_seconds(self, state: vf.State) -> float:
        return time.time() - state["timing"]["start_time"]

    async def sandbox_oom(self, state: vf.State) -> float:
        return float(bool(state.get("sandbox_oom")))

    async def sandbox_timeout(self, state: vf.State) -> float:
        return float(bool(state.get("sandbox_timeout")))


class MiniSweRubric(vf.Rubric):
    def __init__(self, harness: Literal["r2e", "swebench"]):
        super().__init__()
        self.harness = harness
        self.add_reward_func(self.solved)

    def solved(self, state: vf.State, info: vf.Info) -> int:
        if state.get("error") is not None:
            return 0
        if state.get("max_command_timeouts_reached"):
            return 0
        if self.harness == "swebench":
            return self.swebench_reward(state, info)
        return self.r2e_reward(state, info)

    def swebench_reward(self, state: vf.State, info: vf.Info) -> int:
        output = str(state.get("test_output") or "")
        constants = swebench_constants()
        grading = swebench_grading()
        test_spec = make_swebench_test_spec(info)
        eval_status_map, _ = get_logs_eval(test_spec, output)
        eval_ref = {
            constants.KEY_INSTANCE_ID: test_spec.instance_id,
            constants.FAIL_TO_PASS: test_spec.FAIL_TO_PASS,
            constants.PASS_TO_PASS: test_spec.PASS_TO_PASS,
        }
        eval_type = (
            constants.EvalType.FAIL_ONLY
            if test_spec.repo in constants.FAIL_ONLY_REPOS
            else constants.EvalType.PASS_AND_FAIL
        )
        report = grading.get_eval_tests_report(
            eval_status_map,
            eval_ref,
            eval_type=eval_type,
        )
        return int(
            grading.get_resolution_status(report) == constants.ResolvedStatus.FULL.value
        )

    def r2e_reward(self, state: vf.State, info: vf.Info) -> int:
        output = str(state.get("test_output") or "")
        parsed = decolor_dict_keys(parse_log_fn(info["repo_name"])(output))
        expected = decolor_dict_keys(json.loads(info["expected_output_json"]))
        parsed = {key.split(" - ")[0]: parsed[key] for key in sorted(parsed)}
        expected = {key.split(" - ")[0]: expected[key] for key in sorted(expected)}
        return int(bool(parsed) and parsed == expected)


class MiniSweHarness(vf.Harness):
    def __init__(
        self,
        system_prompt: str | None = SYSTEM_PROMPT,
        sandbox: vf.SandboxSpec | None = None,
        harness: Literal["r2e", "swebench"] = "r2e",
        sandbox_command_timeout: int = 90,
        test_timeout: int = 900,
        total_timeout_minutes: int = 360,
        rollout_timeout_seconds: float = 5400.0,
        max_command_timeouts: int = 5,
        allow_git: bool = False,
        sandbox_wait_for_creation_max_attempts: int = 120,
        sandbox_client_max_workers: int = 64,
        labels: list[str] | None = None,
        max_turns: int = 200,
        rubric: vf.Rubric | None = None,
        tools: Iterable[object] | None = None,
    ):
        rubrics: list[vf.Rubric] = [MiniSweMonitorRubric()]
        if rubric is not None:
            rubrics.insert(0, rubric)
        rubric = vf.RubricGroup(rubrics)
        base_tools = [self.execute_bash, self.edit_via_str_replace]
        super().__init__(
            rubric=rubric,
            system_prompt=system_prompt,
            tools=[*base_tools, *(tools or [])],
            max_turns=max_turns,
        )
        self.sandbox_spec = sandbox or vf.SandboxSpec(
            cpu_cores=4,
            memory_gb=4,
            disk_size_gb=2,
            timeout_minutes=total_timeout_minutes,
            labels=labels or ["mini-swe-agent-plus-th"],
        )
        self.harness = harness
        self.sandbox_command_timeout = sandbox_command_timeout
        self.test_timeout = test_timeout
        self.rollout_timeout_seconds = rollout_timeout_seconds
        self.max_command_timeouts = max_command_timeouts
        self.allow_git = allow_git
        self.sandbox_wait_for_creation_max_attempts = (
            sandbox_wait_for_creation_max_attempts
        )
        self.sandbox_client_max_workers = sandbox_client_max_workers

    def channels(self, task: vf.Task | None = None) -> vf.ChannelMap:
        channels = super().channels(task)
        channels["sandbox"] = {
            "spec": self.sandbox_spec,
            "runtime": {"client_max_workers": self.sandbox_client_max_workers},
        }
        return channels

    async def setup_state(self, task: vf.Task, resources: vf.Resources) -> vf.State:
        state = await super().setup_state(task, resources)
        state["command_timeout_count"] = 0
        state["sandbox_state"] = {"command_execution_times": []}
        await self.setup_sandbox(task, state, resources)
        return state

    def resolve_sandbox_spec(self, resources: vf.Resources) -> vf.SandboxSpec:
        seed = resources.get("sandbox_request")
        spec = self.sandbox_spec
        if isinstance(seed, vf.SandboxSpec):
            return seed
        if not isinstance(seed, vf.SandboxSeed):
            return spec
        return vf.SandboxSpec(
            image=seed.image or spec.image,
            cpu_cores=seed.cpu_cores or spec.cpu_cores,
            memory_gb=seed.memory_gb or spec.memory_gb,
            disk_size_gb=seed.disk_size_gb or spec.disk_size_gb,
            gpu_count=seed.gpu_count if seed.gpu_count is not None else spec.gpu_count,
            network_access=(
                seed.network_access
                if seed.network_access is not None
                else spec.network_access
            ),
            timeout_minutes=seed.timeout_minutes or spec.timeout_minutes,
            start_command=seed.start_command or spec.start_command,
            environment_vars={**spec.environment_vars, **seed.environment_vars},
            labels=[*spec.labels, *seed.labels],
        )

    async def setup_sandbox(
        self, task: vf.Task, state: vf.State, resources: vf.Resources
    ) -> None:
        from prime_sandboxes import CreateSandboxRequest

        spec = self.resolve_sandbox_spec(resources)
        request = CreateSandboxRequest(
            name=f"mini-swe-{task.example_id}-{os.urandom(4).hex()}",
            docker_image=spec.image,
            start_command=spec.start_command,
            cpu_cores=spec.cpu_cores,
            memory_gb=spec.memory_gb,
            disk_size_gb=spec.disk_size_gb,
            gpu_count=spec.gpu_count,
            gpu_type=spec.gpu_type,
            vm=spec.vm if spec.vm is not None else spec.gpu_count > 0,
            network_access=spec.network_access,
            timeout_minutes=spec.timeout_minutes,
            environment_vars=spec.environment_vars,
            secrets=spec.secrets,
            team_id=spec.team_id,
            advanced_configs=spec.advanced_configs,
            registry_credentials_id=spec.registry_credentials_id,
            labels=spec.labels,
        )
        runtime = resources.require("sandbox_runtime", SandboxResources)
        sandbox_id = await runtime.create(
            request,
            max_attempts=self.sandbox_wait_for_creation_max_attempts,
        )
        state["sandbox_id"] = sandbox_id
        await self.setup_repo(state)

    async def setup_repo(self, state: vf.State) -> None:
        if self.harness == "swebench":
            await self.execute_command_raise_on_exit_code(
                state, "ln -s /opt/miniconda3/envs/testbed /root/.venv"
            )
            return
        for command in (
            f"ln -s {REPO_PATH}/.venv {ALT_PATH}/.venv",
            f"ln -s {REPO_PATH}/.venv/bin/python {ALT_PATH}/.local/bin/python",
            f"ln -s {REPO_PATH}/.venv/bin/python {ALT_PATH}/.local/bin/python3",
            f"find {REPO_PATH}/.venv/bin -type f -executable -exec ln -sf {{}} {ALT_PATH}/.local/bin/ \\;",
            f"mv /r2e_tests {ALT_PATH}/r2e_tests",
            f"ln -s {ALT_PATH}/r2e_tests {REPO_PATH}/r2e_tests",
        ):
            await self.execute_command_raise_on_exit_code(state, command)

    async def get_env_messages(
        self, task: vf.Task, state: vf.State, resources: vf.Resources
    ) -> vf.Messages:
        tool_calls = self.get_tool_calls(state)
        if len(tool_calls) > 1:
            return [
                vf.UserMessage(
                    content=render_template(
                        FORMAT_ERROR_TEMPLATE,
                        actions=len(tool_calls),
                    )
                )
            ]
        messages = await super().get_env_messages(task, state, resources)
        for message in messages:
            content = str(getattr(message, "content", ""))
            if "MINI_SWE_AGENT_FINAL_OUTPUT" in content:
                state["agent_signaled_done"] = True
                state["is_completed"] = True
        return messages

    async def _execute_command(
        self,
        state: vf.State,
        command: str,
        timeout: int | None = None,
        working_dir: str | None = None,
    ) -> tuple[int, str]:
        resources = cast(vf.Resources, self.resources)
        runtime = resources.require("sandbox_runtime", SandboxResources)
        start = time.time()
        try:
            result = await runtime.with_retry(runtime.client.execute_command)(
                state["sandbox_id"],
                command,
                timeout=timeout or self.sandbox_command_timeout,
                working_dir=working_dir,
            )
        except (SandboxOOMError, SandboxTimeoutError) as e:
            state[
                "sandbox_oom" if isinstance(e, SandboxOOMError) else "sandbox_timeout"
            ] = True
            raise vf.SandboxError(f"Sandbox failed while executing command: {e}") from e
        except CommandTimeoutError:
            state["command_timeout_count"] = (
                int(state.get("command_timeout_count", 0)) + 1
            )
            state["sandbox_state"]["command_execution_times"].append(
                time.time() - start
            )
            return (
                -1,
                f"The last command <command>{command}</command> timed out and has been killed.\n"
                "Try another command and avoid interactive input.",
            )
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        parts = [stdout] if stdout else []
        if stderr:
            parts.append(f"stderr:\n{stderr}")
        state["sandbox_state"]["command_execution_times"].append(time.time() - start)
        return result.exit_code, "\n".join(parts) if parts else "(no output)"

    async def execute_command_raise_on_exit_code(
        self,
        state: vf.State,
        command: str,
        working_dir: str | None = None,
        timeout: int | None = None,
    ):
        exit_code, output = await self._execute_command(
            state, command, timeout, working_dir
        )
        if exit_code != 0:
            raise vf.SandboxError(f"Error executing command: {command}\n{output}")

    async def execute_bash(self, command: str, state: vf.State) -> str:
        """Execute a bash command in the task sandbox."""
        blocked = {"ipython", "jupyter", "nohup"}
        if not self.allow_git:
            blocked.add("git")
        for segment in re.split(r"&&|\|\||;|\|", command):
            first = segment.strip().split()[0] if segment.strip() else ""
            if first in blocked:
                return f"Bash command '{first}' is not allowed. Please use a different command or tool."
        exit_code, output = await self._execute_command(
            state,
            f"{ENV_VARS} {command}",
            timeout=self.sandbox_command_timeout,
            working_dir=REPO_PATH,
        )
        if exit_code == -1:
            return output
        return render_template(
            ACTION_OBSERVATION_TEMPLATE,
            exit_code=exit_code,
            output=output,
        )

    async def edit_via_str_replace(
        self,
        path: str,
        old_str: str,
        new_str: str,
        state: vf.State,
        context_lines: int = 3,
        encoding: str = "utf-8",
    ) -> str:
        """Replace one exact string occurrence in a file inside the task sandbox."""
        resources = cast(vf.Resources, self.resources)
        runtime = resources.require("sandbox_runtime", SandboxResources)
        target = f"{REPO_PATH}/{path.lstrip('/')}"
        result = await runtime.with_retry(runtime.client.read_file)(
            state["sandbox_id"], target
        )
        text = result.content
        occurrences = [match.start() for match in re.finditer(re.escape(old_str), text)]
        if not occurrences:
            return (
                f"No replacement performed: old_str did not appear verbatim in {path}."
            )
        if len(occurrences) > 1:
            lines = [text.count("\n", 0, idx) + 1 for idx in occurrences]
            return f"No replacement performed. Multiple occurrences of old_str at lines {lines}."
        start = occurrences[0]
        replacement_line = text.count("\n", 0, start) + 1
        new_content = text[:start] + new_str + text[start + len(old_str) :]
        with tempfile.NamedTemporaryFile("w", delete=False, encoding=encoding) as f:
            f.write(new_content)
            local_path = f.name
        try:
            await runtime.with_retry(runtime.client.upload_file)(
                state["sandbox_id"], target, local_path
            )
        finally:
            Path(local_path).unlink(missing_ok=True)
        lines = new_content.splitlines()
        snippet_start = max(1, replacement_line - context_lines)
        snippet_end = min(
            len(lines), replacement_line + context_lines + new_str.count("\n")
        )
        width = len(str(snippet_end))
        snippet = "\n".join(
            f"{idx:>{width}} | {lines[idx - 1]}"
            for idx in range(snippet_start, snippet_end + 1)
        )
        return f"The file {path} has been edited successfully.\n{snippet}"

    async def run_background_job(
        self,
        state: vf.State,
        command: str,
        timeout: int,
        working_dir: str | None = None,
        poll_interval: int = 3,
    ):
        resources = cast(vf.Resources, self.resources)
        runtime = resources.require("sandbox_runtime", SandboxResources)
        job = await runtime.with_retry(runtime.client.start_background_job)(
            sandbox_id=state["sandbox_id"],
            command=command,
            working_dir=working_dir,
        )
        for _ in range(0, timeout + poll_interval, poll_interval):
            result = await runtime.with_retry(runtime.client.get_background_job)(
                state["sandbox_id"], job
            )
            if result.completed:
                return result
            await asyncio.sleep(poll_interval)
        raise CommandTimeoutError(
            sandbox_id=state["sandbox_id"], command=command, timeout=timeout
        )

    async def run_tests_swebench(self, state: vf.State) -> str:
        test_spec = make_swebench_test_spec(state["info"])
        with tempfile.NamedTemporaryFile(suffix=".sh", mode="w") as eval_file:
            eval_file.write(test_spec.eval_script)
            eval_file.flush()
            resources = cast(vf.Resources, self.resources)
            runtime = resources.require("sandbox_runtime", SandboxResources)
            await runtime.with_retry(runtime.client.upload_file)(
                state["sandbox_id"], "/eval.sh", eval_file.name
            )
        await self.execute_command_raise_on_exit_code(state, "chmod +x /eval.sh")
        result = await self.run_background_job(
            state,
            f"{ENV_VARS} /eval.sh > /test_output.txt 2>&1",
            self.test_timeout,
        )
        if result.exit_code > 1:
            raise vf.SandboxError(f"Error running tests: {result}")
        resources = cast(vf.Resources, self.resources)
        runtime = resources.require("sandbox_runtime", SandboxResources)
        output = await runtime.with_retry(runtime.client.execute_command)(
            state["sandbox_id"], "cat /test_output.txt", timeout=self.test_timeout
        )
        return output.stdout

    async def run_tests_r2e(self, state: vf.State) -> str:
        result = await self.run_background_job(
            state,
            f"{ENV_VARS} ln -sf /root/r2e_tests r2e_tests && /bin/bash run_tests.sh > test_output.txt 2>&1",
            self.test_timeout,
            working_dir=REPO_PATH,
        )
        if result.exit_code > 1:
            raise vf.SandboxError(f"Error running tests: {result}")
        resources = cast(vf.Resources, self.resources)
        runtime = resources.require("sandbox_runtime", SandboxResources)
        output = await runtime.with_retry(runtime.client.execute_command)(
            state["sandbox_id"],
            f"cat {REPO_PATH}/test_output.txt",
            timeout=self.test_timeout,
        )
        return output.stdout

    async def run_tests(self, state: vf.State) -> str:
        if self.harness == "swebench":
            return await self.run_tests_swebench(state)
        return await self.run_tests_r2e(state)

    @vf.cleanup(priority=20)
    async def run_tests_cleanup(
        self, task: vf.Task, state: vf.State, resources: vf.Resources
    ) -> None:
        if state.get("error") is not None:
            state["test_output"] = ""
            return
        try:
            state["test_output"] = await self.run_tests(state)
        except Exception as e:
            state["test_output"] = ""
            state["error"] = vf.SandboxError(f"Error running tests: {e}")

    @vf.cleanup(priority=-100)
    async def cleanup_sandbox(
        self, task: vf.Task, state: vf.State, resources: vf.Resources
    ) -> None:
        sandbox_id = state.get("sandbox_id")
        if not sandbox_id:
            return
        runtime = resources.require("sandbox_runtime", SandboxResources)
        try:
            await runtime.delete(str(sandbox_id))
        except Exception as e:
            logger.warning("Failed to delete sandbox %s: %s", sandbox_id, e)

    @vf.stop
    async def agent_signaled_done(
        self, task: vf.Task, state: vf.State, resources: vf.Resources
    ) -> bool:
        return bool(state.get("agent_signaled_done"))

    @vf.stop
    async def max_command_timeouts_reached(
        self, task: vf.Task, state: vf.State, resources: vf.Resources
    ) -> bool:
        if int(state.get("command_timeout_count", 0)) < self.max_command_timeouts:
            return False
        state["max_command_timeouts_reached"] = True
        return True

    @vf.stop
    async def rollout_timeout_reached(
        self, task: vf.Task, state: vf.State, resources: vf.Resources
    ) -> bool:
        elapsed = time.time() - state["timing"]["start_time"]
        if elapsed <= self.rollout_timeout_seconds:
            return False
        state["error"] = vf.InfraError(f"Rollout timeout after {elapsed:.0f}s")
        return True


def dataset_source(
    dataset_name: str,
    filter_repos: list[str] | None,
) -> Callable[[], Dataset]:
    def source() -> Dataset:
        split = "test" if "bench" in dataset_name.lower() else "train"
        dataset = load_dataset(dataset_name, split=split)
        assert isinstance(dataset, Dataset)
        if filter_repos:
            repo_filter = set(filter_repos)
            dataset = dataset.filter(
                lambda row: repo_filter.isdisjoint(
                    (row.get("repo"), row.get("repo_name"))
                )
            )
        return dataset.map(process_example, remove_columns=dataset.column_names)

    return source


def load_taskset(
    dataset_name: Literal[
        "R2E-Gym/R2E-Gym-Subset",
        "SWE-bench/SWE-bench_Verified",
        "PrimeIntellect/SWE-Bench-Verified-Quick",
    ] = "R2E-Gym/R2E-Gym-Subset",
    filter_repos: list[str] | None = None,
    labels: list[str] | None = None,
) -> vf.Taskset:
    harness = cast(Literal["r2e", "swebench"], get_harness(dataset_name))
    return MiniSweTaskset(
        source=dataset_source(dataset_name, filter_repos),
        harness=harness,
        labels=labels or ["mini-swe-agent-plus-th"],
        rubric=MiniSweRubric(harness),
    )


def load_harness(
    dataset_name: str = "R2E-Gym/R2E-Gym-Subset",
    max_turns: int = 200,
    sandbox_command_timeout: int = 90,
    total_timeout_minutes: int = 360,
    test_timeout: int = 900,
    cpu_cores: int = 4,
    memory_gb: int = 4,
    disk_size_gb: int = 2,
    labels: list[str] | None = None,
    sandbox_client_max_workers: int = 64,
    rollout_timeout_seconds: float = 5400.0,
    max_command_timeouts: int = 5,
    allow_git: bool = False,
) -> vf.Harness:
    harness = cast(Literal["r2e", "swebench"], get_harness(dataset_name))
    sandbox = vf.SandboxSpec(
        cpu_cores=cpu_cores,
        memory_gb=memory_gb,
        disk_size_gb=disk_size_gb,
        timeout_minutes=total_timeout_minutes,
        labels=labels or ["mini-swe-agent-plus-th"],
    )
    return MiniSweHarness(
        sandbox=sandbox,
        harness=harness,
        sandbox_command_timeout=sandbox_command_timeout,
        test_timeout=test_timeout,
        total_timeout_minutes=total_timeout_minutes,
        rollout_timeout_seconds=rollout_timeout_seconds,
        max_command_timeouts=max_command_timeouts,
        allow_git=allow_git,
        sandbox_client_max_workers=sandbox_client_max_workers,
        max_turns=max_turns,
    )


def load_environment(
    dataset_name: Literal[
        "R2E-Gym/R2E-Gym-Subset",
        "SWE-bench/SWE-bench_Verified",
        "PrimeIntellect/SWE-Bench-Verified-Quick",
    ] = "R2E-Gym/R2E-Gym-Subset",
    max_turns: int = 200,
    sandbox_command_timeout: int = 90,
    total_timeout_minutes: int = 360,
    test_timeout: int = 900,
    cpu_cores: int = 4,
    memory_gb: int = 4,
    disk_size_gb: int = 2,
    labels: list[str] | None = None,
    sandbox_client_max_workers: int = 64,
    rollout_timeout_seconds: float = 5400.0,
    max_command_timeouts: int = 5,
    allow_git: bool = False,
    filter_repos: list[str] | None = None,
) -> vf.Environment:
    return vf.Env(
        taskset=load_taskset(
            dataset_name=dataset_name,
            filter_repos=filter_repos,
            labels=labels,
        ),
        harness=load_harness(
            dataset_name=dataset_name,
            max_turns=max_turns,
            sandbox_command_timeout=sandbox_command_timeout,
            total_timeout_minutes=total_timeout_minutes,
            test_timeout=test_timeout,
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            disk_size_gb=disk_size_gb,
            labels=labels,
            sandbox_client_max_workers=sandbox_client_max_workers,
            rollout_timeout_seconds=rollout_timeout_seconds,
            max_command_timeouts=max_command_timeouts,
            allow_git=allow_git,
        ),
    )
