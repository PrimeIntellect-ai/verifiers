from __future__ import annotations

import asyncio
import importlib
import json
import logging
import re
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, Literal, cast

from datasets import Dataset, load_dataset
from prime_sandboxes import CommandTimeoutError, SandboxOOMError, SandboxTimeoutError

import verifiers as vf
from verifiers.envs.experimental.channels import SandboxResources

logger = logging.getLogger(__name__)

LoadedSource = Dataset | Iterable[Mapping[str, Any]] | None
Source = LoadedSource | Callable[[], LoadedSource]

REPO_PATH = "/testbed"
ALT_PATH = "/root"
PATH_SWEBENCH = (
    "PATH=/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:"
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)
PATH_R2E = (
    "PATH=/testbed/.venv/bin:/root/.local/bin:/root/.cargo/bin:"
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)
ENV_VARS_SWEBENCH = (
    f"export {PATH_SWEBENCH} PAGER=cat MANPAGER=cat LESS=-R "
    "PIP_PROGRESS_BAR=off TQDM_DISABLE=1;"
)
ENV_VARS_R2E = (
    f"export {PATH_R2E} PAGER=cat MANPAGER=cat LESS=-R "
    "PIP_PROGRESS_BAR=off TQDM_DISABLE=1;"
)
R2E_TESTS_ARCHIVE = "/tmp/.vf_r2e_tests.tar.gz"
FINAL_MARKER = "MINI_SWE_AGENT_FINAL_OUTPUT"
RIPGREP_SETUP_COMMAND = (
    "command -v rg >/dev/null 2>&1 || { "
    "V=14.1.1; "
    "curl -fsSL "
    "https://github.com/BurntSushi/ripgrep/releases/download/"
    "${V}/ripgrep-${V}-x86_64-unknown-linux-musl.tar.gz "
    "| tar xz -C /tmp "
    "&& install -m 755 /tmp/ripgrep-${V}-x86_64-unknown-linux-musl/rg "
    "/usr/local/bin/rg; "
    "} || true"
)
PATCH_BROKE_TESTS_EXIT_CODES = {
    2: "collection error",
    4: "usage error",
    5: "no tests collected",
}

PROMPT_TEMPLATE = """<pr_description>

Consider the following PR description:

{problem_statement}

</pr_description>

<instructions>

# Task Instructions

## Overview

You're a software engineer interacting continuously with a computer by submitting tool calls.

You'll be helping implement necessary changes to meet requirements in the PR description.

Your task is specifically to make changes to non-test files in the current directory in order to fix the issue described in the PR description in a way that is general and consistent with the codebase.

IMPORTANT: This is an interactive process where you will think and issue ONE tool call, see its result, then think and issue your next tool call.

For each response provide exactly ONE tool call to execute.

## Important Boundaries

- MODIFY: Regular source code files

- DO NOT MODIFY: Tests, configuration files (pyproject.toml, setup.cfg, etc.)

## Recommended Workflow

1. Analyze the codebase by finding and reading relevant files

2. Create a script to reproduce the issue

3. Edit the source code to resolve the issue

4. Verify your fix works by running your script again

5. Test edge cases to ensure your fix is robust

## Command Execution Rules

You are operating in an environment where

1. You issue a single tool call with the tool name and arguments

2. The system executes that tool call in a subshell

3. You see the result

4. You issue your next tool call

Each response should include a single tool call with the tool name and arguments.

**CRITICAL REQUIREMENTS:**

- Your response MUST include EXACTLY ONE tool call

- If you include zero or multiple tool calls, or no tool call at all, YOUR RESPONSE WILL FAIL

- Do NOT try to run multiple independent tool calls in one response

- Directory or environment variable changes are not persistent. Every action is executed in a new subshell.

- However, you can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or write/load environment variables from files

If you need to run multiple commands, either:

1. Combine them in one block using && or || using your tool call


command1 && command2 || echo "Error occurred"


2. Wait for the first tool call to complete, see its output, then issue the next tool call in your following response.

## Environment Details

- You have a full Linux shell environment

- Always use non-interactive flags (-y, -f) for commands

- Avoid interactive tools like vi, nano, or any that require user input

- If a command isn't available, you can install it

## Useful Command Examples

### Create a new file:


cat <<'EOF' > newfile.py

import numpy as np

hello = "world"

print(hello)

EOF


### View file content:


# View specific lines with numbers

nl -ba filename.py | sed -n '10,20p'


### Any other command you want to run using your tool call


anything


## Submission

When you've completed your changes or can't make further progress

issue exactly the following command using your tool call:

echo MINI_SWE_AGENT_FINAL_OUTPUT

This command will submit your changes.

You cannot continue working on this task after submitting.

</instructions>"""

SYSTEM_PROMPT = """You are a helpful assistant that can interact multiple times with tools to solve programming tasks.

Your response must contain exactly ONE tool call with the tool name and arguments.

Failure to follow these rules will cause your response to be rejected."""

FORMAT_ERROR_MESSAGE = (
    "Please always provide exactly one tool call. Found {count} tool calls."
)


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
    return module.make_test_spec(cast(Any, dict(info)), namespace="swebench")


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


class MiniSweMonitorRubric(vf.Rubric):
    def __init__(self):
        super().__init__()
        self.add_metric(self.command_timeout_count)
        self.add_metric(self.rollout_duration_seconds)
        self.add_metric(self.sandbox_oom)
        self.add_metric(self.sandbox_timeout)
        self.add_metric(self.sandbox_image_pull_error)
        self.add_metric(self.patch_broke_tests)

    async def command_timeout_count(self, state: vf.State) -> int:
        return int(state.get("command_timeout_count", 0))

    async def rollout_duration_seconds(self, state: vf.State) -> float:
        return time.time() - state["timing"]["start_time"]

    async def sandbox_oom(self, state: vf.State) -> float:
        return float(bool(state.get("sandbox_oom")))

    async def sandbox_timeout(self, state: vf.State) -> float:
        return float(bool(state.get("sandbox_timeout")))

    async def sandbox_image_pull_error(self, state: vf.State) -> float:
        return float(bool(state.get("sandbox_image_pull_error")))

    async def patch_broke_tests(self, state: vf.State) -> float:
        return float(bool(state.get("patch_broke_tests")))


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
        if state.get("patch_broke_tests"):
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


class MiniSweTaskset(vf.Taskset):
    def __init__(
        self,
        source: Source,
        harness: Literal["r2e", "swebench"],
        labels: list[str],
        rubric: vf.Rubric,
        test_timeout: int,
        rollout_timeout_seconds: float,
        max_command_timeouts: int,
        sandbox_client_max_workers: int,
        skip_swebench_install: bool,
    ):
        self.harness = harness
        self.labels = labels
        self.test_timeout = test_timeout
        self.rollout_timeout_seconds = rollout_timeout_seconds
        self.max_command_timeouts = max_command_timeouts
        self.sandbox_client_max_workers = sandbox_client_max_workers
        self.skip_swebench_install = skip_swebench_install
        super().__init__(source=source, rubric=rubric, name="mini-swe-agent-plus-th")

    def channels(self, task: vf.Task | None = None) -> vf.ChannelMap:
        channels = super().channels(task)
        sandbox_channel: dict[str, object] = {
            "runtime": {"client_max_workers": self.sandbox_client_max_workers}
        }
        if task is not None:
            info = dict(task.info)
            image = str(info.get("docker_image") or "")
            if self.harness == "swebench":
                image = make_swebench_test_spec(info).instance_image_key
            sandbox_channel["spec"] = vf.SandboxSeed(
                image=f"us-central1-docker.pkg.dev/prime-intellect-platform/prod-sandbox/{image}",
                labels=[*self.labels, f"mini-swe-task:{task.example_id}"],
                setup_commands=self.setup_commands(),
            )
        channels["sandbox"] = sandbox_channel
        return channels

    def setup_commands(self) -> list[str]:
        if self.harness == "swebench":
            return [
                RIPGREP_SETUP_COMMAND,
                "ln -s /opt/miniconda3/envs/testbed /root/.venv",
            ]
        return [
            RIPGREP_SETUP_COMMAND,
            f"ln -s {REPO_PATH}/.venv {ALT_PATH}/.venv",
            f"ln -s {REPO_PATH}/.venv/bin/python {ALT_PATH}/.local/bin/python",
            f"ln -s {REPO_PATH}/.venv/bin/python {ALT_PATH}/.local/bin/python3",
            f"find {REPO_PATH}/.venv/bin -type f -executable -exec ln -sf {{}} {ALT_PATH}/.local/bin/ \\;",
            f"tar -C / -czf {R2E_TESTS_ARCHIVE} r2e_tests",
            "rm -rf /r2e_tests",
        ]

    async def execute_command(
        self,
        state: vf.State,
        resources: vf.Resources,
        command: str,
        timeout: int,
        working_dir: str | None = None,
    ) -> tuple[int, str]:
        runtime = resources.require("sandbox_runtime", SandboxResources)
        sandbox_id = str(state["sandbox_id"])
        start = time.time()
        try:
            result = await runtime.with_retry(runtime.client.execute_command)(
                sandbox_id,
                command,
                timeout=timeout,
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
            self.record_command_time(state, time.time() - start)
            return (
                -1,
                f"The last command <command>{command}</command> timed out and has been killed.\n"
                "Try another command and avoid interactive input.",
            )
        self.record_command_time(state, time.time() - start)
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        parts = [stdout] if stdout else []
        if stderr:
            parts.append(f"stderr:\n{stderr}")
        return result.exit_code, "\n".join(parts) if parts else "(no output)"

    def record_command_time(self, state: vf.State, seconds: float) -> None:
        tool_states = state.setdefault("sandbox_tools", {})
        sandbox_state = tool_states.setdefault(
            "mini_swe", {"command_execution_times": []}
        )
        command_times = sandbox_state.setdefault("command_execution_times", [])
        command_times.append(seconds)

    async def execute_command_raise_on_exit_code(
        self,
        state: vf.State,
        resources: vf.Resources,
        command: str,
        timeout: int,
        working_dir: str | None = None,
    ) -> None:
        exit_code, output = await self.execute_command(
            state, resources, command, timeout, working_dir
        )
        if exit_code != 0:
            raise vf.SandboxError(f"Error executing command: {command}\n{output}")

    async def run_background_job(
        self,
        state: vf.State,
        resources: vf.Resources,
        command: str,
        timeout: int,
        working_dir: str | None = None,
        poll_interval: int = 3,
    ):
        runtime = resources.require("sandbox_runtime", SandboxResources)
        sandbox_id = str(state["sandbox_id"])
        job = await runtime.with_retry(runtime.client.start_background_job)(
            sandbox_id=sandbox_id,
            command=command,
            working_dir=working_dir,
        )
        for _ in range(0, timeout + poll_interval, poll_interval):
            result = await runtime.with_retry(runtime.client.get_background_job)(
                sandbox_id, job
            )
            if result.completed:
                return result
            await asyncio.sleep(poll_interval)
        raise CommandTimeoutError(
            sandbox_id=sandbox_id, command=command, timeout=timeout
        )

    async def read_test_output_tail(
        self,
        state: vf.State,
        resources: vf.Resources,
        path: str,
        max_chars: int = 4000,
    ) -> str:
        try:
            _, output = await self.execute_command(
                state,
                resources,
                f"tail -c {int(max_chars)} {path} 2>/dev/null",
                self.test_timeout,
            )
            return output[-max_chars:]
        except Exception:
            return ""

    def mark_patch_broke_tests(
        self, state: vf.State, exit_code: int, tail: str, path: str
    ) -> None:
        state["patch_broke_tests"] = True
        state["patch_broke_tests_reason"] = (
            f"pytest exit_code={exit_code} "
            f"({PATCH_BROKE_TESTS_EXIT_CODES[exit_code]}) from {path}"
        )
        state["patch_broke_tests_exit_code"] = exit_code
        state["patch_broke_tests_tail"] = tail[-800:] if tail else ""

    async def run_tests_swebench(self, state: vf.State, resources: vf.Resources) -> str:
        test_spec = make_swebench_test_spec(state["info"])
        eval_script = test_spec.eval_script
        if self.skip_swebench_install:
            try:
                diff_command = (
                    "cd /testbed && git -c core.fileMode=false diff --name-only HEAD"
                )
                exit_code, output = await self.execute_command(
                    state, resources, diff_command, self.test_timeout
                )
                if exit_code == 0:
                    changed = [
                        line.strip() for line in output.splitlines() if line.strip()
                    ]
                    compiled_exts = {
                        ".pyx",
                        ".pxd",
                        ".pxi",
                        ".c",
                        ".cc",
                        ".cpp",
                        ".h",
                        ".hpp",
                        ".f",
                        ".f90",
                    }
                    build_files = {
                        "setup.py",
                        "setup.cfg",
                        "pyproject.toml",
                        "meson.build",
                        "CMakeLists.txt",
                    }
                    needs_rebuild = any(
                        Path(path).name in build_files
                        or "/_build_utils/" in path
                        or "/__check_build/" in path
                        or Path(path).suffix in compiled_exts
                        for path in changed
                    )
                    if not needs_rebuild:
                        specs = swebench_constants().MAP_REPO_VERSION_TO_SPECS[
                            test_spec.repo
                        ][test_spec.version]
                        install_cmd = specs.get("install")
                        if install_cmd:
                            lines = [
                                line
                                for line in test_spec.eval_script_list
                                if line.strip() != install_cmd.strip()
                            ]
                            eval_script = (
                                "\n".join(["#!/bin/bash", "set -uxo pipefail", *lines])
                                + "\n"
                            )
            except Exception as e:
                logger.warning(f"Failed to inspect SWE-bench diff: {e!r}")
        with tempfile.NamedTemporaryFile(suffix=".sh", mode="w") as eval_file:
            eval_file.write(eval_script)
            eval_file.flush()
            runtime = resources.require("sandbox_runtime", SandboxResources)
            await runtime.with_retry(runtime.client.upload_file)(
                str(state["sandbox_id"]), "/eval.sh", eval_file.name
            )
        await self.execute_command_raise_on_exit_code(
            state, resources, "chmod +x /eval.sh", self.test_timeout
        )
        result = await self.run_background_job(
            state,
            resources,
            f"{ENV_VARS_SWEBENCH} /eval.sh > /test_output.txt 2>&1",
            self.test_timeout,
        )
        if result.exit_code in PATCH_BROKE_TESTS_EXIT_CODES:
            tail = await self.read_test_output_tail(
                state, resources, "/test_output.txt"
            )
            self.mark_patch_broke_tests(
                state, result.exit_code, tail, "/test_output.txt"
            )
            return ""
        if result.exit_code > 1:
            raise vf.SandboxError(f"Error running tests: {result}")
        runtime = resources.require("sandbox_runtime", SandboxResources)
        output = await runtime.with_retry(runtime.client.execute_command)(
            str(state["sandbox_id"]),
            "cat /test_output.txt",
            timeout=self.test_timeout,
        )
        return output.stdout

    async def run_tests_r2e(self, state: vf.State, resources: vf.Resources) -> str:
        await self.execute_command_raise_on_exit_code(
            state,
            resources,
            f"tar -C {REPO_PATH} -xzf {R2E_TESTS_ARCHIVE}",
            self.test_timeout,
        )
        result = await self.run_background_job(
            state,
            resources,
            f"{ENV_VARS_R2E} /bin/bash run_tests.sh > test_output.txt 2>&1",
            self.test_timeout,
            working_dir=REPO_PATH,
        )
        if result.exit_code in PATCH_BROKE_TESTS_EXIT_CODES:
            tail = await self.read_test_output_tail(
                state, resources, f"{REPO_PATH}/test_output.txt"
            )
            self.mark_patch_broke_tests(
                state, result.exit_code, tail, f"{REPO_PATH}/test_output.txt"
            )
            return ""
        if result.exit_code > 1:
            tail = await self.read_test_output_tail(
                state, resources, f"{REPO_PATH}/test_output.txt"
            )
            raise vf.SandboxError(f"Error running tests: {result}; tail={tail!r}")
        runtime = resources.require("sandbox_runtime", SandboxResources)
        output = await runtime.with_retry(runtime.client.execute_command)(
            str(state["sandbox_id"]),
            f"cat {REPO_PATH}/test_output.txt",
            timeout=self.test_timeout,
        )
        return output.stdout

    async def run_tests(self, state: vf.State, resources: vf.Resources) -> str:
        if self.harness == "swebench":
            return await self.run_tests_swebench(state, resources)
        return await self.run_tests_r2e(state, resources)

    @vf.cleanup(priority=20)
    async def run_tests_cleanup(
        self, task: vf.Task, state: vf.State, resources: vf.Resources
    ) -> None:
        if state.get("error") is not None or not state.get("sandbox_id"):
            state["test_output"] = ""
            return
        try:
            state["test_output"] = await self.run_tests(state, resources)
        except Exception as e:
            state["test_output"] = ""
            state["error"] = vf.SandboxError(f"Error running tests: {e}")

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
    test_timeout: int = 900,
    rollout_timeout_seconds: float = 5400.0,
    max_command_timeouts: int = 5,
    sandbox_client_max_workers: int = 64,
    skip_swebench_install: bool = True,
) -> vf.Taskset:
    harness = cast(Literal["r2e", "swebench"], get_harness(dataset_name))
    return MiniSweTaskset(
        source=dataset_source(dataset_name, filter_repos),
        harness=harness,
        labels=labels or ["mini-swe-agent-plus-th"],
        rubric=vf.RubricGroup([MiniSweRubric(harness), MiniSweMonitorRubric()]),
        test_timeout=test_timeout,
        rollout_timeout_seconds=rollout_timeout_seconds,
        max_command_timeouts=max_command_timeouts,
        sandbox_client_max_workers=sandbox_client_max_workers,
        skip_swebench_install=skip_swebench_install,
    )


def load_harness(
    dataset_name: str = "R2E-Gym/R2E-Gym-Subset",
    max_turns: int = 200,
    sandbox_command_timeout: int = 90,
    total_timeout_minutes: int = 360,
    cpu_cores: int = 4,
    memory_gb: int = 4,
    disk_size_gb: int = 2,
    labels: list[str] | None = None,
    sandbox_client_max_workers: int = 64,
    allow_git: bool = False,
) -> vf.Harness:
    blocked_commands = {"ipython", "jupyter", "nohup"}
    if not allow_git:
        blocked_commands.add("git")
    sandbox = vf.SandboxSpec(
        cpu_cores=cpu_cores,
        memory_gb=memory_gb,
        disk_size_gb=disk_size_gb,
        timeout_minutes=total_timeout_minutes,
        labels=labels or ["mini-swe-agent-plus-th"],
    )
    sandbox_runtime = {"client_max_workers": sandbox_client_max_workers}
    harness_kind = get_harness(dataset_name)
    env_vars = ENV_VARS_SWEBENCH if harness_kind == "swebench" else ENV_VARS_R2E
    command_prefix = f"export ALLOW_GIT=1; {env_vars}" if allow_git else env_vars
    bash = vf.SandboxBashTool(
        name="execute_bash",
        sandbox=sandbox,
        sandbox_key="mini_swe",
        sandbox_runtime=sandbox_runtime,
        command_timeout=sandbox_command_timeout,
        command_prefix=command_prefix,
        working_dir=REPO_PATH,
        blocked_commands=blocked_commands,
        completion_marker=FINAL_MARKER,
    )
    edit = vf.SandboxEditTool(
        name="edit_via_str_replace",
        sandbox=sandbox,
        sandbox_key="mini_swe",
        sandbox_runtime=sandbox_runtime,
        command_timeout=sandbox_command_timeout,
        root_dir=REPO_PATH,
    )
    return vf.Harness(
        system_prompt=SYSTEM_PROMPT,
        tools=[bash, edit],
        run=vf.RunConfig(
            max_turns=max_turns,
            stop_errors=(vf.SandboxError,),
            max_tool_calls_per_turn=1,
            tool_call_limit_message=FORMAT_ERROR_MESSAGE,
        ),
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
    skip_swebench_install: bool = True,
) -> vf.Environment:
    return vf.Env(
        taskset=load_taskset(
            dataset_name=dataset_name,
            filter_repos=filter_repos,
            labels=labels,
            test_timeout=test_timeout,
            rollout_timeout_seconds=rollout_timeout_seconds,
            max_command_timeouts=max_command_timeouts,
            sandbox_client_max_workers=sandbox_client_max_workers,
            skip_swebench_install=skip_swebench_install,
        ),
        harness=load_harness(
            dataset_name=dataset_name,
            max_turns=max_turns,
            sandbox_command_timeout=sandbox_command_timeout,
            total_timeout_minutes=total_timeout_minutes,
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            disk_size_gb=disk_size_gb,
            labels=labels,
            sandbox_client_max_workers=sandbox_client_max_workers,
            allow_git=allow_git,
        ),
    )
