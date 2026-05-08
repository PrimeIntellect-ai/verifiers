from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from datasets import load_dataset
from pydantic import Field

import verifiers.v1 as vf

logger = logging.getLogger(__name__)

REGISTRY_PREFIX = "us-central1-docker.pkg.dev/prime-intellect-platform/prod-sandbox"
DEFAULT_DATASET_NAME = "R2E-Gym/R2E-Gym-Subset"
DEFAULT_REPO_PATH = "/testbed"
DEFAULT_ALT_PATH = "/root"


class RlmSweTasksetConfig(vf.TasksetConfig):
    dataset_name: str = DEFAULT_DATASET_NAME
    repo_path: str = DEFAULT_REPO_PATH
    alt_path: str = DEFAULT_ALT_PATH
    filter_repos: list[str] | None = None
    ds_num_proc: int | None = None
    ds_keep_in_memory: bool = True
    timeout_minutes: int | None = None
    hide_tests_from_agent: bool = True
    env: dict[str, object] = Field(default_factory=dict)


class R2ESWETaskset(vf.Taskset):
    config_type = RlmSweTasksetConfig

    def __init__(
        self,
        dataset_name: str | None = None,
        repo_path: str | None = None,
        alt_path: str | None = None,
        filter_repos: list[str] | None = None,
        ds_num_proc: int | None = None,
        ds_keep_in_memory: bool | None = None,
        timeout_minutes: int | None = None,
        hide_tests_from_agent: bool | None = None,
        env: Mapping[str, object] | None = None,
        config: RlmSweTasksetConfig | Mapping[str, object] | None = None,
        **kwargs: Any,
    ):
        config = RlmSweTasksetConfig(config)
        self.dataset_name = dataset_name or config.dataset_name
        self.repo_path = repo_path or config.repo_path
        self.alt_path = alt_path or config.alt_path
        self.filter_repos = (
            filter_repos if filter_repos is not None else config.filter_repos
        )
        self.ds_num_proc = (
            ds_num_proc if ds_num_proc is not None else config.ds_num_proc
        )
        self.ds_keep_in_memory = (
            ds_keep_in_memory
            if ds_keep_in_memory is not None
            else config.ds_keep_in_memory
        )
        self.timeout_minutes = (
            timeout_minutes if timeout_minutes is not None else config.timeout_minutes
        )
        self.hide_tests_from_agent = (
            hide_tests_from_agent
            if hide_tests_from_agent is not None
            else config.hide_tests_from_agent
        )
        self.env = {**config.env, **dict(env or {})}
        super().__init__(
            source=self.load_rows,
            taskset_id="swe/r2e",
            config=config,
            **kwargs,
        )

    def load_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for index, row in enumerate(self.load_dataset_rows()):
            row = dict(row)
            info = dict(row["info"])
            instruction = str(info["problem_statement"])
            task_row: dict[str, Any] = {
                "example_id": index,
                "task_id": info.get("instance_id") or index,
                "question": row.get("question", instruction),
                "instruction": instruction,
                "prompt": [{"role": "user", "content": instruction}],
                "answer": row.get("answer", ""),
                "info": info,
                "sandbox": self.sandbox_config(info),
                "program": {"env": {"AGENT_WORKDIR": self.repo_path}},
            }
            rows.append(task_row)
        return rows

    def load_dataset_rows(self):
        dataset_kwargs = dict(
            num_proc=self.ds_num_proc,
            keep_in_memory=self.ds_keep_in_memory,
            load_from_cache_file=False,
        )
        dataset = load_dataset(
            self.dataset_name,
            split="train",
            keep_in_memory=self.ds_keep_in_memory,
            num_proc=self.ds_num_proc,
        )
        if self.filter_repos:
            filter_set = frozenset(self.filter_repos)
            dataset = dataset.filter(
                lambda row: row.get("repo_name") not in filter_set,
                **dataset_kwargs,
            )
        return dataset.map(
            process_r2e_example,
            remove_columns=dataset.column_names,
            **dataset_kwargs,
        )

    def sandbox_config(self, info: Mapping[str, object]) -> dict[str, object]:
        config: dict[str, object] = {
            "image": f"{REGISTRY_PREFIX}/{info['docker_image']}",
            "cpu_cores": 4,
            "memory_gb": 4,
            "disk_size_gb": 10,
            "gpu_count": 0,
            "workdir": self.repo_path,
            "scope": "rollout",
        }
        if self.timeout_minutes is not None:
            config["timeout_minutes"] = self.timeout_minutes
        return config

    def get_env_vars(self) -> dict[str, str]:
        return {
            "PATH": (
                "/opt/miniconda3/bin:/testbed/.venv/bin:/root/.local/bin:"
                "/root/.cargo/bin:/go/bin:/usr/local/go/bin:/usr/local/cargo:"
                "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
            ),
            "PAGER": "cat",
            "MANPAGER": "cat",
            "LESS": "-R",
            "PIP_PROGRESS_BAR": "off",
            "TQDM_DISABLE": "1",
            **{str(key): str(value) for key, value in self.env.items()},
        }

    @vf.setup(priority=250)
    async def setup_r2e_sandbox(self, task, state, sandbox=None) -> None:
        if sandbox is None:
            raise RuntimeError("R2E SWE setup requires the active program sandbox.")
        state["_rlm_swe_sandbox"] = sandbox
        state["sandbox_id"] = getattr(sandbox, "id", state.get("sandbox_id"))
        sandbox_config = task.get("sandbox")
        if isinstance(sandbox_config, Mapping):
            timeout_minutes = int(sandbox_config.get("timeout_minutes") or 60)
            state.setdefault("test_timeout", timeout_minutes * 60)
        await self.setup_sandbox(sandbox, state)

    async def setup_sandbox(self, sandbox: Any, state: dict[str, Any]) -> None:
        async def exec_checked(
            command: str, working_dir: str | None = None, timeout: int = 90
        ):
            result = await sandbox.execute(
                command, working_dir=working_dir, timeout=timeout
            )
            if result.exit_code != 0:
                raise RuntimeError(
                    f"Setup command failed: {command} exit_code={result.exit_code}"
                )
            return result

        link_commands = [
            f"ln -s {self.repo_path}/.venv {self.alt_path}/.venv",
            f"ln -s {self.repo_path}/.venv/bin/python {self.alt_path}/.local/bin/python",
            f"ln -s {self.repo_path}/.venv/bin/python {self.alt_path}/.local/bin/python3",
            f"find {self.repo_path}/.venv/bin -type f -executable -exec ln -sfn {{}} {self.alt_path}/.local/bin/ \\;",
        ]
        for command in link_commands:
            await exec_checked(command)

        try:
            cleanup_commands = [
                (
                    "timeout 30 bash -c 'shopt -s globstar; rm -rf **/*.pyc **/__pycache__' 2>/dev/null || timeout 30 find . -name '*.pyc' -delete || true",
                    self.repo_path,
                ),
                (
                    "timeout 30 bash -c 'shopt -s globstar; rm -rf **/__pycache__' 2>/dev/null || timeout 30 find . -name '__pycache__' -exec rm -rf {} + || true",
                    self.repo_path,
                ),
                (
                    "timeout 30 bash -c 'shopt -s globstar; rm -rf /r2e_tests/**/*.pyc /r2e_tests/**/__pycache__' 2>/dev/null || timeout 30 find /r2e_tests -name '*.pyc' -delete || true",
                    None,
                ),
                (
                    "timeout 30 bash -c 'shopt -s globstar; rm -rf /r2e_tests/**/__pycache__' 2>/dev/null || timeout 30 find /r2e_tests -name '__pycache__' -exec rm -rf {} + || true",
                    None,
                ),
            ]
            for command, working_dir in cleanup_commands:
                await exec_checked(command, working_dir=working_dir)
        except Exception as exc:
            logger.warning("Continuing without deleting pycache: %r", exc)

        if not self.hide_tests_from_agent:
            await exec_checked(f"mv /r2e_tests {self.repo_path}/r2e_tests", timeout=60)
            return

        remote_archive = "/tmp/r2e_tests.tar.gz"
        local_archive_path = str(Path("/tmp") / f"r2e_tests_{sandbox.id}.tar.gz")
        await exec_checked(f"tar -C / -czf {remote_archive} r2e_tests", timeout=300)
        await sandbox.download_file(remote_archive, local_archive_path, timeout=300)
        state["r2e_tests_archive_local_path"] = local_archive_path
        await exec_checked("rm -rf /r2e_tests", timeout=300)
        await exec_checked(f"rm -f {remote_archive}", timeout=300)

    @vf.reward(weight=1.0)
    async def solved(self, task, state) -> float:
        if state.get("error") is not None:
            return 0.0
        sandbox = state.get("_rlm_swe_sandbox")
        if sandbox is None:
            return 0.0
        try:
            test_output = await self.run_tests(
                sandbox,
                state,
                int(state.get("test_timeout", 900)),
            )
            state["test_output"] = test_output
        except Exception as exc:
            logger.warning("Test execution failed: %r", exc)
            state["test_output"] = f"ERROR: {exc}"
            return 0.0
        return float(self.calculate_reward(test_output, task["info"]))

    async def run_tests(
        self,
        sandbox: Any,
        state: dict[str, Any],
        test_timeout: int,
    ) -> str:
        local_archive_path = state.get("r2e_tests_archive_local_path")
        if local_archive_path and Path(str(local_archive_path)).exists():
            remote_archive = "/tmp/r2e_tests_roundtrip.tar.gz"
            await sandbox.upload_file(
                remote_archive, str(local_archive_path), timeout=300
            )
            result = await sandbox.execute(
                f"tar -C {self.repo_path} -xzf {remote_archive}",
                timeout=300,
            )
            if result.exit_code != 0:
                raise RuntimeError(
                    f"Failed to extract r2e_tests: exit_code={result.exit_code}"
                )
            Path(str(local_archive_path)).unlink(missing_ok=True)
            del state["r2e_tests_archive_local_path"]
        elif self.hide_tests_from_agent:
            raise RuntimeError(
                f"Missing cached r2e_tests archive: {local_archive_path}"
            )

        env_str = " ".join(
            f"{key}={value}" for key, value in self.get_env_vars().items()
        )
        command = f"export {env_str}; /bin/bash run_tests.sh > test_output.txt 2>&1"
        result = await sandbox.run_background_job(
            command, timeout=test_timeout, working_dir=self.repo_path
        )
        if result.exit_code > 1:
            raise RuntimeError(f"Error running tests: exit_code={result.exit_code}")
        result = await sandbox.execute(
            f"cat {self.repo_path}/test_output.txt", timeout=300
        )
        return result.stdout or ""

    def calculate_reward(self, test_output: str, info: Mapping[str, object]) -> float:
        parsed = parse_log_pytest(test_output)
        parsed = decolor_dict_keys(parsed)
        expected_raw = info["expected_output_json"]
        expected: dict[str, str] = json.loads(str(expected_raw))
        expected = decolor_dict_keys(expected)
        parsed = {key.split(" - ")[0]: parsed[key] for key in sorted(parsed.keys())}
        expected = {
            key.split(" - ")[0]: expected[key] for key in sorted(expected.keys())
        }
        if len(parsed) != len(expected):
            return 0.0
        for key in parsed:
            if not key:
                continue
            if key not in expected or parsed[key] != expected[key]:
                return 0.0
        return 1.0

    async def apply_gold_patch(self, sandbox: Any, state) -> None:
        info = state["info"]
        patch = extract_gold_patch(
            str(info["parsed_commit_content"]),
            test_file=False,
            only_python=True,
        )
        if not patch.strip():
            raise RuntimeError(
                "Empty gold patch reconstructed from parsed_commit_content"
            )

        await sandbox.upload_bytes("/tmp/gold.patch", patch.encode(), "gold.patch")
        result = await sandbox.execute(
            "git apply --whitespace=fix /tmp/gold.patch",
            working_dir=self.repo_path,
            timeout=30,
        )
        if result.exit_code != 0:
            stderr = (result.stderr or "")[:500]
            raise RuntimeError(
                f"git apply failed: exit_code={result.exit_code} stderr={stderr}"
            )

    async def validate_instance(self, state) -> bool:
        sandbox = state["_rlm_swe_sandbox"]
        await self.apply_gold_patch(sandbox, state)
        test_output = await self.run_tests(
            sandbox,
            state,
            int(state.get("test_timeout", 900)),
        )
        state["test_output"] = test_output
        return self.calculate_reward(test_output, state.get("info") or {}) > 0

    @vf.cleanup(priority=100)
    async def cleanup_r2e_state(self, task, state) -> None:
        archive = state.pop("r2e_tests_archive_local_path", None)
        if isinstance(archive, str):
            Path(archive).unlink(missing_ok=True)
        state.pop("_rlm_swe_sandbox", None)


def process_r2e_example(row: Mapping[str, Any]) -> dict[str, Any]:
    info = dict(row)
    info.setdefault("instance_id", row.get("commit_hash"))
    info.setdefault("repo", row.get("repo_name"))
    return {
        "question": row["problem_statement"],
        "info": info,
        "answer": "",
    }


def parse_log_pytest(log: str | None) -> dict[str, str]:
    if log is None or "short test summary info" not in log:
        return {}
    test_status_map = {}
    for line in log.split("short test summary info", 1)[1].strip().splitlines():
        if "PASSED" in line:
            test_name = ".".join(line.split("::")[1:])
            test_status_map[test_name] = "PASSED"
        elif "FAILED" in line:
            test_name = ".".join(line.split("::")[1:]).split(" - ")[0]
            test_status_map[test_name] = "FAILED"
        elif "ERROR" in line:
            test_name = ".".join(line.split("::")[1:]) if "::" in line else line
            test_status_map[test_name.split(" - ")[0]] = "ERROR"
    return test_status_map


def decolor_dict_keys(values: Mapping[str, str]) -> dict[str, str]:
    return {re.sub(r"\u001b\[\d+m", "", key): value for key, value in values.items()}


def extract_gold_patch(
    parsed_commit_content: str, test_file: bool = False, only_python: bool = True
) -> str:
    data = json.loads(parsed_commit_content)
    patch = ""
    for file_diff in data.get("file_diffs", []):
        path = file_diff.get("header", {}).get("file", {}).get("path", "")
        if not path:
            continue
        if only_python and not path.endswith(".py"):
            continue
        parts = path.split("/")
        is_test = (
            path.endswith("_test.py")
            or path.split("/")[-1].startswith("test_")
            or any(part in {"tests", "Tests", "test", "Test"} for part in parts)
        )
        if is_test and not test_file:
            continue

        header = file_diff.get("header", {})
        misc_line = header.get("misc_line")
        patch += f"diff --git a/{path} b/{path}\n"
        if misc_line:
            patch += misc_line + "\n"

        index_line = file_diff.get("index_line")
        if index_line:
            old_hash = index_line.get("old_commit_hash", "")
            new_hash = index_line.get("new_commit_hash", "")
            mode = index_line.get("mode", "")
            patch += f"index {old_hash}..{new_hash}{' ' if mode else ''}{mode}\n"

        if file_diff.get("is_binary_file"):
            binary_line = file_diff.get("binary_line", "")
            if binary_line:
                patch += binary_line + "\n"

        minus_file = file_diff.get("minus_file")
        plus_file = file_diff.get("plus_file")
        if minus_file and plus_file:
            patch += f"--- {minus_file['path']}\n"
            patch += f"+++ {plus_file['path']}\n"

        for hunk in file_diff.get("hunks", []):
            descriptor = hunk.get("descriptor", {})
            old_range = descriptor.get("old_range", {})
            new_range = descriptor.get("new_range", {})
            old_str = str(old_range.get("start", 0))
            if old_range.get("length") is not None:
                old_str += f",{old_range['length']}"
            new_str = str(new_range.get("start", 0))
            if new_range.get("length") is not None:
                new_str += f",{new_range['length']}"
            section = descriptor.get("section", "")
            hunk_header = f"@@ -{old_str} +{new_str} @@"
            if section:
                hunk_header += f" {section}"
            patch += hunk_header + "\n"

            for line in hunk.get("line_group", {}).get("all_lines", []):
                content = line.get("content", "")
                line_type = line.get("type", "")
                if line_type == "context":
                    patch += f" {content}\n"
                elif line_type == "added":
                    patch += f"+{content}\n"
                elif line_type == "deleted":
                    patch += f"-{content}\n"
                elif line_type == "note":
                    patch += f"\\ {content}\n"
    return patch


def load_taskset(
    config: RlmSweTasksetConfig | Mapping[str, object] | None = None,
    dataset_name: str | None = None,
    repo_path: str | None = None,
    alt_path: str | None = None,
    filter_repos: list[str] | None = None,
    ds_num_proc: int | None = None,
    ds_keep_in_memory: bool | None = None,
    timeout_minutes: int | None = None,
    hide_tests_from_agent: bool | None = None,
    env: Mapping[str, object] | None = None,
) -> R2ESWETaskset:
    return R2ESWETaskset(
        dataset_name=dataset_name,
        repo_path=repo_path,
        alt_path=alt_path,
        filter_repos=filter_repos,
        ds_num_proc=ds_num_proc,
        ds_keep_in_memory=ds_keep_in_memory,
        timeout_minutes=timeout_minutes,
        hide_tests_from_agent=hide_tests_from_agent,
        env=env,
        config=config,
    )


def load_harness(
    config: vf.HarnessConfig | None = None,
    workdir: str = DEFAULT_REPO_PATH,
    gh_token: str | None = None,
    rlm_tools: list[str] | None = None,
    skills: str | Path | None = None,
    **rlm_kwargs: Any,
) -> vf.RLM:
    token = gh_token or os.environ.get("GH_TOKEN")
    return vf.RLM(
        workdir=workdir,
        gh_token=token,
        rlm_tools=rlm_tools,
        skills=skills,
        config=config,
        **rlm_kwargs,
    )


def load_environment(
    config: vf.EnvConfig | None = None,
    dataset_name: str | None = None,
    repo_path: str | None = None,
    alt_path: str | None = None,
    filter_repos: list[str] | None = None,
    ds_num_proc: int | None = None,
    ds_keep_in_memory: bool | None = None,
    timeout_minutes: int | None = None,
    hide_tests_from_agent: bool | None = None,
    env: Mapping[str, object] | None = None,
    instruction_path: str | None = None,
    rlm_repo_url: str | None = None,
    rlm_ref: str | None = None,
    rlm_max_turns: int | None = None,
    rlm_exec_timeout: int | None = None,
    rlm_max_depth: int | None = None,
    summarize_at_tokens: int | tuple[int, int] | list[int] | None = None,
    include_sub_rlm_trajectories: bool | None = None,
    append_to_system_prompt: str | None = None,
    local_checkout: str | Path | None = None,
    gh_token: str | None = None,
    rlm_tools: list[str] | None = None,
    rlm_env: dict[str, object] | None = None,
    skills: str | Path | None = None,
) -> vf.Env:
    config = vf.EnvConfig(
        config,
        taskset=RlmSweTasksetConfig(
            dataset_name=dataset_name or DEFAULT_DATASET_NAME,
            repo_path=repo_path or DEFAULT_REPO_PATH,
            alt_path=alt_path or DEFAULT_ALT_PATH,
            filter_repos=filter_repos,
            ds_num_proc=ds_num_proc,
            ds_keep_in_memory=ds_keep_in_memory
            if ds_keep_in_memory is not None
            else True,
            timeout_minutes=timeout_minutes,
            hide_tests_from_agent=hide_tests_from_agent
            if hide_tests_from_agent is not None
            else True,
            env=dict(env or {}),
        ),
    )
    taskset = load_taskset(config=config.taskset)
    merged_rlm_env = {**dict(rlm_env or {}), **taskset.get_env_vars()}
    rlm_kwargs = {
        key: value
        for key, value in {
            "instruction_path": instruction_path,
            "rlm_repo_url": rlm_repo_url,
            "rlm_ref": rlm_ref,
            "rlm_max_turns": rlm_max_turns,
            "rlm_exec_timeout": rlm_exec_timeout,
            "rlm_max_depth": rlm_max_depth,
            "summarize_at_tokens": summarize_at_tokens,
            "include_sub_rlm_trajectories": include_sub_rlm_trajectories,
            "append_to_system_prompt": append_to_system_prompt,
            "local_checkout": local_checkout,
            "rlm_env": merged_rlm_env,
        }.items()
        if value is not None
    }
    harness = load_harness(
        config=config.harness,
        workdir=taskset.repo_path,
        gh_token=gh_token,
        rlm_tools=rlm_tools,
        skills=skills,
        **rlm_kwargs,
    )
    return vf.Env(taskset=taskset, harness=harness)
