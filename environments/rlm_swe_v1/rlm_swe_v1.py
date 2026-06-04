import json
import logging
import re
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from datasets import load_dataset
import verifiers as vf
from harnesses import RLM, RLMConfig

logger = logging.getLogger(__name__)

REGISTRY_PREFIX = "us-central1-docker.pkg.dev/prime-intellect-platform/prod-sandbox"


class RlmSweTasksetConfig(vf.TasksetConfig):
    taskset_id: str = "swe/r2e"
    """Identifier surfaced to the runtime and stamped on each task row."""

    dataset_name: str = "R2E-Gym/R2E-Gym-Subset"
    """HuggingFace dataset to load SWE instances from."""

    repo_path: str = "/testbed"
    """In-sandbox path to the checked-out repo. Also exported as
    `AGENT_WORKDIR` and used as the sandbox `workdir`."""

    filter_repos: list[str] | None = None
    """Optional `repo_name` allow-out list; rows whose `repo_name` is in this
    set are dropped before task rows are built."""

    timeout_minutes: int | None = None
    """Per-task sandbox timeout in minutes. None leaves the sandbox default
    (60). Also drives `test_timeout` (`timeout_minutes * 60`)."""

    hide_tests_from_agent: bool = True
    """When True, `/r2e_tests` is archived off the sandbox during setup and
    restored just before scoring so the agent can't read or modify the test
    suite. When False, `/r2e_tests` is moved into `repo_path/r2e_tests` and
    left visible."""

    env: vf.ConfigData | None = None
    """Extra environment variables to merge on top of the SWE baseline
    (`PATH`, `PAGER`, etc.). Exported into both the agent program env and
    the test-run env."""


def sandbox_config(
    *, info: vf.JsonData, repo_path: str, timeout_minutes: int | None
) -> vf.JsonData:
    config: vf.JsonData = {
        "image": f"{REGISTRY_PREFIX}/{info['docker_image']}",
        "cpu_cores": 4,
        "memory_gb": 4,
        "disk_size_gb": 10,
        "gpu_count": 0,
        "workdir": repo_path,
        "scope": "rollout",
    }
    if timeout_minutes is not None:
        config["timeout_minutes"] = timeout_minutes
    return config


def env_vars(*, repo_path: str, env: vf.ConfigData) -> dict[str, str]:
    return {
        "PATH": (
            f"/opt/miniconda3/bin:{repo_path}/.venv/bin:/root/.local/bin:"
            "/root/.cargo/bin:/go/bin:/usr/local/go/bin:/usr/local/cargo:"
            "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        ),
        "PAGER": "cat",
        "MANPAGER": "cat",
        "LESS": "-R",
        "PIP_PROGRESS_BAR": "off",
        "TQDM_DISABLE": "1",
        **{str(key): str(value) for key, value in env.items()},
    }


class R2ESWETaskset(vf.Taskset[RlmSweTasksetConfig]):
    def sandbox_config(self, info: vf.JsonData) -> vf.JsonData:
        return sandbox_config(
            info=info,
            repo_path=self.config.repo_path,
            timeout_minutes=self.config.timeout_minutes,
        )

    def get_env_vars(self) -> dict[str, str]:
        return env_vars(
            repo_path=self.config.repo_path, env=dict(self.config.env or {})
        )

    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        repo_path = self.config.repo_path
        task_env = dict(self.config.env or {})
        dataset_kwargs = dict(keep_in_memory=True, load_from_cache_file=False)
        dataset = load_dataset(
            self.config.dataset_name, split="train", keep_in_memory=True
        )
        if self.config.filter_repos:
            filter_set = frozenset(self.config.filter_repos)
            dataset = dataset.filter(
                lambda row: row.get("repo_name") not in filter_set,
                **dataset_kwargs,
            )
        dataset = dataset.map(
            process_r2e_example,
            remove_columns=dataset.column_names,
            **dataset_kwargs,
        )
        rows: list[vf.JsonData] = []
        for index, row in enumerate(dataset):
            row = dict(row)
            info = dict(row["info"])
            instruction = str(info["problem_statement"])
            program_env = env_vars(repo_path=repo_path, env=task_env)
            agent_path = program_env.pop("PATH", None)
            if agent_path is not None:
                program_env.setdefault("AGENT_PATH", agent_path)
            program_env.setdefault("AGENT_WORKDIR", repo_path)
            rows.append(
                {
                    "example_id": index,
                    "task_id": info.get("instance_id") or index,
                    "question": row.get("question", instruction),
                    "instruction": instruction,
                    "prompt": [{"role": "user", "content": instruction}],
                    "answer": row.get("answer", ""),
                    "info": info,
                    "sandbox": sandbox_config(
                        info=info,
                        repo_path=repo_path,
                        timeout_minutes=self.config.timeout_minutes,
                    ),
                    "program": {"env": program_env},
                }
            )
        return rows

    @vf.setup(priority=250)
    async def setup_r2e_sandbox(self, task, state, sandbox=None) -> None:
        if sandbox is None:
            raise RuntimeError("R2E SWE setup requires the active program sandbox.")
        state["sandbox"] = sandbox
        state["sandbox_id"] = getattr(sandbox, "id", state.get("sandbox_id"))
        sandbox_config = task.get("sandbox")
        if isinstance(sandbox_config, Mapping):
            timeout_minutes = int(sandbox_config.get("timeout_minutes") or 60)
            state.setdefault("test_timeout", timeout_minutes * 60)
        await self.setup_sandbox(sandbox, state)

    async def setup_sandbox(self, sandbox, state: vf.State) -> None:
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

        try:
            cleanup_commands = [
                (
                    "timeout 30 bash -c 'shopt -s globstar; rm -rf **/*.pyc **/__pycache__' 2>/dev/null || timeout 30 find . -name '*.pyc' -delete || true",
                    self.config.repo_path,
                ),
                (
                    "timeout 30 bash -c 'shopt -s globstar; rm -rf **/__pycache__' 2>/dev/null || timeout 30 find . -name '__pycache__' -exec rm -rf {} + || true",
                    self.config.repo_path,
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

        if not self.config.hide_tests_from_agent:
            await exec_checked(
                f"mv /r2e_tests {self.config.repo_path}/r2e_tests", timeout=60
            )
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
        sandbox = state.get("sandbox")
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
        sandbox,
        state: vf.State,
        test_timeout: int,
    ) -> str:
        local_archive_path = state.get("r2e_tests_archive_local_path")
        if local_archive_path and Path(str(local_archive_path)).exists():
            remote_archive = "/tmp/r2e_tests_roundtrip.tar.gz"
            await sandbox.upload_file(
                remote_archive, str(local_archive_path), timeout=300
            )
            result = await sandbox.execute(
                f"tar -C {self.config.repo_path} -xzf {remote_archive}",
                timeout=300,
            )
            if result.exit_code != 0:
                raise RuntimeError(
                    f"Failed to extract r2e_tests: exit_code={result.exit_code}"
                )
            Path(str(local_archive_path)).unlink(missing_ok=True)
            del state["r2e_tests_archive_local_path"]
        elif self.config.hide_tests_from_agent:
            raise RuntimeError(
                f"Missing cached r2e_tests archive: {local_archive_path}"
            )

        env_str = " ".join(
            f"{shlex.quote(key)}={shlex.quote(value)}"
            for key, value in self.get_env_vars().items()
        )
        command = f"export {env_str}; /bin/bash run_tests.sh > test_output.txt 2>&1"
        result = await sandbox.run_background_job(
            command, timeout=test_timeout, working_dir=self.config.repo_path
        )
        if result.exit_code > 1:
            raise RuntimeError(f"Error running tests: exit_code={result.exit_code}")
        result = await sandbox.execute(
            f"cat {self.config.repo_path}/test_output.txt", timeout=300
        )
        return result.stdout or ""

    def calculate_reward(self, test_output: str, info: vf.JsonData) -> float:
        parsed = parse_log_pytest(test_output)
        parsed = decolor_dict_keys(parsed)
        expected_raw = info["expected_output_json"]
        expected: dict[str, str] = json.loads(str(expected_raw))
        expected = decolor_dict_keys(expected)
        parsed = {key.split(" - ")[0]: parsed[key] for key in sorted(parsed.keys())}
        expected = {
            key.split(" - ")[0]: expected[key] for key in sorted(expected.keys())
        }
        if any(not key for key in parsed):
            return 0.0
        if len(parsed) != len(expected):
            return 0.0
        for key in parsed:
            if key not in expected or parsed[key] != expected[key]:
                return 0.0
        return 1.0

    async def apply_gold_patch(self, sandbox, state: vf.State) -> None:
        info = cast(vf.JsonData, state["info"])
        assert isinstance(info, Mapping)
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
            working_dir=self.config.repo_path,
            timeout=30,
        )
        if result.exit_code != 0:
            stderr = (result.stderr or "")[:500]
            raise RuntimeError(
                f"git apply failed: exit_code={result.exit_code} stderr={stderr}"
            )

    async def validate_instance(self, state: vf.State) -> bool:
        sandbox = state["sandbox"]
        await self.apply_gold_patch(sandbox, state)
        test_timeout = state.get("test_timeout", 900)
        assert isinstance(test_timeout, int)
        test_output = await self.run_tests(
            sandbox,
            state,
            test_timeout,
        )
        state["test_output"] = test_output
        info = cast(vf.JsonData, state["info"])
        assert isinstance(info, Mapping)
        return self.calculate_reward(test_output, info) > 0

    @vf.cleanup(priority=100)
    async def cleanup_r2e_state(self, task, state) -> None:
        archive = state.pop("r2e_tests_archive_local_path", None)
        if isinstance(archive, str):
            Path(archive).unlink(missing_ok=True)
        state.pop("sandbox", None)


def process_r2e_example(row: vf.JsonData) -> vf.JsonData:
    info = cast(vf.JsonData, dict(row))
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
        status, _, test_ref = line.strip().partition(" ")
        if status in {"PASSED", "FAILED", "ERROR"}:
            test_name = (
                ".".join(test_ref.split("::")[1:]) if "::" in test_ref else test_ref
            )
            test_status_map[test_name.split(" - ")[0]] = status
    return test_status_map


def decolor_dict_keys(values: dict[str, str]) -> dict[str, str]:
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


def load_taskset(config: RlmSweTasksetConfig) -> R2ESWETaskset:
    return R2ESWETaskset(config=config)


def load_harness(config: RLMConfig) -> RLM:
    return RLM(config=config)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(
        taskset=vf.load_taskset(config=config.taskset),
        harness=vf.load_harness(config=config.harness),
    )
