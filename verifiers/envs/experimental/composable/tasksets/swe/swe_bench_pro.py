from __future__ import annotations

import ast
import json
import logging
import re
import shlex
import tempfile
import urllib.request
from pathlib import Path
from textwrap import dedent
from typing import Any

import verifiers as vf
from datasets import Dataset, load_dataset

from verifiers.envs.experimental.composable import SandboxSpec, SandboxTaskSet

DEFAULT_DATASET_NAME = "ScaleAI/SWE-bench_Pro"
DEFAULT_DATASET_SPLIT = "test"
DEFAULT_AGENT_WORKDIR = "/app"
DEFAULT_DOCKER_IMAGE_REPOSITORY = "jefzda/sweap-images"
DEFAULT_RUN_SCRIPTS_URL = (
    "https://raw.githubusercontent.com/scaleapi/SWE-bench_Pro-os/main/run_scripts"
)

logger = logging.getLogger(__name__)

_INSTRUCTION_TEMPLATE = """\
# Task

{problem_statement}

---

**Repo:** `{repo}`
**Base commit:** `{base_commit}`
**Instance ID:** `{instance_id}`
"""

_CAPTURE_PATCH_COMMAND = """\
set -e
mkdir -p /logs/verifier /workspace
cd {agent_workdir}

git status --short > /logs/verifier/repo_status.txt || true
git add -N . >/dev/null 2>&1 || true
git diff --name-only {base_commit} > /logs/verifier/changed_files.txt || true
git diff --stat {base_commit} > /logs/verifier/generated_patch_stat.txt || true
git diff --binary {base_commit} > /logs/verifier/generated_patch.diff || true
"""

_RUN_TESTS_COMMAND = """\
set -e
mkdir -p /logs/verifier /workspace
cd {agent_workdir}

git reset --hard {base_commit}
git clean -fd
git checkout {base_commit}

if [ -s /logs/verifier/generated_patch.diff ]; then
  git apply -v /logs/verifier/generated_patch.diff
fi

{test_setup_command}

bash /workspace/run_script.sh {selected_tests} > /workspace/stdout.log 2> /workspace/stderr.log || true
python /workspace/parser.py /workspace/stdout.log /workspace/stderr.log /workspace/output.json
echo SWEBENCH_PRO_OUTPUT_START
cat /workspace/output.json
echo SWEBENCH_PRO_OUTPUT_END
"""


class SWEBenchProRubric(vf.Rubric):
    def __init__(self, taskset: "SWEBenchProTaskSet", **kwargs: Any):
        super().__init__(**kwargs)
        self.taskset = taskset
        self.add_reward_func(self.solved)

    async def solved(self, state: vf.State, info: dict, **kwargs: Any) -> float:
        if isinstance(state.get("error"), vf.InfraError):
            return 0.0
        sandbox_client = state.get("sandbox_client")
        sandbox_id = state.get("sandbox_id")
        if not sandbox_client or not sandbox_id:
            return 0.0

        try:
            test_output = await self.taskset._run_tests(
                sandbox_client, sandbox_id, state, state.get("test_timeout", 900)
            )
            state["test_output"] = test_output
        except Exception as e:
            logger.warning(f"SWE-bench Pro test execution failed: {e}")
            state["test_output"] = f"ERROR: {e}"
            return 0.0
        return float(self.taskset._calculate_reward(test_output, info))

    @vf.cleanup
    async def cleanup_sandbox(self, state: vf.State) -> None:
        sandbox_client = state.get("sandbox_client")
        sandbox_id = state.get("sandbox_id")
        if sandbox_client and sandbox_id:
            try:
                await sandbox_client.delete(sandbox_id)
            except Exception:
                pass


class SWEBenchProTaskSet(SandboxTaskSet):
    """SWE-bench Pro using ScaleAI rows and DockerHub images."""

    default_workdir = DEFAULT_AGENT_WORKDIR

    def __init__(
        self,
        dataset_name: str = DEFAULT_DATASET_NAME,
        dataset_split: str = DEFAULT_DATASET_SPLIT,
        instance_ids: list[str] | None = None,
        repos: list[str] | None = None,
        max_examples: int = -1,
        agent_workdir: str = DEFAULT_AGENT_WORKDIR,
        docker_image_repository: str = DEFAULT_DOCKER_IMAGE_REPOSITORY,
        run_scripts_url: str = DEFAULT_RUN_SCRIPTS_URL,
        cpu_cores: int = 4,
        memory_gb: int = 8,
        disk_size_gb: int = 20,
        gpu_count: int = 0,
        timeout_minutes: int = 180,
        name: str = "swe/swebench-pro",
    ):
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.instance_ids = set(instance_ids or [])
        self.repos = set(repos or [])
        self.max_examples = max_examples
        self.agent_workdir = agent_workdir
        self.docker_image_repository = docker_image_repository
        self.run_scripts_url = run_scripts_url.rstrip("/")
        self.cpu_cores = cpu_cores
        self.memory_gb = memory_gb
        self.disk_size_gb = disk_size_gb
        self.gpu_count = gpu_count
        self.timeout_minutes = timeout_minutes
        super().__init__(dataset=self._build_dataset(), name=name)

    def _build_dataset(self) -> Dataset:
        rows = []
        for raw_example in load_dataset(self.dataset_name, split=self.dataset_split):
            example = self._normalize_example(raw_example)
            if self.instance_ids and example["instance_id"] not in self.instance_ids:
                continue
            if self.repos and example["repo"] not in self.repos:
                continue
            rows.append(
                {
                    "question": self.get_instruction(example),
                    "info": example,
                    "answer": "",
                }
            )
            if self.max_examples > -1 and len(rows) >= self.max_examples:
                break

        if not rows:
            raise ValueError("No SWE-bench Pro examples matched the requested filters.")
        return Dataset.from_list(rows)

    def _normalize_example(self, example: dict[str, Any]) -> dict[str, Any]:
        row = {
            "repo": str(example["repo"]),
            "instance_id": str(example["instance_id"]),
            "base_commit": str(example["base_commit"]),
            "patch": str(example["patch"]),
            "test_patch": str(example["test_patch"]),
            "problem_statement": str(example["problem_statement"]),
            "requirements": str(example["requirements"]),
            "interface": str(example["interface"]),
            "repo_language": str(example["repo_language"]),
            "fail_to_pass": str(example["fail_to_pass"]),
            "pass_to_pass": str(example["pass_to_pass"]),
            "issue_specificity": str(example["issue_specificity"]),
            "issue_categories": str(example["issue_categories"]),
            "before_repo_set_cmd": str(example["before_repo_set_cmd"]),
            "selected_test_files_to_run": str(example["selected_test_files_to_run"]),
            "dockerhub_tag": str(example["dockerhub_tag"]),
        }
        row["docker_image"] = f"{self.docker_image_repository}:{row['dockerhub_tag']}"
        return row

    def get_instruction(self, info: dict) -> str:
        return _INSTRUCTION_TEMPLATE.format(
            problem_statement=dedent(info["problem_statement"]).strip(),
            repo=info["repo"],
            base_commit=info["base_commit"],
            instance_id=info["instance_id"],
        )

    def get_sandbox_spec(self, info: dict) -> SandboxSpec:
        return SandboxSpec(
            image=info["docker_image"],
            cpu_cores=self.cpu_cores,
            memory_gb=self.memory_gb,
            disk_size_gb=self.disk_size_gb,
            gpu_count=self.gpu_count,
            timeout_minutes=self.timeout_minutes,
        )

    def get_workdir(self, info: dict) -> str:
        return self.agent_workdir

    def get_rubric(self) -> vf.Rubric:
        return SWEBenchProRubric(self)

    async def _run_tests(
        self,
        sandbox_client: Any,
        sandbox_id: str,
        state: dict,
        test_timeout: int,
    ) -> str:
        info = state["info"]
        base_commit = shlex.quote(info["base_commit"])
        agent_workdir = shlex.quote(self.agent_workdir)

        result = await sandbox_client.execute_command(
            sandbox_id,
            _CAPTURE_PATCH_COMMAND.format(
                agent_workdir=agent_workdir,
                base_commit=base_commit,
            ),
            timeout=120,
        )
        if result.exit_code != 0:
            output = ((result.stdout or "") + (result.stderr or ""))[:500]
            raise RuntimeError(f"patch capture failed: {output}")

        instance_id = info["instance_id"]
        for script_name in ("run_script.sh", "parser.py"):
            url = f"{self.run_scripts_url}/{instance_id}/{script_name}"
            with urllib.request.urlopen(url, timeout=60) as response:
                content = response.read().decode()
            with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
                f.write(content)
                f.flush()
                local_path = f.name
            try:
                await sandbox_client.upload_file(
                    sandbox_id, f"/workspace/{script_name}", local_path
                )
            finally:
                Path(local_path).unlink(missing_ok=True)

        selected_tests = ",".join(ast.literal_eval(info["selected_test_files_to_run"]))
        command = _RUN_TESTS_COMMAND.format(
            agent_workdir=agent_workdir,
            base_commit=base_commit,
            test_setup_command=info["before_repo_set_cmd"].strip(),
            selected_tests=shlex.quote(selected_tests),
        )
        result = await sandbox_client.run_background_job(
            sandbox_id,
            f"bash -lc {shlex.quote(command)} > /logs/verifier/scoring.log 2>&1",
            timeout=test_timeout,
        )
        if result.exit_code != 0:
            raise RuntimeError(f"test command failed: exit_code={result.exit_code}")
        result = await sandbox_client.execute_command(
            sandbox_id,
            "cat /logs/verifier/scoring.log",
            timeout=60,
        )
        return (result.stdout or "").strip()

    def _calculate_reward(self, test_output: str, info: dict) -> float:
        if not test_output:
            return 0.0
        match = re.search(
            r"SWEBENCH_PRO_OUTPUT_START\s*(\{.*?\})\s*SWEBENCH_PRO_OUTPUT_END",
            test_output,
            re.DOTALL,
        )
        if not match:
            return 0.0
        parsed = json.loads(match.group(1))
        passed_tests = {
            test["name"]
            for test in parsed.get("tests", [])
            if test["status"] == "PASSED"
        }
        expected = set(ast.literal_eval(info["fail_to_pass"])) | set(
            ast.literal_eval(info["pass_to_pass"])
        )
        if not expected:
            return 0.0
        return float(expected <= passed_tests)

    async def _apply_gold_patch(
        self, sandbox_client: Any, sandbox_id: str, state: dict
    ) -> None:
        patch = state["info"].get("patch", "")
        if not patch.strip():
            raise RuntimeError("No gold patch in info['patch']")

        with tempfile.NamedTemporaryFile(suffix=".patch", mode="w", delete=False) as f:
            f.write(patch)
            f.flush()
            local_path = f.name
        try:
            await sandbox_client.upload_file(sandbox_id, "/tmp/gold.patch", local_path)
        finally:
            Path(local_path).unlink(missing_ok=True)

        result = await sandbox_client.execute_command(
            sandbox_id,
            "git apply -v /tmp/gold.patch",
            working_dir=self.agent_workdir,
            timeout=60,
        )
        if result.exit_code != 0:
            output = ((result.stdout or "") + (result.stderr or ""))[:500]
            raise RuntimeError(f"gold patch apply failed: {output}")

    async def validate_instance(self, state) -> bool:
        sandbox_client = state["sandbox_client"]
        sandbox_id = state["sandbox_id"]
        await self._apply_gold_patch(sandbox_client, sandbox_id, state)
        test_output = await self._run_tests(
            sandbox_client, sandbox_id, state, state.get("test_timeout", 900)
        )
        state["test_output"] = test_output
        return self._calculate_reward(test_output, state["info"]) > 0
