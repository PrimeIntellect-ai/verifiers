from __future__ import annotations

import shlex
from textwrap import dedent
from typing import Any

import verifiers as vf
from datasets import Dataset, load_dataset

from verifiers.envs.experimental.composable import SandboxSpec, SandboxTaskSet

DEFAULT_DATASET_NAME = "ScaleAI/SWE-bench_Pro"
DEFAULT_DATASET_SPLIT = "test"
DEFAULT_AGENT_WORKDIR = "/app"
DEFAULT_DOCKER_IMAGE_REPOSITORY = "jefzda/sweap-images"

_INSTRUCTION_TEMPLATE = """\
# Task

{problem_statement}

---

**Repo:** `{repo}`
**Base commit:** `{base_commit}`
**Instance ID:** `{instance_id}`
"""

_PATCH_CHECK_COMMAND = """\
set -e
mkdir -p /logs/verifier
cd {agent_workdir}

git status --short > /logs/verifier/repo_status.txt || true
git diff --name-only > /logs/verifier/changed_files.txt || true
git diff --stat > /logs/verifier/generated_patch_stat.txt || true
git diff --binary > /logs/verifier/generated_patch.diff || true

if [ -s /logs/verifier/generated_patch.diff ]; then
  echo 1
else
  echo 0
fi
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

        result = await sandbox_client.execute_command(
            sandbox_id,
            _PATCH_CHECK_COMMAND.format(
                agent_workdir=shlex.quote(self.taskset.agent_workdir)
            ),
            timeout=900,
        )
        output = (result.stdout or "").strip()
        state["test_output"] = output
        return float(output or 0)

    @vf.cleanup
    async def cleanup_sandbox(self, state: vf.State) -> None:
        sandbox_client = state.get("sandbox_client")
        sandbox_id = state.get("sandbox_id")
        if sandbox_client and sandbox_id:
            await sandbox_client.delete(sandbox_id)


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
