from __future__ import annotations

import json
import shlex
import tempfile
from pathlib import Path
from textwrap import dedent
from typing import Any

from datasets import Dataset, load_dataset

import verifiers as vf
from verifiers.envs.experimental.tasksets.base import SandboxSpec
from verifiers.envs.experimental.tasksets.harbor_base import HarborTask, HarborTaskSet
from verifiers.types import State

DEFAULT_DATASET_NAME = "princeton-nlp/SWE-bench_Verified"
DEFAULT_DATASET_SPLIT = "test"
DEFAULT_AGENT_WORKDIR = "/testbed"

_HARBOR_INSTRUCTION_TEMPLATE = """\
# Task

{problem_statement}

---

**Repo:** `{repo}`
**Version:** `{version}`
**Base commit:** `{base_commit}`
**Instance ID:** `{instance_id}`
"""

_HARBOR_TASK_TOML_TEMPLATE = """\
[metadata]
author_name = "unknown"
author_email = "unknown"
difficulty = {difficulty}
category = "debugging"
tags = ["debugging", "swe-bench"]

[verifier]
timeout_sec = {timeout_seconds}

[agent]
timeout_sec = {timeout_seconds}
{harness_toml}

[environment]
docker_image = {docker_image}
start_command = {start_command}
cpus = {cpu_cores}
memory = {memory}
storage = {storage}
gpus = {gpu_count}
"""

_HARBOR_DOCKERFILE_TEMPLATE = """\
FROM {docker_image}

WORKDIR {agent_workdir}
"""

_HARBOR_TEST_SH_TEMPLATE = """\
#!/bin/bash
set -euo pipefail

mkdir -p /logs/verifier
cd {agent_workdir}

git status --short > /logs/verifier/repo_status.txt || true
git diff --name-only > /logs/verifier/changed_files.txt || true
git diff --stat > /logs/verifier/generated_patch_stat.txt || true
git diff > /logs/verifier/generated_patch.diff || true

if [ -s /logs/verifier/generated_patch.diff ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi

exit 0
"""

_HARBOR_SOLVE_SH_TEMPLATE = """\
#!/bin/bash
set -euo pipefail

cat > {agent_workdir}/solution_patch.diff << '__SOLUTION__'
{patch}
__SOLUTION__

cd {agent_workdir}

git apply --whitespace=fix solution_patch.diff || patch --fuzz=5 -p1 -i solution_patch.diff
"""


def _parse_test_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str) and value:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return []


def build_swebench_image_name(
    instance_id: str,
    *,
    namespace: str = "swebench",
    arch: str = "x86_64",
    tag: str = "latest",
) -> str:
    image = f"sweb.eval.{arch}.{instance_id.lower()}:{tag}".replace("__", "_1776_")
    if namespace:
        return f"{namespace}/{image}"
    return image


class SWEBenchVerifiedMonitorRubric(vf.Rubric):
    """Monitor metrics for generated SWE-bench patches."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.add_metric(self.has_patch)
        self.add_metric(self.changed_files_count)

    async def has_patch(self, state: State) -> float:
        return float(bool((state.get("generated_patch") or "").strip()))

    async def changed_files_count(self, state: State) -> float:
        changed_files = state.get("changed_files") or ""
        return float(len([line for line in changed_files.splitlines() if line.strip()]))


class SWEBenchVerifiedTask(HarborTask):
    """Harbor-backed SWE-bench task."""

    def __init__(
        self,
        *,
        info: dict[str, Any],
        agent_workdir: str,
        docker_image: str = "python:3.11-slim",
        team_id: str | None = None,
        advanced_configs: Any | None = None,
        labels: list[str] | None = None,
    ):
        super().__init__(
            task_dir=info["task_dir"],
            config=dict(info.get("config") or {}),
            task_name=str(info.get("task_name") or info.get("instance_id") or ""),
            agent_workdir=agent_workdir,
            docker_image=docker_image,
        )
        self.info = dict(info)
        if self.sandbox is not None:
            self.sandbox.team_id = team_id
            self.sandbox.advanced_configs = advanced_configs
            if labels is not None:
                self.sandbox.labels = list(labels)

    async def build_env_vars(self, state: State) -> dict[str, str]:
        env_vars = await super().build_env_vars(state)
        env_vars.update(
            {
                "SWE_INSTANCE_ID": str(self.info.get("instance_id", "")),
                "SWE_REPO": str(self.info.get("repo", "")),
                "SWE_BASE_COMMIT": str(self.info.get("base_commit", "")),
            }
        )
        return env_vars

    async def post_rollout(self, env: Any, state: State) -> None:
        await super().post_rollout(env, state)

        sandbox_id = state.get("sandbox_id")
        if not sandbox_id:
            return

        artifact_paths = {
            "changed_files": "/logs/verifier/changed_files.txt",
            "generated_patch": "/logs/verifier/generated_patch.diff",
            "generated_patch_stat": "/logs/verifier/generated_patch_stat.txt",
            "repo_status": "/logs/verifier/repo_status.txt",
        }
        for state_key, artifact_path in artifact_paths.items():
            result = await env.sandbox_client.execute_command(
                sandbox_id,
                f"if [ -f {shlex.quote(artifact_path)} ]; then cat {shlex.quote(artifact_path)}; fi",
                working_dir=None,
            )
            stdout = getattr(result, "stdout", "")
            state[state_key] = stdout if isinstance(stdout, str) else str(stdout)


class SWEBenchVerifiedTaskSet(HarborTaskSet):
    """SWE-bench Verified implemented as generated Harbor tasks."""

    def __init__(
        self,
        dataset_name: str = DEFAULT_DATASET_NAME,
        dataset_split: str = DEFAULT_DATASET_SPLIT,
        instance_ids: list[str] | None = None,
        repos: list[str] | None = None,
        max_examples: int = -1,
        agent_workdir: str = DEFAULT_AGENT_WORKDIR,
        docker_namespace: str = "swebench",
        docker_arch: str = "x86_64",
        docker_tag: str = "latest",
        start_command: str = "tail -f /dev/null",
        cpu_cores: int = 4,
        memory_gb: int = 8,
        disk_size_gb: int = 20,
        gpu_count: int = 0,
        timeout_minutes: int = 180,
        team_id: str | None = None,
        advanced_configs: Any | None = None,
        labels: list[str] | None = None,
        harness_config: dict[str, Any] | None = None,
    ):
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.instance_ids = set(instance_ids or [])
        self.repos = set(repos or [])
        self.max_examples = max_examples
        self.agent_workdir = agent_workdir
        self.docker_namespace = docker_namespace
        self.docker_arch = docker_arch
        self.docker_tag = docker_tag
        self.start_command = start_command
        self.cpu_cores = cpu_cores
        self.memory_gb = memory_gb
        self.disk_size_gb = disk_size_gb
        self.gpu_count = gpu_count
        self.timeout_minutes = timeout_minutes
        self.team_id = team_id
        self.advanced_configs = advanced_configs
        self.labels = list(labels or ["swebench-verified"])
        self.harness_config = dict(
            harness_config or {"transport": "interceptor", "agent": "opencode"}
        )
        self._generated_tasks_dir = tempfile.TemporaryDirectory(
            prefix="swebench_harbor_"
        )
        self.generated_tasks_path = Path(self._generated_tasks_dir.name)
        self._generate_harbor_tasks()
        super().__init__(
            dataset_path=self.generated_tasks_path,
            agent_workdir=agent_workdir,
        )

    def create_task(self, info: dict[str, Any]) -> HarborTask:
        return SWEBenchVerifiedTask(
            info=info,
            agent_workdir=self.agent_workdir,
            team_id=self.team_id,
            advanced_configs=self.advanced_configs,
            labels=self.labels,
        )

    def build_monitor_rubric(self) -> vf.Rubric | None:
        return SWEBenchVerifiedMonitorRubric()

    def _generate_harbor_tasks(self) -> None:
        dataset_obj = load_dataset(self.dataset_name, split=self.dataset_split)
        if not isinstance(dataset_obj, Dataset):
            raise TypeError(
                "Expected a Dataset for the requested SWE-bench Verified split."
            )

        generated_count = 0
        for raw_example in dataset_obj:
            repo = str(raw_example["repo"])
            instance_id = str(raw_example["instance_id"])
            if self.instance_ids and instance_id not in self.instance_ids:
                continue
            if self.repos and repo not in self.repos:
                continue

            example = self.normalize_example(raw_example)
            self.generate_harbor_task(example, self.generated_tasks_path)
            generated_count += 1

            if self.max_examples > -1 and generated_count >= self.max_examples:
                break

        if generated_count == 0:
            raise ValueError(
                "No SWE-bench Verified examples matched the requested filters."
            )

    def normalize_example(self, example: dict[str, Any]) -> dict[str, Any]:
        return {
            **dict(example),
            "FAIL_TO_PASS": _parse_test_list(example.get("FAIL_TO_PASS")),
            "PASS_TO_PASS": _parse_test_list(example.get("PASS_TO_PASS")),
        }

    def build_sandbox_spec(self, example: dict[str, Any]) -> SandboxSpec:
        return SandboxSpec(
            docker_image=build_swebench_image_name(
                str(example["instance_id"]),
                namespace=self.docker_namespace,
                arch=self.docker_arch,
                tag=self.docker_tag,
            ),
            start_command=self.start_command,
            cpu_cores=self.cpu_cores,
            memory_gb=self.memory_gb,
            disk_size_gb=self.disk_size_gb,
            gpu_count=self.gpu_count,
            timeout_minutes=self.timeout_minutes,
            team_id=self.team_id,
            advanced_configs=self.advanced_configs,
            labels=list(self.labels),
        )

    def render_task_toml(self, example: dict[str, Any], sandbox: SandboxSpec) -> str:
        return _HARBOR_TASK_TOML_TEMPLATE.format(
            difficulty=json.dumps(str(example.get("difficulty") or "hard")),
            timeout_seconds=float(self.timeout_minutes * 60),
            harness_toml=self.render_harness_toml(),
            docker_image=json.dumps(sandbox.docker_image),
            start_command=json.dumps(sandbox.start_command),
            cpu_cores=sandbox.cpu_cores,
            memory=json.dumps(f"{sandbox.memory_gb}G"),
            storage=json.dumps(f"{sandbox.disk_size_gb}G"),
            gpu_count=sandbox.gpu_count,
        )

    def render_harness_toml(self) -> str:
        lines = ["[agent.harness]"]
        for key, value in self.harness_config.items():
            if value is None:
                continue
            lines.append(f"{key} = {self._toml_value(value)}")
        return "\n".join(lines)

    def _toml_value(self, value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str):
            return json.dumps(value)
        if isinstance(value, list):
            return "[" + ", ".join(self._toml_value(item) for item in value) + "]"
        raise TypeError(f"Unsupported TOML value for harness config: {value!r}")

    def render_instruction(self, example: dict[str, Any]) -> str:
        return _HARBOR_INSTRUCTION_TEMPLATE.format(
            problem_statement=dedent(str(example["problem_statement"])).strip(),
            repo=example["repo"],
            version=example["version"],
            base_commit=example["base_commit"],
            instance_id=example["instance_id"],
        )

    def render_test_script(self) -> str:
        return _HARBOR_TEST_SH_TEMPLATE.format(agent_workdir=self.agent_workdir)

    def render_solution_script(self, example: dict[str, Any]) -> str:
        return _HARBOR_SOLVE_SH_TEMPLATE.format(
            agent_workdir=self.agent_workdir,
            patch=(example.get("patch") or "").strip(),
        )

    def render_dockerfile(self, sandbox: SandboxSpec) -> str:
        return _HARBOR_DOCKERFILE_TEMPLATE.format(
            docker_image=sandbox.docker_image,
            agent_workdir=self.agent_workdir,
        )

    def generate_harbor_task(
        self,
        example: dict[str, Any],
        output_dir: str | Path,
    ) -> Path:
        output_dir = Path(output_dir)
        task_dir = output_dir / str(example["instance_id"])
        (task_dir / "environment").mkdir(parents=True, exist_ok=True)
        (task_dir / "tests").mkdir(parents=True, exist_ok=True)
        (task_dir / "solution").mkdir(parents=True, exist_ok=True)

        sandbox = self.build_sandbox_spec(example)
        (task_dir / "instruction.md").write_text(self.render_instruction(example))
        (task_dir / "task.toml").write_text(self.render_task_toml(example, sandbox))
        (task_dir / "info.json").write_text(json.dumps(example, indent=2, default=str))

        test_sh_path = task_dir / "tests" / "test.sh"
        test_sh_path.write_text(self.render_test_script())
        test_sh_path.chmod(0o755)

        (task_dir / "environment" / "Dockerfile").write_text(
            self.render_dockerfile(sandbox)
        )

        solve_sh_path = task_dir / "solution" / "solve.sh"
        solve_sh_path.write_text(self.render_solution_script(example))
        solve_sh_path.chmod(0o755)
        return task_dir
