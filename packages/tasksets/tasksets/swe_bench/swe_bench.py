from __future__ import annotations

import ast
import json
import tempfile
from pathlib import Path
from textwrap import dedent
from typing import Any

from datasets import Dataset, load_dataset

from tasksets.harbor import HarborTaskSet

DEFAULT_PRO_DATASET_NAME = "ScaleAI/SWE-bench_Pro"
DEFAULT_VERIFIED_DATASET_NAME = "princeton-nlp/SWE-bench_Verified"
DEFAULT_DATASET_SPLIT = "test"
DEFAULT_AGENT_WORKDIR = "/app"
DEFAULT_DOCKERHUB_USERNAME = "jefzda"
DEFAULT_DOCKER_IMAGE_REPO = "sweap-images"

_INSTRUCTION_TEMPLATE = """\
# Task

{problem_statement}

---

**Repo:** `{repo}`
**Version:** `{version}`
**Base commit:** `{base_commit}`
**Instance ID:** `{instance_id}`
"""

_TASK_TOML_TEMPLATE = """\
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
start_command = "tail -f /dev/null"
cpus = {cpu_cores}
memory = {memory}
storage = {storage}
gpus = {gpu_count}
"""

_DOCKERFILE_TEMPLATE = """\
FROM {docker_image}

WORKDIR {agent_workdir}
"""

_TEST_SH_TEMPLATE = """\
#!/bin/bash
set -euo pipefail

mkdir -p /logs/verifier
cd {agent_workdir}

git status --short > /logs/verifier/repo_status.txt || true
git diff --name-only > /logs/verifier/changed_files.txt || true
git diff --stat > /logs/verifier/generated_patch_stat.txt || true
git diff --binary > /logs/verifier/generated_patch.diff || true

if [ -s /logs/verifier/generated_patch.diff ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
"""

_SOLVE_SH_TEMPLATE = """\
#!/bin/bash
set -euo pipefail

cat > {agent_workdir}/solution_patch.diff << '__SOLUTION__'
{patch}
__SOLUTION__

cd {agent_workdir}
git apply --whitespace=fix solution_patch.diff || patch --fuzz=5 -p1 -i solution_patch.diff
"""


def _parse_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if not isinstance(value, str) or not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(value)
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


class SWEBenchProTaskSet(HarborTaskSet):
    """SWE-bench Pro as generated Harbor tasks."""

    env_id = "swebench-pro"
    default_workdir = DEFAULT_AGENT_WORKDIR

    def __init__(
        self,
        dataset_name: str = DEFAULT_PRO_DATASET_NAME,
        dataset_split: str = DEFAULT_DATASET_SPLIT,
        instance_ids: list[str] | None = None,
        repos: list[str] | None = None,
        max_examples: int = -1,
        agent_workdir: str = DEFAULT_AGENT_WORKDIR,
        dockerhub_username: str = DEFAULT_DOCKERHUB_USERNAME,
        docker_image_repo: str = DEFAULT_DOCKER_IMAGE_REPO,
        docker_namespace: str = "swebench",
        docker_arch: str = "x86_64",
        docker_tag: str = "latest",
        cpu_cores: int = 4,
        memory_gb: int = 8,
        disk_size_gb: int = 20,
        gpu_count: int = 0,
        timeout_minutes: int = 180,
        harness_config: dict[str, Any] | None = None,
        name: str = "swe/swebench-pro",
    ):
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.instance_ids = set(instance_ids or [])
        self.repos = set(repos or [])
        self.max_examples = max_examples
        self.agent_workdir = agent_workdir
        self.dockerhub_username = dockerhub_username
        self.docker_image_repo = docker_image_repo
        self.docker_namespace = docker_namespace
        self.docker_arch = docker_arch
        self.docker_tag = docker_tag
        self.cpu_cores = cpu_cores
        self.memory_gb = memory_gb
        self.disk_size_gb = disk_size_gb
        self.gpu_count = gpu_count
        self.timeout_minutes = timeout_minutes
        self.harness_config = dict(
            harness_config
            or {
                "agent": "openclaw",
                "agent_workdir": agent_workdir,
            }
        )
        self._generated_tasks_dir = tempfile.TemporaryDirectory(
            prefix="swebench_pro_harbor_"
        )
        self.generated_tasks_path = Path(self._generated_tasks_dir.name)
        self._generate_harbor_tasks()
        super().__init__(
            dataset_path=self.generated_tasks_path,
            agent_workdir=agent_workdir,
            name=name,
        )

    def get_env_vars(self, info: dict | None = None) -> dict[str, str]:
        env_vars = super().get_env_vars(info)
        if info:
            env_vars.update(
                {
                    "SWE_INSTANCE_ID": str(info.get("instance_id", "")),
                    "SWE_REPO": str(info.get("repo", "")),
                    "SWE_BASE_COMMIT": str(info.get("base_commit", "")),
                }
            )
        return env_vars

    def _generate_harbor_tasks(self) -> None:
        dataset_obj = load_dataset(self.dataset_name, split=self.dataset_split)
        if not isinstance(dataset_obj, Dataset):
            raise TypeError("Expected a Dataset for the requested SWE-bench split.")

        generated_count = 0
        for raw_example in dataset_obj:
            example = self.normalize_example(raw_example)
            if self.instance_ids and example["instance_id"] not in self.instance_ids:
                continue
            if self.repos and example["repo"] not in self.repos:
                continue

            self.generate_harbor_task(example, self.generated_tasks_path)
            generated_count += 1
            if self.max_examples > -1 and generated_count >= self.max_examples:
                break

        if generated_count == 0:
            raise ValueError("No SWE-bench examples matched the requested filters.")

    def normalize_example(self, example: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(example)
        normalized["instance_id"] = str(normalized["instance_id"])
        normalized["repo"] = str(
            normalized.get("repo") or normalized.get("repo_name", "")
        )
        normalized["problem_statement"] = str(
            normalized.get("problem_statement")
            or normalized.get("issue")
            or normalized.get("prompt")
            or ""
        )
        normalized["base_commit"] = str(normalized.get("base_commit", ""))
        normalized["version"] = str(normalized.get("version", ""))
        normalized["FAIL_TO_PASS"] = _parse_list(
            normalized.get("FAIL_TO_PASS", normalized.get("fail_to_pass"))
        )
        normalized["PASS_TO_PASS"] = _parse_list(
            normalized.get("PASS_TO_PASS", normalized.get("pass_to_pass"))
        )
        return normalized

    def build_docker_image(self, example: dict[str, Any]) -> str:
        dockerhub_tag = example.get("dockerhub_tag")
        if dockerhub_tag:
            return f"{self.dockerhub_username}/{self.docker_image_repo}:{dockerhub_tag}"
        return build_swebench_image_name(
            example["instance_id"],
            namespace=self.docker_namespace,
            arch=self.docker_arch,
            tag=self.docker_tag,
        )

    def render_instruction(self, example: dict[str, Any]) -> str:
        return _INSTRUCTION_TEMPLATE.format(
            problem_statement=dedent(example["problem_statement"]).strip(),
            repo=example["repo"],
            version=example["version"],
            base_commit=example["base_commit"],
            instance_id=example["instance_id"],
        )

    def render_harness_toml(self) -> str:
        lines = ["[agent.harness]"]
        for key, value in self.harness_config.items():
            if value is not None:
                lines.append(f"{key} = {self.toml_value(value)}")
        return "\n".join(lines)

    def toml_value(self, value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int | float):
            return str(value)
        if isinstance(value, list):
            return "[" + ", ".join(self.toml_value(item) for item in value) + "]"
        return json.dumps(str(value))

    def render_task_toml(self, example: dict[str, Any], docker_image: str) -> str:
        return _TASK_TOML_TEMPLATE.format(
            difficulty=json.dumps(str(example.get("difficulty") or "hard")),
            timeout_seconds=float(self.timeout_minutes * 60),
            harness_toml=self.render_harness_toml(),
            docker_image=json.dumps(docker_image),
            cpu_cores=self.cpu_cores,
            memory=json.dumps(f"{self.memory_gb}G"),
            storage=json.dumps(f"{self.disk_size_gb}G"),
            gpu_count=self.gpu_count,
        )

    def generate_harbor_task(self, example: dict[str, Any], output_dir: Path) -> Path:
        task_dir = output_dir / example["instance_id"]
        (task_dir / "environment").mkdir(parents=True, exist_ok=True)
        (task_dir / "tests").mkdir(parents=True, exist_ok=True)
        (task_dir / "solution").mkdir(parents=True, exist_ok=True)

        docker_image = self.build_docker_image(example)
        (task_dir / "instruction.md").write_text(self.render_instruction(example))
        (task_dir / "task.toml").write_text(
            self.render_task_toml(example, docker_image)
        )
        (task_dir / "info.json").write_text(json.dumps(example, indent=2, default=str))
        (task_dir / "environment" / "Dockerfile").write_text(
            _DOCKERFILE_TEMPLATE.format(
                docker_image=docker_image,
                agent_workdir=self.agent_workdir,
            )
        )

        test_sh = task_dir / "tests" / "test.sh"
        test_sh.write_text(_TEST_SH_TEMPLATE.format(agent_workdir=self.agent_workdir))
        test_sh.chmod(0o755)

        solve_sh = task_dir / "solution" / "solve.sh"
        solve_sh.write_text(
            _SOLVE_SH_TEMPLATE.format(
                agent_workdir=self.agent_workdir,
                patch=(example.get("patch") or "").strip(),
            )
        )
        solve_sh.chmod(0o755)
        return task_dir


class SWEBenchVerifiedTaskSet(SWEBenchProTaskSet):
    def __init__(self, **kwargs: Any):
        kwargs.setdefault("dataset_name", DEFAULT_VERIFIED_DATASET_NAME)
        kwargs.setdefault("agent_workdir", "/testbed")
        kwargs.setdefault("name", "swe/swebench-verified")
        super().__init__(**kwargs)


SWEBenchTaskSet = SWEBenchProTaskSet
