from __future__ import annotations

import json
import shlex
from typing import Any

from datasets import Dataset, load_dataset

import verifiers as vf
from verifiers.envs.experimental.tasksets.base import SandboxSpec, Task, TaskSet
from verifiers.types import Messages, State, UserMessage

DEFAULT_DATASET_NAME = "princeton-nlp/SWE-bench_Verified"
DEFAULT_DATASET_SPLIT = "test"
DEFAULT_AGENT_WORKDIR = "/testbed"

PROMPT_TEMPLATE = """\
You are working inside a prebuilt SWE-bench Verified sandbox.

Repository: {repo}
Base commit: {base_commit}
Working directory: {agent_workdir}

Problem statement:
{problem_statement}
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_metric(self.has_patch)
        self.add_metric(self.changed_files_count)

    async def has_patch(self, state: State) -> float:
        return float(bool((state.get("generated_patch") or "").strip()))

    async def changed_files_count(self, state: State) -> float:
        changed_files = state.get("changed_files") or ""
        return float(len([line for line in changed_files.splitlines() if line.strip()]))


class SWEBenchVerifiedTaskSet(TaskSet, Task):
    """TaskSet for SWE-bench Verified instance images without the swebench package."""

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
        include_hints: bool = True,
    ):
        Task.__init__(self)
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
        self.include_hints = include_hints
        self.dataset = self._load_dataset()

    def get_dataset(self) -> Dataset:
        return self.dataset

    def get_task(self, state: State) -> Task:
        return self

    def build_monitor_rubric(self) -> vf.Rubric | None:
        return SWEBenchVerifiedMonitorRubric()

    def _load_dataset(self) -> Dataset:
        dataset_obj = load_dataset(self.dataset_name, split=self.dataset_split)
        if not isinstance(dataset_obj, Dataset):
            raise TypeError(
                "Expected a Dataset for the requested SWE-bench Verified split."
            )

        rows: list[dict[str, Any]] = []
        for example in dataset_obj:
            repo = str(example["repo"])
            instance_id = str(example["instance_id"])

            if self.instance_ids and instance_id not in self.instance_ids:
                continue
            if self.repos and repo not in self.repos:
                continue

            fail_to_pass = _parse_test_list(example.get("FAIL_TO_PASS"))
            pass_to_pass = _parse_test_list(example.get("PASS_TO_PASS"))
            info = {
                **dict(example),
                "FAIL_TO_PASS": fail_to_pass,
                "PASS_TO_PASS": pass_to_pass,
                "docker_image": build_swebench_image_name(
                    instance_id,
                    namespace=self.docker_namespace,
                    arch=self.docker_arch,
                    tag=self.docker_tag,
                ),
                "agent_workdir": self.agent_workdir,
            }

            rows.append(
                {
                    "example_id": len(rows),
                    "task": instance_id,
                    "question": self.render_prompt(info),
                    "answer": "",
                    "info": info,
                }
            )

            if self.max_examples > -1 and len(rows) >= self.max_examples:
                break

        if not rows:
            raise ValueError(
                "No SWE-bench Verified examples matched the requested filters."
            )

        return Dataset.from_list(rows)

    def render_prompt(self, info: dict[str, Any]) -> str:
        prompt = PROMPT_TEMPLATE.format(
            repo=info["repo"],
            base_commit=info["base_commit"],
            agent_workdir=self.agent_workdir,
            problem_statement=info["problem_statement"].strip(),
        ).rstrip()

        if self.include_hints:
            hints_text = str(info.get("hints_text") or "").strip()
            if hints_text:
                prompt += f"\n\nHints:\n{hints_text}"

        fail_to_pass = info.get("FAIL_TO_PASS") or []
        if fail_to_pass:
            prompt += "\n\nTests that should pass after your fix:\n"
            prompt += "\n".join(f"- {test_name}" for test_name in fail_to_pass)

        pass_to_pass = info.get("PASS_TO_PASS") or []
        if pass_to_pass:
            prompt += "\n\nRegression tests to preserve:\n"
            prompt += "\n".join(f"- {test_name}" for test_name in pass_to_pass)

        prompt += (
            "\n\nInspect the repository, make the minimal code changes needed, "
            "and leave the working tree with your patch applied."
        )
        return prompt

    async def prompt(self, state: State) -> Messages:
        return [UserMessage(content=str(state["question"]))]

    async def get_sandbox_spec(self, state: State) -> SandboxSpec:
        info = state.get("info") or {}
        return SandboxSpec(
            docker_image=info["docker_image"],
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

    async def build_env_vars(self, state: State) -> dict[str, str]:
        info = state.get("info") or {}
        return {
            "AGENT_WORKDIR": self.agent_workdir,
            "SWE_INSTANCE_ID": str(info.get("instance_id", "")),
            "SWE_REPO": str(info.get("repo", "")),
            "SWE_BASE_COMMIT": str(info.get("base_commit", "")),
        }

    async def setup(self, env: Any, state: State) -> None:
        state["repo_path"] = self.agent_workdir

    async def post_rollout(self, env: Any, state: State) -> None:
        sandbox_id = state.get("sandbox_id")
        if not sandbox_id:
            return

        workdir = shlex.quote(self.agent_workdir)
        commands = {
            "changed_files": f"git -C {workdir} diff --name-only",
            "generated_patch": f"git -C {workdir} diff",
            "generated_patch_stat": f"git -C {workdir} diff --stat",
            "repo_status": f"git -C {workdir} status --short",
        }

        for state_key, command in commands.items():
            try:
                result = await env.sandbox_client.execute_command(
                    sandbox_id,
                    command,
                    working_dir=None,
                )
            except Exception as exc:
                state[f"{state_key}_error"] = str(exc)
                continue

            stdout = getattr(result, "stdout", "")
            state[state_key] = stdout if isinstance(stdout, str) else str(stdout)
