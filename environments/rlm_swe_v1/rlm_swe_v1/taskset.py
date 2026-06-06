import json
import re
from typing import cast

from datasets import load_dataset
from harnesses import RLM, RLMConfig

import verifiers.v1 as vf

REGISTRY_PREFIX = "us-central1-docker.pkg.dev/prime-intellect-platform/prod-sandbox"
DEFAULT_DATASET_NAME = "R2E-Gym/R2E-Gym-Subset"
DEFAULT_REPO_PATH = "/testbed"
DEFAULT_RLM_TOOLS = ("bash", "edit")


class RlmSweTasksetConfig(vf.TasksetConfig):
    id: str = "swe/r2e"
    dataset_name: str = DEFAULT_DATASET_NAME
    repo_path: str = DEFAULT_REPO_PATH
    filter_repos: list[str] | None = None
    ds_num_proc: int | None = None
    ds_keep_in_memory: bool = True
    timeout_minutes: int | None = None
    env: vf.JsonData | None = None


class RlmSweHarnessConfig(RLMConfig):
    cwd: str | None = DEFAULT_REPO_PATH
    tools: list[str] = list(DEFAULT_RLM_TOOLS)


class R2ESWETask(vf.Task):
    question: str
    instruction: str
    answer: str
    info: vf.JsonData
    sandbox: vf.JsonData
    program: vf.JsonData


class R2ESWETaskset(vf.Taskset[RlmSweTasksetConfig]):
    task_type = R2ESWETask

    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        if split == "eval":
            return []
        return load_tasks(
            dataset_name=self.config.dataset_name,
            repo_path=self.config.repo_path,
            filter_repos=self.config.filter_repos,
            ds_num_proc=self.config.ds_num_proc,
            ds_keep_in_memory=self.config.ds_keep_in_memory,
            timeout_minutes=self.config.timeout_minutes,
            env=dict(self.config.env or {}),
        )

    @vf.reward(weight=1.0)
    async def solved(self, task: R2ESWETask, state: vf.State) -> float:
        test_output = state.artifacts.get("test_output")
        if not isinstance(test_output, str):
            command = state.artifacts.get("command")
            if isinstance(command, dict):
                test_output = str(command.get("stdout") or command.get("stderr") or "")
        return float(calculate_reward(test_output or "", task.info))


def load_tasks(
    dataset_name: str = DEFAULT_DATASET_NAME,
    repo_path: str = DEFAULT_REPO_PATH,
    filter_repos: list[str] | None = None,
    ds_num_proc: int | None = None,
    ds_keep_in_memory: bool = True,
    timeout_minutes: int | None = None,
    env: dict[str, object] | None = None,
) -> list[vf.JsonData]:
    dataset_kwargs = dict(
        num_proc=ds_num_proc,
        keep_in_memory=ds_keep_in_memory,
        load_from_cache_file=False,
    )
    dataset = load_dataset(
        dataset_name,
        split="train",
        keep_in_memory=ds_keep_in_memory,
        num_proc=ds_num_proc,
    )
    if filter_repos:
        filter_set = frozenset(filter_repos)
        dataset = dataset.filter(
            lambda row: row.get("repo_name") not in filter_set,
            **dataset_kwargs,
        )
    dataset = dataset.map(
        process_r2e_example,
        remove_columns=dataset.column_names,
        **dataset_kwargs,
    )
    task_env = dict(env or {})
    rows: list[vf.JsonData] = []
    for index, row in enumerate(dataset):
        row = dict(row)
        info = dict(row["info"])
        instruction = str(info["problem_statement"])
        program_env = env_vars(repo_path=repo_path, env=task_env)
        program_env.setdefault("AGENT_WORKDIR", repo_path)
        rows.append(
            {
                "row_id": index,
                "task_id": info.get("instance_id") or index,
                "question": row.get("question", instruction),
                "instruction": instruction,
                "prompt": [{"role": "user", "content": instruction}],
                "answer": row.get("answer", ""),
                "info": info,
                "sandbox": sandbox_config(
                    info=cast(vf.JsonData, info),
                    repo_path=repo_path,
                    timeout_minutes=timeout_minutes,
                ),
                "program": {"env": program_env},
            }
        )
    return rows


def sandbox_config(
    *, info: vf.JsonData, repo_path: str, timeout_minutes: int | None
) -> vf.JsonData:
    config: vf.JsonData = {
        "image": f"{REGISTRY_PREFIX}/{info['docker_image']}",
        "cpu_cores": 4,
        "memory_gb": 4,
        "disk_size_gb": 10,
        "workdir": repo_path,
    }
    if timeout_minutes is not None:
        config["timeout_minutes"] = timeout_minutes
    return config


def env_vars(*, repo_path: str, env: dict[str, object]) -> dict[str, str]:
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


def process_r2e_example(row: vf.JsonData) -> vf.JsonData:
    info = cast(vf.JsonData, dict(row))
    info.setdefault("instance_id", row.get("commit_hash"))
    info.setdefault("repo", row.get("repo_name"))
    return {
        "question": row["problem_statement"],
        "info": info,
        "answer": "",
    }


def calculate_reward(test_output: str, info: vf.JsonData) -> float:
    parsed = decolor_dict_keys(parse_log_pytest(test_output))
    expected_raw = info["expected_output_json"]
    expected: dict[str, str] = json.loads(str(expected_raw))
    expected = decolor_dict_keys(expected)
    parsed = {key.split(" - ")[0]: parsed[key] for key in sorted(parsed.keys())}
    expected = {key.split(" - ")[0]: expected[key] for key in sorted(expected.keys())}
    if any(not key for key in parsed):
        return 0.0
    if len(parsed) != len(expected):
        return 0.0
    for key in parsed:
        if key not in expected or parsed[key] != expected[key]:
            return 0.0
    return 1.0


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
    parsed_commit_content: str,
    *,
    test_file: bool = False,
    only_python: bool = True,
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
        patch += f"diff --git a/{path} b/{path}\n"
        for hunk in file_diff.get("hunks", []):
            descriptor = hunk.get("descriptor", {})
            old_range = descriptor.get("old_range", {})
            new_range = descriptor.get("new_range", {})
            patch += (
                f"@@ -{old_range.get('start', 0)} +{new_range.get('start', 0)} @@\n"
            )
            for line in hunk.get("line_group", {}).get("all_lines", []):
                content = line.get("content", "")
                line_type = line.get("type", "")
                if line_type == "context":
                    patch += f" {content}\n"
                elif line_type == "added":
                    patch += f"+{content}\n"
                elif line_type == "deleted":
                    patch += f"-{content}\n"
    return patch


def load_taskset(config: RlmSweTasksetConfig) -> R2ESWETaskset:
    return R2ESWETaskset(config=config)


def load_harness(config: RlmSweHarnessConfig) -> RLM:
    return RLM(config=config)
