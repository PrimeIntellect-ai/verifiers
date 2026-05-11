from __future__ import annotations

import shlex
from collections.abc import Sequence
from typing import Any

import verifiers.v1 as vf

TASKSET_ID = "paperbench"
DEFAULT_SPLIT = "dev"
DEFAULT_IMAGE = "pb-env:latest"
DEFAULT_WORKDIR = "/home"
DEFAULT_PAPER_DIR = "/home/paper"
DEFAULT_SUBMISSION_DIR = "/home/submission"

SPLITS = {
    "debug": ["rice"],
    "testing": ["lca-on-the-line"],
    "dev": [
        "semantic-self-consistency",
        "self-expansion",
        "self-composing-policies",
    ],
    "lite": [
        "fre",
        "test-time-model-adaptation",
        "mechanistic-understanding",
        "all-in-one",
        "stay-on-topic-with-classifier-free-guidance",
    ],
    "human": [
        "all-in-one",
        "stay-on-topic-with-classifier-free-guidance",
        "mechanistic-understanding",
        "test-time-model-adaptation",
        "fre",
    ],
    "all": [
        "fre",
        "mechanistic-understanding",
        "bridging-data-gaps",
        "test-time-model-adaptation",
        "all-in-one",
        "sequential-neural-score-estimation",
        "robust-clip",
        "what-will-my-model-forget",
        "pinn",
        "stay-on-topic-with-classifier-free-guidance",
        "rice",
        "sample-specific-masks",
        "adaptive-pruning",
        "sapg",
        "lca-on-the-line",
        "stochastic-interpolants",
        "bbox",
        "lbcs",
        "bam",
        "ftrl",
    ],
}

BASE_INSTRUCTIONS = """\
You are tasked with reproducing a research paper from PaperBench.

The paper materials are available in /home/paper, including the paper in PDF and
markdown form, an addendum, and a blacklist of resources that must not be used.
Replicate as many core contributions as possible within the available time.

Your submitted repository must be located at /home/submission and committed as a
git repository. Include a README.md explaining what you implemented and how the
code maps to the paper's contributions. Do not include heavy artifacts in git."""

FULL_REPRODUCTION_INSTRUCTIONS = """\
Full PaperBench mode additionally requires /home/submission/reproduce.sh. The
official evaluator copies the submitted repository into a fresh Ubuntu container
and runs `bash reproduce.sh` from the repository root. Any graded artifacts
should be produced by that script."""


def split_ids(split: str, paper_ids: Sequence[str] | None = None) -> list[str]:
    if paper_ids is not None:
        return list(paper_ids)
    try:
        return list(SPLITS[split])
    except KeyError as exc:
        raise ValueError(f"Unknown PaperBench split: {split}") from exc


def build_prompt(
    paper_id: str,
    paper_dir: str,
    submission_dir: str,
    code_only: bool,
) -> str:
    extra = "" if code_only else f"\n\n{FULL_REPRODUCTION_INSTRUCTIONS}"
    return (
        f"Paper ID: {paper_id}\n\n"
        f"{BASE_INSTRUCTIONS}{extra}\n\n"
        f"Paper directory: {paper_dir}\n"
        f"Submission directory: {submission_dir}"
    )


def make_record(
    paper_id: str,
    split: str,
    image: str,
    workdir: str,
    paper_dir: str,
    submission_dir: str,
    code_only: bool,
) -> dict[str, Any]:
    prompt = build_prompt(paper_id, paper_dir, submission_dir, code_only)
    return {
        "task_id": paper_id,
        "prompt": [{"role": "user", "content": prompt}],
        "question": prompt,
        "answer": submission_dir,
        "info": {
            "paper_id": paper_id,
            "split": split,
            "paper_dir": paper_dir,
            "submission_dir": submission_dir,
            "code_only": code_only,
        },
        "sandbox": {
            "image": image,
            "cpu_cores": 36,
            "memory_gb": 220,
            "disk_size_gb": 256,
            "gpu_count": 1,
            "workdir": workdir,
            "scope": "rollout",
            "timeout_minutes": 1440,
            "network_access": True,
        },
        "program": {"env": {"AGENT_WORKDIR": workdir}},
    }


def direct_submission_layout(task: dict[str, Any]) -> dict[str, str]:
    info = task["info"]
    return {
        "paper_id": str(info["paper_id"]),
        "submission_dir": str(info["submission_dir"]),
        "code_only": str(bool(info["code_only"])).lower(),
    }


class PaperBenchTaskset(vf.Taskset):
    def __init__(
        self,
        split: str = DEFAULT_SPLIT,
        paper_ids: Sequence[str] | None = None,
        image: str = DEFAULT_IMAGE,
        workdir: str = DEFAULT_WORKDIR,
        paper_dir: str = DEFAULT_PAPER_DIR,
        submission_dir: str = DEFAULT_SUBMISSION_DIR,
        code_only: bool = True,
        limit: int | None = None,
        config: vf.TasksetConfig | None = None,
    ):
        self.split = split
        self.paper_ids = split_ids(split, paper_ids)
        if limit is not None and limit >= 0:
            self.paper_ids = self.paper_ids[:limit]
        self.image = image
        self.workdir = workdir
        self.paper_dir = paper_dir
        self.submission_dir = submission_dir
        self.code_only = code_only
        super().__init__(
            source=self.load_rows,
            taskset_id=TASKSET_ID,
            system_prompt=BASE_INSTRUCTIONS,
            metrics=[
                submission_dir_exists,
                git_repo_exists,
                readme_exists,
                reproduce_script_exists,
            ],
            rewards=[valid_submission_layout],
            config=config,
        )

    def load_rows(self) -> list[dict[str, Any]]:
        return [
            make_record(
                paper_id,
                self.split,
                self.image,
                self.workdir,
                self.paper_dir,
                self.submission_dir,
                self.code_only,
            )
            for paper_id in self.paper_ids
        ]

    @vf.setup(priority=250)
    async def capture_sandbox(self, task, state, sandbox=None) -> None:
        if sandbox is not None:
            state["_paperbench_sandbox"] = sandbox

    @vf.cleanup(priority=100)
    async def cleanup_sandbox(self, task, state) -> None:
        state.pop("_paperbench_sandbox", None)


async def _test_path(
    task: vf.Task, state: vf.State, path: str, test_flag: str
) -> float:
    sandbox = state.get("_paperbench_sandbox")
    if sandbox is None:
        return 0.0
    result = await sandbox.execute(
        f"test {test_flag} {shlex.quote(path)}",
        timeout=30,
    )
    return 1.0 if result.exit_code == 0 else 0.0


async def submission_dir_exists(task: vf.Task, state: vf.State) -> float:
    return await _test_path(
        task,
        state,
        str(task["info"]["submission_dir"]),
        "-d",
    )


async def git_repo_exists(task: vf.Task, state: vf.State) -> float:
    return await _test_path(
        task,
        state,
        f"{task['info']['submission_dir']}/.git",
        "-d",
    )


async def readme_exists(task: vf.Task, state: vf.State) -> float:
    return await _test_path(
        task,
        state,
        f"{task['info']['submission_dir']}/README.md",
        "-s",
    )


async def reproduce_script_exists(task: vf.Task, state: vf.State) -> float:
    return await _test_path(
        task,
        state,
        f"{task['info']['submission_dir']}/reproduce.sh",
        "-s",
    )


async def valid_submission_layout(task: vf.Task, state: vf.State) -> float:
    required = [
        await submission_dir_exists(task, state),
        await git_repo_exists(task, state),
        await readme_exists(task, state),
    ]
    if not bool(task["info"]["code_only"]):
        required.append(await reproduce_script_exists(task, state))
    return 1.0 if all(score == 1.0 for score in required) else 0.0


def load_taskset(
    split: str = DEFAULT_SPLIT,
    paper_ids: Sequence[str] | None = None,
    image: str = DEFAULT_IMAGE,
    workdir: str = DEFAULT_WORKDIR,
    paper_dir: str = DEFAULT_PAPER_DIR,
    submission_dir: str = DEFAULT_SUBMISSION_DIR,
    code_only: bool = True,
    limit: int | None = None,
    config: vf.TasksetConfig | None = None,
) -> PaperBenchTaskset:
    return PaperBenchTaskset(
        split=split,
        paper_ids=paper_ids,
        image=image,
        workdir=workdir,
        paper_dir=paper_dir,
        submission_dir=submission_dir,
        code_only=code_only,
        limit=limit,
        config=config,
    )


def load_harness(
    max_turns: int | None = None,
    config: vf.HarnessConfig | None = None,
) -> vf.OpenCode:
    return vf.OpenCode(
        sandbox=True,
        max_turns=max_turns,
        config=config,
    )


def load_environment(
    split: str = DEFAULT_SPLIT,
    paper_ids: Sequence[str] | None = None,
    image: str = DEFAULT_IMAGE,
    workdir: str = DEFAULT_WORKDIR,
    paper_dir: str = DEFAULT_PAPER_DIR,
    submission_dir: str = DEFAULT_SUBMISSION_DIR,
    code_only: bool = True,
    limit: int | None = None,
    max_turns: int | None = None,
    config: vf.EnvConfig | None = None,
) -> vf.Env:
    config = config or vf.EnvConfig()
    return vf.Env(
        taskset=load_taskset(
            split=split,
            paper_ids=paper_ids,
            image=image,
            workdir=workdir,
            paper_dir=paper_dir,
            submission_dir=submission_dir,
            code_only=code_only,
            limit=limit,
            config=config.taskset,
        ),
        harness=load_harness(max_turns=max_turns, config=config.harness),
    )
