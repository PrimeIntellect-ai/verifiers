from __future__ import annotations

import shlex
from collections.abc import Mapping, Sequence
from typing import Any

import verifiers.v1 as vf

TASKSET_ID = "mle-bench"
DEFAULT_SPLIT = "low"
DEFAULT_IMAGE = "mlebench-env"
DEFAULT_WORKDIR = "/home"
DEFAULT_SUBMISSION_PATH = "/home/submission/submission.csv"
DEFAULT_VALIDATE_SCRIPT = "/home/validate_submission.sh"

LOW_COMPETITIONS = [
    "aerial-cactus-identification",
    "aptos2019-blindness-detection",
    "denoising-dirty-documents",
    "detecting-insults-in-social-commentary",
    "dog-breed-identification",
    "dogs-vs-cats-redux-kernels-edition",
    "histopathologic-cancer-detection",
    "jigsaw-toxic-comment-classification-challenge",
    "leaf-classification",
    "mlsp-2013-birds",
    "new-york-city-taxi-fare-prediction",
    "nomad2018-predict-transparent-conductors",
    "plant-pathology-2020-fgvc7",
    "random-acts-of-pizza",
    "ranzcr-clip-catheter-line-classification",
    "siim-isic-melanoma-classification",
    "spooky-author-identification",
    "tabular-playground-series-dec-2021",
    "tabular-playground-series-may-2022",
    "text-normalization-challenge-english-language",
    "text-normalization-challenge-russian-language",
    "the-icml-2013-whale-challenge-right-whale-redux",
]

DEV_COMPETITIONS = [
    "invasive-species-monitoring",
    "ml2021spring-hw2",
    "movie-review-sentiment-analysis-kernels-only",
    "paddy-disease-classification",
    "plant-seedlings-classification",
    "playground-series-s3e18",
    "spaceship-titanic",
]

BENCHMARK_INSTRUCTIONS = """\
You are participating in MLE-Bench, an offline version of Kaggle competitions
adapted to test machine-learning engineering agents.

Read the competition-specific instructions in /home/data/description.md, inspect
the dataset files in /home/data, train or construct a model, and write your final
submission to /home/submission/submission.csv. The submission must be a CSV in
the benchmark-required format. You may use /home/validate_submission.sh to check
that the submission file is structurally valid, but the validator does not reveal
the score.

Do not hand-label the test set, copy external solutions, or use information that
would violate the benchmark rules. The final score is determined by the
competition metric from the benchmark grader."""


def split_ids(split: str, competition_ids: Sequence[str] | None = None) -> list[str]:
    if competition_ids is not None:
        return list(competition_ids)
    if split == "dev":
        return list(DEV_COMPETITIONS)
    if split in {"low", "lite"}:
        return list(LOW_COMPETITIONS)
    if split == "all":
        return load_registry_ids() or list(LOW_COMPETITIONS)
    raise ValueError(f"Unknown MLE-Bench split: {split}")


def load_registry_ids() -> list[str] | None:
    try:
        from mlebench.registry import registry

        return list(registry.list_competition_ids())
    except Exception:
        return None


def load_registry_competition(competition_id: str) -> Mapping[str, Any] | None:
    try:
        from mlebench.registry import registry

        competition = registry.get_competition(competition_id)
        return {
            "id": competition.id,
            "name": competition.name,
            "description": competition.description,
            "competition_type": competition.competition_type,
            "sample_submission": str(competition.sample_submission),
            "answers": str(competition.answers),
        }
    except Exception:
        return None


def build_prompt(
    competition_id: str,
    description: str,
    submission_path: str,
    validate_script: str,
) -> str:
    return (
        f"Competition ID: {competition_id}\n\n"
        f"{BENCHMARK_INSTRUCTIONS}\n\n"
        f"Required submission path: {submission_path}\n"
        f"Validation command: {validate_script} {submission_path}\n\n"
        "Competition description:\n"
        f"{description.strip() or '(description unavailable in this runtime)'}"
    )


def make_record(
    competition_id: str,
    split: str,
    image: str,
    workdir: str,
    submission_path: str,
    validate_script: str,
) -> dict[str, Any]:
    registry_data = load_registry_competition(competition_id) or {}
    description = str(registry_data.get("description") or "")
    info = {
        "competition_id": competition_id,
        "split": split,
        "competition_type": registry_data.get("competition_type"),
        "sample_submission": registry_data.get("sample_submission"),
        "answers": registry_data.get("answers"),
        "submission_path": submission_path,
        "validate_script": validate_script,
    }
    return {
        "task_id": competition_id,
        "prompt": [
            {
                "role": "user",
                "content": build_prompt(
                    competition_id,
                    description,
                    submission_path,
                    validate_script,
                ),
            }
        ],
        "question": build_prompt(
            competition_id,
            description,
            submission_path,
            validate_script,
        ),
        "answer": submission_path,
        "info": info,
        "sandbox": {
            "image": image,
            "cpu_cores": 36,
            "memory_gb": 440,
            "disk_size_gb": 256,
            "gpu_count": 1,
            "workdir": workdir,
            "scope": "rollout",
            "timeout_minutes": 1440,
        },
        "program": {"env": {"AGENT_WORKDIR": workdir}},
    }


def grading_submission_row(task: Mapping[str, Any]) -> dict[str, str]:
    info = task["info"]
    return {
        "competition_id": str(info["competition_id"]),
        "submission_path": str(info["submission_path"]),
    }


def grading_submission_jsonl(task: Mapping[str, Any]) -> str:
    import json

    return json.dumps(grading_submission_row(task), sort_keys=True) + "\n"


class MLEBenchTaskset(vf.Taskset):
    def __init__(
        self,
        split: str = DEFAULT_SPLIT,
        competition_ids: Sequence[str] | None = None,
        image: str = DEFAULT_IMAGE,
        workdir: str = DEFAULT_WORKDIR,
        submission_path: str = DEFAULT_SUBMISSION_PATH,
        validate_script: str = DEFAULT_VALIDATE_SCRIPT,
        limit: int | None = None,
        config: vf.TasksetConfig | None = None,
    ):
        self.split = split
        self.competition_ids = split_ids(split, competition_ids)
        if limit is not None and limit >= 0:
            self.competition_ids = self.competition_ids[:limit]
        self.image = image
        self.workdir = workdir
        self.submission_path = submission_path
        self.validate_script = validate_script
        super().__init__(
            source=self.load_rows,
            taskset_id=TASKSET_ID,
            system_prompt=BENCHMARK_INSTRUCTIONS,
            metrics=[submission_exists, submission_nonempty, validator_available],
            rewards=[valid_submission],
            config=config,
        )

    def load_rows(self) -> list[dict[str, Any]]:
        return [
            make_record(
                competition_id,
                self.split,
                self.image,
                self.workdir,
                self.submission_path,
                self.validate_script,
            )
            for competition_id in self.competition_ids
        ]

    @vf.setup(priority=250)
    async def capture_sandbox(self, task, state, sandbox=None) -> None:
        if sandbox is not None:
            state["_mle_bench_sandbox"] = sandbox

    @vf.cleanup(priority=100)
    async def cleanup_sandbox(self, task, state) -> None:
        state.pop("_mle_bench_sandbox", None)


async def submission_exists(task: vf.Task, state: vf.State) -> float:
    sandbox = state.get("_mle_bench_sandbox")
    if sandbox is None:
        return 0.0
    submission_path = str(task["info"]["submission_path"])
    result = await sandbox.execute(
        f"test -f {shlex.quote(submission_path)}",
        timeout=30,
    )
    return 1.0 if result.exit_code == 0 else 0.0


async def submission_nonempty(task: vf.Task, state: vf.State) -> float:
    sandbox = state.get("_mle_bench_sandbox")
    if sandbox is None:
        return 0.0
    submission_path = str(task["info"]["submission_path"])
    result = await sandbox.execute(
        f"test -s {shlex.quote(submission_path)}",
        timeout=30,
    )
    return 1.0 if result.exit_code == 0 else 0.0


async def validator_available(task: vf.Task, state: vf.State) -> float:
    sandbox = state.get("_mle_bench_sandbox")
    if sandbox is None:
        return 0.0
    validate_script = str(task["info"]["validate_script"])
    result = await sandbox.execute(
        f"test -x {shlex.quote(validate_script)}",
        timeout=30,
    )
    return 1.0 if result.exit_code == 0 else 0.0


async def valid_submission(task: vf.Task, state: vf.State) -> float:
    sandbox = state.get("_mle_bench_sandbox")
    if sandbox is None:
        return 0.0
    info = task["info"]
    submission_path = str(info["submission_path"])
    validate_script = str(info["validate_script"])
    command = f"{shlex.quote(validate_script)} {shlex.quote(submission_path)}"
    result = await sandbox.execute(command, timeout=300, working_dir=DEFAULT_WORKDIR)
    state["validation_stdout"] = result.stdout or ""
    state["validation_stderr"] = result.stderr or ""
    state["validation_exit_code"] = result.exit_code
    return 1.0 if result.exit_code == 0 and validator_accepts(result.stdout) else 0.0


def validator_accepts(stdout: str | None) -> bool:
    for line in (stdout or "").splitlines():
        if line.strip().lower() == "submission is valid.":
            return True
    return False


def load_taskset(
    split: str = DEFAULT_SPLIT,
    competition_ids: Sequence[str] | None = None,
    image: str = DEFAULT_IMAGE,
    workdir: str = DEFAULT_WORKDIR,
    submission_path: str = DEFAULT_SUBMISSION_PATH,
    validate_script: str = DEFAULT_VALIDATE_SCRIPT,
    limit: int | None = None,
    config: vf.TasksetConfig | None = None,
) -> MLEBenchTaskset:
    return MLEBenchTaskset(
        split=split,
        competition_ids=competition_ids,
        image=image,
        workdir=workdir,
        submission_path=submission_path,
        validate_script=validate_script,
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
    competition_ids: Sequence[str] | None = None,
    image: str = DEFAULT_IMAGE,
    workdir: str = DEFAULT_WORKDIR,
    submission_path: str = DEFAULT_SUBMISSION_PATH,
    validate_script: str = DEFAULT_VALIDATE_SCRIPT,
    limit: int | None = None,
    max_turns: int | None = None,
    config: vf.EnvConfig | None = None,
) -> vf.Env:
    config = config or vf.EnvConfig()
    return vf.Env(
        taskset=load_taskset(
            split=split,
            competition_ids=competition_ids,
            image=image,
            workdir=workdir,
            submission_path=submission_path,
            validate_script=validate_script,
            limit=limit,
            config=config.taskset,
        ),
        harness=load_harness(max_turns=max_turns, config=config.harness),
    )
