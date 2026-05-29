import difflib
import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

from datasets import load_dataset

import verifiers as vf
from verifiers.v1.utils.config_utils import coerce_config

DEFAULT_DATASET_NAME = "princeton-nlp/SWE-bench_Verified"
DEFAULT_DATASET_SPLIT = "test"
DEFAULT_TASKSET_ID = "swe-bench/verified"

PATCH_TAG_RE = re.compile(r"<patch>\s*(.*?)\s*</patch>", re.DOTALL | re.IGNORECASE)
FENCED_PATCH_RE = re.compile(
    r"```(?:diff|patch)?\s*(diff --git[\s\S]*?)```", re.IGNORECASE
)
DIFF_HEADER_RE = re.compile(r"^diff --git a/(.*?) b/(.*?)$", re.MULTILINE)
PLUS_FILE_RE = re.compile(r"^\+\+\+ b/(.*?)$", re.MULTILINE)


class SWEBenchVerifiedTasksetConfig(vf.TasksetConfig):
    taskset_id: str = DEFAULT_TASKSET_ID
    dataset_name: str = DEFAULT_DATASET_NAME
    dataset_split: str = DEFAULT_DATASET_SPLIT
    max_examples: int | None = None
    streaming: bool = False
    include_test_names: bool = True


class SWEBenchVerifiedTaskset(vf.Taskset[SWEBenchVerifiedTasksetConfig]):
    def __init__(self, config: SWEBenchVerifiedTasksetConfig | None = None):
        config = coerce_config(SWEBenchVerifiedTasksetConfig, config)
        self.dataset_name = config.dataset_name
        self.dataset_split = config.dataset_split
        self.max_examples = config.max_examples
        self.streaming = config.streaming
        self.include_test_names = config.include_test_names
        super().__init__(config=config)

    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        # The v1 taskset API passes train/eval here; the HF dataset split remains
        # configurable because SWE-bench Verified commonly uses the "test" split.
        _ = split
        return load_swe_bench_verified_tasks(
            dataset_name=self.dataset_name,
            dataset_split=self.dataset_split,
            max_examples=self.max_examples,
            streaming=self.streaming,
            include_test_names=self.include_test_names,
        )

    @vf.reward(weight=1.0)
    async def patch_reward(self, task, state) -> float:
        prediction = extract_patch(state.get("completion") or "")
        gold_patch = str(task.get("answer") or "")
        exact = normalize_patch(prediction) == normalize_patch(gold_patch)
        similarity = patch_similarity(prediction, gold_patch)
        overlap = changed_file_overlap(prediction, gold_patch)
        reward = 1.0 if exact and prediction.strip() else min(
            0.99, 0.75 * similarity + 0.25 * overlap
        )

        state["swe_bench_verified_patch"] = prediction
        state["swe_bench_verified_submission"] = official_submission(task, prediction)
        state["swe_bench_verified_exact_match"] = exact
        state["swe_bench_verified_patch_similarity"] = similarity
        state["swe_bench_verified_changed_file_overlap"] = overlap
        return reward


def load_swe_bench_verified_tasks(
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_split: str = DEFAULT_DATASET_SPLIT,
    max_examples: int | None = None,
    streaming: bool = False,
    include_test_names: bool = True,
) -> list[vf.ConfigData]:
    dataset = load_dataset(dataset_name, split=dataset_split, streaming=streaming)
    rows: list[vf.ConfigData] = []
    for row in limited_rows(dataset, max_examples):
        rows.append(
            process_swe_bench_row(
                row,
                dataset_name=dataset_name,
                include_test_names=include_test_names,
            )
        )
    return rows


def limited_rows(dataset: Iterable[Mapping[str, Any]], limit: int | None):
    if limit is None:
        yield from dataset
        return

    if hasattr(dataset, "select") and hasattr(dataset, "__len__"):
        count = min(limit, len(dataset))  # type: ignore[arg-type]
        yield from dataset.select(range(count))  # type: ignore[attr-defined]
        return

    for index, row in enumerate(dataset):
        if index >= limit:
            break
        yield row


def process_swe_bench_row(
    row: Mapping[str, Any],
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    include_test_names: bool = True,
) -> vf.ConfigData:
    info = {
        "instance_id": str(row.get("instance_id") or ""),
        "repo": str(row.get("repo") or ""),
        "base_commit": str(row.get("base_commit") or ""),
        "environment_setup_commit": str(row.get("environment_setup_commit") or ""),
        "version": str(row.get("version") or ""),
        "problem_statement": str(row.get("problem_statement") or ""),
        "hints_text": str(row.get("hints_text") or ""),
        "fail_to_pass": normalize_test_names(row.get("FAIL_TO_PASS")),
        "pass_to_pass": normalize_test_names(row.get("PASS_TO_PASS")),
        "test_patch": str(row.get("test_patch") or ""),
        "dataset_name": dataset_name,
        "official_submission_format": {
            "instance_id": str(row.get("instance_id") or ""),
            "model_patch": "<generated unified diff>",
        },
    }
    prompt = build_prompt(info, include_test_names=include_test_names)
    return {
        "task_id": info["instance_id"],
        "prompt": [{"role": "user", "content": prompt}],
        "answer": str(row.get("patch") or ""),
        "info": info,
    }


def normalize_test_names(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list | tuple):
        return [str(item) for item in value]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        if isinstance(decoded, list):
            return [str(item) for item in decoded]
        return [str(decoded)]
    return [str(value)]


def build_prompt(info: Mapping[str, Any], *, include_test_names: bool = True) -> str:
    lines = [
        "You are fixing a SWE-bench Verified repository issue.",
        "",
        f"Repository: {info.get('repo')}",
        f"Base commit: {info.get('base_commit')}",
        f"Instance ID: {info.get('instance_id')}",
        "",
        "Issue:",
        str(info.get("problem_statement") or "").strip(),
    ]
    hints = str(info.get("hints_text") or "").strip()
    if hints:
        lines.extend(["", "Hints:", hints])
    if include_test_names:
        fail_to_pass = info.get("fail_to_pass") or []
        pass_to_pass = info.get("pass_to_pass") or []
        if fail_to_pass:
            lines.extend(["", "Fail-to-pass tests:", json.dumps(fail_to_pass)])
        if pass_to_pass:
            lines.extend(["", "Pass-to-pass tests:", json.dumps(pass_to_pass)])
    lines.extend(
        [
            "",
            "Return only a unified diff that applies to the base commit.",
            "Wrap the diff in <patch>...</patch> tags.",
        ]
    )
    return "\n".join(lines)


def completion_to_text(completion: object) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, Mapping):
        return str(completion.get("content") or "")
    if isinstance(completion, list | tuple):
        for message in reversed(completion):
            if isinstance(message, Mapping):
                role = message.get("role")
                if role is None or role == "assistant":
                    return str(message.get("content") or "")
            content = getattr(message, "content", None)
            if content is not None:
                return str(content)
    return str(completion or "")


def extract_patch(completion: object) -> str:
    text = completion_to_text(completion)
    tagged = PATCH_TAG_RE.search(text)
    if tagged:
        return tagged.group(1).strip()

    fenced = FENCED_PATCH_RE.search(text)
    if fenced:
        return fenced.group(1).strip()

    marker = text.find("diff --git ")
    if marker >= 0:
        return text[marker:].strip()
    return text.strip()


def normalize_patch(patch: str) -> str:
    lines = patch.replace("\r\n", "\n").splitlines()
    return "\n".join(line.rstrip() for line in lines).strip()


def changed_files(patch: str) -> set[str]:
    normalized = normalize_patch(patch)
    files: set[str] = set()
    for match in DIFF_HEADER_RE.finditer(normalized):
        files.add(match.group(2))
    for match in PLUS_FILE_RE.finditer(normalized):
        path = match.group(1)
        if path != "/dev/null":
            files.add(path)
    return files


def changed_file_overlap(prediction: str, gold_patch: str) -> float:
    predicted_files = changed_files(prediction)
    gold_files = changed_files(gold_patch)
    if not predicted_files and not gold_files:
        return 1.0
    if not predicted_files or not gold_files:
        return 0.0
    return len(predicted_files & gold_files) / len(predicted_files | gold_files)


def patch_similarity(prediction: str, gold_patch: str) -> float:
    predicted = normalize_patch(prediction)
    gold = normalize_patch(gold_patch)
    if not predicted and not gold:
        return 1.0
    if not predicted or not gold:
        return 0.0
    return difflib.SequenceMatcher(None, predicted, gold).ratio()


def official_submission(task: Mapping[str, Any], model_patch: str) -> dict[str, str]:
    info = task.get("info") or {}
    if not isinstance(info, Mapping):
        info = {}
    instance_id = str(info.get("instance_id") or task.get("task_id") or "")
    return {
        "instance_id": instance_id,
        "model_patch": model_patch,
    }


def load_taskset(
    config: SWEBenchVerifiedTasksetConfig,
) -> SWEBenchVerifiedTaskset:
    return SWEBenchVerifiedTaskset(config=config)


class SWEBenchVerifiedEnvConfig(vf.EnvConfig):
    taskset: SWEBenchVerifiedTasksetConfig = SWEBenchVerifiedTasksetConfig()
    harness: vf.HarnessConfig = vf.HarnessConfig(max_turns=1)


def load_environment(config: SWEBenchVerifiedEnvConfig) -> vf.Env:
    return vf.Env(
        taskset=load_taskset(config=config.taskset),
        harness=vf.Harness(config=config.harness),
    )
