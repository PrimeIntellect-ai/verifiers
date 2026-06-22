"""S1 DeepResearch composable search taskset."""

import json
import os
from collections.abc import Iterable
from typing import Any

import verifiers as vf
from datasets import Dataset
from huggingface_hub import hf_hub_download
from verifiers.envs.experimental.composable import SandboxSpec, SandboxTaskSet
from verifiers.envs.experimental.composable.tasksets.search.redsearcher.taskset import (
    DEFAULT_ANSWER_FILE,
    DEFAULT_JUDGE_API_KEY_VAR,
    DEFAULT_JUDGE_BASE_URL,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_SANDBOX_IMAGE,
    DEFAULT_WORKDIR,
    RedSearcherRubric,
)

DEFAULT_DATASET_NAME = "ScienceOne-AI/S1-DeepResearch-15k"
DEFAULT_SPLIT = "train"
DEFAULT_DATA_FILE = "data.jsonl"
DEFAULT_LANGUAGE = "en"
VERIFIABLE_TASK_TYPE = "Closed-ended Multi-hop Resolution"


def _iter_s1_records(*, dataset_name: str, data_file: str) -> Iterable[dict[str, Any]]:
    path = hf_hub_download(repo_id=dataset_name, filename=data_file, repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _build_s1_rows(
    records: Iterable[dict[str, Any]],
    *,
    dataset_name: str,
    split: str,
    answer_file: str,
    language: str | None = DEFAULT_LANGUAGE,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(records):
        meta = row["meta"]
        if meta["type"] != VERIFIABLE_TASK_TYPE:
            continue
        if language is not None and meta["language"] != language:
            continue
        question = meta["question"].strip()
        answer = meta["answer"].strip()
        rows.append(
            {
                "question": question,
                "answer": answer,
                "info": {
                    "question": question,
                    "answer": answer,
                    "source_dataset": dataset_name,
                    "source_id": meta["id"],
                    "split": split,
                    "row_index": row_index,
                    "language": meta["language"],
                    "task_type": meta["type"],
                    "answer_file": answer_file,
                },
            }
        )
    return rows


class S1DeepResearchTaskSet(SandboxTaskSet):
    """ScienceOne S1 DeepResearch closed-ended multi-hop taskset."""

    default_workdir = DEFAULT_WORKDIR

    def __init__(
        self,
        dataset_name: str = DEFAULT_DATASET_NAME,
        split: str = DEFAULT_SPLIT,
        filter_fn: str | None = None,
        data_file: str = DEFAULT_DATA_FILE,
        language: str | None = DEFAULT_LANGUAGE,
        sandbox_image: str = DEFAULT_SANDBOX_IMAGE,
        sandbox_cpu_cores: int = 2,
        sandbox_memory_gb: int = 2,
        sandbox_disk_size_gb: int = 5,
        sandbox_timeout_minutes: int | None = None,
        answer_file: str = DEFAULT_ANSWER_FILE,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        judge_base_url: str | None = DEFAULT_JUDGE_BASE_URL,
        judge_api_key_var: str = DEFAULT_JUDGE_API_KEY_VAR,
        judge_sampling_args: dict[str, Any] | None = None,
        judge_max_retries: int = 5,
        use_exact_match_shortcut: bool = True,
    ) -> None:
        if split != DEFAULT_SPLIT:
            raise ValueError("S1 DeepResearch currently exposes only the train split")
        self.dataset_name = dataset_name
        self.split = split
        self.data_file = data_file
        self.language = language
        self.answer_file = answer_file
        self._sandbox_spec = SandboxSpec(
            image=sandbox_image,
            cpu_cores=sandbox_cpu_cores,
            memory_gb=sandbox_memory_gb,
            disk_size_gb=sandbox_disk_size_gb,
            timeout_minutes=sandbox_timeout_minutes,
        )
        self._judge_model = judge_model
        self._judge_base_url = judge_base_url
        self._judge_api_key_var = judge_api_key_var
        self._judge_sampling_args = dict(judge_sampling_args or {})
        self._judge_max_retries = judge_max_retries
        self._use_exact_match_shortcut = use_exact_match_shortcut
        super().__init__(
            dataset=self._build_dataset,
            name="search/s1_deepresearch",
            filter_fn=filter_fn,
        )

    def _build_dataset(self) -> Dataset:
        rows = _build_s1_rows(
            _iter_s1_records(
                dataset_name=self.dataset_name,
                data_file=self.data_file,
            ),
            dataset_name=self.dataset_name,
            split=self.split,
            answer_file=self.answer_file,
            language=self.language,
        )
        return Dataset.from_list(rows)

    def get_instruction(self, info: dict) -> str:
        question = info["question"]
        return (
            f"{question}\n\n"
            f"When you have the final response, write it to {self.answer_file} using a tool call, "
            "then stop. The task is incomplete unless that file exists. Provide the requested answer "
            "directly, include supporting URLs/citations when useful, and do not include scratch "
            "reasoning or tool traces."
        )

    def get_sandbox_spec(self, info: dict) -> SandboxSpec:
        return self._sandbox_spec

    def get_workdir(self, info: dict) -> str:
        return self.default_workdir

    def get_env_vars(self) -> dict[str, str]:
        env_vars: dict[str, str] = {}
        value = os.environ.get("SERPER_API_KEY")
        if value:
            env_vars["SERPER_API_KEY"] = value
        return env_vars

    async def setup(self, state: vf.State) -> None:
        sandbox_client = state["sandbox_client"]
        sandbox_id = state["sandbox_id"]
        await sandbox_client.execute_command(
            sandbox_id, f"mkdir -p {self.default_workdir} /task", timeout=10
        )

    def get_rubric(self) -> vf.Rubric:
        return RedSearcherRubric(
            answer_file=self.answer_file,
            judge_model=self._judge_model,
            judge_base_url=self._judge_base_url,
            judge_api_key_var=self._judge_api_key_var,
            judge_sampling_args=self._judge_sampling_args,
            judge_max_retries=self._judge_max_retries,
            use_exact_match_shortcut=self._use_exact_match_shortcut,
        )
