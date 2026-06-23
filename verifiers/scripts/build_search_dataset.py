"""Build a unified search dataset from supported closed-ended backends."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from typing import Any

from datasets import Dataset, DatasetDict, Features, Value, load_dataset

from verifiers.envs.experimental.composable.tasksets.search.openseeker.taskset import (
    DEFAULT_DATASET_NAME as OPENSEEKER_DATASET,
)
from verifiers.envs.experimental.composable.tasksets.search.redsearcher.taskset import (
    DEFAULT_DATASET_NAME as REDSEARCHER_DATASET,
)
from verifiers.envs.experimental.composable.tasksets.search.s1_deepresearch.taskset import (
    DEFAULT_DATA_FILE as S1_DATA_FILE,
)
from verifiers.envs.experimental.composable.tasksets.search.s1_deepresearch.taskset import (
    DEFAULT_DATASET_NAME as S1_DATASET,
)
from verifiers.envs.experimental.composable.tasksets.search.s1_deepresearch.taskset import (
    DEFAULT_LANGUAGE as S1_LANGUAGE,
)
from verifiers.envs.experimental.composable.tasksets.search.s1_deepresearch.taskset import (
    VERIFIABLE_TASK_TYPE as S1_VERIFIABLE_TASK_TYPE,
)
from verifiers.envs.experimental.composable.tasksets.search.s1_deepresearch.taskset import (
    _iter_s1_records,
)

DEFAULT_SPLIT = "train"

UNIFIED_SEARCH_FEATURES = Features(
    {
        "question": Value("string"),
        "answer": Value("string"),
        "info": {
            "source": Value("string"),
            "language": Value("string"),
            "source_task_type": Value("string"),
        },
    }
)


def _base_info(
    *,
    source: str,
    language: str,
    source_task_type: str,
) -> dict[str, Any]:
    return {
        "source": source,
        "language": language,
        "source_task_type": source_task_type,
    }


def normalize_openseeker_row(
    row: dict[str, Any],
    *,
    row_index: int,
    dataset_name: str = OPENSEEKER_DATASET,
    split: str = DEFAULT_SPLIT,
) -> dict[str, Any]:
    return {
        "question": row["question"].strip(),
        "answer": row["answer"].strip(),
        "info": {
            **_base_info(
                source="openseeker",
                language="en",
                source_task_type="binary_semantic_answer",
            ),
        },
    }


def normalize_redsearcher_row(
    row: dict[str, Any],
    *,
    row_index: int,
    dataset_name: str = REDSEARCHER_DATASET,
    split: str = DEFAULT_SPLIT,
) -> dict[str, Any]:
    return {
        "question": row["problem"].strip(),
        "answer": row["answer"].strip(),
        "info": {
            **_base_info(
                source="redsearcher",
                language="en",
                source_task_type="text_rl_query",
            ),
        },
    }


def normalize_s1_record(
    record: dict[str, Any],
    *,
    row_index: int,
    dataset_name: str = S1_DATASET,
    split: str = DEFAULT_SPLIT,
) -> dict[str, Any]:
    meta = record["meta"]
    return {
        "question": meta["question"].strip(),
        "answer": meta["answer"].strip(),
        "info": _base_info(
            source="s1_deepresearch",
            language=meta["language"],
            source_task_type=meta["type"],
        ),
    }


def _take_limit(rows: Iterable[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(row)
        if limit is not None and len(out) >= limit:
            break
    return out


def build_unified_search_rows(
    *,
    split: str = DEFAULT_SPLIT,
    s1_language: str | None = S1_LANGUAGE,
    limit_per_source: int | None = None,
) -> list[dict[str, Any]]:
    openseeker = load_dataset(OPENSEEKER_DATASET, split=split)
    redsearcher = load_dataset(REDSEARCHER_DATASET, split=split)

    rows: list[dict[str, Any]] = []
    rows.extend(
        _take_limit(
            (
                normalize_openseeker_row(row, row_index=row_index, split=split)
                for row_index, row in enumerate(openseeker)
            ),
            limit_per_source,
        )
    )
    rows.extend(
        _take_limit(
            (
                normalize_redsearcher_row(row, row_index=row_index, split=split)
                for row_index, row in enumerate(redsearcher)
            ),
            limit_per_source,
        )
    )
    rows.extend(
        _take_limit(
            (
                normalize_s1_record(record, row_index=row_index, split=split)
                for row_index, record in enumerate(
                    _iter_s1_records(dataset_name=S1_DATASET, data_file=S1_DATA_FILE)
                )
                if record["meta"]["type"] == S1_VERIFIABLE_TASK_TYPE
                and (s1_language is None or record["meta"]["language"] == s1_language)
            ),
            limit_per_source,
        )
    )
    return rows


def build_unified_search_dataset(
    *,
    split: str = DEFAULT_SPLIT,
    s1_language: str | None = S1_LANGUAGE,
    limit_per_source: int | None = None,
) -> DatasetDict:
    rows = build_unified_search_rows(
        split=split,
        s1_language=s1_language,
        limit_per_source=limit_per_source,
    )
    dataset = Dataset.from_list(rows, features=UNIFIED_SEARCH_FEATURES)
    return DatasetDict({split: dataset})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a unified OpenSeeker/REDSearcher/S1 DeepResearch dataset."
    )
    parser.add_argument("--repo-id", help="Hugging Face repo id, e.g. user/search-tasks")
    parser.add_argument(
        "--output-dir",
        default="unified_search_dataset",
        help="Local directory for save_to_disk output.",
    )
    parser.add_argument(
        "--s1-language",
        default=S1_LANGUAGE,
        help="S1 language filter. Use 'all' to include every language.",
    )
    parser.add_argument(
        "--limit-per-source",
        type=int,
        default=None,
        help="Optional debugging limit per source dataset.",
    )
    parser.add_argument("--push", action="store_true", help="Push to --repo-id.")
    parser.add_argument("--private", action="store_true", help="Create/push private HF dataset.")
    args = parser.parse_args()

    s1_language = None if args.s1_language == "all" else args.s1_language
    dataset = build_unified_search_dataset(
        s1_language=s1_language,
        limit_per_source=args.limit_per_source,
    )
    dataset.save_to_disk(args.output_dir)
    print(dataset)
    print(f"Saved to {args.output_dir}")

    if args.push:
        if not args.repo_id:
            raise ValueError("--repo-id is required with --push")
        dataset.push_to_hub(args.repo_id, private=args.private)
        print(f"Pushed to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
