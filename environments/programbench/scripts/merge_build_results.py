#!/usr/bin/env python3
"""
merge_build_results.py

Reads build_results.jsonl from the build script and updates
PrimeIntellect/programbench-processed with binary metadata fields.

Usage:
    HF_TOKEN=... uv run python3 scripts/merge_build_results.py \
        --results data/binaries/build_results.jsonl
"""

from __future__ import annotations
import argparse
import json
import logging
import os
from pathlib import Path

from datasets import Dataset

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

HF_TOKEN = (
    os.environ.get("HF_TOKEN")
    or open(Path.home() / ".cache/huggingface/token").read().strip()
)
OUT_REPO = "PrimeIntellect/programbench-processed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Path to build_results.jsonl")
    args = parser.parse_args()

    results: dict[str, dict] = {}
    with open(args.results) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("success"):
                results[rec["task_id"]] = rec

    log.info("Loaded %d successful build results", len(results))

    from datasets import load_dataset

    ds = load_dataset(OUT_REPO, split="train", token=HF_TOKEN)
    log.info("Loaded %d rows from %s", len(ds), OUT_REPO)

    rows = []
    updated = 0
    for row in ds:
        row = dict(row)
        tid = row["task_id"]
        if tid in results:
            r = results[tid]
            row["binary_size"] = r.get("binary_size")
            row["file_type"] = r.get("file_type", "")
            row["strings_output"] = r.get("strings_output", "")
            row["nm_output"] = r.get("nm_output", "")
            row["objdump_head"] = r.get("objdump_head", "")
            row["binary_hf_repo"] = r.get("binary_hf_repo", "")
            row["binary_hf_filename"] = r.get("binary_hf_filename", "")
            updated += 1
        rows.append(row)

    log.info("Updated %d rows", updated)

    ds_new = Dataset.from_list(rows)
    ds_new.push_to_hub(OUT_REPO, token=HF_TOKEN, private=True)
    log.info("Pushed updated dataset to %s", OUT_REPO)


if __name__ == "__main__":
    main()
