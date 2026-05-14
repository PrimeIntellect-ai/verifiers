#!/usr/bin/env python3
"""
Preprocess Go subset of ProgramBench tasks.
Collects metadata for known Go tasks from the HuggingFace dataset
programbench/ProgramBench-Tests and writes a JSONL file for use in
the prime-rl/verifiers environment.
"""

import json
import os
import re
import sys
from pathlib import Path

import requests
from huggingface_hub import HfApi, hf_hub_url

# Known Go projects to look for in ProgramBench
KNOWN_GO_PREFIXES = [
    "antonmedv__fx",
    "antonmedv__walk",
    "ariga__atlas",
    "direnv__direnv",
    "dundee__gdu",
    "filosofottile__age",
    "junegunn__fzf",
    "jesseduffield__lazygit",
    "johnkerl__miller",
    "go-critic__go-critic",
]

DATASET_REPO = "programbench/ProgramBench-Tests"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "go_subset.jsonl"


def list_dataset_files(token: str | None = None) -> list[str]:
    """List all files in the HuggingFace dataset repo."""
    api = HfApi(token=token)
    files = api.list_repo_files(DATASET_REPO, repo_type="dataset")
    return list(files)


def parse_attribution(attribution_text: str) -> tuple[str, str, str, str]:
    """
    Parse ATTRIBUTION.md content to extract GitHub URL and commit hash.
    Returns (owner, repo, commit, full_url).
    """
    # Look for a GitHub tree link: https://github.com/owner/repo/tree/commithash
    pattern = r"https://github\.com/([^/]+)/([^/]+)/tree/([a-f0-9]+)"
    match = re.search(pattern, attribution_text)
    if match:
        owner, repo, commit = match.group(1), match.group(2), match.group(3)
        full_url = f"https://github.com/{owner}/{repo}/tree/{commit}"
        return owner, repo, commit, full_url
    # Fallback: just a github.com/owner/repo link
    pattern2 = r"https://github\.com/([^/\s]+)/([^/\s]+)"
    match2 = re.search(pattern2, attribution_text)
    if match2:
        owner, repo = match2.group(1), match2.group(2).rstrip(")")
        return owner, repo, "", f"https://github.com/{owner}/{repo}"
    return "", "", "", ""


def fetch_readme(owner: str, repo: str, commit: str) -> str:
    """
    Fetch README from GitHub raw URL.
    Tries README.md, readme.md, README.rst, README as fallbacks.
    """
    ref = commit if commit else "main"
    candidates = ["README.md", "readme.md", "README.rst", "README"]
    for name in candidates:
        url = f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{name}"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                print(f"  Fetched README from {url}")
                return resp.text[:8000]  # cap at 8K chars
        except requests.RequestException as e:
            print(f"  Warning: failed to fetch {url}: {e}")
    print(f"  No README found for {owner}/{repo}@{ref}")
    return ""


def fetch_attribution(
    owner: str, repo_name: str, task_id: str, token: str | None = None
) -> str:
    """Fetch ATTRIBUTION.md content from HuggingFace dataset."""
    # ATTRIBUTION.md is at the task directory root
    file_path = f"{task_id}/ATTRIBUTION.md"
    url = hf_hub_url(DATASET_REPO, file_path, repo_type="dataset")
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            return resp.text
    except requests.RequestException as e:
        print(f"  Warning: failed to fetch ATTRIBUTION.md for {task_id}: {e}")
    return ""


def get_github_description(owner: str, repo: str, token: str | None = None) -> str:
    """Fetch brief repo description from GitHub API."""
    url = f"https://api.github.com/repos/{owner}/{repo}"
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("description") or ""
    except requests.RequestException:
        pass
    return ""


def extract_readme_first_paragraph(readme: str) -> str:
    """Extract the first non-heading, non-empty paragraph from a README."""
    lines = readme.splitlines()
    paragraph_lines = []
    in_paragraph = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_paragraph:
                break
            continue
        if stripped.startswith("#"):
            continue
        # Skip badge lines (common in Go READMEs)
        if stripped.startswith("[![") or stripped.startswith("!["):
            continue
        in_paragraph = True
        paragraph_lines.append(stripped)
        if len(paragraph_lines) >= 5:
            break
    return " ".join(paragraph_lines)


def main():
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("Using HF_TOKEN from environment.")
    else:
        print("No HF_TOKEN found, attempting unauthenticated access.")

    print(f"\nListing files in {DATASET_REPO}...")
    try:
        all_files = list_dataset_files(token=hf_token)
    except Exception as e:
        print(f"ERROR listing dataset files: {e}")
        sys.exit(1)

    print(f"Total files in dataset: {len(all_files)}")

    # Find all unique task directories
    task_dirs = set()
    for f in all_files:
        parts = f.split("/")
        if len(parts) >= 1 and parts[0]:
            task_dirs.add(parts[0])

    print(f"Unique task directories: {len(task_dirs)}")

    # Match Go tasks: task dir starts with one of our known Go prefixes
    go_tasks = []
    for task_dir in sorted(task_dirs):
        for prefix in KNOWN_GO_PREFIXES:
            if task_dir.startswith(prefix):
                go_tasks.append(task_dir)
                break

    print(f"\nFound {len(go_tasks)} Go task directories:")
    for t in go_tasks:
        print(f"  {t}")

    # Process first 5
    to_process = go_tasks[:5]
    print(f"\nProcessing first {len(to_process)} tasks...")

    # Build per-task file index for fast lookup
    file_index: dict[str, list[str]] = {}
    for f in all_files:
        parts = f.split("/", 1)
        if len(parts) == 2:
            task, rel = parts
            file_index.setdefault(task, []).append(rel)

    records = []
    for task_id in to_process:
        print(f"\n--- {task_id} ---")
        task_files = file_index.get(task_id, [])

        # Find test branches (files under tests/ without .tar.gz extension)
        test_branches = []
        for rel in task_files:
            if rel.startswith("tests/") and rel.endswith(".tar.gz"):
                branch = rel[len("tests/") : -len(".tar.gz")]
                test_branches.append(branch)

        print(
            f"  Test branches ({len(test_branches)}): {test_branches[:3]}{'...' if len(test_branches) > 3 else ''}"
        )

        # Fetch ATTRIBUTION.md
        attribution_text = fetch_attribution("", "", task_id, token=hf_token)
        owner, repo, commit, gh_url = "", "", "", ""
        if attribution_text:
            owner, repo, commit, gh_url = parse_attribution(attribution_text)
            print(f"  GitHub: {gh_url}")
        else:
            # Infer from task_id: format is owner__repo.shortcommit
            base = task_id.split(".")[0]  # strip short commit
            commit_short = task_id.split(".")[-1] if "." in task_id else ""
            parts = base.split("__", 1)
            if len(parts) == 2:
                owner, repo = parts
                commit = commit_short  # short hash only
                gh_url = f"https://github.com/{owner}/{repo}"
            print(
                f"  ATTRIBUTION.md not fetched, inferred: owner={owner} repo={repo} commit={commit}"
            )

        # Fetch README
        readme = ""
        if owner and repo:
            readme = fetch_readme(owner, repo, commit)

        # Get GitHub description
        description = ""
        if owner and repo:
            description = get_github_description(owner, repo)
            if description:
                print(f"  GitHub description: {description[:80]}")

        # Build docs from readme first paragraph if no description
        docs = description or extract_readme_first_paragraph(readme)

        # Infer compile hint
        compile_hint = "cd /workspace/src && go build -o /workspace/executable ."

        record = {
            "task_id": task_id,
            "language": "go",
            "difficulty": None,
            "readme": readme,
            "docs": docs,
            "strings_output": "",
            "nm_output": "",
            "objdump_head": "",
            "file_type": "",
            "binary_size": 0,
            "binary_hf_repo": "",
            "binary_hf_filename": "",
            "test_branches": test_branches,
            "compile_hint": compile_hint,
            "example_io": [],
        }
        records.append(record)
        print(
            f"  Record built: test_branches={len(test_branches)}, readme_len={len(readme)}, docs_len={len(docs)}"
        )

    # Write JSONL
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"\nWrote {len(records)} records to {OUTPUT_PATH}")

    # Pretty-print first record summary
    if records:
        first = records[0]
        display = {
            k: (v[:200] + "..." if isinstance(v, str) and len(v) > 200 else v)
            for k, v in first.items()
        }
        print("\n=== First record (truncated) ===")
        print(json.dumps(display, indent=2))

    return records


if __name__ == "__main__":
    main()
