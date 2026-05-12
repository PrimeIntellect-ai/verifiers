#!/usr/bin/env bash
# build_binaries_linux.sh
#
# Run on a Linux x86_64 machine with Go, Rust, GCC, and standard binutils installed.
# For each task in go_subset.jsonl:
#   1. Clone the GitHub repo at the pinned commit
#   2. Build the binary with the provided compile_hint
#   3. Run strings, nm, file, and objdump analysis
#   4. Emit a per-task <task_id>_analysis.json in the output dir
#
# Usage:
#   bash build_binaries_linux.sh [JSONL_PATH] [OUTPUT_DIR]
#
# Defaults:
#   JSONL_PATH  = ../data/go_subset.jsonl  (relative to this script)
#   OUTPUT_DIR  = ../data/binaries/
#
# Dependencies (must be on PATH):
#   go, git, file, strings, nm, objdump, jq, python3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JSONL_PATH="${1:-${SCRIPT_DIR}/../data/go_subset.jsonl}"
OUTPUT_DIR="${2:-${SCRIPT_DIR}/../data/binaries}"
WORKSPACE="$(mktemp -d /tmp/programbench_build_XXXXXX)"

echo "=== ProgramBench Go Binary Builder ==="
echo "JSONL:      $JSONL_PATH"
echo "Output dir: $OUTPUT_DIR"
echo "Workspace:  $WORKSPACE"

mkdir -p "$OUTPUT_DIR"

cleanup() {
    echo "Cleaning up workspace $WORKSPACE ..."
    rm -rf "$WORKSPACE"
}
trap cleanup EXIT

check_tool() {
    if ! command -v "$1" &>/dev/null; then
        echo "ERROR: $1 not found on PATH. Please install it." >&2
        exit 1
    fi
}

for tool in go git file strings nm objdump python3; do
    check_tool "$tool"
done

go version
echo ""

# Read each JSON line from the JSONL file
while IFS= read -r line; do
    task_id="$(echo "$line" | python3 -c "import sys,json; print(json.load(sys.stdin)['task_id'])")"
    compile_hint="$(echo "$line" | python3 -c "import sys,json; print(json.load(sys.stdin)['compile_hint'])")"
    readme="$(echo "$line" | python3 -c "import sys,json; print(json.load(sys.stdin).get('readme',''))")"

    echo ""
    echo "=== Task: $task_id ==="

    # Parse owner/repo/commit from task_id: format is owner__repo.shortcommit
    base="${task_id%%.*}"
    short_commit="${task_id##*.}"
    owner="${base%%__*}"
    repo="${base##*__}"
    clone_url="https://github.com/${owner}/${repo}.git"

    task_workspace="${WORKSPACE}/${task_id}"
    src_dir="${task_workspace}/src"
    executable="${task_workspace}/executable"
    mkdir -p "$src_dir"

    echo "Cloning $clone_url ..."
    if ! git clone --depth=50 "$clone_url" "$src_dir" 2>&1; then
        echo "ERROR: git clone failed for $task_id" >&2
        continue
    fi

    # Checkout the pinned commit (short hash)
    echo "Checking out $short_commit ..."
    if ! (cd "$src_dir" && git checkout "$short_commit" 2>&1); then
        echo "WARNING: exact checkout of $short_commit failed, trying fetch ..."
        (cd "$src_dir" && git fetch --depth=50 origin "$short_commit" && git checkout FETCH_HEAD 2>&1) || true
    fi

    echo "Building binary with hint: $compile_hint"
    export WORKSPACE="$task_workspace"
    # Replace /workspace references with the actual task workspace path
    actual_cmd="${compile_hint//\/workspace/$task_workspace}"

    if ! eval "$actual_cmd" 2>&1; then
        echo "ERROR: build failed for $task_id" >&2
        # Try fallback: go build -o executable in root
        echo "Trying fallback: go build ./..."
        (cd "$src_dir" && go build -o "$executable" ./... 2>&1) || true
    fi

    if [ ! -f "$executable" ]; then
        echo "ERROR: no executable produced for $task_id" >&2
        continue
    fi

    echo "Binary built: $executable"
    binary_size="$(stat -c%s "$executable" 2>/dev/null || stat -f%z "$executable")"
    file_type="$(file -b "$executable")"

    echo "Running strings ..."
    strings_output="$(strings "$executable" | head -500 | tr '\0' '\n')" || strings_output=""

    echo "Running nm ..."
    nm_output="$(nm --defined-only "$executable" 2>/dev/null | head -300)" || nm_output=""

    echo "Running objdump ..."
    objdump_head="$(objdump -d "$executable" 2>/dev/null | head -200)" || objdump_head=""

    # Write per-task analysis JSON
    analysis_file="${OUTPUT_DIR}/${task_id}_analysis.json"
    python3 - <<PYEOF
import json, sys

record = {
    "task_id": "$task_id",
    "binary_size": $binary_size,
    "file_type": """$file_type""",
    "strings_output": """${strings_output//\"/\\\"}""",
    "nm_output": """${nm_output//\"/\\\"}""",
    "objdump_head": """${objdump_head//\"/\\\"}""",
}

with open("$analysis_file", "w") as f:
    json.dump(record, f, indent=2)
print(f"Wrote analysis to $analysis_file")
PYEOF

    echo "Done: $task_id (binary_size=$binary_size bytes)"

done < "$JSONL_PATH"

echo ""
echo "=== Build complete. Analysis files in $OUTPUT_DIR ==="
echo ""
echo "To merge analysis back into go_subset.jsonl, run:"
echo "  python3 ${SCRIPT_DIR}/merge_binary_analysis.py \\"
echo "    --jsonl ${JSONL_PATH} \\"
echo "    --analysis-dir ${OUTPUT_DIR} \\"
echo "    --output ${JSONL_PATH%.jsonl}_with_binaries.jsonl"
