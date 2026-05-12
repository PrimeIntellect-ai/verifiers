#!/usr/bin/env bash
# build_all_binaries.sh
#
# Run on a Linux x86_64 machine.
# For each task in all_tasks.jsonl (or go_subset.jsonl):
#   1. Clone GitHub repo at pinned commit
#   2. Build native Linux binary using compile_hint
#   3. Run strings/nm/file/objdump analysis
#   4. Upload binary to HuggingFace
#   5. Update dataset row with binary_hf_repo, binary_hf_filename, analysis fields
#
# Usage:
#   HF_TOKEN=... GITHUB_TOKEN=... bash build_all_binaries.sh [JSONL_PATH]
#
# Dependencies: go, cargo, gcc, g++, cmake, make, git, python3, jq,
#               file, strings, nm, objdump, huggingface-hub (pip)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JSONL_PATH="${1:-${SCRIPT_DIR}/../data/all_tasks.jsonl}"
OUTPUT_DIR="${SCRIPT_DIR}/../data/binaries"
BINARY_HF_REPO="PrimeIntellect/programbench-processed"
WORKSPACE_ROOT="$(mktemp -d /tmp/programbench_build_XXXXXX)"
HF_TOKEN="${HF_TOKEN:-$(cat ~/.cache/huggingface/token 2>/dev/null)}"

echo "=== ProgramBench Multi-Language Binary Builder ==="
echo "JSONL:       $JSONL_PATH"
echo "Output dir:  $OUTPUT_DIR"
echo "HF repo:     $BINARY_HF_REPO"
echo "Workspace:   $WORKSPACE_ROOT"
echo ""

mkdir -p "$OUTPUT_DIR"

cleanup() {
    echo "Cleaning up $WORKSPACE_ROOT ..."
    rm -rf "$WORKSPACE_ROOT"
}
trap cleanup EXIT

# Tool checks
for tool in git python3 jq file nm objdump; do
    command -v "$tool" &>/dev/null || { echo "ERROR: $tool not found"; exit 1; }
done
command -v go    &>/dev/null && echo "go:    $(go version)"
command -v cargo &>/dev/null && echo "cargo: $(cargo --version)"
command -v gcc   &>/dev/null && echo "gcc:   $(gcc --version | head -1)"
command -v g++   &>/dev/null && echo "g++:   $(g++ --version | head -1)"
echo ""

RESULTS_FILE="${OUTPUT_DIR}/build_results.jsonl"
> "$RESULTS_FILE"

build_count=0
skip_count=0
fail_count=0

while IFS= read -r line; do
    task_id="$(echo "$line" | python3 -c "import sys,json; print(json.load(sys.stdin)['task_id'])")"
    lang="$(echo     "$line" | python3 -c "import sys,json; print(json.load(sys.stdin).get('language','c'))")"
    compile_hint="$(echo "$line" | python3 -c "import sys,json; print(json.load(sys.stdin).get('compile_hint',''))")"

    echo ""
    echo "══════════════════════════════════════════════"
    echo "Task:  $task_id  [$lang]"
    echo "══════════════════════════════════════════════"

    # Already built?
    binary_out="${OUTPUT_DIR}/${task_id}_executable"
    if [ -f "$binary_out" ]; then
        echo "Binary already exists, skipping build."
        ((skip_count++)) || true
        # Still re-run analysis so results file is complete
    fi

    # Parse owner/repo/commit
    base="${task_id%%.*}"
    short_commit="${task_id##*.}"
    owner="${base%%__*}"
    repo="${base##*__}"
    clone_url="https://github.com/${owner}/${repo}.git"

    task_ws="${WORKSPACE_ROOT}/${task_id}"
    src_dir="${task_ws}/src"
    executable="${task_ws}/executable"
    mkdir -p "$src_dir"

    if [ ! -f "$binary_out" ]; then
        # Clone
        echo "Cloning $clone_url ..."
        if ! git clone --depth=200 "$clone_url" "$src_dir" 2>&1; then
            echo "ERROR: clone failed" >&2
            ((fail_count++)) || true
            echo '{"task_id":"'"$task_id"'","success":false,"error":"clone_failed"}' >> "$RESULTS_FILE"
            continue
        fi

        # Checkout pinned commit
        echo "Checkout $short_commit ..."
        (cd "$src_dir" && git checkout "$short_commit" 2>&1) || \
        (cd "$src_dir" && git fetch --depth=200 origin "$short_commit" && git checkout FETCH_HEAD 2>&1) || \
        echo "WARNING: exact checkout failed, using HEAD"

        # Build
        echo "Build hint: $compile_hint"
        export PATH="/usr/local/go/bin:/root/.cargo/bin:$PATH"

        # Replace /workspace refs with actual paths
        actual_cmd="${compile_hint//\/workspace\/src/$src_dir}"
        actual_cmd="${actual_cmd//\/workspace\/executable/$executable}"
        actual_cmd="${actual_cmd//\/workspace/$task_ws}"

        build_ok=false
        if eval "$actual_cmd" 2>&1; then
            build_ok=true
        else
            echo "Primary compile_hint failed; trying auto-detect ..."

            if [ "$lang" = "go" ]; then
                (cd "$src_dir" && go build -o "$executable" ./... 2>&1) && build_ok=true
            elif [ "$lang" = "rust" ]; then
                (cd "$src_dir" && cargo build --release 2>&1) && \
                find "$src_dir/target/release" -maxdepth 1 -type f -executable \
                    ! -name "*.d" ! -name "*.rlib" ! -name "*.so" ! -name "*.a" \
                    | head -1 | xargs -I{} cp {} "$executable" && build_ok=true
            elif [ "$lang" = "c" ] || [ "$lang" = "cpp" ]; then
                # Try cmake first
                if [ -f "$src_dir/CMakeLists.txt" ]; then
                    mkdir -p "$src_dir/build"
                    (cd "$src_dir/build" && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j"$(nproc)" 2>&1) && \
                    find "$src_dir/build" -maxdepth 2 -type f -executable | head -1 | xargs -I{} cp {} "$executable" && build_ok=true
                fi
                # Try make
                if ! $build_ok && [ -f "$src_dir/Makefile" ]; then
                    (cd "$src_dir" && make -j"$(nproc)" 2>&1) && \
                    find "$src_dir" -maxdepth 2 -type f -executable ! -name "*.sh" | head -1 | xargs -I{} cp {} "$executable" && build_ok=true
                fi
                # Try autoconf
                if ! $build_ok && [ -f "$src_dir/configure" ]; then
                    (cd "$src_dir" && ./configure && make -j"$(nproc)" 2>&1) && \
                    find "$src_dir" -maxdepth 2 -type f -executable ! -name "*.sh" | head -1 | xargs -I{} cp {} "$executable" && build_ok=true
                fi
            fi
        fi

        if ! $build_ok || [ ! -f "$executable" ]; then
            echo "ERROR: build failed for $task_id" >&2
            ((fail_count++)) || true
            echo '{"task_id":"'"$task_id"'","success":false,"error":"build_failed"}' >> "$RESULTS_FILE"
            continue
        fi

        cp "$executable" "$binary_out"
        echo "Binary saved: $binary_out ($(stat -c%s "$binary_out") bytes)"
    else
        cp "$binary_out" "$executable"
    fi

    # Analysis
    binary_size="$(stat -c%s "$binary_out" 2>/dev/null || echo 0)"
    file_type="$(file -b "$executable" 2>/dev/null || echo '')"
    strings_out="$(strings "$executable" 2>/dev/null | head -500 | tr '\0' '\n' | head -500)" || strings_out=""
    nm_out="$(nm --defined-only "$executable" 2>/dev/null | head -300)" || nm_out=""
    objdump_head="$(objdump -d "$executable" 2>/dev/null | head -200)" || objdump_head=""

    # Upload binary to HF
    echo "Uploading binary to HF: $BINARY_HF_REPO / binaries/${task_id}/executable ..."
    upload_ok=false
    if python3 - <<PYEOF
import sys, os
from huggingface_hub import HfApi
api = HfApi(token="${HF_TOKEN}")
try:
    api.upload_file(
        path_or_fileobj="${binary_out}",
        path_in_repo="binaries/${task_id}/executable",
        repo_id="${BINARY_HF_REPO}",
        repo_type="dataset",
    )
    print("upload ok")
except Exception as e:
    print(f"upload error: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
    then
        upload_ok=true
        echo "Upload OK"
    else
        echo "WARNING: HF upload failed for $task_id" >&2
    fi

    binary_hf_filename="binaries/${task_id}/executable"
    binary_hf_repo="$BINARY_HF_REPO"

    # Write result record
    python3 - <<PYEOF
import json
record = {
    "task_id": "$task_id",
    "success": True,
    "binary_size": $binary_size,
    "file_type": """$file_type""",
    "strings_output": ${strings_out:+$(echo "$strings_out" | python3 -c "import sys,json; print(json.dumps(sys.stdin.read()))")},
    "nm_output": ${nm_out:+$(echo "$nm_out" | python3 -c "import sys,json; print(json.dumps(sys.stdin.read()))")},
    "objdump_head": ${objdump_head:+$(echo "$objdump_head" | python3 -c "import sys,json; print(json.dumps(sys.stdin.read()))")},
    "binary_hf_repo": "$binary_hf_repo",
    "binary_hf_filename": "$binary_hf_filename",
}
with open("${RESULTS_FILE}", "a") as f:
    f.write(json.dumps(record) + "\n")
print("Result written.")
PYEOF

    ((build_count++)) || true
    echo "Done: $task_id"

done < "$JSONL_PATH"

echo ""
echo "═══════════════════════════════════════════════"
echo "Build complete."
echo "  Built:   $build_count"
echo "  Skipped: $skip_count"
echo "  Failed:  $fail_count"
echo "Results:   $RESULTS_FILE"
echo ""
echo "Next: run merge_build_results.py to update the HF dataset."
