#!/bin/bash

# Generate /home/fix.patch from the current working-tree diff using the same
# fix_patch/test_patch path split as upstream Multi-SWE dataset construction.

set -euo pipefail

REPO_DIR="${1:-$(pwd)}"
BASE_COMMIT="${2:-HEAD}"
OUTPUT_PATH="${3:-/home/fix.patch}"
cd "$REPO_DIR"

if [ -z "$BASE_COMMIT" ]; then
    BASE_COMMIT="HEAD"
fi

FIX_FILES=()
TEST_FILES=()
UNTRACKED_FILES=()

while IFS= read -r -d '' file; do
    UNTRACKED_FILES+=("$file")
done < <(git ls-files --others --exclude-standard -z)

if [ "${#UNTRACKED_FILES[@]}" -gt 0 ]; then
    git add --intent-to-add -- "${UNTRACKED_FILES[@]}"
fi

while IFS= read -r -d '' file; do
    lowered="$(printf '%s' "$file" | tr '[:upper:]' '[:lower:]')"
    if [[ "$lowered" == *"test"* ]] || [[ "$lowered" == *"tests"* ]] || [[ "$lowered" == *"e2e"* ]] || [[ "$lowered" == *"testing"* ]]; then
        TEST_FILES+=("$file")
    else
        FIX_FILES+=("$file")
    fi
done < <(git diff --name-only -z "$BASE_COMMIT" --)

if [ "${#FIX_FILES[@]}" -eq 0 ]; then
    : > "$OUTPUT_PATH"
else
    git diff --binary "$BASE_COMMIT" -- "${FIX_FILES[@]}" > "$OUTPUT_PATH"
fi

if [ "${#TEST_FILES[@]}" -gt 0 ]; then
    printf 'Dropped %d test-like file(s) from Multi-SWE fix patch.\n' "${#TEST_FILES[@]}" >&2
fi

if [ ! -s "$OUTPUT_PATH" ]; then
    : > "$OUTPUT_PATH"
fi

git reset --hard "$BASE_COMMIT" >/dev/null
if [ "${#UNTRACKED_FILES[@]}" -gt 0 ]; then
    git clean -fd -- "${UNTRACKED_FILES[@]}" >/dev/null
fi
