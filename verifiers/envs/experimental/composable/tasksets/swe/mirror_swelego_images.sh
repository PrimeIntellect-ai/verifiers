#!/usr/bin/env bash
# Mirror SWE-Lego Docker Hub images to the Prime Intellect sandbox registry.
#
# Usage:
#   bash mirror_swelego_images.sh [--dry-run] [--workers N]
#
# Prerequisites:
#   - docker CLI authenticated to both Docker Hub and GCR
#   - gcloud auth configure-docker us-central1-docker.pkg.dev
#   - uv / Python with `datasets` installed
#
# Each image is pulled once and re-tagged; existing tags in GCR are skipped.

set -euo pipefail

REGISTRY_PREFIX="us-central1-docker.pkg.dev/prime-intellect-platform/prod-sandbox"
DATASET="SWE-Lego/SWE-Lego-Synthetic-Data"
SPLIT="resolved"
DRY_RUN=0
WORKERS=4

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --workers) WORKERS="$2"; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
    shift
done

# Collect unique image names from the dataset
echo "Loading image list from $DATASET ($SPLIT)..."
IMAGES_FILE="$(mktemp)"
trap 'rm -f "$IMAGES_FILE"' EXIT

python - <<PYEOF > "$IMAGES_FILE"
from datasets import load_dataset
ds = load_dataset("$DATASET", split="$SPLIT", keep_in_memory=False)
images = sorted(set(ds["image_name"]))
for img in images:
    print(img)
PYEOF

TOTAL=$(wc -l < "$IMAGES_FILE")
echo "Found $TOTAL unique images to mirror."

mirror_one() {
    local src="$1"
    local dst="${REGISTRY_PREFIX}/${src}"

    # Check if already in registry (docker manifest inspect is fast)
    if docker manifest inspect "$dst" >/dev/null 2>&1; then
        echo "[SKIP] $src already in registry"
        return 0
    fi

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY-RUN] would mirror $src → $dst"
        return 0
    fi

    echo "[MIRROR] $src"
    docker pull "$src"
    docker tag "$src" "$dst"
    docker push "$dst"
    docker rmi "$src" "$dst" 2>/dev/null || true
    echo "[DONE] $src"
}

export -f mirror_one
export REGISTRY_PREFIX DRY_RUN

xargs -P "$WORKERS" -I{} bash -c 'mirror_one "$@"' _ {} < "$IMAGES_FILE"

echo "Done."
