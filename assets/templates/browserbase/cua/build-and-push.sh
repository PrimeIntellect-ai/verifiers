#!/bin/bash
# Build and push a CUA server runtime image to Prime Images.
#
# Usage:
#   ./build-and-push.sh                      # Push cua-server:latest
#   ./build-and-push.sh my-tag              # Push cua-server:my-tag
#   ./build-and-push.sh owner/cua-server:latest  # Push an explicit fully qualified ref

set -euo pipefail

IMAGE_NAME=${IMAGE_NAME:-"cua-server"}
WAIT_TIMEOUT_SECONDS=${WAIT_TIMEOUT_SECONDS:-600}
WAIT_INTERVAL_SECONDS=${WAIT_INTERVAL_SECONDS:-5}

if ! command -v prime >/dev/null 2>&1; then
    echo "ERROR: prime CLI not found. Install it first."
    exit 1
fi

if ! command -v node >/dev/null 2>&1; then
    echo "ERROR: node is required to parse Prime image metadata."
    exit 1
fi

RAW_TARGET=${1:-latest}
case "${RAW_TARGET}" in
    */*:*|*:* )
        REQUESTED_IMAGE="${RAW_TARGET}"
        ;;
    * )
        REQUESTED_IMAGE="${IMAGE_NAME}:${RAW_TARGET}"
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

resolve_from_list() {
    local requested_image="$1"
    REQUESTED_IMAGE="${requested_image}" prime images list --output json | node -e '
const fs = require("fs");

const requested = process.env.REQUESTED_IMAGE;
const raw = fs.readFileSync(0, "utf8");

let data;
try {
  data = JSON.parse(raw);
} catch {
  process.exit(0);
}

const items = Array.isArray(data) ? data : data.items || data.data || data.images || [];

function extractRef(item) {
  if (!item || typeof item !== "object") return null;
  for (const key of ["displayRef", "fullImagePath"]) {
    const value = item[key];
    if (typeof value === "string" && value) return value;
  }
  const imageName = item.imageName;
  const imageTag = item.imageTag;
  if (typeof imageName === "string" && imageName) {
    return typeof imageTag === "string" && imageTag ? `${imageName}:${imageTag}` : imageName;
  }
  for (const key of ["image", "image_reference", "image_ref", "name", "ref"]) {
    const value = item[key];
    if (typeof value === "string" && value) return value;
  }
  const image = item.image;
  if (image && typeof image === "object") {
    const name = image.name;
    const tag = image.tag;
    if (typeof name === "string" && name) {
      return typeof tag === "string" && tag ? `${name}:${tag}` : name;
    }
  }
  return null;
}

function extractTimestamp(item) {
  if (!item || typeof item !== "object") return 0;
  const raw = item.pushedAt || item.createdAt || item.updatedAt || item.created_at || item.updated_at;
  if (typeof raw !== "string" || !raw) return 0;
  const timestamp = Date.parse(raw);
  return Number.isNaN(timestamp) ? 0 : timestamp;
}

const matches = [];
for (const item of items) {
  const ref = extractRef(item);
  if (!ref) continue;
  if (ref === requested || ref.endsWith(`/${requested}`) || ref.endsWith(requested)) {
    matches.push({ ref, timestamp: extractTimestamp(item) });
  }
}

matches.sort((a, b) => b.timestamp - a.timestamp);
if (matches.length > 0) {
  process.stdout.write(matches[0].ref);
}
' || true
}

get_status() {
    local image_ref="$1"
    IMAGE_REF="${image_ref}" prime images list --output json | node -e '
const fs = require("fs");

const target = process.env.IMAGE_REF;
const raw = fs.readFileSync(0, "utf8");

let data;
try {
  data = JSON.parse(raw);
} catch {
  process.exit(0);
}

const items = Array.isArray(data) ? data : data.items || data.data || data.images || [];

function extractRef(item) {
  if (!item || typeof item !== "object") return null;
  for (const key of ["displayRef", "fullImagePath"]) {
    const value = item[key];
    if (typeof value === "string" && value) return value;
  }
  const imageName = item.imageName;
  const imageTag = item.imageTag;
  if (typeof imageName === "string" && imageName) {
    return typeof imageTag === "string" && imageTag ? `${imageName}:${imageTag}` : imageName;
  }
  for (const key of ["image", "image_reference", "image_ref", "name", "ref"]) {
    const value = item[key];
    if (typeof value === "string" && value) return value;
  }
  const image = item.image;
  if (image && typeof image === "object") {
    const name = image.name;
    const tag = image.tag;
    if (typeof name === "string" && name) {
      return typeof tag === "string" && tag ? `${name}:${tag}` : name;
    }
  }
  return null;
}

function extractTimestamp(item) {
  if (!item || typeof item !== "object") return 0;
  const raw = item.pushedAt || item.createdAt || item.updatedAt || item.created_at || item.updated_at;
  if (typeof raw !== "string" || !raw) return 0;
  const timestamp = Date.parse(raw);
  return Number.isNaN(timestamp) ? 0 : timestamp;
}

const matches = [];
for (const item of items) {
  const ref = extractRef(item);
  if (ref !== target) continue;
  const status = item.status || item.state || "";
  if (status) {
    matches.push({ status: String(status), timestamp: extractTimestamp(item) });
  }
}

matches.sort((a, b) => b.timestamp - a.timestamp);
if (matches.length > 0) {
  process.stdout.write(matches[0].status);
}
' || true
}

to_lower() {
    printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

echo "============================================"
echo "Building CUA Server Runtime Image"
echo "Requested image: ${REQUESTED_IMAGE}"
echo "============================================"

push_output_file="$(mktemp)"
trap 'rm -f "${push_output_file}"' EXIT

echo ""
echo "[1/2] Submitting image build to Prime Images..."
prime images push "${REQUESTED_IMAGE}" --dockerfile Dockerfile.runtime --context . 2>&1 | tee "${push_output_file}"

resolved_image="$(sed -n 's/^Image:[[:space:]]*//p' "${push_output_file}" | tail -n 1)"
if [ -z "${resolved_image}" ]; then
    resolved_image="$(resolve_from_list "${REQUESTED_IMAGE}")"
fi

if [ -z "${resolved_image}" ]; then
    echo "ERROR: Could not resolve the fully qualified Prime image reference."
    echo "Run 'prime images list --output json' and check the newly created image."
    exit 1
fi

echo ""
echo "[2/2] Waiting for image to become ready..."
deadline=$((SECONDS + WAIT_TIMEOUT_SECONDS))
last_status=""
while [ "${SECONDS}" -lt "${deadline}" ]; do
    status="$(get_status "${resolved_image}")"
    if [ -n "${status}" ]; then
        last_status="${status}"
        echo "Current status: ${status}"
        case "$(to_lower "${status}")" in
            ready|succeeded|completed)
                break
                ;;
            failed|error)
                echo "ERROR: Prime image build failed for ${resolved_image}."
                exit 1
                ;;
        esac
    fi
    sleep "${WAIT_INTERVAL_SECONDS}"
done

case "$(to_lower "${last_status}")" in
    ready|succeeded|completed)
        ;;
    *)
        echo "ERROR: Timed out waiting for ${resolved_image} to become ready."
        echo "Run 'prime images list' to check the image status."
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "Prime image is ready: ${resolved_image}"
echo "============================================"
echo ""
echo "To use this image with BrowserEnv:"
echo ""
echo "  env = BrowserEnv("
echo "      mode='cua',"
echo "      use_prebuilt_image=True,"
echo "      prebuilt_image='${resolved_image}',"
echo "      ..."
echo "  )"
echo ""
