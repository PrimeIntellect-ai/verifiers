#!/usr/bin/env bash
# Build and push the ProgramBench toolchain image.
# Requires: docker login (to push to Docker Hub or internal registry)
#
# Usage:
#   ./build.sh                     # build + push latest
#   TAG=v1.2 ./build.sh            # build + push with custom tag
#
# Note: buildx with --platform linux/amd64 requires Docker Buildx and QEMU
# if you are building from an ARM host (e.g. Apple Silicon Mac).
# Run once to set up: docker buildx create --use

set -euo pipefail

IMAGE="primeintellect/programbench-toolchain"
TAG="${TAG:-latest}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Building ${IMAGE}:${TAG} for linux/amd64 ..."

docker buildx build \
    --platform linux/amd64 \
    --tag "${IMAGE}:${TAG}" \
    --push \
    "${SCRIPT_DIR}"

echo "Pushed ${IMAGE}:${TAG}"
