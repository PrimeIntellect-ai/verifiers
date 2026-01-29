#!/bin/bash
# Build and push CUA server runtime image to Docker Hub
#
# This script:
# 1. Builds the binary via Dockerfile.build
# 2. Builds the runtime image via Dockerfile.runtime
# 3. Pushes to Docker Hub
#
# Usage:
#   ./build-and-push.sh                    # Push as :latest
#   ./build-and-push.sh v1.0.0             # Push as :v1.0.0
#   DOCKERHUB_USER=myuser ./build-and-push.sh  # Use different Docker Hub user

set -e

# Configuration
DOCKERHUB_USER=${DOCKERHUB_USER:-"deepdream19"}
VERSION=${1:-latest}
IMAGE_NAME="cua-server"
FULL_IMAGE="${DOCKERHUB_USER}/${IMAGE_NAME}:${VERSION}"

echo "============================================"
echo "Building CUA Server Runtime Image"
echo "Target: ${FULL_IMAGE}"
echo "============================================"

# Ensure we're in the right directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Step 1: Login to Docker Hub
echo ""
echo "[1/4] Authenticating with Docker Hub..."
docker login

# Step 2: Build the binary
echo ""
echo "[2/4] Building binary via Dockerfile.build..."
docker build --platform linux/amd64 -f Dockerfile.build -t cua-builder .

# Step 3: Extract binary from builder
echo ""
echo "[3/4] Extracting binary from builder container..."
mkdir -p dist/sea
docker run --rm --platform linux/amd64 -v "$(pwd)/dist:/output" cua-builder

# Verify binary was extracted
if [ ! -f "dist/sea/cua-server-linux-x64" ]; then
    echo "ERROR: Binary not found at dist/sea/cua-server-linux-x64"
    exit 1
fi

echo "Binary size: $(du -h dist/sea/cua-server-linux-x64 | cut -f1)"

# Step 4: Build runtime image
echo ""
echo "[4/4] Building runtime image via Dockerfile.runtime..."
docker build --platform linux/amd64 -f Dockerfile.runtime -t "${FULL_IMAGE}" .

# Step 5: Push to Docker Hub
echo ""
echo "Pushing to Docker Hub..."
docker push "${FULL_IMAGE}"

# Also tag and push as latest if we're pushing a version tag
if [ "$VERSION" != "latest" ]; then
    echo "Also tagging as :latest..."
    docker tag "${FULL_IMAGE}" "${DOCKERHUB_USER}/${IMAGE_NAME}:latest"
    docker push "${DOCKERHUB_USER}/${IMAGE_NAME}:latest"
fi

echo ""
echo "============================================"
echo "Successfully published: ${FULL_IMAGE}"
echo "============================================"
echo ""
echo "To use this image with BrowserEnv:"
echo ""
echo "  env = BrowserEnv("
echo "      mode='cua',"
echo "      use_prebuilt_image=True,"
echo "      prebuilt_image='${FULL_IMAGE}',"
echo "      ..."
echo "  )"
echo ""
