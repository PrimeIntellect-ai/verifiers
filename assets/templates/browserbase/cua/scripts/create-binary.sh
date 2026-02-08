#!/bin/bash
# Create SEA binary from prepared blob
#
# This script creates the final executable by:
# 1. Detecting the current platform
# 2. Copying the Node.js executable
# 3. Injecting the SEA blob using postject
#
# Usage: bash scripts/create-binary.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Detect platform
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
    Linux)
        PLATFORM="linux"
        ;;
    Darwin)
        PLATFORM="darwin"
        ;;
    *)
        echo "Unsupported OS: $OS"
        exit 1
        ;;
esac

case "$ARCH" in
    x86_64)
        ARCH_NAME="x64"
        ;;
    arm64|aarch64)
        ARCH_NAME="arm64"
        ;;
    *)
        echo "Unsupported architecture: $ARCH"
        exit 1
        ;;
esac

BINARY_NAME="cua-server-${PLATFORM}-${ARCH_NAME}"
NODE_PATH="$(which node)"
OUTPUT_PATH="dist/sea/${BINARY_NAME}"

echo "Creating binary for ${PLATFORM}-${ARCH_NAME}..."

# Copy Node executable
echo "Copying Node.js executable..."
cp "$NODE_PATH" "$OUTPUT_PATH"

# On macOS, we need to remove the code signature first
if [ "$PLATFORM" = "darwin" ]; then
    echo "Removing code signature (macOS)..."
    codesign --remove-signature "$OUTPUT_PATH" 2>/dev/null || true
fi

# Inject the SEA blob
echo "Injecting SEA blob..."
npx postject "$OUTPUT_PATH" NODE_SEA_BLOB dist/sea/sea-prep.blob \
    --sentinel-fuse NODE_SEA_FUSE_fce680ab2cc467b6e072b8b5df1996b2

# On macOS, re-sign the binary
if [ "$PLATFORM" = "darwin" ]; then
    echo "Re-signing binary (macOS)..."
    codesign --sign - "$OUTPUT_PATH" 2>/dev/null || true
fi

# Make executable
chmod +x "$OUTPUT_PATH"

echo "Binary created at: $OUTPUT_PATH"
ls -lh "$OUTPUT_PATH"
