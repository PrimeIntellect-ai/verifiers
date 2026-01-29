#!/bin/bash
# Build SEA (Single Executable Application) binary for CUA server
#
# This script bundles the CUA server with esbuild and creates a standalone binary
# that can be distributed without requiring npm install on the target machine.
#
# Requirements:
# - Node.js 22+ (for SEA support)
# - pnpm (for installing dependencies)
# - esbuild (installed as devDependency)
# - postject (installed as devDependency)
#
# Usage: bash scripts/build-binary.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "Building CUA server SEA binary..."

# Create output directory
mkdir -p dist/sea

# Step 1: Bundle with esbuild
echo "Bundling with esbuild..."
npx esbuild index.ts \
  --bundle \
  --platform=node \
  --target=node22 \
  --format=cjs \
  --outfile=dist/sea/bundle.cjs \
  --external:playwright \
  --external:playwright-core

echo "Bundle created at dist/sea/bundle.cjs"

# Step 2: Generate SEA blob
echo "Generating SEA blob..."
node --experimental-sea-config sea-config.json

echo "SEA blob created at dist/sea/sea-prep.blob"

# Step 3: Create the binary
echo "Creating platform binary..."
bash scripts/create-binary.sh

echo "Build complete!"
