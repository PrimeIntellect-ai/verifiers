#!/bin/bash
# CUA Server Sandbox Setup Script
# This script sets up and starts the CUA server inside a sandbox container.
#
# Expected environment:
# - Node.js 18+ available (use node:18-slim docker image)
# - Server files already uploaded to /app/cua-server/
#
# Usage: This script is called by CUASandboxMode via start_background_job()

set -e

cd /app/cua-server

# Install curl if not present (needed for health checks)
if ! command -v curl &> /dev/null; then
    echo "Installing curl..."
    apt-get update -qq && apt-get install -y -qq curl
fi

# Install pnpm if not present
if ! command -v pnpm &> /dev/null; then
    echo "Installing pnpm..."
    npm install -g pnpm
fi

# Remove any existing node_modules to avoid pnpm interactive prompts
if [ -d "node_modules" ]; then
    echo "Removing existing node_modules..."
    rm -rf node_modules
fi

# Install dependencies (CI=true makes pnpm non-interactive)
echo "Installing dependencies..."
CI=true pnpm install --frozen-lockfile 2>/dev/null || CI=true pnpm install

# Set server configuration
export CUA_SERVER_PORT="${CUA_SERVER_PORT:-3000}"
export CUA_SERVER_HOST="${CUA_SERVER_HOST:-0.0.0.0}"

echo "Starting CUA server on ${CUA_SERVER_HOST}:${CUA_SERVER_PORT}..."

# Start server (this keeps running in the foreground)
exec pnpm tsx index.ts
