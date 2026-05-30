#!/bin/bash
# CUA Server Binary Setup Script
# This script runs the pre-built SEA binary inside a sandbox container.
#
# Expected environment:
# - Binary already uploaded to /app/cua-server/cua-server-linux-x64
#
# Usage: This script is called by CUASandboxMode via start_background_job()

set -e

cd /app/cua-server

# Install curl if not present (needed for health checks).
# Acquire::Retries=3 mitigates transient archive.ubuntu.com CDN sync mismatches
# that fail fresh-sandbox apt-get update mid-rollout (launchpad bug #1876035).
if ! command -v curl &> /dev/null; then
    echo "Installing curl..."
    apt-get -o Acquire::Retries=3 update -qq && apt-get -o Acquire::Retries=3 install -y -qq curl
fi

# Set server configuration
export CUA_SERVER_PORT="${CUA_SERVER_PORT:-3000}"
export CUA_SERVER_HOST="${CUA_SERVER_HOST:-0.0.0.0}"

echo "Starting CUA server binary on ${CUA_SERVER_HOST}:${CUA_SERVER_PORT}..."

# Make binary executable and run
chmod +x ./cua-server-linux-x64
exec ./cua-server-linux-x64
