#!/usr/bin/env bash
set -euo pipefail

if [ "$(cat /app/hello.txt 2>/dev/null || true)" = "Hello, world!" ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
