#!/usr/bin/env bash
set -euo pipefail

mkdir -p /logs/verifier

if [ "$(cat /app/answer.txt 2>/dev/null || true)" = "verifiers-v1" ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
