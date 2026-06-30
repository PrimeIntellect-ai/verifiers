#!/bin/sh
# Pristine test suite for the example task — staged into each candidate's _tests/ and run by the
# judge against that candidate's /repo. Exit non-zero on failure. Replace with the real suite.
set -e
python3 -c "from repo.config import parse_config; assert parse_config('') == {}, 'empty file should parse to {}'"
echo "ok"
