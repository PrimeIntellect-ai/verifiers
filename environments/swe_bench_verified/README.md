# swe-bench-verified

SWE-bench Verified patch-generation taskset for Verifiers.

## Overview

- Environment ID: `swe-bench-verified`
- Dataset: `princeton-nlp/SWE-bench_Verified`
- Default split: `test`
- Task type: single-turn patch generation

Each task asks the model to produce a unified diff inside `<patch>...</patch>` tags.
The task metadata preserves the SWE-bench `instance_id`, repository, base commit,
environment setup commit, version, fail-to-pass tests, pass-to-pass tests, and test
patch. The environment records the generated patch and the official SWE-bench JSONL
submission row shape:

```json
{"instance_id": "...", "model_patch": "diff --git ..."}
```

## Quickstart

```bash
prime env install swe-bench-verified
prime eval run swe-bench-verified -n 5 -r 1
```

Use environment args to cap local dataset loading during smoke tests:

```bash
prime eval run swe-bench-verified -a '{"taskset": {"max_examples": 5}}'
```

## Scoring

The default reward is deterministic and local. It combines exact normalized patch
match, patch text similarity, and changed-file overlap against the gold patch. This
is a cheap training/evaluation signal and an official-submission handoff, not a
replacement for the full SWE-bench execution harness.

## Configuration

| Field | Default | Description |
| --- | --- | --- |
| `dataset_name` | `princeton-nlp/SWE-bench_Verified` | Hugging Face dataset name. |
| `dataset_split` | `test` | Dataset split to load. |
| `max_examples` | `null` | Optional cap for smoke tests. |
| `streaming` | `false` | Load the dataset through HF streaming. |
| `include_test_names` | `true` | Include fail/pass test names in the prompt. |
