# Example Environments

This directory contains native v1 taskset/harness packages and legacy v0 compatibility examples.
Use native packages as templates for new work.

## Native v1 examples

Native packages import `verifiers.v1 as vf` and export one `vf.Taskset` subclass through
`__all__`. A package may also export a custom `vf.Harness`.

| Package | Pattern |
| --- | --- |
| `reverse_text_v1` | Minimal single-turn taskset and pure trace reward. |
| `gsm8k_v1` | In-runtime PEP 723 verifier and model-free validation. |
| `code_golf_v1` | Cross-rollout `@vf.group_reward`. |
| `alphabet_sort_v1` | Stateful multi-turn `vf.User` simulator. |
| `glossary_v1` | Custom vf-native MCP `Toolset`. |
| `wiki_search_v1` | Shared read-only tool server and LLM judge. |
| `scratchpad_v1` | Shared writable tool server with per-rollout typed state. |
| `deepwiki_v1` | Existing remote MCP endpoint. |
| `color_codeword_v1` | Multimodal task prompt. |
| `wordle_v1` | Thin taskset over the built-in TextArena integration. |
| `compact` | Custom branching/context-compaction harness. |

The repository also ships built-in tasksets and harnesses under `verifiers/v1/tasksets` and
`verifiers/v1/harnesses`.

## Run examples

The workspace sync installs checked-in v1 packages. Run them directly:

```bash
uv run eval reverse-text-v1 -n 5
uv run eval @ configs/gsm8k.toml
uv run validate gsm8k-v1 -n 20 --runtime.type subprocess
```

Switch harness/runtime independently:

```bash
uv run eval gsm8k-v1 -n 1 --harness.id rlm
uv run eval harbor -n 1 --harness.id bash --harness.runtime.type docker
```

Use `--dry-run` to resolve a config without model calls:

```bash
uv run eval @ configs/wiki_search.toml --dry-run
```

## Create another native package

```bash
uv run init my-task-v1
uv run init my-agent-v1 -T -U -H
```

Then install and test the package boundary:

```bash
uv pip install -e environments/my_task_v1
uv run validate my-task-v1 -n 5 --runtime.type subprocess
uv run eval my-task-v1 -n 5 -r 2
```

For repository package-install coverage:

```bash
uv run pytest tests/test_envs.py -k my_task_v1 -vv
```

## Legacy v0 examples

Directories without the `_v1` suffix generally exercise the preserved v0 environment hierarchy:
`load_environment()`, datasets, rubrics/parsers, and classes such as `SingleTurnEnv`,
`MultiTurnEnv`, or `ToolEnv`. They remain for compatibility and legacy test coverage.

Run one through the v1 bridge with `--id`:

```bash
uv run eval --id reverse-text -n 5
```

Do not copy a legacy example when authoring a native v1 package. See
[`docs/environments.md`](../docs/environments.md) for the current taskset, toolset, user, harness,
runtime, and packaging contracts.
