# terminal-bench-2

### Overview
- **Environment ID**: `terminal-bench-2`
- **Short description**: Terminal-Bench 2 composed from `tasksets` and `harnesses`
- **Tags**: terminal-bench, cli_agent, composable

### Quickstart

```bash
prime eval run terminal-bench-2 -n 1 -r 1
```

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `git_ref` | str | `main` | Harbor-native Terminal-Bench 2 repo ref to download |
| `dataset_path` | str | unset | Local Harbor-format Terminal-Bench task directory |
| `task_ids` / `tasks` | list[str] | unset | Optional task IDs |
| `max_examples` | int | `-1` | Limit examples; `-1` uses all |
| `limit` | int | unset | Alias for `max_examples` |
| `harness` | str/table | unset | Harness name or generic harness factory config |
| `harness_config` | table | `{agent = "openclaw"}` | Harness factory config |

The taskset uses the Harbor-native Terminal-Bench 2 task repository directly.
Each task already has `instruction.md`, `task.toml`, tests, solutions, and a
prebuilt `environment.docker_image`. Harness config is resolved by
`harnesses.build_harness_from_config`, so existing harness names like `opencode`,
`codex`, `mini-swe-agent`, and `terminus-2` work without environment-specific
branching.
