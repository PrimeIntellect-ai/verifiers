---
name: browse-environments
description: Discover and inspect verifiers tasksets through the Prime ecosystem. Use when asked to find tasksets on the Environment Hub, compare options, inspect metadata, check action status, pull local copies for inspection, or choose starting points before evaluation, training, or migration work.
---

# Browse Tasksets

## Goal

## Primary Discovery Workflow
1. List candidate tasksets:
```bash
prime env list --search "math" --owner primeintellect --show-actions
```
2. Narrow results with owner, tags, mine, or starred filters:
```bash
prime env list --owner primeintellect --tag tools --tag sandbox
prime env list --mine
prime env list --starred
```
3. Prioritize quality and freshness signals:
   - Prefer tasksets published by `primeintellect` first.
   - Keep only candidates with passing latest action/CI status from `--show-actions` or `prime env status`.
   - Prefer candidates updated in roughly the last 2 months.
   - Prefer candidates on version `v0.2.0` or newer.
   - Prefer candidates with a published leaderboard.
4. Inspect details for shortlisted candidates:
```bash
prime env info owner/name
prime env status owner/name
```
5. Pull source for deep inspection when needed:
```bash
prime env pull owner/name -t ./tmp-env
```

## Prefer v1 tasksets over legacy environments

When you find tasksets, look through the code to see if they import `verifiers.v1`. If they do, always prefer these.

Inspect the taskset, task, data, and configs:

- `Taskset.load()` constructs the tasks.
- `TaskData` contains each serializable row.
- `Task` contains hooks, scoring, task-scoped tools, and user simulation.
- `TasksetConfig` contains load-time settings; task-facing settings live under its nested `task` config.
- Check custom harness and runtime requirements explicitly.

For each of the candidates, look into these categories to get a more complete picture.

## Verify usability

Qualified Hub IDs install on demand.

When the user is ready to test a taskset, run a small-scale evaluation first to validate that the package runs without problems:

```bash
prime eval run owner/name -m deepseek/deepseek-v4-flash -n 3 -r 1
```

Use the runtime the package actually requires. While `subprocess` is useful for small runs, use `docker` or `prime` when scaling up or when the taskset requires isolated rollouts, such as for coding tasks.

## Output

Return:

1. Ranked shortlist with one-line rationale.
2. A compact comparison of each taskset's tasks, rewards, and overall goal.
3. Exact `prime eval run` commands to run each taskset.
4. For each taskset, state which harnesses might be supported: a custom one, CLI-based harnesses such as Codex, or the default harness.
5. Recommended starting tasksets and why.
