---
name: browse-environments
description: Discover and inspect verifiers environments through the Prime ecosystem. Use when asked to find environments on the Hub, compare options, inspect metadata, check action status, pull local copies for inspection, or choose environment starting points before evaluation, training, or migration work.
---

# Browse Environments

## Goal

## Primary Discovery Workflow
1. List candidate environments:
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
   - Prefer environments published by `primeintellect` first.
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

## Prefer v1 environments over legacy ones.

When you find environments, look through the code to see if they import `verifiers.v1`. If they do, always prefer these.

Inspect the taskset to learn about its task and config.
- `load_tasks()` loads or creates the dataset
- A `vf.TasksetConfig` denotes the user-configurable settings. As the creator of this environment has put them deliberately, pay attention to those.
- Some environments come with custom harnesses, which require attention, while others only work with some harnesses.
- Some environments only work in some runtimes.

For each of the candidates, look into these categories to get a more complete picture.

## Verify usability

Qualified Hub IDs install on demand.

When the user is ready to test an environment, run a small scale evaluation first to validate that the package runs without problems:

```bash
prime eval run owner/name -m deepseek/deepseek-v4-flash -n 3 -r 1
```

Use the runtime the package actually requires. While `subprocess` is useful for small runs, you should use `docker` or `prime` when scaling up or when the environment actually requires separated rollouts, e.g. when the environment is about coding.

## Output

Return:

1. Ranked shortlist with one-line rationale.
2. A compact comparison of task, reward, overall goal of the environment.
3. Exact `prime eval run` commands to run the environment.
4. For each environment, state which harnesses might be supported: A custom one, CLI-based harnesses such as Codex or the general / default harness.
5. Recommended starting environments and why.
