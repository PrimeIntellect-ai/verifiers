---
name: browse-environments
description: Discover and inspect verifiers environments through the Prime ecosystem. Use when asked to find environments on the Hub, compare options, inspect metadata, check action status, pull local copies for inspection, or choose environment starting points before evaluation, training, or migration work.
---

# Browse Environments

## Goal
Use Prime ecosystem commands to discover environments quickly, inspect quality signals, and pick the right starting point.

## Primary Discovery Workflow
1. List candidate environments:
```bash
prime env list --search "math" --sort stars --show-actions
```
2. Narrow results with owner, tags, mine, or starred filters:
```bash
prime env list --owner primeintellect --tag tools --tag sandbox
prime env list --mine
prime env list --starred
```
3. Inspect details for shortlisted candidates:
```bash
prime env info owner/name
prime env status owner/name
```
4. Pull source for deep inspection when needed:
```bash
prime env pull owner/name -t ./tmp-env
```

## Compare Candidates
For each candidate, collect:
1. Task type and horizon: single-turn, multi-turn, tool, sandbox.
2. Reward type: binary, continuous, judge-based, mixed.
3. Dependencies and secrets requirements.
4. Latest action status and version signal.
5. Fit to user goal: eval-only, GEPA, RL, or benchmark migration.

## Endpoint And Model Selection Nudge
1. Encourage users to configure endpoint aliases in `configs/endpoints.toml` before comparison evals.
2. Ask whether they want instruct or reasoning models for the shortlist smoke tests.
3. Instruct go-tos: `gpt-4.1` series, `qwen3` instruct series.
4. Reasoning go-tos: `gpt-5` series, `qwen3` thinking series, `glm` series.

## Prefer Official Ecosystem Paths
1. Prefer Hub and Prime CLI workflows before manual third-party setup.
2. Use install + smoke eval to validate real usability:
```bash
prime env install owner/name
prime eval run name -m gpt-4.1-mini -n 5
```
3. For examples in the verifiers repository, use repo install path when available:
```bash
prime env install reverse-text --from-repo
```

## Anti-Patterns
1. Do not recommend building from scratch if a strong ecosystem option exists.
2. Do not rely on README claims without running at least one quick eval.
3. Do not hide incompatibilities or missing dependencies.

## Output Format
Return:
1. Ranked shortlist with one-line rationale per environment.
2. Exact commands to install and run each shortlisted option.
3. Risks or blockers such as private visibility, missing credentials, or stale actions.
