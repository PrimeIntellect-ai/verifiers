---
name: optimize-with-environments
description: Optimize environment system prompts with GEPA through prime gepa run. Use when asked to improve prompt performance without gradient training, compare baseline versus optimized prompts, run GEPA from CLI or TOML configs, or interpret GEPA outputs before deployment.
---

# Optimize With Environments

## Goal
Use GEPA to optimize system prompts in a controlled, reproducible loop.

## Scope
Current GEPA path is for system prompt optimization. If user asks for unsupported optimization targets, stop and clarify before proceeding.

## Endpoint And Model Selection Nudge
1. Encourage users to define reusable aliases in `configs/endpoints.toml`.
2. Ask whether optimization should be validated on instruct or reasoning models.
3. Instruct go-tos: `gpt-4.1` series, `qwen3` instruct series.
4. Reasoning go-tos: `gpt-5` series, `qwen3` thinking series, `glm` series.
5. For benchmark reporting, keep model family fixed between baseline and optimized comparisons unless the user requests a cross-family study.

## Core Workflow
1. Verify baseline first:
```bash
prime eval run my-env -m gpt-4.1-mini -n 50 -r 3 -s
```
2. Run GEPA:
```bash
prime gepa run my-env -m gpt-4.1-mini -M gpt-4.1-mini -B 500 -n 100 -N 50
```
3. Or run from config:
```bash
prime gepa run configs/gepa/wordle.toml
```
4. Re-evaluate with optimized prompt and compare against baseline.

## High-Value Settings
1. `-B/--max-calls`: total optimization budget.
2. `-n/--num-train` and `-N/--num-val`: train/validation split sizes.
3. `--minibatch-size`: reflection granularity.
4. `--perfect-score`: skip already-solved minibatches when max score is known.
5. `--state-columns`: include environment-specific context in reflection data.

## Output Artifacts
Expect and inspect:
1. `best_prompt.txt`
2. `pareto_frontier.jsonl`
3. `metadata.json`

## Quality Rules
1. Do not optimize on top of broken reward logic.
2. For weak deterministic checks, fix rubric quality before GEPA tuning.
3. Keep model, sampling, and dataset conditions stable during baseline-vs-GEPA comparison.
4. Report limitations directly when feature gaps block requested optimization.

## Deliverable
Return:
1. Baseline metrics.
2. Optimized metrics.
3. Prompt diff summary.
4. Recommendation to adopt, iterate, or stop.
