# RLM Metrics Contract

This document defines a stable schema for RLM-related metrics stored in `state`.
It is intended for environments under `environments/` that want to expose RLM
metrics consistently, regardless of recursion depth or tool configuration.

The contract is **additive-only**: new keys may be introduced, but existing key
names and semantics should not change.

## Scope

These metrics are meant to be produced by a shared helper (e.g. an
`rlm_metrics_utils.add_metrics_to_state(...)` function) and consumed by
rubrics or evaluation pipelines. All keys are scalar and meaningful for any
RLM configuration, including recursion disabled.

## Required Scalar Metrics

All required metrics should always be present. If no sub-LLM calls occurred,
use `0` (or `0.0` for floats).

### Sub-LLM Aggregates

- `sub_llm_call_count` (int)
  - Number of sub-LLM requests (i.e., batch items) across the rollout.
- `sub_llm_total_turns` (int)
  - Total LLM turns across all sub-LLM calls (including tool loops and the
    final forced answer turn when max tool turns are reached).
- `sub_llm_prompt_tokens` (int)
  - Sum of prompt tokens across all sub-LLM LLM calls.
- `sub_llm_completion_tokens` (int)
  - Sum of completion tokens across all sub-LLM LLM calls.
- `sub_llm_total_tool_calls` (int)
  - Total tool calls made by sub-LLMs (including recursion tool calls).
- `sub_llm_batch_count` (int)
  - Number of distinct sub-LLM batches issued.
- `sub_llm_max_batch_size` (int)
  - Maximum batch size among all sub-LLM batches.
- `sub_llm_mean_batch_size` (float)
  - Mean batch size across all sub-LLM batches.

### Sub-LLM Depth Summary

Depth is 1-indexed for sub-LLM calls (the first level below the root model).

- `sub_llm_depth_max` (int)
  - Maximum depth observed (0 if no sub-LLM calls).
- `sub_llm_depth_mean` (float)
  - Mean depth across sub-LLM calls (0.0 if no calls).
- `sub_llm_depth_gt1_frac` (float)
  - Fraction of sub-LLM calls with depth > 1 (0.0 if no calls).

### Sub-LLM Per-Call Averages

These are simple derived scalars for rubric-friendly reporting:

- `sub_llm_prompt_tokens_per_call` (float)
- `sub_llm_completion_tokens_per_call` (float)
- `sub_llm_tool_calls_per_call` (float)
- `sub_llm_turns_per_call` (float)

All averages should be `0.0` when `sub_llm_call_count == 0`.

### Main RLM Aggregates (Root Model)

These already exist in `RLMEnv` and remain unchanged:

- `main_rlm_turns` (int)
- `main_rlm_prompt_tokens` (int)
- `main_rlm_completion_tokens` (int)

### REPL Timing

These already exist in `RLMEnv` and remain unchanged:

- `repl_total_time_seconds` (float)
- `repl_call_count` (int)
- `repl_mean_time_seconds` (float)

## Optional Raw Breakdown (Not for Rubrics)

These are useful for debugging or analysis but should not be used as rubric
inputs because they are structured and not scalar.

- `sub_llm_calls_by_depth` (dict[int,int])
- `sub_llm_prompt_tokens_by_depth` (dict[int,int])
- `sub_llm_completion_tokens_by_depth` (dict[int,int])
- `sub_llm_tool_calls_by_depth` (dict[int,int])
- `sub_llm_turns_by_depth` (dict[int,int])

## Invariants

- All required keys exist on the final `state`.
- All scalar metrics are valid numbers (no NaN/inf).
- Metrics are meaningful regardless of:
  - `max_sub_llm_depth` being `0`, `1`, `N`, or `None`
  - presence/absence of sub-LLM tools
  - recursion tool enablement

