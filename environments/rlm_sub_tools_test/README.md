# RLM Sub-Tools Test Environment

Test environment for validating the RLM sub-agent tools feature.

## Overview

This environment tests the ability of RLMEnv to:
1. Pass tools to sub-LLMs (not the root model)
2. Document available sub-agent tools in the system prompt
3. Execute tool calls within sub-LLM interactions
4. Work correctly with no tools configured (fallback behavior)

## Tools

The environment provides two simple, deterministic tools:

- `calculate(expression: str)` - Evaluate mathematical expressions
- `lookup_data(key: str)` - Look up values from a predefined dictionary

## Task Design

Tasks are designed to **require** tool use for correct answers:
- Mathematical expressions that need calculation
- Data lookups that need the `lookup_data` tool
- Combined tasks that require both tools

The root model uses `llm_batch()` to delegate to sub-LLMs, which autonomously use the tools.

## Reward Functions

- `exact_match_reward` (weight: 1.0) - Correct final answer
- `tools_mentioned_reward` (weight: 0.0) - Metric: checks if model mentioned tools in reasoning

## Usage

```bash
# Install
vf-install rlm_sub_tools_test

# Run with tools (default)
vf-eval -s rlm_sub_tools_test -m gpt-4.1-mini -n 5

# Run without tools (fallback test)
vf-eval -s rlm_sub_tools_test -m gpt-4.1-mini -n 5 --env-args '{"with_tools": false}'
```

## Dependencies

- `verifiers` (with envs extras for sandbox support)

