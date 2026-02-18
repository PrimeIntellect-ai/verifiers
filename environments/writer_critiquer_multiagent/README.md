# writer-critiquer-multiagent

Two-agent collaborative writing environment built on `MultiAgentEnv`.

## Overview
- Two actors alternate turns: `writer` then `critiquer`.
- Two role-specific tools with hidden state injection:
  - `submit_draft(writing)` for writer
  - `submit_feedback(feedback)` for critiquer
- Dynamic tool filtering updates `state["tool_defs"]` per turn so each actor only sees its own tool.
- State preserves full progression in append-only histories:
  - `draft_history`
  - `feedback_history`
  - `collaboration_log`
- Writer prompt always includes latest feedback; critiquer prompt always includes latest draft.
- Reward uses `JudgeRubric` with a detailed writing-quality rubric.

## Required Environment Variables
- `OPENAI_API_KEY` (or pass a different `judge_api_key_var` in `load_environment(...)`)

## Quickstart
```bash
prime eval run writer-critiquer-multiagent
```
