---
name: brainstorm
description: Run interactive brainstorming across verifiers environments, evaluations, GEPA, and RL training. Use when the user wants ideation, literature scanning, concept teaching, roadmap planning, or research program design grounded in local CLI sources, verifiers, and RL trainer code.
---

# Brainstorm

## Goal
Run structured, interactive ideation that turns ambiguous research goals into concrete environment and evaluation plans.

## Interaction Style
1. Drive an iterative conversation, not a one-shot dump.
2. Ask focused clarifying questions before proposing large plans.
3. Keep suggestions toolchain-native: CLI, verifiers, and RL trainer workflows.

## Discovery Workflow
1. Clarify objective, model family, budget, and timeline.
2. Map objective to workflow levers:
- environment creation or migration
- benchmark/eval design
- GEPA prompt optimization
- RL training
3. Build a short option set, then deepen only selected options.
4. Nudge model-family intent explicitly:
- Instruct-first exploration defaults: `gpt-4.1` series, `qwen3` instruct series.
- Reasoning-first exploration defaults: `gpt-5` series, `qwen3` thinking series, `glm` series.
- Recommend endpoint aliases in `configs/endpoints.toml` for repeatable experiments.

## Required Grounding Sources
1. Read local source before proposing workflows:
- `~/dev/prime-cli`
- `~/dev/prime-rl` (clone to `/tmp` only if needed)
- current verifiers workspace docs/configs
2. For literature and external eval ideas, browse web sources and prioritize mid-2025 onward unless the user asks otherwise.
3. Include dates when discussing recent papers or benchmarks.

## Concept Teaching Mode
When asked to explain RL or environment concepts:
1. Anchor explanations in prime-rl and verifiers terminology.
2. Use concrete config and rollout examples.
3. Distinguish binary-reward and continuous-reward training implications.

## Planning Output Format
Produce:
1. Problem framing and assumptions.
2. Candidate environment or eval ideas, ranked by expected value and implementation effort.
3. Experiment plan with milestones, metrics, and go/no-go gates.
4. Risks, dependencies, and required decisions from the user.
5. Distribution plan for mature environments: recommend Hub push after smoke-test stability and ask whether visibility should be `PUBLIC` or `PRIVATE`.

## Quality Guardrails
1. Do not make hidden assumptions about benchmark prompt formatting or scoring contracts.
2. Flag platform limitations clearly and pause for user direction when blocked.
3. Prefer official first-party capabilities before suggesting custom third-party tooling.
