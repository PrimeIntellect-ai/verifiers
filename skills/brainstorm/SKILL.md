---
name: brainstorm
description: Help plan tasksets, evaluations, prompt optimization, and reinforcement learning experiments. Use to explore ideas, explain options, review research, or turn a goal into an experiment.
---

# Brainstorm

Help the user turn an idea into a concrete next step.

1. Establish the goal, budget, and what a useful result would look like. Use
   constraints already given; ask only about missing choices that matter.
2. Check existing tasksets on the Environment Hub and read the relevant code or
   research before proposing something new.
3. Explain the options and their tradeoffs in plain language. Assume the user is
   knowledgeable, but may not know every API or research term.
4. Agree on what to measure and outline the work needed to measure it.

For an existing taskset, use [evaluate-environments](../evaluate-environments/SKILL.md)
to plan and run the comparison. For a new taskset or interaction, use
[create-environments](../create-environments/SKILL.md). For an unexpected result,
use [debug-environments](../debug-environments/SKILL.md) before redesigning it.
Use [audit-envs](../audit-envs/SKILL.md) to check whether the tasks and scores
measure the intended ability.

If the goal is to improve a system prompt against a measurable reward, consider
[gepa](../gepa/SKILL.md). Keep final-test tasks separate from both the tasks used
to improve the prompt and those used to select it. Problems with tools, task
design, or the harness may need changes to those parts instead.
