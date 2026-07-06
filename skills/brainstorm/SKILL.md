---
name: brainstorm
description: Run interactive brainstorming across verifiers environments, evaluations, GEPA, and RL training. Use when the user wants ideation, literature scanning, concept teaching, roadmap planning, or research program design grounded in local CLI sources, verifiers, and RL trainer code.
---

# Brainstorm

## Goal

Run an interactive idea session with the user to turn their idea into a concrete environment outline.

## Interaction style

- Find out what the goal of the user is, what their budget is, and ultimately what they want to achieve.
- Ask a lot of clarifying questions and outline the concrete steps before you start implementing.
- When something is ambiguous, do research first and present the user with different options to choose from.
- Don't overload the user with terminology, but guide them in the right direction. Assume a knowledgeable user, but not an in-depth expert at every topic.

## Discovery workflow

1. Find out what the user wants:
  - Is it running existing evaluations or training with common environments? If so, use the environments hub to see whether those envs already exist.
  - Is it optimizing an existing workflow? In this case, prompt optimization with GEPA is the right tool.
  - Else: For training and evaluations, building an environment together is the right call.
2. After you have found this out, look at the other skills and choose the appropriate one to proceed on the technical level, while still keeping the interactive session with the user alive.
