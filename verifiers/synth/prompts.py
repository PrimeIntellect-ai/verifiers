from __future__ import annotations

import json
from typing import Any


def render_env_spec(spec: dict[str, Any]) -> str:
    """Render an EnvSpec dict into a human-readable description for LLM prompts."""
    lines: list[str] = []
    lines.append(f"Environment type: {spec['env_type']} (max_turns={spec['max_turns']})")

    if spec.get("system_prompt"):
        lines.append(f'System prompt: "{spec["system_prompt"]}"')

    tools = spec.get("tools")
    if tools:
        lines.append("Available tools:")
        for t in tools:
            params = t.get("parameters", {})
            props = params.get("properties", {})
            param_strs = [
                f"{name}: {p.get('type', 'any')}" for name, p in props.items()
            ]
            sig = ", ".join(param_strs)
            lines.append(f"  - {t['name']}({sig}): {t.get('description', '')}")

    reward_fns = spec.get("reward_functions", [])
    if reward_fns:
        lines.append("Scoring criteria:")
        for rf in reward_fns:
            doc = rf.get("doc") or "no description"
            lines.append(f"  - {rf['name']} (weight={rf.get('weight', 1.0)}): {doc}")

    if spec.get("parser_info"):
        lines.append(f"Expected output format: {spec['parser_info']}")

    example_rows = spec.get("example_rows", [])
    if example_rows:
        lines.append("Example dataset rows:")
        for row in example_rows[:2]:
            display = {k: v for k, v in row.items() if k != "prompt"}
            if "prompt" in row:
                msgs = row["prompt"]
                if isinstance(msgs, list) and msgs:
                    last = msgs[-1]
                    content = last.get("content", "") if isinstance(last, dict) else str(last)
                    display["prompt_preview"] = content[:200]
            lines.append(f"  {json.dumps(display, default=str)}")

    return "\n".join(lines)


PLAN_PROMPT = """\
You are a synthetic data planner. Given source material and an environment \
specification, identify distinct subtopics that can be used to generate diverse \
training tasks.

## Environment
{env_spec}

## Source Material (seed)
{seed_content}

## Instructions
Identify exactly {num_subtopics} distinct subtopics from the source material \
that are relevant to the environment's task domain. Each subtopic should be \
specific enough to generate focused tasks, but broad enough to support \
{samples_per_subtopic} unique samples.

Return a JSON array of subtopic strings. Example:
["subtopic one", "subtopic two", "subtopic three"]

Return ONLY the JSON array, no other text.
"""


BACKTRANSLATE_PROMPT = """\
You are a synthetic data generator. Given source material, an environment \
specification, and a subtopic, generate a training task with a golden answer.

## Environment
{env_spec}

## Source Material
{seed_content}

## Subtopic
{subtopic}

## Instructions
Using the source material above, create a task that:
1. Matches the environment's dataset format (see example rows above)
2. Has a clear, verifiable answer derivable from the source material
3. Focuses on the specified subtopic
4. Would require access to the source material to answer correctly
5. Is self-contained — the question makes sense without seeing the source

Return a single JSON object with these fields:
- "question": the task question (string)
- "answer": the golden answer (string)
- "info": additional metadata as a JSON object (include at minimum \
{{"seed_id": "{seed_id}", "subtopic": "{subtopic}"}})

Return ONLY the JSON object, no other text.
"""


FILTER_JUDGE_PROMPT = """\
You are a judge evaluating whether a response correctly answers a question.

## Question
{question}

## Golden Answer
{golden_answer}

## Model Response
{response}

## Instructions
Determine whether the model's response is correct by comparing it to the \
golden answer. The response does not need to match exactly, but must convey \
the same core information.

Return a JSON object:
{{"correct": true/false, "score": 0.0-1.0}}

Return ONLY the JSON object, no other text.
"""
