from __future__ import annotations

import json
from typing import Any


def render_env_spec(spec: dict[str, Any]) -> str:
    """Render an EnvSpec dict into a human-readable description for LLM prompts."""
    lines: list[str] = []
    lines.append(
        f"Environment type: {spec['env_type']} (max_turns={spec['max_turns']})"
    )

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

    ds_schema = spec.get("dataset_schema") or {}
    if ds_schema:
        lines.append("Dataset schema from the environment (exact full feature spec):")
        lines.append(json.dumps(ds_schema, indent=2, default=str))

    return "\n".join(lines)


PLAN_PROMPT = """\
You are a synthetic data planner. You receive an environment specification and \
at most a small seed sample. Infer what task distribution this environment is \
meant to capture, then propose subtopics for diverse synthetic data.

## Environment
{env_spec}

## Seed sample (bounded, complete rows/documents, JSON or text)
{examples_json}

## Instructions
1. Read every field in each seed example. Infer the intent of the environment \
and what a good synthetic row should look like.
2. Propose distinct subtopics that naturally partition the task space. Each \
subtopic should support {samples_per_subtopic} unique new rows.{max_subtopics_line}

3. The dataset schema shown above, when present, is authoritative. Do not invent \
or rename columns.
4. **generation_guidance**: Brief instructions for a downstream generator so it \
produces rows that match real examples in structure and style.

Return a single JSON object with exactly these keys:
- "subtopics": string array
- "generation_guidance": string

Return ONLY the JSON object, no other text.
"""


GENERATE_PROMPT = """\
You are a synthetic data generator. Produce **one** new dataset row that matches \
the schema and style of the seed examples.

## Environment
{env_spec}

## Reference material (seed rows and/or documents)
{source_section}

## Subtopic
{subtopic}

## Generation guidance
{generation_guidance}

## Required output schema
The environment schema is authoritative. Output a single JSON object that matches \
this schema exactly, including keys and value structure:
{schema_json}

## Instructions
1. Output a single JSON object that matches the schema exactly.
2. Match the shape and nesting of the seed examples for every field.
3. The row must be plausible for the subtopic and consistent with the environment \
and reward criteria.
4. Do not include helper keys like "subtopic" unless they are real dataset columns.

Return ONLY the JSON object, no other text.
"""


FILTER_JUDGE_PROMPT = """\
You are a judge evaluating whether a response correctly answers a task.

## Task
{task_text}

## Golden reference
{golden_answer}

## Model response
{response}

## Instructions
Determine whether the model's response is correct by comparing it to the \
golden reference. The response does not need to match exactly, but must convey \
the same core information.

Return a JSON object:
{{"correct": true/false, "score": 0.0-1.0}}

Return ONLY the JSON object, no other text.
"""

_NO_SOURCE_DIRECTIVE = (
    "No external source material provided. Use the environment "
    "specification above to infer what tasks should look like."
)
