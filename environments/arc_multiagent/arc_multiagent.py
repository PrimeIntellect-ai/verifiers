"""
ARC-AGI Multi-Agent Environment.

Recreates the ARC-AGI solver pipeline using the verifiers multi-agent framework:
- SingleSolverEnv: One actor solves the task (basic/deep/hint variants)
- ObjectsPipelineEnv: Three actors in sequence (extract -> transform -> solve)
- JudgeEnv: Evaluates candidate solutions

All orchestrated via Protocol.spawn() for parallel execution.

Usage:
    prime env install arc-multiagent
    prime eval run arc-multiagent -m qwen3-30b-i -n 5 -r 1 --debug
"""

from __future__ import annotations

import json
import logging
import random
import re
from typing import Any, List, Optional

from datasets import Dataset

import verifiers as vf
from verifiers.envs.actor import Actor
from verifiers.envs.multiagent_env import MultiAgentEnv
from verifiers.envs.protocol import Protocol
from verifiers.types import Messages, State
from verifiers.rubrics.multiagent_rubric import MultiAgentRubric
from verifiers.utils.client_utils import get_actor_client

logger = logging.getLogger(__name__)


# =============================================================================
# Grid Utilities
# =============================================================================

Grid = List[List[int]]


def format_grid(grid: Grid) -> str:
    """Format grid as CSV (comma-separated rows). Used in solver prompts."""
    if grid is None:
        return ""
    return "\n".join(",".join(str(c) for c in row) for row in grid)


def grid_to_string(grid: Optional[Grid]) -> str:
    """Format grid with size header and dense rows. Used in Logic Judge prompts."""
    if not grid:
        return "(Empty Grid)"
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    lines = [f"Size: {rows}x{cols}"]
    for row in grid:
        lines.append("".join(str(c) for c in row))
    return "\n".join(lines)


def grid_to_csv_rows(grid: Optional[Grid], padding: str = "      ") -> str:
    """Format grid as padded CSV rows. Used in Consistency Judge prompts."""
    if not grid:
        return ""
    lines = []
    for row in grid:
        lines.append(padding + ",".join(map(str, row)))
    return "\n".join(lines)


def parse_grid_from_text(text: str) -> Grid:
    """
    Parse a CSV grid from model output, handling noise and labels.

    Strategy (from ARC-AGI solver):
    1. Identify candidate rows (lines with comma-separated numbers)
    2. Group consecutive rows into blocks
    3. Markdown code fences are HARD separators
    4. Small gaps (blank lines/text) allowed within blocks (MAX_GAP=2)
    5. Return the LAST valid block (assumed to be the final answer)
    """
    text = text.strip()
    lines = text.splitlines()

    candidate_rows = []
    hard_separators = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Hard separator: markdown code fences
        if stripped.startswith("```"):
            candidate_rows.append(None)
            hard_separators.append(i)
            continue

        if not stripped:
            candidate_rows.append(None)
            continue

        # Skip standalone row labels
        if re.match(r'^Row\s+\d+:?$', stripped, re.IGNORECASE):
            candidate_rows.append(None)
            continue

        # Skip markdown list items
        if stripped.startswith(("-", "*", "+")):
            candidate_rows.append(None)
            continue

        row = None
        try:
            # Clean formatting artifacts
            clean_line = stripped.replace("`", " ").replace("[", " ").replace("]", " ").strip()

            # Handle numbered lists: "1. 8,8,8" or "1) 8,8,8"
            numbered_match = re.match(r'^\d+[\.\)]\s+', clean_line)
            if numbered_match:
                clean_line = clean_line[numbered_match.end():]

            tokens = clean_line.split(",")
            if len(tokens) > 0 and all(t.strip().isdigit() for t in tokens):
                row = [int(t.strip()) for t in tokens]
            else:
                # Fallback: try splitting by colon
                if ":" in clean_line:
                    clean_line = clean_line.split(":")[-1].strip()

                match = re.search(r'\d', clean_line)
                if match:
                    last_digit_idx = -1
                    for idx, char in enumerate(clean_line):
                        if char.isdigit():
                            last_digit_idx = idx

                    if last_digit_idx != -1 and last_digit_idx >= match.start():
                        candidate_sub = clean_line[match.start():last_digit_idx + 1]
                        sub_tokens = candidate_sub.split(",")
                        if len(sub_tokens) > 1 and all(t.strip().isdigit() for t in sub_tokens):
                            remainder = clean_line[last_digit_idx + 1:].strip()
                            if not any(c.isalpha() for c in remainder):
                                row = [int(t.strip()) for t in sub_tokens]

        except ValueError:
            pass

        candidate_rows.append(row)

    # Block reconstruction
    MAX_GAP = 2
    blocks = []
    current_block = []
    last_row_index = -1

    for i, row in enumerate(candidate_rows):
        if row is not None:
            if not current_block:
                current_block = [row]
                last_row_index = i
            else:
                has_hard_sep = any(last_row_index < sep_idx < i for sep_idx in hard_separators)
                gap_size = i - last_row_index - 1
                width_diff = abs(len(row) - len(current_block[0]))
                width_match = width_diff <= 5

                if has_hard_sep or gap_size > MAX_GAP or not width_match:
                    blocks.append(current_block)
                    current_block = [row]
                    last_row_index = i
                else:
                    current_block.append(row)
                    last_row_index = i

    if current_block:
        blocks.append(current_block)

    if not blocks:
        raise ValueError("Could not parse grid")

    return blocks[-1]


def grids_equal(a: Grid | None, b: Grid | None) -> bool:
    """Check if two grids are identical."""
    if a is None or b is None:
        return False
    if len(a) != len(b):
        return False
    return all(row_a == row_b for row_a, row_b in zip(a, b))


def grid_similarity(predicted: Grid | None, expected: Grid | None) -> float:
    """Compute cell-level similarity (0.0 to 1.0)."""
    if predicted is None or expected is None:
        return 0.0
    max_rows = max(len(predicted), len(expected))
    max_cols = max(
        max((len(r) for r in predicted), default=0),
        max((len(r) for r in expected), default=0),
    )
    if max_rows == 0 or max_cols == 0:
        return 0.0

    matches = 0
    total = max_rows * max_cols
    for i in range(max_rows):
        for j in range(max_cols):
            pred_val = predicted[i][j] if i < len(predicted) and j < len(predicted[i]) else -1
            exp_val = expected[i][j] if i < len(expected) and j < len(expected[i]) else -1
            if pred_val == exp_val:
                matches += 1
    return matches / total


# =============================================================================
# Prompt Templates
# =============================================================================

def build_prompt(
    train_pairs: list[dict],
    test_input: Grid,
    strategy: str = None,
    image_path: str = None,
    trigger_deep_thinking: bool = False,
    objects_insertion: str = None,
) -> str:
    """
    Build the main solving prompt.

    Args:
        train_pairs: List of {"input": Grid, "output": Grid}
        test_input: The test input grid
        strategy: Optional hint text to inject
        image_path: Optional image path reference
        trigger_deep_thinking: Enable deep reasoning prompt
        objects_insertion: Optional objects + transformation text
    """
    lines = [
        "You are solving an ARC (Abstraction and Reasoning Corpus) task.",
        "Each grid cell is an integer 0-9 representing a color.",
        "Use the solved examples to infer the transformation and apply it to the test input.",
        "",
        "Solved examples:",
    ]

    for idx, pair in enumerate(train_pairs, start=1):
        lines.append(f"Example {idx}:")
        lines.append("input:")
        lines.append(format_grid(pair["input"]))
        lines.append("output:")
        lines.append(format_grid(pair["output"]))
        lines.append("")

    lines.append("Test input:")
    lines.append(format_grid(test_input))
    lines.append("")

    if strategy:
        lines.append("Below are a few hints that you might find helpful:")
        lines.append(strategy)
        lines.append("")

    if image_path:
        lines.append("Attached you'll find an image that shows the input/output example pairs. Use this image to find objects, patterns and transformations")
        lines.append("")

    if trigger_deep_thinking:
        lines.append("PROTOCOL OVERRIDE: ENGAGE ARC NEURO-SYMBOLIC LOGIC ENGINE")
        lines.append("")
        lines.append("Silently enter maximal test-time reasoning mode. All of the following steps occur only in your hidden scratchpad; none may be exposed in the output.")
        lines.append("")
        lines.append("Perform hierarchical object decomposition of each grid into foreground objects and background fields; track shapes, colors, connectivity, and object persistence. Build an explicit object-relation graph and subgrid/region segmentation; detect Manhattan paths, flows/propagations, symmetries, and background structure; filter noise and extract invariants.")
        lines.append("")
        lines.append("Enumerate multiple candidate transformation rules/programs (at least three distinct hypotheses). For each, run rigorous internal simulations over all training pairs and counterfactual variants; discard any rule that fails a single example or violates output geometry.")
        lines.append("")
        lines.append("Triangulate using three paradigms in parallel: geometric (positions, topology, symmetries, paths), symbolic (predicates, programs, rewrite rules, counting), and counterexample-based search (actively seek minimal failure cases to refine or reject rules).")
        lines.append("")
        lines.append("Explicitly check for adversarial traps, spurious shortcuts, and degenerate memorization. Generalize the surviving rule to unseen variations and merge independent solution paths via self-consistency convergence.")
        lines.append("")
        lines.append("Apply the final rule to the test input using stepwise internal simulation only.")
        lines.append("")
        lines.append("OUTPUT CONSTRAINT (STRICT): Reveal ONLY the final answer grid. Never reveal chain-of-thought, intermediate states, or search traces.")
        lines.append("")

    if objects_insertion:
        lines.append("To solve this problem, please consider using the description of the input/output data below:")
        lines.append("")
        lines.append(objects_insertion)
        lines.append("")
        lines.append("Respond with an explanation of your thinking that is detailed enough that someone can reconstruct your solution. Afterwards, you MUST also respond with the completed output grid.")
    else:
        lines.append("Respond with an explanation of your thinking that is detailed enough that someone can reconstruct your solution. Afterwards, you MUST also respond with the completed output grid.")

    return "\n".join(lines)


def build_objects_extraction_prompt(
    train_pairs: list[dict],
    test_input: Grid,
) -> str:
    """Build prompt for Phase A: Object extraction."""
    lines = [
        "Describe the types of objects involved in the grids below. Do not infer rules, transformations, or relationships between grids. Focus solely on the types of objects that are involved in the various grids and their attributes, and describe them generally - not for each grid.",
        "When describing colors, use ONLY the numeric values (0-9), e.g., 'color 0', 'color 5'. Do not use color names like 'black' or 'gray'.",
        "",
        "IMPORTANT: You MUST end your response with a concise summary inside <objects_summary>...</objects_summary> tags. This summary will be used by the next step. Keep the summary under 500 words.",
        "",
    ]

    # Collect all grids and shuffle (matches ARC solver behavior)
    all_grids = []
    for pair in train_pairs:
        all_grids.append(pair["input"])
        if pair.get("output"):
            all_grids.append(pair["output"])
    all_grids.append(test_input)

    random.shuffle(all_grids)

    for grid in all_grids:
        lines.append(format_grid(grid))
        lines.append("")

    return "\n".join(lines)


def build_objects_transformation_prompt(
    train_pairs: list[dict],
    test_input: Grid,
    objects_text: str,
) -> str:
    """Build prompt for Phase B: Transformation detection."""
    lines = [
        "Below is a set of input / output grids and a description of the objects in each of these grids. Your task is to describe all potential transformations that are involved in changing the objects in the input to the objects in the output. Please ensure that you list ALL possible transformations that you have identified.",
        "Use strictly numeric values (0-9) for colors in your description and summary.",
        "",
        "IMPORTANT: You MUST end your response with a concise summary inside <transformation_summary>...</transformation_summary> tags. This summary will be used by the next step. Keep the summary under 500 words.",
        "",
    ]

    for idx, pair in enumerate(train_pairs, start=1):
        lines.append(f"Example {idx}:")
        lines.append("input:")
        lines.append(format_grid(pair["input"]))
        lines.append("output:")
        lines.append(format_grid(pair["output"]))
        lines.append("")

    lines.append("Test input:")
    lines.append(format_grid(test_input))
    lines.append("")
    lines.append("## Objects Description")
    lines.append(objects_text)

    return "\n".join(lines)


HINT_PROMPT = """You are analyzing a single ARC-AGI training example presented as an image.
The image shows one or more **input -> output** grid pairs.

Your task is **NOT** to solve the test task.
Your task is to extract **generalizable insights** about what the transformation is *doing*, even if the example is incomplete or ambiguous.

Follow all steps precisely.

---

## **1. Describe the observable structures**

Describe what you see **without guessing** the rule:

* What objects exist? (shapes, colors, connected components)
* How are they arranged? (locations, orientations, bounding boxes)
* What changes from input to output? (additions, removals, movements)
* What stays the same? (stable background, preserved shapes)

Be very literal and avoid hypothesizing here.

---

## **2. Identify transformation categories**

For each, say "present", "maybe", or "absent":

* **Object movement**
* **Object recoloring**
* **Bordering / outlining**
* **Filling holes or cavities**
* **Removing noise / extracting main shape**
* **Copying / pasting / repositioning**
* **Using a key anchor location (corners, edges, center)**
* **Using highest-frequency or rarest colors**
* **Size-based filtering or selection**
* **Symmetry or reflection operations**
* **Growth, dilation, shrinking, erosion**
* **Cluster -> icon or icon -> cluster mapping**

This helps narrow down plausible rule families.

---

## **3. Extract the key non-obvious insights**

List 3-6 insights that are **crucial and not immediately obvious**.
Examples of the kind of insights to extract:

* "Only the *largest* blue cluster receives a border; smaller ones are ignored."
* "New colors in the output correspond to *inside* regions of hollow shapes."
* "The transformation only acts on objects touching the top edge."
* "The rule depends on which color appears in the **top-left tile** of the input."
* "Objects are replaced with templated icons depending on their color."
* "Internal holes are treated as objects and recolored independently."

Focus on the subtle but general behavior.

---

## **4. Hypothesize a general rule (high level)**

In 2-4 sentences, propose a **general transformation mechanism** that would explain the mapping *across inputs and outputs*.

Avoid low-level pixel descriptions; describe conceptual behavior.

Example styles:

* "The task identifies hollow shapes and outlines them in red while recoloring their interior according to hole depth."
* "The task compresses all meaningful objects to the top-left quadrant while preserving their original shapes."
* "All scattered singletons are deleted and replaced by large structured blocks based on color frequency."

If ambiguous, propose the *two or three* most likely rule families.

---

## **5. List checks you would perform when solving**

Give 3-5 quick algorithmic checks for validation, e.g.:

* "Verify whether recoloring depends on hole depth or on adjacency to perimeter."
* "Check if the anchor color is always the top-left non-background pixel."
* "Test whether clusters are preserved or converted into icons."
* "Determine whether the largest cluster always receives the transformation."

These guide a solver toward the correct general rule.

---

## **6. Summarize final insights**

In 3-4 bullet points, give the distilled essence -- short, hard, and generalizable.
**IMPORTANT: Start this section with the exact line `HINT_START` and end it with the exact line `HINT_END`.**

Example:
HINT_START
* The task outlines certain shapes using a new color.
* Only shapes with internal cavities are processed.
* Inner cavities receive a secondary recolor distinct from the border.
HINT_END
"""


def build_hint_extraction_prompt(
    train_pairs: list[dict],
    test_input: Grid,
) -> str:
    """
    Build hint extraction prompt WITH the actual grid data.

    Combines the task grids (so the model can see the patterns)
    with the HINT_PROMPT instructions (so it knows what to extract).
    """
    lines = [
        "You are analyzing an ARC (Abstraction and Reasoning Corpus) task.",
        "Each grid cell is an integer 0-9 representing a color.",
        "",
        "Solved examples:",
    ]

    for idx, pair in enumerate(train_pairs, start=1):
        lines.append(f"Example {idx}:")
        lines.append("input:")
        lines.append(format_grid(pair["input"]))
        lines.append("output:")
        lines.append(format_grid(pair["output"]))
        lines.append("")

    lines.append("Test input:")
    lines.append(format_grid(test_input))
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(HINT_PROMPT)

    return "\n".join(lines)


# Judge prompt constants

LOGIC_JUDGE_SYSTEM = """<SYSTEM_ROLE>
You are the **ARC LOGIC AUDITOR**.
You are NOT a creative solver. You are a skeptical Critic and Verifier.
Your task is to review a list of pre-grouped "Candidate Clusters" (proposed solutions for an ARC puzzle) and rank them based on logical validity.

Your Core Principle is **FALSIFICATION**:
1. Trust NO ONE.
2. Assume every candidate is "hallucinating" until they prove otherwise.
3. The "Ground Truth" (Solved Examples) is the absolute law.
</SYSTEM_ROLE>"""

LOGIC_JUDGE_INSTRUCTIONS = """<AUDIT_PROTOCOL>
Execute this pipeline sequentially for every Candidate. You must output your thinking process inside <AUDIT_LOG> tags.

### PHASE 1: LOGIC SELECTION & CRYSTALLIZATION
- **Selection:** If a Candidate contains multiple <REASONING> blocks, read them all and select the **single most detailed and logical** explanation to audit.
- **Crystallization:** Convert that text into a strict "IF-THEN" algorithm.
  - *Bad:* "The pattern involves moving blue pixels." (Too vague to audit)
  - *Good:* "IF a pixel is Blue, THEN move it 1 step Right. Else, preserve color."
- *Constraint:* If the reasoning is incoherent or vague, mark the Candidate as "INVALID - VAGUE".

### PHASE 2: THE GROUND TRUTH AUDIT (CRITICAL)
- You must "Back-Test" the Crystallized Rule against the {{SOLVED_EXAMPLES}}.
- For **EACH** Solved Example pair (Input -> Output), you must strictly perform this 3-step check:
  1. **Hypothesis:** "If I apply the Candidate's Rule to this Input, exactly what *should* happen?"
  2. **Observation:** "Look at the actual Official Output. What *actually* happened?"
  3. **Verdict:** "Do they match?"
- **Fatal Contradictions to Watch For:**
  * **Scope Error:** Rule applies to specific colors (e.g., "Blue"), but the example changes "Red" pixels.
  * **Geometry Error:** Rule says "rotate 90," but example shows a flip.
  * **Object Error:** Rule treats pixels individually, but example shows objects moving as blocks.
- *Constraint:* Record exactly how many Solved Examples the candidate PASSED vs FAILED. **Do not stop at the first failure**; check all examples to determine the severity of the failure (e.g., "Passed 2/3 Examples").

### PHASE 3: EXECUTION CONSISTENCY
- For Candidates that survived Phase 2 (or passed at least one example):
- Look at the {{TEST_INPUT}} and the Candidate's <PROPOSED_SOLUTION> grid.
- Does the proposed output actually follow the Crystallized Rule?
- *Common Hallucination:* The text says "Move Blue," but the grid shows Blue staying still. Mark this as **INTERNAL_CONTRADICTION**.

### PHASE 4: STACK RANKING & TIE-BREAKING
- Rank ALL candidates from Best to Worst based on this hierarchy:
  1. **GOLD (Tier 1):** Passed ALL Solved Examples + Consistent Execution on Test Input.
  2. **SILVER (Tier 2):** Passed ALL Solved Examples + Minor Execution Error on Test Input.
  3. **BRONZE (Tier 3):** Passed MOST Solved Examples (Partial Logic).
  4. **INVALID (Tier 4):** Failed ALL/MOST Solved Examples, Vague Reasoning, or Severe Internal Contradictions.

- **Tie-Breaking:** If two candidates are in the same Tier, rank the one with the **Simplest Rule** (Occam's Razor) higher.
</AUDIT_PROTOCOL>

<OUTPUT_FORMAT>
Return a single JSON object with the following structure:

{
  "candidates": [
    {
      "candidate_id": 0,
      "score": 8.7,
      "tier": "GOLD",
      "example_audit": {
        "per_example": {
          "1": "Pass",
          "2": "Pass",
          "3": "Partial"
        },
        "summary": "Rule matches main behaviors across examples; minor ambiguity in example 3."
      },
      "test_grid_consistency": "Plausible",
      "rule_summary": "Short, 1-3 sentence description of this candidate's representative rule."
    }
  ]
}
</OUTPUT_FORMAT>"""


CONSISTENCY_JUDGE_SYSTEM = """<SYSTEM_ROLE>
You are an ARC Solution Auditor.

Your primary ability is NOT to solve new ARC tasks from scratch.
Instead, you are excellent at:
- Checking whether a proposed rule is logically consistent
- Verifying that a rule matches known solved examples
- Verifying that a candidate's test output actually follows its own stated rule

You are skeptical and detail-oriented. If a candidate's explanation says X
but the examples show not-X, you must call that out.
</SYSTEM_ROLE>"""

CONSISTENCY_JUDGE_INSTRUCTIONS = """<INSTRUCTIONS>
You must behave as an AUDITOR, not a solver.

Your overall goal:
- For each candidate, select the **single most detailed and logical** explanation and treat it as that
  candidate's proposed rule.
- Audit that rule against all training examples.
- Check whether the candidate's predicted test OUTPUT_GRID actually follows
  that rule.
- Assign each candidate a score from 0 to 10 and rank all candidates.

Follow these steps:

STEP 1 -- SELECT THE BEST RULE PER CANDIDATE
STEP 2 -- EXAMPLE CONSISTENCY AUDIT (DO NOT USE THE TEST INPUT HERE)
STEP 3 -- RULE-TO-TEST-GRID CONSISTENCY
STEP 4 -- SCORING AND GLOBAL RANKING

For each candidate, assign a numeric SCORE from 0 to 10:
  - 10: Rule is simple and coherent, strongly consistent with all training examples, and the test grid fits the rule.
  - 7-9: Mostly consistent with examples; minor ambiguities or small issues.
  - 4-6: Some consistency, but noticeable contradictions or hand-wavy parts.
  - 1-3: Major contradictions with examples or test grid; rule not credible.
  - 0: Completely incompatible with examples; or explanation is nonsense.

Then rank all candidates in descending order of SCORE.
</INSTRUCTIONS>

<OUTPUT_FORMAT>
Return a single JSON object with the following structure:

{
  "candidates": [
    {
      "candidate_id": 0,
      "score": 8.7,
      "tier": "GOLD",
      "example_audit": {
        "per_example": {
          "1": "Pass",
          "2": "Pass",
          "3": "Partial"
        },
        "summary": "Rule matches main behaviors across examples; minor ambiguity in example 3."
      },
      "test_grid_consistency": "Plausible",
      "rule_summary": "Short, 1-3 sentence description of this candidate's representative rule."
    }
  ],
  "final_ranking_by_candidate": [0, 4, 5, 1]
}
</OUTPUT_FORMAT>"""


def build_logic_judge_prompt(
    train_pairs: list[dict],
    test_input: Grid,
    candidates: list[dict],
) -> str:
    """
    Build the Logic Judge prompt with candidates and their reasoning.

    candidates: list of {
        "id": int,
        "grid": Grid,
        "reasoning": {model_id: response_text, ...},
        "models": [model_ids],
    }
    """
    parts = [LOGIC_JUDGE_SYSTEM, "\n<INPUT_DATA>"]

    parts.append("1. {SOLVED_EXAMPLES}:")
    for i, pair in enumerate(train_pairs, 1):
        parts.append(f"<EXAMPLE_{i}>")
        parts.append("<INPUT>")
        parts.append(grid_to_string(pair["input"]))
        parts.append("</INPUT>")
        parts.append("<OUTPUT>")
        parts.append(grid_to_string(pair["output"]))
        parts.append("</OUTPUT>")
        parts.append(f"</EXAMPLE_{i}>")

    parts.append("\n2. {TEST_INPUT}:")
    parts.append(grid_to_string(test_input))

    parts.append("\n3. {CANDIDATES}:")
    for cand in candidates:
        c_id = cand["id"]
        parts.append(f"<CANDIDATE {c_id}>")
        parts.append("<PROPOSED_SOLUTION>")
        parts.append(grid_to_string(cand["grid"]))
        parts.append("</PROPOSED_SOLUTION>")
        for j, model_id in enumerate(cand.get("models", [])):
            alias = chr(65 + j)
            reasoning = cand.get("reasoning", {}).get(model_id, "(Reasoning not found)")
            parts.append(f'<REASONING_MODEL_{alias} model_id="{model_id}">')
            parts.append(reasoning)
            parts.append(f"</REASONING_MODEL_{alias}>")
        parts.append(f"</CANDIDATE {c_id}>")

    parts.append("</INPUT_DATA>\n")
    parts.append(LOGIC_JUDGE_INSTRUCTIONS)
    return "\n".join(parts)


def build_consistency_judge_prompt(
    train_pairs: list[dict],
    test_input: Grid,
    candidates: list[dict],
) -> str:
    """Build the Consistency Judge prompt."""
    parts = [CONSISTENCY_JUDGE_SYSTEM]

    parts.append("\n<PROBLEM>")
    for i, pair in enumerate(train_pairs, 1):
        parts.append(f'  <TRAIN_EXAMPLE index="{i}">')
        parts.append("    <INPUT_GRID>")
        parts.append(grid_to_csv_rows(pair["input"]))
        parts.append("    </INPUT_GRID>")
        parts.append("    <OUTPUT_GRID>")
        parts.append(grid_to_csv_rows(pair["output"]))
        parts.append("    </OUTPUT_GRID>")
        parts.append("  </TRAIN_EXAMPLE>")

    parts.append("  <TEST_INPUT>")
    parts.append("    <INPUT_GRID>")
    parts.append(grid_to_csv_rows(test_input))
    parts.append("    </INPUT_GRID>")
    parts.append("  </TEST_INPUT>")
    parts.append("</PROBLEM>\n")

    parts.append("<CANDIDATES>")
    for cand in candidates:
        c_id = cand["id"]
        parts.append(f'  <CANDIDATE id="{c_id}">')
        for j, model_id in enumerate(cand.get("models", [])):
            alias = chr(65 + j)
            reasoning = cand.get("reasoning", {}).get(model_id, "(Reasoning not found)")
            parts.append(f'    <ANSWER id="{alias}" model_id="{model_id}">')
            parts.append("      <EXPLANATION>")
            parts.append(reasoning)
            parts.append("      </EXPLANATION>")
            parts.append("      <OUTPUT_GRID>")
            parts.append(grid_to_csv_rows(cand["grid"]))
            parts.append("      </OUTPUT_GRID>")
            parts.append(f"    </ANSWER>")
        parts.append("  </CANDIDATE>")
    parts.append("</CANDIDATES>\n")

    parts.append(CONSISTENCY_JUDGE_INSTRUCTIONS)
    return "\n".join(parts)


# =============================================================================
# Model Configuration
# =============================================================================
# Configure models for each role. Endpoint names reference configs/endpoints.py.
# None/missing endpoint = use the default model from the -m flag.
#
# Available endpoints: See configs/endpoints.py
#   e.g., "sonnet", "opus", "gemini-2.5-flash", "gpt-4.1", "qwen3-235b-i"
# =============================================================================

SOLVER_CONFIGS = [
    # Each entry adds a solver model. The first is the "orchestrator" (does
    # the initial shallow solve in turn 1). Additional entries add parallel
    # solvers with different models -- more models = more candidate diversity.
    #
    # Uncomment entries to enable multi-model solving:
    # {"endpoint": "sonnet", "is_trainable": False},
    # {"endpoint": "gemini-2.5-flash", "is_trainable": False},
    # {"endpoint": "gpt-4.1", "is_trainable": False},
    #
    # Empty list = single-model mode (uses the -m flag model for everything)
]

JUDGE_ENDPOINT = None           # e.g., "gemini-2.5-flash"
OBJECTS_ENDPOINT = None         # e.g., "sonnet"
HINT_EXTRACTOR_ENDPOINT = None  # e.g., "sonnet"


# =============================================================================
# Dataset Helpers
# =============================================================================

# Default HuggingFace dataset repo — set after uploading with upload_dataset.py
HF_DATASET_REPO = "bhoy/arc-agi-2"


def load_arc_dataset_hf(
    hf_repo: str = HF_DATASET_REPO,
    split: str = "train",
) -> Dataset:
    """
    Load ARC tasks from HuggingFace Hub.

    Returns dataset with columns: task_id, train_pairs, test_input, test_output
    (all JSON strings).
    """
    from datasets import load_dataset

    ds = load_dataset(hf_repo, split=split)
    logger.info(f"Loaded {len(ds)} tasks from HF: {hf_repo} (split={split})")
    return ds


def prepare_dataset_for_eval(raw_dataset: Dataset) -> Dataset:
    """
    Transform raw ARC dataset into the format vf-eval expects.

    Takes the HF dataset (task_id, train_pairs, test_input, test_output)
    and adds the columns the framework needs:
        - prompt: chat message for the orchestrator's shallow solve
        - answer: expected output for grading
        - info: JSON bag with grid data for child envs to rebuild prompts
        - task: "arc_pipeline" (routes to ArcPipelineEnv)
        - example_id: row index for rollout grouping
    """
    records = []
    for i in range(len(raw_dataset)):
        row = raw_dataset[i]
        train_pairs = json.loads(row["train_pairs"])
        test_input = json.loads(row["test_input"])
        prompt_text = build_prompt(train_pairs, test_input)

        records.append({
            "prompt": [{"role": "user", "content": prompt_text}],
            "answer": row.get("test_output", "") or "",
            "info": json.dumps({
                "train_pairs": row["train_pairs"],
                "test_input": row["test_input"],
                "task_id": row["task_id"],
            }),
            "task": "arc_pipeline",
            "example_id": i,
        })
    return Dataset.from_list(records)



def _get_train_pairs(state: State) -> list[dict]:
    """Extract train_pairs from state's input data."""
    raw = state.get("info", {}).get("train_pairs", "[]")
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def _get_test_input(state: State) -> Grid:
    """Extract test_input from state's input data."""
    raw = state.get("info", {}).get("test_input", "[]")
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def _get_test_output(state: State) -> Grid | None:
    """Extract expected test output from state."""
    raw = state.get("answer", "")
    if not raw:
        return None
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None
    return raw


def _extract_tag_content(text: str, tag_name: str) -> str | None:
    """Extract content between <tag>...</tag>."""
    pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: check for opening tag without closing (truncated response)
    open_pattern = f"<{tag_name}>"
    open_match = re.search(open_pattern, text)
    if open_match:
        # Response was truncated after opening tag - extract what we have
        content = text[open_match.end():].strip()
        if content:
            logger.warning(
                f"Tag <{tag_name}> found but not closed (truncated response). "
                f"Extracted {len(content)} chars after opening tag."
            )
            return content
    return None


def _truncate_fallback(text: str, max_chars: int = 2000) -> str:
    """Truncate fallback text to avoid cascading token bloat in multi-step pipelines."""
    if len(text) <= max_chars:
        return text
    # Take last portion (more likely to contain the summary)
    truncated = text[-max_chars:]
    # Find a clean break point
    newline_idx = truncated.find("\n")
    if newline_idx > 0 and newline_idx < 200:
        truncated = truncated[newline_idx + 1:]
    return f"[...truncated...]\n{truncated}"


def _get_last_response_text(state: State) -> str:
    """Get the text content of the last model response."""
    trajectory = state.get("trajectory", [])
    if not trajectory:
        return ""
    last_completion = trajectory[-1].get("completion", [])
    if not last_completion:
        return ""
    return last_completion[-1].get("content", "")


# =============================================================================
# SingleSolverEnv
# =============================================================================

class SingleSolverEnv(MultiAgentEnv):
    """
    Single-actor environment for basic ARC solving.

    One actor gets the task prompt and produces a solution.
    Used for: shallow search, deep thinking, hint-based solving.

    The prompt variant (basic/deep/hint) is set per-rollout via the input's
    prompt field, not baked into the environment.

    Each model gets its own env instance so Protocol routes to the right actor:
        SingleSolverEnv(actor_id="solver_gpt")  -> name="solver_gpt"
        SingleSolverEnv(actor_id="solver_claude") -> name="solver_claude"
    """

    def __init__(self, actor_id: str = "solver", env_name: str | None = None, **kwargs):
        self._actor_id = actor_id
        self.actors = [actor_id]
        self.name = env_name or actor_id
        super().__init__(max_turns=1, **kwargs)

    def get_initial_actor(self, state: State) -> str:
        return self._actor_id

    def get_next_actor(self, state: State) -> str:
        return self._actor_id

    async def build_actor_prompt(self, actor_id: str, state: State) -> Messages:
        actor = self.get_actor(actor_id)
        messages: Messages = []

        if actor.system_prompt:
            messages.append({"role": "system", "content": actor.system_prompt})

        # Use the prompt from the input (already formatted with the right variant)
        for msg in state.get("prompt", []):
            if msg.get("role") != "system":
                messages.append(msg)

        return messages

    async def on_turn_complete(self, state: State) -> None:
        """Parse the grid from the response and store it."""
        response_text = _get_last_response_text(state)
        if not response_text:
            return

        try:
            predicted_grid = parse_grid_from_text(response_text)
            state["extras"]["predicted_grid"] = predicted_grid
            state["extras"]["full_response"] = response_text

            expected = _get_test_output(state)
            if expected is not None:
                state["extras"]["is_correct"] = grids_equal(predicted_grid, expected)
                state["extras"]["similarity"] = grid_similarity(predicted_grid, expected)
            else:
                state["extras"]["is_correct"] = None

        except ValueError:
            state["extras"]["predicted_grid"] = None
            state["extras"]["is_correct"] = False
            state["extras"]["parse_error"] = True

    @vf.stop
    async def single_turn_done(self, state: State) -> bool:
        """Stop after 1 turn."""
        return len(state.get("trajectory", [])) >= 1


# =============================================================================
# DeepThinkingEnv
# =============================================================================

class DeepThinkingEnv(SingleSolverEnv):
    """
    Same as SingleSolverEnv but the prompt includes the deep thinking trigger.
    The orchestrator builds the deep thinking prompt before spawning.
    """

    def __init__(self, actor_id: str = "solver", **kwargs):
        super().__init__(actor_id=actor_id, env_name=f"deep_{actor_id}", **kwargs)


# =============================================================================
# HintSolverEnv
# =============================================================================

class HintSolverEnv(SingleSolverEnv):
    """
    Single-actor env that solves with hints injected into the prompt.
    The orchestrator extracts hints first, then spawns this with an enriched prompt.
    """

    def __init__(self, actor_id: str = "solver", **kwargs):
        super().__init__(actor_id=actor_id, env_name=f"hint_{actor_id}", **kwargs)


# =============================================================================
# HintExtractorEnv
# =============================================================================

class HintExtractorEnv(SingleSolverEnv):
    """
    Single-actor env that extracts hints (HINT_START...HINT_END) from the task.
    Does NOT solve the task. Just produces insights for downstream solvers.
    """

    def __init__(self, **kwargs):
        super().__init__(actor_id="hint_extractor", env_name="hint_extractor", **kwargs)

    async def on_turn_complete(self, state: State) -> None:
        """Extract the hint between HINT_START/HINT_END markers."""
        response_text = _get_last_response_text(state)
        match = re.search(r"HINT_START(.*?)HINT_END", response_text, re.DOTALL)
        if match:
            hint = match.group(1).strip()
            state["extras"]["hint"] = hint
            logger.info(f"HintExtractor: extracted hint ({len(hint)} chars)")
        else:
            # Check if HINT_START exists but HINT_END was truncated
            if "HINT_START" in response_text:
                idx = response_text.index("HINT_START") + len("HINT_START")
                hint = response_text[idx:].strip()
                state["extras"]["hint"] = hint
                logger.warning(f"HintExtractor: HINT_START found but HINT_END missing (truncated?). Extracted {len(hint)} chars.")
            else:
                state["extras"]["hint"] = None
                logger.warning("HintExtractor: no HINT_START/HINT_END markers found in response")
        state["extras"]["full_response"] = response_text


# =============================================================================
# ObjectsPipelineEnv
# =============================================================================

class ObjectsPipelineEnv(MultiAgentEnv):
    """
    Three-step objects pipeline: Extract -> Transform -> Solve.

    Three actors in sequence:
    1. extractor: Identifies objects in the grids
    2. transformer: Identifies transformations between input/output
    3. solver: Uses object + transformation descriptions to solve

    Context flows forward: each actor sees the previous actors' output.
    """

    actors = ["extractor", "transformer", "pipeline_solver"]
    name = "objects_pipeline"

    def __init__(self, **kwargs):
        super().__init__(max_turns=3, **kwargs)

    def get_initial_actor(self, state: State) -> str:
        return "extractor"

    def get_next_actor(self, state: State) -> str:
        current = state["extras"].get("current_actor_id")
        order = {
            "extractor": "transformer",
            "transformer": "pipeline_solver",
        }
        next_actor = order.get(current)
        if next_actor is None:
            logger.warning(f"ObjectsPipeline: unexpected actor '{current}', falling back to pipeline_solver")
            return "pipeline_solver"
        return next_actor

    async def build_actor_prompt(self, actor_id: str, state: State) -> Messages:
        actor = self.get_actor(actor_id)
        train_pairs = _get_train_pairs(state)
        test_input = _get_test_input(state)

        messages: Messages = []
        if actor.system_prompt:
            messages.append({"role": "system", "content": actor.system_prompt})

        if actor_id == "extractor":
            prompt_text = build_objects_extraction_prompt(train_pairs, test_input)
            messages.append({"role": "user", "content": prompt_text})

        elif actor_id == "transformer":
            # Get objects from extractor
            objects_text = state["extras"].get("objects_summary", "")
            prompt_text = build_objects_transformation_prompt(
                train_pairs, test_input, objects_text
            )
            messages.append({"role": "user", "content": prompt_text})

        elif actor_id == "pipeline_solver":
            # Combine objects + transformations into enriched prompt
            objects_text = state["extras"].get("objects_summary", "")
            transform_text = state["extras"].get("transformation_summary", "")
            insertion = f"## Objects Description\n\n{objects_text}\n\n## Transformation Description\n\n{transform_text}"
            prompt_text = build_prompt(
                train_pairs, test_input, objects_insertion=insertion
            )
            messages.append({"role": "user", "content": prompt_text})

        return messages

    async def on_turn_complete(self, state: State) -> None:
        """Extract structured output from each actor's response."""
        current = state["extras"]["current_actor_id"]
        response_text = _get_last_response_text(state)
        logger.info(f"ObjectsPipeline [{current}]: response length = {len(response_text)} chars")

        if current == "extractor":
            summary = _extract_tag_content(response_text, "objects_summary")
            if summary:
                state["extras"]["objects_summary"] = summary
            else:
                logger.warning("Extractor: no <objects_summary> tag found, using truncated fallback")
                state["extras"]["objects_summary"] = _truncate_fallback(response_text)
            print(f"\n{'='*60}")
            print(f"OBJECTS EXTRACTOR OUTPUT ({len(response_text)} chars):")
            print(f"{'='*60}")
            print(f"Summary extracted: {state['extras']['objects_summary'][:500]}")
            print(f"{'='*60}\n")

        elif current == "transformer":
            summary = _extract_tag_content(response_text, "transformation_summary")
            if summary:
                state["extras"]["transformation_summary"] = summary
            else:
                logger.warning("Transformer: no <transformation_summary> tag found, using truncated fallback")
                state["extras"]["transformation_summary"] = _truncate_fallback(response_text)
            print(f"\n{'='*60}")
            print(f"OBJECTS TRANSFORMER OUTPUT ({len(response_text)} chars):")
            print(f"{'='*60}")
            print(f"Summary extracted: {state['extras']['transformation_summary'][:500]}")
            print(f"{'='*60}\n")

        elif current == "pipeline_solver":
            try:
                predicted_grid = parse_grid_from_text(response_text)
                state["extras"]["predicted_grid"] = predicted_grid
                state["extras"]["full_response"] = response_text

                expected = _get_test_output(state)
                if expected is not None:
                    is_correct = grids_equal(predicted_grid, expected)
                    similarity = grid_similarity(predicted_grid, expected)
                    state["extras"]["is_correct"] = is_correct
                    state["extras"]["similarity"] = similarity
                    print(f"\n{'='*60}")
                    print(f"OBJECTS SOLVER: correct={is_correct}, similarity={similarity:.2f}")
                    print(f"Predicted: {predicted_grid}")
                    print(f"Expected:  {expected}")
                    print(f"{'='*60}\n")
                else:
                    state["extras"]["is_correct"] = None
            except ValueError:
                state["extras"]["predicted_grid"] = None
                state["extras"]["is_correct"] = False
                print(f"\n{'='*60}")
                print(f"OBJECTS SOLVER: FAILED TO PARSE GRID")
                print(f"Response (last 500 chars): {response_text[-500:]}")
                print(f"{'='*60}\n")

    @vf.stop
    async def pipeline_done(self, state: State) -> bool:
        """Stop after all 3 actors have gone."""
        return len(state.get("trajectory", [])) >= 3


# =============================================================================
# JudgeEnv - Logic and Consistency Judges
# =============================================================================

class JudgeEnv(MultiAgentEnv):
    """
    Judge environment for evaluating candidate solutions.

    The judge receives a prompt containing all candidates + their reasoning,
    and outputs a JSON with scores for each candidate.

    Used for both Logic Judge and Consistency Judge - the prompt determines
    which judging protocol is used.
    """

    actors = ["judge"]
    name = "judge"

    def __init__(self, env_name: str = "judge", **kwargs):
        self.name = env_name
        super().__init__(max_turns=1, **kwargs)

    def get_initial_actor(self, state: State) -> str:
        return "judge"

    def get_next_actor(self, state: State) -> str:
        return "judge"

    async def build_actor_prompt(self, actor_id: str, state: State) -> Messages:
        actor = self.get_actor(actor_id)
        messages: Messages = []

        if actor.system_prompt:
            messages.append({"role": "system", "content": actor.system_prompt})

        # The judge prompt is passed in via the input
        for msg in state.get("prompt", []):
            if msg.get("role") != "system":
                messages.append(msg)

        return messages

    async def on_turn_complete(self, state: State) -> None:
        """Parse the JSON response from the judge."""
        response_text = _get_last_response_text(state)
        state["extras"]["full_response"] = response_text

        # Try to extract JSON from the response
        parsed = _extract_judge_json(response_text)
        state["extras"]["parsed_json"] = parsed

        if parsed and "candidates" in parsed:
            # Extract scores keyed by candidate_id
            scores = {}
            for cand in parsed["candidates"]:
                cid = cand.get("candidate_id")
                if isinstance(cid, str):
                    try:
                        cid = int(cid)
                    except (ValueError, TypeError):
                        pass
                score = cand.get("score", 0.0)
                if cid is not None:
                    scores[cid] = score
            state["extras"]["scores"] = scores
        else:
            state["extras"]["scores"] = {}

    @vf.stop
    async def judge_done(self, state: State) -> bool:
        """Stop after 1 turn."""
        return len(state.get("trajectory", [])) >= 1


class LogicJudgeEnv(JudgeEnv):
    """Logic Judge - falsification-based auditing."""
    name = "logic_judge"

    def __init__(self, **kwargs):
        super().__init__(env_name="logic_judge", **kwargs)


class ConsistencyJudgeEnv(JudgeEnv):
    """Consistency Judge - internal coherence checking."""
    name = "consistency_judge"

    def __init__(self, **kwargs):
        super().__init__(env_name="consistency_judge", **kwargs)


def _extract_judge_json(text: str) -> dict | None:
    """
    Extract JSON from judge response.

    Handles:
    - JSON in markdown code blocks
    - Raw JSON in response
    - Prioritizes objects with 'candidates' key
    """
    if not text:
        return None

    text = text.strip()

    # Try markdown code block first
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(1))
            if isinstance(obj, dict) and "candidates" in obj:
                return obj
        except json.JSONDecodeError:
            pass

    # Scan for any '{' and try to decode
    for match in re.finditer(r"\{", text):
        start_idx = match.start()
        try:
            decoder = json.JSONDecoder()
            obj, _ = decoder.raw_decode(text, idx=start_idx)
            if isinstance(obj, dict) and "candidates" in obj:
                return obj
        except json.JSONDecodeError:
            continue

    return None


def _build_candidates_for_judge(
    candidates: dict[tuple, dict],
    reasoning_store: dict[str, str],
) -> list[dict]:
    """
    Convert candidates dict to list format for judge prompts.

    Returns list of:
    {
        "id": int,
        "grid": Grid,
        "models": [run_ids],
        "reasoning": {run_id: response_text, ...},
    }
    """
    result = []
    for idx, (grid_tuple, val) in enumerate(candidates.items()):
        reasoning = {}
        for model_id in val.get("models", []):
            if model_id in reasoning_store:
                reasoning[model_id] = reasoning_store[model_id]

        result.append({
            "id": idx,
            "grid": val["grid"],
            "models": val.get("models", []),
            "reasoning": reasoning,
            "count": val.get("count", 1),
        })

    return result


async def run_judges(
    protocol: Protocol,
    candidates: dict[tuple, dict],
    reasoning_store: dict[str, str],
    train_pairs: list[dict],
    test_input: Grid,
    client: Any,
    default_model: str,
    enable_consistency: bool = True,
) -> dict[int, float]:
    """
    Run Logic and Consistency judges on candidates.

    Returns dict mapping candidate_id -> max(logic_score, consistency_score)
    """
    if not candidates:
        return {}

    # Convert candidates for judge prompts
    cand_list = _build_candidates_for_judge(candidates, reasoning_store)

    # Filter: if 2+ candidates have 2+ votes, only judge those
    multi_vote = [c for c in cand_list if c["count"] >= 2]
    if len(multi_vote) >= 2:
        candidates_for_judging = multi_vote
    else:
        candidates_for_judging = cand_list

    # Build judge prompts
    logic_prompt = build_logic_judge_prompt(train_pairs, test_input, candidates_for_judging)
    consistency_prompt = build_consistency_judge_prompt(train_pairs, test_input, candidates_for_judging)

    # Spawn both judges in parallel
    spawn_inputs = [
        {
            "task": "logic_judge",
            "prompt": [{"role": "user", "content": logic_prompt}],
            "answer": "",
            "info": {},
        },
    ]

    if enable_consistency:
        spawn_inputs.append({
            "task": "consistency_judge",
            "prompt": [{"role": "user", "content": consistency_prompt}],
            "answer": "",
            "info": {},
        })

    judge_states = await protocol.spawn(
        inputs=spawn_inputs,
        client=client,
        model=default_model,
        score=False,
    )

    # Aggregate scores: max of both judges per candidate
    all_scores: dict[int, float] = {c["id"]: 0.0 for c in cand_list}

    for state in judge_states:
        scores = state.get("extras", {}).get("scores", {})
        for cid, score in scores.items():
            if cid in all_scores:
                all_scores[cid] = max(all_scores[cid], score)

    return all_scores


# =============================================================================
# Candidate Collection
# =============================================================================

def collect_candidates(
    candidates: dict[tuple, dict],
    reasoning_store: dict[str, str],
    states: list[State],
) -> None:
    """
    Collect candidate grids from completed rollout states.

    Matches the ARC solver's candidates_object structure:
    candidates[grid_tuple] = {
        "grid": Grid,
        "count": int,
        "models": [run_ids],
        "is_correct": bool | None,
    }
    """
    for state in states:
        extras = state.get("extras", {})
        grid = extras.get("predicted_grid")
        if grid is None:
            actor_id = extras.get("current_actor_id", "unknown")
            logger.warning(f"collect_candidates: no predicted_grid from {actor_id}, skipping")
            continue

        grid_tuple = tuple(tuple(row) for row in grid)

        # Track actor/model info
        actor_id = extras.get("current_actor_id", "unknown")
        model = state.get("model", "unknown")
        run_id = f"{actor_id}_{model}"

        if grid_tuple not in candidates:
            candidates[grid_tuple] = {
                "grid": grid,
                "count": 0,
                "models": [],
                "is_correct": extras.get("is_correct"),
            }

        candidates[grid_tuple]["count"] += 1
        candidates[grid_tuple]["models"].append(run_id)

        # Store reasoning for judges
        full_response = extras.get("full_response", "")
        if full_response:
            reasoning_store[run_id] = full_response


def is_solved(candidates: dict[tuple, dict]) -> bool:
    """
    Check if we have a confident solution via consensus.

    From ARC solver: >25% of votes AND >=7 votes AND all others have exactly 1.
    """
    if not candidates:
        return False

    total = sum(g["count"] for g in candidates.values())
    if total == 0:
        return False

    sorted_groups = sorted(candidates.values(), key=lambda g: g["count"], reverse=True)
    top = sorted_groups[0]
    max_count = top["count"]
    percentage = max_count / total

    if not (percentage > 0.25 and max_count >= 7):
        return False

    for group in sorted_groups[1:]:
        if group["count"] != 1:
            return False

    return True


def pick_best_solution(
    candidates: dict[tuple, dict],
    judge_scores: dict[int, float] | None = None,
) -> list[dict]:
    """
    Pick top solutions using consensus + judge scores.

    Attempt 1: Highest vote count (tie-break by judge score)
    Attempt 2: Highest judge score (excluding attempt 1)
    """
    if not candidates:
        return []

    cand_list = []
    for idx, (grid_tuple, val) in enumerate(candidates.items()):
        cand_list.append({
            "id": idx,
            "grid": val["grid"],
            "count": val["count"],
            "models": val["models"],
            "is_correct": val.get("is_correct"),
        })

    scores = judge_scores or {c["id"]: 0.0 for c in cand_list}

    # Attempt 1: consensus (votes, then score)
    by_consensus = sorted(
        cand_list, key=lambda c: (c["count"], scores.get(c["id"], 0)), reverse=True
    )
    attempt_1 = by_consensus[0]

    # Attempt 2: best score excluding attempt 1
    by_score = sorted(
        cand_list, key=lambda c: scores.get(c["id"], 0), reverse=True
    )
    attempt_2 = None
    for c in by_score:
        if c["id"] != attempt_1["id"]:
            attempt_2 = c
            break

    results = [attempt_1]
    if attempt_2:
        results.append(attempt_2)

    return results


# =============================================================================
# ARC Pipeline Environment 
# =============================================================================

class ArcPipelineEnv(MultiAgentEnv):
    """
    Orchestrates multi-strategy solving + dual judge evaluation:
    1. Orchestrator does initial shallow solve (turn 1)
    2. on_turn_complete spawns additional strategies in parallel
    3. Collects candidates, runs judges, picks best solution
    4. Reward = correctness of best pick (pass@2)

    Follows the proposer_solver pattern: main env spawns child episodes
    via Protocol.spawn().

    Single-model usage (default):
        All actors use the model provided. Strategy diversity
        comes from different prompt variants (shallow, deep, objects, hint).

    Multi-model usage (via SOLVER_CONFIGS):
        Each entry gets its own Actor (and thus its own model).
    """

    name = "arc_pipeline"
    actors = ["orchestrator"]

    def __init__(
        self,
        strategies: list[str] | None = None,
        solver_actor_ids: list[str] | None = None,
        enable_judges: bool = True,
        enable_consistency_judge: bool = True,
        **kwargs,
    ):
        self.strategies = strategies or ["shallow", "deep", "objects"]
        self.solver_actor_ids = solver_actor_ids or []
        self.enable_judges = enable_judges
        self.enable_consistency_judge = enable_consistency_judge
        super().__init__(max_turns=1, **kwargs)

    def get_initial_actor(self, state: State) -> str:
        return "orchestrator"

    def get_next_actor(self, state: State) -> str:
        return "orchestrator"

    async def setup_state(self, state: State) -> State:
        state = await super().setup_state(state)
        state["extras"]["pipeline_done"] = False
        state["extras"]["picks"] = []
        state["extras"]["num_candidates"] = 0
        state["extras"]["consensus_solved"] = False
        state["extras"]["is_correct"] = False
        state["extras"]["is_correct_2"] = False
        return state

    async def build_actor_prompt(self, actor_id: str, state: State) -> Messages:
        """Build prompt for orchestrator's initial shallow solve."""
        actor = self.get_actor(actor_id)
        messages: Messages = []
        if actor.system_prompt:
            messages.append({"role": "system", "content": actor.system_prompt})
        for msg in state.get("prompt", []):
            if msg.get("role") != "system":
                messages.append(msg)
        return messages

    async def on_turn_complete(self, state: State) -> None:
        """
        After orchestrator's initial solve:
        1. Parse orchestrator's grid
        2. Spawn additional strategies (deep, objects, hint, extra solvers)
        3. Collect all candidates
        4. Run judges
        5. Pick best solution
        """
        train_pairs = _get_train_pairs(state)
        test_input = _get_test_input(state)
        expected = _get_test_output(state)
        answer_str = state.get("answer", "")
        info_dict = state.get("info", {})
        idx = state.get("example_id", 0)

        candidates: dict[tuple, dict] = {} #accumlates all unique grids 
        reasoning_store: dict[str, str] = {} #full reponse text

        # --- Parse orchestrator's initial solve ---
        response_text = _get_last_response_text(state)
        if response_text:
            try:
                grid = parse_grid_from_text(response_text)
                grid_tuple = tuple(tuple(row) for row in grid)
                candidates[grid_tuple] = {
                    "grid": grid,
                    "count": 1,
                    "models": ["orchestrator"],
                    "is_correct": grids_equal(grid, expected) if expected else None,
                }
                reasoning_store["orchestrator"] = response_text
            except ValueError:
                logger.warning("Failed to parse orchestrator's grid response")

        # --- Build spawn inputs for additional strategies ---
        spawn_inputs = []

        # Shallow solve with additional solver models
        if "shallow" in self.strategies:
            basic_prompt = build_prompt(train_pairs, test_input)
            for sid in self.solver_actor_ids:
                spawn_inputs.append({
                    "task": sid,
                    "prompt": [{"role": "user", "content": basic_prompt}],
                    "answer": answer_str,
                    "example_id": idx,
                    "info": info_dict,
                })

        # Deep thinking (orchestrator + additional solvers)
        if "deep" in self.strategies:
            deep_prompt = build_prompt(
                train_pairs, test_input, trigger_deep_thinking=True
            )
            spawn_inputs.append({
                "task": "deep_orchestrator",
                "prompt": [{"role": "user", "content": deep_prompt}],
                "answer": answer_str,
                "example_id": idx,
                "info": info_dict,
            })
            for sid in self.solver_actor_ids:
                spawn_inputs.append({
                    "task": f"deep_{sid}",
                    "prompt": [{"role": "user", "content": deep_prompt}],
                    "answer": answer_str,
                    "example_id": idx,
                    "info": info_dict,
                })

        # Objects pipeline
        if "objects" in self.strategies:
            spawn_inputs.append({
                "task": "objects_pipeline",
                "prompt": [{"role": "user", "content": ""}],
                "answer": answer_str,
                "example_id": idx,
                "info": info_dict,
            })

        # Spawn all parallel strategies
        if spawn_inputs:
            strategy_states = await self.protocol.spawn(
                inputs=spawn_inputs,
                client=state["client"],
                model=state["model"],
                sampling_args=state.get("sampling_args"),
                score=False,
            )
            collect_candidates(candidates, reasoning_store, strategy_states)
            for s in strategy_states:
                state["child_states"].append(s)

        # --- Hint strategy (two-phase: extract then solve) ---
        if "hint" in self.strategies:
            hint_prompt = build_hint_extraction_prompt(train_pairs, test_input)
            hint_states = await self.protocol.spawn(
                inputs=[{
                    "task": "hint_extractor",
                    "prompt": [{"role": "user", "content": hint_prompt}],
                    "answer": "",
                    "example_id": idx,
                    "info": info_dict,
                }],
                client=state["client"],
                model=state["model"],
                score=False,
            )

            hint_text = (
                hint_states[0].get("extras", {}).get("hint")
                if hint_states else None
            )
            if hint_text:
                hint_solve_prompt = build_prompt(
                    train_pairs, test_input, strategy=hint_text
                )
                hint_inputs = [{
                    "task": "hint_orchestrator",
                    "prompt": [{"role": "user", "content": hint_solve_prompt}],
                    "answer": answer_str,
                    "example_id": idx,
                    "info": info_dict,
                }]
                for sid in self.solver_actor_ids:
                    hint_inputs.append({
                        "task": f"hint_{sid}",
                        "prompt": [{"role": "user", "content": hint_solve_prompt}],
                        "answer": answer_str,
                        "example_id": idx,
                        "info": info_dict,
                    })
                hint_solve_states = await self.protocol.spawn(
                    inputs=hint_inputs,
                    client=state["client"],
                    model=state["model"],
                    score=False,
                )
                collect_candidates(candidates, reasoning_store, hint_solve_states)
                for s in hint_solve_states:
                    state["child_states"].append(s)

        # --- Log candidate summary ---
        for gt, val in candidates.items():
            logger.info(
                f"Candidate: correct={val.get('is_correct')}, "
                f"votes={val['count']}, models={val['models']}"
            )

        # --- Check consensus ---, not used currently for anything besides metric tracking at the moment
        consensus = is_solved(candidates)

        # --- Run judges ---
        judge_scores = {}
        if self.enable_judges and len(candidates) > 1:
            judge_scores = await run_judges(
                protocol=self.protocol,
                candidates=candidates,
                reasoning_store=reasoning_store,
                train_pairs=train_pairs,
                test_input=test_input,
                client=state["client"],
                default_model=state["model"],
                enable_consistency=self.enable_consistency_judge,
            )

        # --- Pick best solution(s) ---
        picks = pick_best_solution(candidates, judge_scores)

        # --- Store results ---
        state["extras"]["picks"] = picks
        state["extras"]["num_candidates"] = len(candidates)
        state["extras"]["consensus_solved"] = consensus
        state["extras"]["total_runs"] = sum(
            v["count"] for v in candidates.values()
        )

        if picks:
            top = picks[0]
            state["extras"]["is_correct"] = bool(top.get("is_correct"))
            state["extras"]["top_grid"] = top.get("grid")
            if len(picks) > 1:
                state["extras"]["is_correct_2"] = bool(
                    picks[1].get("is_correct")
                )

        state["extras"]["pipeline_done"] = True

        # DEBUG: print final pipeline scoring state
        print(f"\n{'='*60}")
        print(f"PIPELINE FINAL STATE:")
        print(f"  num_candidates: {len(candidates)}")
        print(f"  total_runs: {sum(v['count'] for v in candidates.values())}")
        print(f"  picks: {len(picks)}")
        if picks:
            top = picks[0]
            print(f"  top pick is_correct: {top.get('is_correct')} (type: {type(top.get('is_correct'))})")
            print(f"  top pick grid: {top.get('grid')}")
            print(f"  state extras is_correct: {state['extras'].get('is_correct')}")
            print(f"  state extras is_correct_2: {state['extras'].get('is_correct_2')}")
        print(f"  expected answer: {expected}")
        print(f"  judge_scores: {judge_scores}")
        for gt, val in candidates.items():
            print(f"  candidate: correct={val.get('is_correct')}, votes={val['count']}, models={val['models']}")
        print(f"{'='*60}\n")

    @vf.stop
    async def pipeline_done(self, state: State) -> bool:
        """Stop after orchestrator turn + spawns complete."""
        return state.get("extras", {}).get("pipeline_done", False)


# =============================================================================
# Rubric
# =============================================================================

def create_arc_rubric():
    """
    Create rubric for ARC pipeline.

    Orchestrator reward: 1.0 if top pick correct, 0.5 if second pick correct
    (pass@2), 0.0 otherwise. Metrics track candidates and consensus.
    """
    rubric = MultiAgentRubric()

    def orchestrator_reward(state: State, **kwargs) -> float:
        """Reward based on correctness of best pick (pass@2)."""
        extras = state.get("extras") or {}
        if extras.get("is_correct"):
            return 1.0
        if extras.get("is_correct_2"):
            return 0.5
        return 0.0

    def num_candidates_metric(state: State, **kwargs) -> float:
        """Track number of unique candidate grids."""
        extras = state.get("extras") or {}
        return float(extras.get("num_candidates", 0))

    def consensus_metric(state: State, **kwargs) -> float:
        """Track whether consensus was reached."""
        extras = state.get("extras") or {}
        return 1.0 if extras.get("consensus_solved") else 0.0

    def total_runs_metric(state: State, **kwargs) -> float:
        """Track total number of solve attempts."""
        extras = state.get("extras") or {}
        return float(extras.get("total_runs", 0))

    rubric.add_actor_reward_func("orchestrator", orchestrator_reward, weight=1.0)
    rubric.add_actor_metric("orchestrator", num_candidates_metric)
    rubric.add_actor_metric("orchestrator", consensus_metric)
    rubric.add_actor_metric("orchestrator", total_runs_metric)

    return rubric


# =============================================================================
# Actor + Env Factory
# =============================================================================

def create_actors(
    strategy_list: list[str],
    enable_judges: bool = True,
) -> tuple[list[Actor], list[str]]:
    """
    Create actors for ARC pipeline from SOLVER_CONFIGS.

    Each solver in SOLVER_CONFIGS gets its own Actor (and thus its own model).
    Utility actors (judge, objects, hint_extractor) use their respective
    endpoint configs, or the default -m model if None.

    Returns:
        (actors, solver_actor_ids) where solver_actor_ids are the IDs
        of additional solvers beyond the orchestrator.
    """
    actors = []
    solver_actor_ids = []

    # --- Solver actors (from SOLVER_CONFIGS) ---
    for i, config in enumerate(SOLVER_CONFIGS):
        if i == 0:
            actor_id = "orchestrator"
        else:
            actor_id = f"solver_{i}"
            solver_actor_ids.append(actor_id)

        client, model = get_actor_client(config.get("endpoint"))
        actors.append(Actor(
            id=actor_id,
            model=model,
            client=client,
            max_tokens=16384,
            is_trainable=config.get("is_trainable", False),
        ))

    # If no configs, create default orchestrator (uses -m model)
    if not SOLVER_CONFIGS:
        actors.append(Actor(
            id="orchestrator", max_tokens=16384, is_trainable=False,
        ))

    # --- Judge actor ---
    if enable_judges:
        client, model = get_actor_client(JUDGE_ENDPOINT)
        actors.append(Actor(
            id="judge", model=model, client=client,
            max_tokens=16384, is_trainable=False,
        ))

    # --- Objects pipeline actors (all share OBJECTS_ENDPOINT) ---
    if "objects" in strategy_list:
        client, model = get_actor_client(OBJECTS_ENDPOINT)
        for role_id in ("extractor", "transformer", "pipeline_solver"):
            actors.append(Actor(
                id=role_id, model=model, client=client,
                max_tokens=16384, is_trainable=False,
            ))

    # --- Hint extractor ---
    if "hint" in strategy_list:
        client, model = get_actor_client(HINT_EXTRACTOR_ENDPOINT)
        actors.append(Actor(
            id="hint_extractor", model=model, client=client,
            max_tokens=16384, is_trainable=False,
        ))

    return actors, solver_actor_ids


def create_envs(
    pipeline_env: ArcPipelineEnv,
    strategy_list: list[str],
    solver_actor_ids: list[str],
    enable_judges: bool = True,
) -> list:
    """
    Create all environments for the Protocol routing table.

    Each env's name becomes a route: spawn({"task": env.name}) → that env.
    Only pipeline_env has a dataset — the rest are workers that run when spawned.
    """
    envs: list = [pipeline_env]

    # Per-solver envs (multi-model mode — only runs if SOLVER_CONFIGS is populated)
    for sid in solver_actor_ids:
        envs.append(SingleSolverEnv(actor_id=sid))
        if "deep" in strategy_list:
            envs.append(DeepThinkingEnv(actor_id=sid))
        if "hint" in strategy_list:
            envs.append(HintSolverEnv(actor_id=sid))

    if "deep" in strategy_list:
        envs.append(DeepThinkingEnv(actor_id="orchestrator"))

    if "hint" in strategy_list:
        envs.append(HintExtractorEnv())
        envs.append(HintSolverEnv(actor_id="orchestrator"))

    if "objects" in strategy_list:
        envs.append(ObjectsPipelineEnv())

    if enable_judges:
        envs.append(LogicJudgeEnv())
        envs.append(ConsistencyJudgeEnv())

    return envs


# =============================================================================
# Environment Loader
# =============================================================================

def load_environment(
    hf_dataset: str = HF_DATASET_REPO,
    split: str = "train",
    strategies: str = "shallow,deep,objects",
    enable_judges: bool = True,
) -> ArcPipelineEnv:
    """
    Entry point called by vf-eval. Loads data, builds the full pipeline, returns it.

    All args are passable from CLI via -a flag:
        -a '{"split": "eval", "strategies": "shallow,deep,hint", "enable_judges": false}'

    Args:
        hf_dataset: HuggingFace dataset repo. Default: "bhoy/arc-agi-2".
        split: Dataset split — "train" (1000 tasks) or "eval" (120 tasks).
        strategies: Comma-separated solving strategies. Options: shallow, deep, objects, hint.
        enable_judges: Run Logic + Consistency judges to rank candidates. Default: True.

    """

    strategy_list = [s.strip() for s in strategies.split(",")]

    # 1. Load ARC tasks from HuggingFace → transform into vf-eval format
    raw_dataset = load_arc_dataset_hf(hf_dataset, split=split)
    dataset = prepare_dataset_for_eval(raw_dataset)

    # 2. Scoring: reward = 1.0 (top pick correct), 0.5 (2nd pick correct), 0.0
    rubric = create_arc_rubric()

    # 3. Actors: one per role (orchestrator, judge, extractor, etc.)
    #    In single-model mode (default), all use the -m model.
    #    solver_actor_ids is empty unless SOLVER_CONFIGS has extra models.
    actors, solver_actor_ids = create_actors(strategy_list, enable_judges)

    # 4. Pipeline env: the only env with a dataset — framework iterates over it
    pipeline_env = ArcPipelineEnv(
        strategies=strategy_list,
        solver_actor_ids=solver_actor_ids,
        enable_judges=enable_judges,
        rubric=rubric,
        dataset=dataset,
    )

    # 5. Worker envs + Protocol: routing table so spawn({"task": name}) works
    envs = create_envs(pipeline_env, strategy_list, solver_actor_ids, enable_judges)
    Protocol(actors=actors, envs=envs)

    return pipeline_env
