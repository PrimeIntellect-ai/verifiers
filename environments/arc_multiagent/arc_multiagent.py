"""
ARC-AGI Multi-Agent Environment.

Recreates the ARC-AGI solver pipeline using the verifiers multi-agent framework:
- SingleSolverEnv: One actor solves the task (basic/deep/hint variants)
- ObjectsPipelineEnv: Three actors in sequence (extract -> transform -> solve)
- JudgeEnv: Evaluates candidate solutions

All orchestrated via Registry.spawn() for parallel execution.

Usage:
    prime env install arc-multiagent
    prime eval run arc-multiagent -m qwen3-30b-i -n 5 -r 1 --debug
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import platform
import random
import re
import subprocess
import sys
import tempfile
import textwrap
from typing import Any, List, Optional

from datasets import Dataset

import verifiers as vf
from verifiers.agent import Agent
from verifiers.envs.multiagent_env import MultiAgentEnv
from verifiers.envs.registry import Registry
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


def extract_all_grids(text: str) -> list[Grid]:
    """
    Extract ALL grid blocks from text (not just the last one).
    Used by DuoPickJudge to get the judge's two grid picks.
    Same parsing logic as parse_grid_from_text but returns all blocks.
    """
    if not text:
        return []

    text = text.strip()
    lines = text.splitlines()

    candidate_rows = []
    hard_separators = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```"):
            candidate_rows.append(None)
            hard_separators.append(i)
            continue
        if not stripped:
            candidate_rows.append(None)
            continue
        if re.match(r'^Row\s+\d+:?$', stripped, re.IGNORECASE):
            candidate_rows.append(None)
            continue
        if stripped.startswith(("-", "*", "+")):
            candidate_rows.append(None)
            continue

        row = None
        try:
            clean_line = stripped.replace("`", " ").replace("[", " ").replace("]", " ").strip()
            numbered_match = re.match(r'^\d+[\.\)]\s+', clean_line)
            if numbered_match:
                clean_line = clean_line[numbered_match.end():]

            tokens = clean_line.split(",")
            if len(tokens) > 0 and all(t.strip().isdigit() for t in tokens):
                row = [int(t.strip()) for t in tokens]
            else:
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

    return blocks


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
For each CANDIDATE:

  1. It may have multiple ANSWER blocks, all with the same OUTPUT_GRID.
  2. Among its ANSWERs, select the **single most detailed and logical** explanation.
     - Prefer the explanation that is:
       - most rigorous and complete,
       - least self-contradictory,
       - best grounded in the grid data.
  3. Treat that explanation as the candidate's rule.
  4. Treat the OUTPUT_GRID from any ANSWER as the candidate's predicted
     test output (they are guaranteed identical within that candidate).

STEP 2 -- EXAMPLE CONSISTENCY AUDIT (DO NOT USE THE TEST INPUT HERE)
For each candidate's representative rule:

  1. Using only the training examples:
     For each TRAIN_EXAMPLE (in index order: 1, 2, 3, ...):

       - Check whether the described rule correctly explains the transformation
         from that example's INPUT_GRID to OUTPUT_GRID.

       - Be strict:
         * If the explanation states a universal rule that is clearly violated
           by any training example, mark that as a serious contradiction.
         * If the explanation fails to mention an obvious systematic behavior
           visible in multiple examples, note that as a weakness.

     For each training example, assign:
       - "Pass": fits the example with no obvious contradictions.
       - "Partial": roughly fits but has ambiguities or minor mismatches.
       - "Fail": clear contradiction with that example.

  2. Summarize, across all training examples:
     - How well does this rule fit the *set* of training examples taken together?
     - Does the rule feel overfitted, overcomplicated, or ad hoc?

STEP 3 -- RULE-TO-TEST-GRID CONSISTENCY
For each candidate:

  1. Take its representative rule (from STEP 1).
  2. Apply the rule *conceptually* to the TEST_INPUT:
     - You do not need to compute a perfect output from scratch;
       focus on key structural consequences of the rule.
  3. Check whether the candidate's test OUTPUT_GRID is a reasonable outcome
     of that rule.
     - If the grid blatantly violates the described rule, mark this as a
       contradiction.
     - If the grid is broadly consistent, mark it as plausible.

STEP 4 -- SCORING AND GLOBAL RANKING

For each candidate, assign a numeric SCORE from 0 to 10:
  - 10: Rule is simple and coherent, strongly consistent with all training
        examples, and the test grid fits the rule.
  - 7-9: Mostly consistent with examples; minor ambiguities or small issues.
  - 4-6: Some consistency, but noticeable contradictions or hand-wavy parts.
  - 1-3: Major contradictions with examples or test grid; rule not credible.
  - 0: Completely incompatible with examples; or explanation is nonsense.

Then:
  - Rank all candidates in descending order of SCORE.
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
    },
    {
      "candidate_id": 1,
      "score": 6.0,
      "tier": "INVALID",
      "example_audit": {
        "per_example": {
          "1": "Partial",
          "2": "Fail"
        },
        "summary": "Contradiction in example 2; seems overfitted."
      },
      "test_grid_consistency": "Contradictory",
      "rule_summary": "..."
    }
  ],
  "final_ranking_by_candidate": [
    0,
    4,
    5,
    1
  ]
}

Constraints:
- Do not add any fields outside this schema.
- All candidate_id values must match the id attributes in the <CANDIDATE> tags.
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
# Codegen Prompt Templates
# =============================================================================

def build_codegen_prompt(
    train_pairs: list[dict],
    test_input: Grid,
    version: str = "v1b",
) -> str:
    """
    Build codegen prompt asking model to write a solver() function.

    version="v1b": Basic prompt with probes (test input shown as unlabeled probe).
    version="v4": Expert-level scientific verification protocol.
    """
    if version == "v4":
        return _build_codegen_prompt_v4(train_pairs, test_input)
    return _build_codegen_prompt_v1b(train_pairs, test_input)


def _build_codegen_prompt_v1b(
    train_pairs: list[dict],
    test_input: Grid,
) -> str:
    lines = [
        "Below is an ARC AGI task. You're given the training input/output pairs. "
        "Your task is to write a python function solver(input_grid) that returns the output grid. "
        "The input_grid is a 2D numpy array. The solver() function must solve all the input/output pairs. "
        "You're also given some input-only training data to help you ensure your solution is generalizable.",
        "",
        "You may use numpy, scipy, and cv2 (OpenCV) for grid manipulation.",
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

    # Show test input as an unlabeled "probe" (no expected output)
    lines.append("Input-only training data:")
    lines.append("Probe 1:")
    lines.append("input:")
    lines.append(format_grid(test_input))
    lines.append("")

    lines.append("Only output the python code for the solver() function")
    return "\n".join(lines)


def _build_codegen_prompt_v4(
    train_pairs: list[dict],
    test_input: Grid,
) -> str:
    lines = [
        "[ARC TASK DATA START]",
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

    lines.append("Input-only training data (Probe Inputs):")
    lines.append("Probe 1:")
    lines.append("input:")
    lines.append(format_grid(test_input))
    lines.append("")
    lines.append("[ARC TASK DATA END]")
    lines.append("")

    # Scientific verification protocol (works well with most models)
    lines.extend([
        "*** INSTRUCTIONS: SCIENTIFIC VERIFICATION PROTOCOL ***",
        "",
        "You are an expert ARC-AGI Solver Architect.",
        "Your goal is to write a final, robust `solver(input_grid)` function.",
        "The `input_grid` provided to `solver` will be a **2D NumPy array**.",
        "You are provided with **Solved Training Pairs** (to derive the rule) "
        "and **Unlabeled Probe Inputs** (to test generalizability).",
        "",
        "**CRITICAL RULE:** Do NOT guess. You must PROVE your solution works.",
        "",
        "### PHASE 1: ANALYSIS",
        "1. Load and analyze all training grids.",
        "2. Identify the transformation rule.",
        "3. Check that probe inputs are compatible with your rule.",
        "",
        "### PHASE 2: IMPLEMENTATION & VERIFICATION",
        "1. Draft a candidate `solver` function.",
        "2. Verify it produces correct output for ALL training pairs.",
        "3. Ensure it doesn't crash on probe inputs.",
        "4. Refine if any test fails.",
        "",
        "### PHASE 3: FINAL OUTPUT",
        "Output the final, standalone `solver(input_grid)` function.",
        "Precede it with: `### FINAL SOLUTION ###`",
        "",
        "Format:",
        "### FINAL SOLUTION ###",
        "```python",
        "import numpy as np",
        "",
        "def solver(input_grid):",
        "    # input_grid is a 2D numpy array",
        "    # ...",
        "```",
    ])

    return "\n".join(lines)


def _extract_solver_code(llm_response: str) -> str | None:
    """
    Extract Python code containing def solver() from LLM response.
    Multi-stage extraction for robustness (matches original codegen.py).
    """
    code_search_area = None

    # Stage 1: Explicit marker search (v4 style)
    if "### FINAL SOLUTION ###" in llm_response:
        parts = llm_response.split("### FINAL SOLUTION ###")
        code_search_area = parts[-1]

    # Stage 2: Search markdown blocks in reverse for 'def solver'
    pattern = r"```python(.*?)```"
    if not code_search_area:
        blocks = re.findall(pattern, llm_response, re.DOTALL)
        for block in reversed(blocks):
            if "def solver" in block:
                return block.strip()

    # Stage 3: Extract from search area
    if code_search_area:
        match = re.search(pattern, code_search_area, re.DOTALL)
        if match:
            return match.group(1).strip()
        # No markdown block after marker — find def solver directly
        if "def solver" in code_search_area:
            lines = code_search_area.splitlines()
            for i, line in enumerate(lines):
                if "def solver" in line:
                    return "\n".join(lines[i:])

    # Stage 4: Ultimate fallback — search entire response
    if "def solver" in llm_response:
        lines = llm_response.splitlines()
        for i, line in enumerate(lines):
            if "def solver" in line:
                return "\n".join(lines[i:])

    return None


# =============================================================================
# Sandbox Execution (Codegen)
# =============================================================================

_SANDBOX_DRIVER = r"""
import json
import sys
import traceback
import math
import itertools
from collections import Counter, deque, defaultdict
from typing import List, Optional, Tuple, Any, Dict, Set
import copy

try:
    import numpy as np
except ImportError:
    np = None

try:
    import scipy
    import scipy.ndimage
except ImportError:
    scipy = None

try:
    import cv2
except ImportError:
    cv2 = None

def convert_to_numpy(obj):
    if np is None: return obj
    if isinstance(obj, list):
        return np.array(obj)
    return obj

def sanitize_output(obj):
    if isinstance(obj, list):
        return [sanitize_output(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(sanitize_output(x) for x in obj)
    if isinstance(obj, dict):
        return {k: sanitize_output(v) for k, v in obj.items()}
    if np and isinstance(obj, (np.integer, int)):
        return int(obj)
    if np and isinstance(obj, (np.floating, float)):
        return float(obj)
    if np and isinstance(obj, np.ndarray):
        return sanitize_output(obj.tolist())
    return obj

def secure_runtime():
    import sys
    banned_modules = [
        'socket', 'ssl', 'asyncio', 'requests', 'urllib3', 'ftplib',
        'poplib', 'imaplib', 'nntplib', 'smtplib', 'telnetlib',
        'httpx', 'aiohttp', 'websockets', 'paramiko', 'boto3',
        'botocore', 'google', 'azure', 'subprocess',
    ]
    for mod in banned_modules:
        sys.modules[mod] = None

def main():
    try:
        secure_runtime()
        input_data = sys.stdin.read()
        if not input_data:
            raise ValueError("No input received on stdin")
        payload = json.loads(input_data)
        code = payload["code"]
        inp_raw = payload["input"]
        inp = convert_to_numpy(inp_raw)

        local_scope = {
            "np": np, "cv2": cv2, "scipy": scipy,
            "Counter": Counter, "deque": deque, "defaultdict": defaultdict,
            "List": List, "Optional": Optional, "Tuple": Tuple,
            "Any": Any, "Dict": Dict, "Set": Set,
            "copy": copy.copy, "deepcopy": copy.deepcopy,
            "gcd": math.gcd, "math": math, "itertools": itertools,
            "Grid": List[List[int]],
        }
        exec(code, local_scope)
        if "solver" not in local_scope:
            raise RuntimeError("No 'solver' function defined in code.")
        solver = local_scope["solver"]
        if not callable(solver):
            raise RuntimeError("'solver' is not callable.")
        raw_out = solver(inp)
        out = sanitize_output(raw_out)
        json.dump({"ok": True, "output": out}, sys.stdout)
    except Exception as e:
        json.dump({"ok": False, "error": f"{type(e).__name__}: {str(e)}"}, sys.stdout)

if __name__ == "__main__":
    main()
"""


def run_solver_in_sandbox(
    code: str,
    input_grid: list,
    timeout_s: float = 10.0,
) -> list | None:
    """
    Run solver code in a sandboxed subprocess.
    Returns the output grid (list of lists) or None on failure.
    """
    payload = {"code": code, "input": input_grid}

    # Write driver to temp file
    driver_fd, driver_path = tempfile.mkstemp(suffix=".py", prefix="arc_sandbox_")
    try:
        with os.fdopen(driver_fd, "w", encoding="utf-8") as f:
            f.write(_SANDBOX_DRIVER)

        # Platform-specific subprocess kwargs
        kwargs = {}
        if platform.system() == "Windows":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["preexec_fn"] = os.setsid

        p = subprocess.Popen(
            [sys.executable, "-u", driver_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            **kwargs,
        )

        try:
            stdout_data, stderr_data = p.communicate(
                input=json.dumps(payload), timeout=timeout_s
            )
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
            return None

        if p.returncode != 0 or not stdout_data.strip():
            return None

        try:
            result = json.loads(stdout_data)
        except json.JSONDecodeError:
            return None

        if result.get("ok"):
            out = result["output"]
            if isinstance(out, list) and len(out) > 0 and isinstance(out[0], list):
                return out
        return None

    except Exception:
        return None
    finally:
        try:
            os.remove(driver_path)
        except OSError:
            pass


# =============================================================================
# Image Generation
# =============================================================================

ARC_COLORS = [
    '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
    '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25',
]


def generate_arc_image(
    train_pairs: list[dict],
    test_input: Grid,
) -> str | None:
    """
    Generate a base64-encoded PNG image showing ARC training pairs.
    Returns base64 string or None if matplotlib is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import colors as mcolors
        import numpy as np
    except ImportError:
        logger.warning("matplotlib/numpy not available, skipping image generation")
        return None

    cmap = mcolors.ListedColormap(ARC_COLORS)
    norm = mcolors.BoundaryNorm(list(range(11)), cmap.N)

    num_pairs = len(train_pairs)
    if num_pairs == 0:
        return None

    # Calculate figure dimensions
    cell_px = 15
    dpi = 100

    grids = []
    for pair in train_pairs:
        inp = np.array(pair["input"])
        out = np.array(pair["output"])
        grids.append((inp, out))

    height_ratios = [max(i.shape[0], o.shape[0]) for i, o in grids]
    total_h = sum(height_ratios)
    max_w = max(i.shape[1] + o.shape[1] for i, o in grids)

    fig_h = total_h * cell_px * 1.2 / dpi
    fig_w = max_w * cell_px * 1.2 / dpi

    fig, axes = plt.subplots(num_pairs, 2, figsize=(max(fig_w, 4), max(fig_h, 3)),
                              gridspec_kw={"height_ratios": height_ratios})
    fig.patch.set_facecolor('#F8F8F4')

    if num_pairs == 1:
        axes = [axes]

    for i, (inp, out) in enumerate(grids):
        ax_in = axes[i][0] if num_pairs > 1 else axes[0]
        ax_out = axes[i][1] if num_pairs > 1 else axes[1]

        ax_in.imshow(inp, cmap=cmap, norm=norm, interpolation='nearest', aspect='equal')
        ax_in.set_xticks([])
        ax_in.set_yticks([])
        ax_in.set_title(f"Input {i+1}", fontsize=10, fontweight='bold')

        ax_out.imshow(out, cmap=cmap, norm=norm, interpolation='nearest', aspect='equal')
        ax_out.set_xticks([])
        ax_out.set_yticks([])
        ax_out.set_title(f"Output {i+1}", fontsize=10, fontweight='bold')

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# =============================================================================
# Duo Pick Judge Prompt
# =============================================================================

def build_duo_pick_prompt(
    train_pairs: list[dict],
    test_input: Grid,
    candidates: list[dict],
    reasoning_store: dict[str, str],
    total_attempts: int,
) -> str:
    """
    Build prompt for Duo Pick Judge (meta-conclusion from council of 3).

    candidates: list of {
        "id": int, "grid": Grid, "models": [run_ids], "count": int
    }
    """
    parts = []

    # Header
    parts.append(f"Below is a prompt that was run {total_attempts} times:")

    # Base prompt (shows the task context)
    base_prompt = build_prompt(train_pairs, test_input)
    parts.append("\n<PROMPT START>")
    parts.append(base_prompt)
    parts.append("<PROMPT STOP>\n")

    # Solution count
    total_solutions = sum(len(c.get("models", [])) for c in candidates)
    parts.append(
        f"Solutions were generated {total_solutions} times, "
        "using different types of solvers. All solutions are represented below:\n"
    )

    # Each solution with its reasoning + predicted grid
    solution_index = 1
    for cand in candidates:
        grid_csv = format_grid(cand["grid"])
        for model_id in cand.get("models", []):
            parts.append(f"<SOLUTION {solution_index} START>")
            content = reasoning_store.get(model_id, "(Reasoning not found)")
            # If codegen, try to extract just the solver function
            if "def solver" in content:
                match = re.search(
                    r"(def solver\(.*?\):.*?\n\s+return\s+.*)",
                    content, re.DOTALL
                )
                if match:
                    content = match.group(1)
            parts.append("<CONTENT>")
            parts.append(content)
            parts.append("</CONTENT>")
            parts.append("<PREDICTED_GRID>")
            parts.append(grid_csv)
            parts.append("</PREDICTED_GRID>")
            parts.append(f"<SOLUTION {solution_index} STOP>\n")
            solution_index += 1

    # Closing instructions
    parts.append(
        "Your task is to understand these solutions, and assess how well they've "
        "understood the problem, and how likely their solutions are to provide "
        "the correct solution to the test input.\n"
        "Often, new mechanics are introduced in the test example for which the "
        "solutions do not generalize well. Please output two solutions that you "
        "think represent the right mechanic for solving the problem.\n"
        "Output your two solutions as grids (in code blocks). Format the grids as "
        "comma-separated values (CSV) with each row on a new line, like this:\n"
        "```\n7,0,0,7\n0,7,7,0\n```\n"
        "Explain how you came to these two solutions being the two most likely. "
        "In coming up with your two solutions, study all the provided solutions "
        "and their reasoning to come up with a meta-conclusion about how to solve "
        "the problem."
    )

    return "\n".join(parts)


# =============================================================================
# Model Configuration
# =============================================================================
# Unified config: each key is a role, each value is a list of endpoint keys
# from configs/endpoints.py. None = use the default model from the -m flag.
#
# Key present = role active. Missing key = role disabled.
# Multiple endpoints per role = parallel diversity (one actor per endpoint).
# Tuple entry (endpoint, count) = spawn N parallel instances of that endpoint.
#
# Available endpoints: See configs/endpoints.py
#   e.g., "sonnet", "opus", "gemini-2.5-flash", "gpt-4.1", "qwen3-235b-i"
# =============================================================================

ConfigEntry = str | None | tuple[str | None, int]
ModelConfig = dict[str, list[ConfigEntry]]


def _parse_entry(entry: ConfigEntry) -> tuple[str | None, int]:
    """Parse a MODEL_CONFIG entry into (endpoint, count)."""
    if isinstance(entry, (list, tuple)):
        return entry[0], entry[1]
    return entry, 1


def _ep_suffix(endpoint: str | None) -> str:
    """Return endpoint suffix for naming. Empty string if None."""
    return f"_{endpoint}" if endpoint else ""


def _actor_id_for(role: str, endpoint: str | None) -> str:
    """Generate actor ID from role and endpoint key."""
    return f"{role}{_ep_suffix(endpoint)}"


MODEL_CONFIG: ModelConfig = {
    "shallow":  [None],                # None = use -m model
    "deep":     [None],
    "codegen":  [None],
    "image":    ["qwen3-vl-235b-i"],   # must be a vision model
    "duo_pick": [None],
    # "judge":  [None],                # logic + consistency judges (fallback)
    # "objects": [None],               # 3-actor objects pipeline
    # "hint":   [None],               # hint extraction + solve
}


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

    Each model gets its own env instance so Registry routes to the right agent:
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

    def __init__(self, actor_id: str = "hint", env_name: str = "hint_extractor", **kwargs):
        super().__init__(actor_id=actor_id, env_name=env_name, **kwargs)

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

    # Roles indexed by turn order
    _ROLES = ("extractor", "transformer", "pipeline_solver")

    def __init__(
        self,
        actor_ids: list[str] | None = None,
        env_name: str | None = None,
        **kwargs,
    ):
        ids = actor_ids or ["extractor", "transformer", "pipeline_solver"]
        self._extractor_id = ids[0]
        self._transformer_id = ids[1]
        self._solver_id = ids[2]
        self.actors = ids
        self.name = env_name or "objects_pipeline"
        # Map actor_id → role for prompt/on_turn routing
        self._role_of = {ids[0]: "extractor", ids[1]: "transformer", ids[2]: "pipeline_solver"}
        self._next_of = {ids[0]: ids[1], ids[1]: ids[2]}
        super().__init__(max_turns=3, **kwargs)

    def get_initial_actor(self, state: State) -> str:
        return self._extractor_id

    def get_next_actor(self, state: State) -> str:
        current = state["extras"].get("current_actor_id")
        next_actor = self._next_of.get(current)
        if next_actor is None:
            logger.warning(f"ObjectsPipeline: unexpected actor '{current}', falling back to solver")
            return self._solver_id
        return next_actor

    async def build_actor_prompt(self, actor_id: str, state: State) -> Messages:
        actor = self.get_actor(actor_id)
        train_pairs = _get_train_pairs(state)
        test_input = _get_test_input(state)
        role = self._role_of.get(actor_id, "pipeline_solver")

        messages: Messages = []
        if actor.system_prompt:
            messages.append({"role": "system", "content": actor.system_prompt})

        if role == "extractor":
            prompt_text = build_objects_extraction_prompt(train_pairs, test_input)
            messages.append({"role": "user", "content": prompt_text})

        elif role == "transformer":
            objects_text = state["extras"].get("objects_summary", "")
            prompt_text = build_objects_transformation_prompt(
                train_pairs, test_input, objects_text
            )
            messages.append({"role": "user", "content": prompt_text})

        elif role == "pipeline_solver":
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
        role = self._role_of.get(current, "pipeline_solver")
        response_text = _get_last_response_text(state)
        logger.info(f"ObjectsPipeline [{role}]: response length = {len(response_text)} chars")

        if role == "extractor":
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

        elif role == "transformer":
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

        elif role == "pipeline_solver":
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

    def __init__(self, actor_id: str = "judge", env_name: str | None = None, **kwargs):
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

    def __init__(self, actor_id: str = "judge", **kwargs):
        super().__init__(actor_id=actor_id, env_name=f"logic_{actor_id}", **kwargs)


class ConsistencyJudgeEnv(JudgeEnv):
    """Consistency Judge - internal coherence checking."""

    def __init__(self, actor_id: str = "judge", **kwargs):
        super().__init__(actor_id=actor_id, env_name=f"consistency_{actor_id}", **kwargs)


# =============================================================================
# CodegenEnv
# =============================================================================

class CodegenEnv(MultiAgentEnv):
    """
    Codegen environment: model writes a solver() function, which is
    verified against all training pairs in a sandbox, then run on test input.

    Turn 1: Send codegen prompt, get response with Python code.
    on_turn_complete: Extract code → verify on training → run on test.
    """

    def __init__(
        self,
        actor_id: str = "orchestrator",
        codegen_version: str = "v1b",
        env_name: str | None = None,
        **kwargs,
    ):
        self._actor_id = actor_id
        self._codegen_version = codegen_version
        self.actors = [actor_id]
        self.name = env_name or f"codegen_{codegen_version}_{actor_id}"
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

        # Use the codegen prompt passed in via input
        for msg in state.get("prompt", []):
            if msg.get("role") != "system":
                messages.append(msg)

        return messages

    async def on_turn_complete(self, state: State) -> None:
        """Extract solver code, verify on training pairs, run on test input."""
        response_text = _get_last_response_text(state)
        state["extras"]["full_response"] = response_text

        if not response_text:
            state["extras"]["predicted_grid"] = None
            state["extras"]["is_correct"] = False
            return

        # Extract code
        code = _extract_solver_code(response_text)
        if not code:
            logger.warning(f"CodegenEnv [{self.name}]: no solver code found in response")
            state["extras"]["predicted_grid"] = None
            state["extras"]["is_correct"] = False
            state["extras"]["codegen_status"] = "NO_CODE"
            return

        state["extras"]["extracted_code"] = code

        # Verify against all training pairs
        train_pairs = _get_train_pairs(state)
        for i, pair in enumerate(train_pairs):
            result = run_solver_in_sandbox(code, pair["input"], timeout_s=10.0)
            if result is None:
                logger.info(f"CodegenEnv [{self.name}]: sandbox failed on training pair {i}")
                state["extras"]["predicted_grid"] = None
                state["extras"]["is_correct"] = False
                state["extras"]["codegen_status"] = f"TRAIN_FAIL_{i}"
                return
            if result != pair["output"]:
                logger.info(f"CodegenEnv [{self.name}]: wrong output on training pair {i}")
                state["extras"]["predicted_grid"] = None
                state["extras"]["is_correct"] = False
                state["extras"]["codegen_status"] = f"TRAIN_MISMATCH_{i}"
                return

        # All training pairs passed! Run on test input.
        test_input = _get_test_input(state)
        predicted = run_solver_in_sandbox(code, test_input, timeout_s=10.0)

        if predicted is None:
            logger.info(f"CodegenEnv [{self.name}]: sandbox failed on test input")
            state["extras"]["predicted_grid"] = None
            state["extras"]["is_correct"] = False
            state["extras"]["codegen_status"] = "TEST_FAIL"
            return

        state["extras"]["predicted_grid"] = predicted
        state["extras"]["codegen_status"] = "SUCCESS"

        expected = _get_test_output(state)
        if expected is not None:
            state["extras"]["is_correct"] = grids_equal(predicted, expected)
            state["extras"]["similarity"] = grid_similarity(predicted, expected)
        else:
            state["extras"]["is_correct"] = None

        print(f"CodegenEnv [{self.name}]: SUCCESS - training verified, test grid produced")

    @vf.stop
    async def codegen_done(self, state: State) -> bool:
        return len(state.get("trajectory", [])) >= 1


# =============================================================================
# ImageSolverEnv
# =============================================================================

class ImageSolverEnv(MultiAgentEnv):
    """
    Image solver: sends standard ARC prompt + a rendered image of training pairs
    as a multimodal content block.
    """

    def __init__(self, actor_id: str = "image", env_name: str | None = None, **kwargs):
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
        train_pairs = _get_train_pairs(state)
        test_input = _get_test_input(state)

        messages: Messages = []
        if actor.system_prompt:
            messages.append({"role": "system", "content": actor.system_prompt})

        # Build text prompt
        prompt_text = build_prompt(train_pairs, test_input, image_path="attached")

        # Generate image
        image_b64 = generate_arc_image(train_pairs, test_input)

        if image_b64:
            # Multimodal message: text + image
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                        },
                    },
                ],
            })
        else:
            # Fallback: text only
            messages.append({"role": "user", "content": prompt_text})

        return messages

    async def on_turn_complete(self, state: State) -> None:
        """Parse the grid from the response."""
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
    async def image_done(self, state: State) -> bool:
        return len(state.get("trajectory", [])) >= 1


# =============================================================================
# DuoPickJudgeEnv
# =============================================================================

class DuoPickJudgeEnv(MultiAgentEnv):
    """
    Duo Pick Judge: sees all candidates + reasoning, picks 2 best grids.
    Spawned 3x in parallel as a "council of judges".
    """

    actors = ["duo_pick"]
    name = "duo_pick"

    def __init__(self, actor_id: str = "duo_pick", **kwargs):
        self._actor_id = actor_id
        self.actors = [actor_id]
        self.name = actor_id
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
        for msg in state.get("prompt", []):
            if msg.get("role") != "system":
                messages.append(msg)
        return messages

    async def on_turn_complete(self, state: State) -> None:
        """Extract the two grid picks from the judge's response."""
        response_text = _get_last_response_text(state)
        state["extras"]["full_response"] = response_text

        grids = extract_all_grids(response_text)

        # Deduplicate
        unique_grids = []
        for g in grids:
            if g not in unique_grids:
                unique_grids.append(g)

        # Take the last 2 unique grids (judge's final picks)
        if len(unique_grids) >= 2:
            state["extras"]["picked_grids"] = unique_grids[-2:]
        elif len(unique_grids) == 1:
            state["extras"]["picked_grids"] = unique_grids
        else:
            state["extras"]["picked_grids"] = []

    @vf.stop
    async def judge_done(self, state: State) -> bool:
        return len(state.get("trajectory", [])) >= 1


def score_duo_pick_results(
    judge_states: list[State],
    candidates: dict[tuple, dict],
) -> list[dict]:
    """
    Score duo pick judge results and return top 2 solutions.

    Scoring: 1st pick = 2 points, 2nd pick = 1 point.
    Aggregated across all judges. Top 2 by total points.
    """
    scoreboard: dict[tuple, dict] = {}  # grid_tuple -> {points, grid, ...}

    # Build candidate lookup
    cand_list = []
    for idx, (grid_tuple, val) in enumerate(candidates.items()):
        cand_list.append({"id": idx, "grid_tuple": grid_tuple, "val": val})

    for state in judge_states:
        picked = state.get("extras", {}).get("picked_grids", [])
        for i, grid in enumerate(picked):
            points = 2 if i == 0 else 1
            grid_tuple = tuple(tuple(row) for row in grid)

            if grid_tuple not in scoreboard:
                # Check if matches existing candidate
                match_id = None
                for cand in cand_list:
                    if cand["grid_tuple"] == grid_tuple:
                        match_id = cand["id"]
                        break

                scoreboard[grid_tuple] = {
                    "points": 0,
                    "grid": grid,
                    "is_existing": match_id is not None,
                    "match_id": match_id,
                    "is_correct": (
                        cand_list[match_id]["val"].get("is_correct")
                        if match_id is not None else None
                    ),
                }

            scoreboard[grid_tuple]["points"] += points

    # Sort by points descending
    sorted_entries = sorted(
        scoreboard.items(), key=lambda x: x[1]["points"], reverse=True
    )

    # Return top 2
    results = []
    for grid_tuple, entry in sorted_entries[:2]:
        results.append({
            "grid": entry["grid"],
            "points": entry["points"],
            "is_existing": entry["is_existing"],
            "is_correct": entry["is_correct"],
        })

    return results


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
    registry: Registry,
    candidates: dict[tuple, dict],
    reasoning_store: dict[str, str],
    train_pairs: list[dict],
    test_input: Grid,
    client: Any,
    default_model: str,
    judge_endpoint: str | None = None,
) -> dict[int, float]:
    """
    Run Logic and Consistency judges on candidates.

    Returns dict mapping candidate_id -> max(logic_score, consistency_score)
    """
    if not candidates:
        return {}

    judge_aid = _actor_id_for("judge", judge_endpoint)

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
            "task": f"logic_{judge_aid}",
            "prompt": [{"role": "user", "content": logic_prompt}],
            "answer": "",
            "info": {},
        },
        {
            "task": f"consistency_{judge_aid}",
            "prompt": [{"role": "user", "content": consistency_prompt}],
            "answer": "",
            "info": {},
        },
    ]

    judge_states = await registry.spawn(
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

    From ARC solver: >25% of votes AND >=11 votes AND all others have exactly 1.
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

    if not (percentage > 0.25 and max_count >= 11):
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
    Orchestrates multi-strategy solving with step-based execution:

    Step 1 — Shallow Search: orchestrator + codegen + image + extra solvers
    Step 2 — Consensus Check: skip to finalize if strong agreement
    Steps 3-4 — Extended Search + Consensus (disabled by default)
    Step 5 — Full Search: deep thinking + more codegen + more image
    Finalize — Duo pick (default) or logic/consistency judges
    """

    name = "arc_pipeline"
    actors = ["orchestrator"]

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        enable_steps_3_4: bool = False,
        **kwargs,
    ):
        self.model_config = model_config or MODEL_CONFIG
        self.enable_steps_3_4 = enable_steps_3_4
        super().__init__(max_turns=1, **kwargs)

    def _has_role(self, role: str) -> bool:
        """Check if a role is active in model_config (key present + non-empty list)."""
        return bool(self.model_config.get(role))

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
        """Step-based pipeline: Step 1 → 2 → (3-4) → 5 → Finalize."""
        train_pairs = _get_train_pairs(state)
        test_input = _get_test_input(state)
        expected = _get_test_output(state)
        answer_str = state.get("answer", "")
        info_dict = state.get("info", {})
        idx = state.get("example_id", 0)
        mc = self.model_config

        candidates: dict[tuple, dict] = {}
        reasoning_store: dict[str, str] = {}

        # === STEP 1 — Shallow Search ===
        print(f"\n{'='*60}\nSTEP 1 — Shallow Search\n{'='*60}")

        # Parse orchestrator's initial solve (turn 1)
        response_text = _get_last_response_text(state)
        if response_text:
            try:
                grid = parse_grid_from_text(response_text)
                grid_tuple = tuple(tuple(row) for row in grid)
                candidates[grid_tuple] = {
                    "grid": grid, "count": 1, "models": ["orchestrator"],
                    "is_correct": grids_equal(grid, expected) if expected else None,
                }
                reasoning_store["orchestrator"] = response_text
            except ValueError:
                logger.warning("Failed to parse orchestrator's grid response")

        step1_inputs = []

        # Extra shallow solvers (2nd endpoint onward)
        shallow_entries = mc.get("shallow", [None])
        if len(shallow_entries) > 1:
            basic_prompt = build_prompt(train_pairs, test_input)
            for entry in shallow_entries[1:]:
                ep, count = _parse_entry(entry)
                aid = _actor_id_for("shallow", ep)
                for _ in range(count):
                    step1_inputs.append({
                        "task": aid,
                        "prompt": [{"role": "user", "content": basic_prompt}],
                        "answer": answer_str, "example_id": idx, "info": info_dict,
                    })

        # Codegen (v1b + v4 per endpoint)
        if self._has_role("codegen"):
            for entry in mc["codegen"]:
                ep, count = _parse_entry(entry)
                aid = _actor_id_for("codegen", ep)
                for version in ["v1b", "v4"]:
                    cg_prompt = build_codegen_prompt(train_pairs, test_input, version=version)
                    for _ in range(count):
                        step1_inputs.append({
                            "task": f"codegen_{version}_{aid}",
                            "prompt": [{"role": "user", "content": cg_prompt}],
                            "answer": answer_str, "example_id": idx, "info": info_dict,
                        })

        # Image solver (per endpoint)
        if self._has_role("image"):
            for entry in mc["image"]:
                ep, count = _parse_entry(entry)
                aid = _actor_id_for("image", ep)
                for _ in range(count):
                    step1_inputs.append({
                        "task": aid,
                        "prompt": [{"role": "user", "content": ""}],
                        "answer": answer_str, "example_id": idx, "info": info_dict,
                    })

        # Objects pipeline (per endpoint)
        if self._has_role("objects"):
            for entry in mc["objects"]:
                ep, count = _parse_entry(entry)
                for _ in range(count):
                    step1_inputs.append({
                        "task": f"objects_pipeline{_ep_suffix(ep)}",
                        "prompt": [{"role": "user", "content": ""}],
                        "answer": answer_str, "example_id": idx, "info": info_dict,
                    })

        if step1_inputs:
            step1_states = await self.registry.spawn(
                inputs=step1_inputs, client=state["client"],
                model=state["model"], sampling_args=state.get("sampling_args"),
                score=False,
            )
            collect_candidates(candidates, reasoning_store, step1_states)
            for s in step1_states:
                state["child_states"].append(s)

        # Hint (two-phase: extract then solve)
        if self._has_role("hint"):
            hint_entries = mc["hint"]
            first_hint_ep, _ = _parse_entry(hint_entries[0])
            hint_prompt = build_hint_extraction_prompt(train_pairs, test_input)
            hint_states = await self.registry.spawn(
                inputs=[{
                    "task": f"hint_extractor{_ep_suffix(first_hint_ep)}",
                    "prompt": [{"role": "user", "content": hint_prompt}],
                    "answer": "", "example_id": idx, "info": info_dict,
                }],
                client=state["client"], model=state["model"], score=False,
            )
            hint_text = (hint_states[0].get("extras", {}).get("hint") if hint_states else None)
            if hint_text:
                hint_solve_prompt = build_prompt(train_pairs, test_input, strategy=hint_text)
                hint_inputs = []
                for entry in hint_entries:
                    ep, count = _parse_entry(entry)
                    aid = _actor_id_for("hint", ep)
                    for _ in range(count):
                        hint_inputs.append({
                            "task": f"hint_{aid}",
                            "prompt": [{"role": "user", "content": hint_solve_prompt}],
                            "answer": answer_str, "example_id": idx, "info": info_dict,
                        })
                hint_solve_states = await self.registry.spawn(
                    inputs=hint_inputs, client=state["client"],
                    model=state["model"], score=False,
                )
                collect_candidates(candidates, reasoning_store, hint_solve_states)
                for s in hint_solve_states:
                    state["child_states"].append(s)

        print(f"  Step 1: {len(candidates)} candidates, "
              f"{sum(v['count'] for v in candidates.values())} total votes")

        # === STEP 2 — Consensus Check ===
        print(f"\n{'='*60}\nSTEP 2 — Consensus Check\n{'='*60}")
        if is_solved(candidates):
            print("  CONSENSUS REACHED — skipping to finalize")
            await self._finalize(state, candidates, reasoning_store,
                                 train_pairs, test_input, expected, True)
            return
        print("  No consensus — continuing")

        # === STEPS 3-4 — Extended Search (disabled by default) ===
        if self.enable_steps_3_4:
            print(f"\n{'='*60}\nSTEPS 3-4 — Extended Search\n{'='*60}")
            step3_inputs = []

            # Re-run codegen + image
            if self._has_role("codegen"):
                for entry in mc["codegen"]:
                    ep, count = _parse_entry(entry)
                    aid = _actor_id_for("codegen", ep)
                    for version in ["v1b", "v4"]:
                        cg_prompt = build_codegen_prompt(train_pairs, test_input, version=version)
                        for _ in range(count):
                            step3_inputs.append({
                                "task": f"codegen_{version}_{aid}",
                                "prompt": [{"role": "user", "content": cg_prompt}],
                                "answer": answer_str, "example_id": idx, "info": info_dict,
                            })
            if self._has_role("image"):
                for entry in mc["image"]:
                    ep, count = _parse_entry(entry)
                    aid = _actor_id_for("image", ep)
                    for _ in range(count):
                        step3_inputs.append({
                            "task": aid,
                            "prompt": [{"role": "user", "content": ""}],
                            "answer": answer_str, "example_id": idx, "info": info_dict,
                        })

            if step3_inputs:
                step3_states = await self.registry.spawn(
                    inputs=step3_inputs, client=state["client"],
                    model=state["model"], sampling_args=state.get("sampling_args"),
                    score=False,
                )
                collect_candidates(candidates, reasoning_store, step3_states)
                for s in step3_states:
                    state["child_states"].append(s)

            # Step 4 — Consensus Check
            print(f"  Steps 3-4: {len(candidates)} candidates")
            if is_solved(candidates):
                print("  CONSENSUS REACHED after Step 4 — skipping to finalize")
                await self._finalize(state, candidates, reasoning_store,
                                     train_pairs, test_input, expected, True)
                return

        # === STEP 5 — Full Search ===
        print(f"\n{'='*60}\nSTEP 5 — Full Search\n{'='*60}")
        step5_inputs = []

        # Deep thinking (per deep endpoint)
        if self._has_role("deep"):
            deep_prompt = build_prompt(train_pairs, test_input, trigger_deep_thinking=True)
            for entry in mc["deep"]:
                ep, count = _parse_entry(entry)
                aid = _actor_id_for("deep", ep)
                for _ in range(count):
                    step5_inputs.append({
                        "task": f"deep_{aid}",
                        "prompt": [{"role": "user", "content": deep_prompt}],
                        "answer": answer_str, "example_id": idx, "info": info_dict,
                    })

        # More codegen
        if self._has_role("codegen"):
            for entry in mc["codegen"]:
                ep, count = _parse_entry(entry)
                aid = _actor_id_for("codegen", ep)
                for version in ["v1b", "v4"]:
                    cg_prompt = build_codegen_prompt(train_pairs, test_input, version=version)
                    for _ in range(count):
                        step5_inputs.append({
                            "task": f"codegen_{version}_{aid}",
                            "prompt": [{"role": "user", "content": cg_prompt}],
                            "answer": answer_str, "example_id": idx, "info": info_dict,
                        })

        # More image
        if self._has_role("image"):
            for entry in mc["image"]:
                ep, count = _parse_entry(entry)
                aid = _actor_id_for("image", ep)
                for _ in range(count):
                    step5_inputs.append({
                        "task": aid,
                        "prompt": [{"role": "user", "content": ""}],
                        "answer": answer_str, "example_id": idx, "info": info_dict,
                    })

        if step5_inputs:
            step5_states = await self.registry.spawn(
                inputs=step5_inputs, client=state["client"],
                model=state["model"], sampling_args=state.get("sampling_args"),
                score=False,
            )
            collect_candidates(candidates, reasoning_store, step5_states)
            for s in step5_states:
                state["child_states"].append(s)

        print(f"  Step 5: {len(candidates)} candidates, "
              f"{sum(v['count'] for v in candidates.values())} total votes")

        # === FINALIZE ===
        await self._finalize(state, candidates, reasoning_store,
                             train_pairs, test_input, expected, is_solved(candidates))

    async def _finalize(
        self,
        state: State,
        candidates: dict[tuple, dict],
        reasoning_store: dict[str, str],
        train_pairs: list[dict],
        test_input: Grid,
        expected: Grid | None,
        consensus: bool,
    ) -> None:
        """Run judges and pick best solution."""
        print(f"\n{'='*60}\nFINALIZE — Pick Solution\n{'='*60}")

        picks = []

        # Duo Pick: council of 3 judges (default)
        if self._has_role("duo_pick") and len(candidates) > 0:
            # Use first duo_pick endpoint for the council
            duo_ep, duo_count = _parse_entry(self.model_config["duo_pick"][0])
            council_size = max(duo_count, 3)  # minimum 3 for council voting
            duo_aid = _actor_id_for("duo_pick", duo_ep)
            total_attempts = sum(v["count"] for v in candidates.values())
            cand_list = _build_candidates_for_judge(candidates, reasoning_store)
            duo_prompt = build_duo_pick_prompt(
                train_pairs, test_input, cand_list, reasoning_store, total_attempts,
            )
            duo_inputs = [
                {"task": duo_aid, "prompt": [{"role": "user", "content": duo_prompt}],
                 "answer": "", "info": {}}
                for _ in range(council_size)
            ]
            duo_states = await self.registry.spawn(
                inputs=duo_inputs, client=state["client"],
                model=state["model"], score=False,
            )
            duo_picks = score_duo_pick_results(duo_states, candidates)
            if duo_picks:
                for dp in duo_picks:
                    grid = dp["grid"]
                    grid_tuple = tuple(tuple(row) for row in grid)
                    is_correct = dp.get("is_correct")
                    if is_correct is None and expected is not None:
                        is_correct = grids_equal(grid, expected)
                    picks.append({
                        "grid": grid, "is_correct": is_correct,
                        "count": candidates.get(grid_tuple, {}).get("count", 0),
                        "models": candidates.get(grid_tuple, {}).get("models", ["duo_pick_synth"]),
                        "duo_pick_points": dp["points"],
                    })
                print(f"  Duo pick: {len(duo_picks)} solutions selected")

        # Fallback: logic/consistency judges
        if not picks and self._has_role("judge") and len(candidates) > 1:
            judge_ep, _ = _parse_entry(self.model_config["judge"][0])
            judge_scores = await run_judges(
                registry=self.registry, candidates=candidates,
                reasoning_store=reasoning_store, train_pairs=train_pairs,
                test_input=test_input, client=state["client"],
                default_model=state["model"],
                judge_endpoint=judge_ep,
            )
            picks = pick_best_solution(candidates, judge_scores)
            print(f"  Logic/Consistency judges: {len(picks)} picks")

        # Final fallback: consensus only
        if not picks and candidates:
            picks = pick_best_solution(candidates, {})
            print(f"  Consensus fallback: {len(picks)} picks")

        # Store results
        state["extras"]["picks"] = picks
        state["extras"]["num_candidates"] = len(candidates)
        state["extras"]["consensus_solved"] = consensus
        state["extras"]["total_runs"] = sum(v["count"] for v in candidates.values())

        if picks:
            top = picks[0]
            state["extras"]["is_correct"] = bool(top.get("is_correct"))
            state["extras"]["top_grid"] = top.get("grid")
            if len(picks) > 1:
                state["extras"]["is_correct_2"] = bool(picks[1].get("is_correct"))

        state["extras"]["pipeline_done"] = True

        # Debug summary
        print(f"\n{'='*60}\nPIPELINE FINAL STATE:")
        print(f"  num_candidates: {len(candidates)}")
        print(f"  total_runs: {sum(v['count'] for v in candidates.values())}")
        print(f"  picks: {len(picks)}")
        if picks:
            top = picks[0]
            print(f"  top pick is_correct: {top.get('is_correct')}")
            print(f"  2nd pick is_correct: {picks[1].get('is_correct') if len(picks) > 1 else 'N/A'}")
            print(f"  top pick grid: {top.get('grid')}")
        print(f"  expected answer: {expected}")
        for gt, val in candidates.items():
            print(f"  candidate: correct={val.get('is_correct')}, "
                  f"votes={val['count']}, models={val['models']}")
        print(f"{'='*60}\n")

    @vf.stop
    async def pipeline_done(self, state: State) -> bool:
        """Stop after orchestrator turn + spawns complete."""
        return state.get("extras", {}).get("pipeline_done", False)


# =============================================================================
# Rubric
# =============================================================================

def create_arc_rubric(model_config: ModelConfig) -> MultiAgentRubric:
    """
    Create rubric for ARC pipeline.

    Registers reward func for every actor in model_config.
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

    # Collect all actor IDs from config (mirrors create_actors logic)
    actor_ids = {"orchestrator"}

    shallow_entries = model_config.get("shallow", [None])
    for entry in shallow_entries[1:]:
        ep, _ = _parse_entry(entry)
        actor_ids.add(_actor_id_for("shallow", ep))

    for role in ("deep", "codegen", "image", "duo_pick", "judge", "hint"):
        for entry in model_config.get(role, []):
            ep, _ = _parse_entry(entry)
            actor_ids.add(_actor_id_for(role, ep))

    for entry in model_config.get("objects", []):
        ep, _ = _parse_entry(entry)
        suffix = _ep_suffix(ep)
        for sub in ("extractor", "transformer", "pipeline_solver"):
            actor_ids.add(f"{sub}{suffix}")

    # Team reward: every actor gets the same score based on the pipeline's
    # final pick (set by _finalize). Individual actor correctness is visible
    # in debug logs but not scored here. For per-actor rewards (e.g. RL
    # training), register different reward funcs per actor instead.
    for aid in actor_ids:
        rubric.add_actor_reward_func(aid, orchestrator_reward, weight=1.0)

    # Metrics only on orchestrator (pipeline-level stats)
    rubric.add_actor_metric("orchestrator", num_candidates_metric)
    rubric.add_actor_metric("orchestrator", consensus_metric)
    rubric.add_actor_metric("orchestrator", total_runs_metric)

    return rubric


# =============================================================================
# Agent + Env Factory
# =============================================================================

def create_agents(model_config: ModelConfig) -> list[Agent]:
    """
    Create agents for ARC pipeline from MODEL_CONFIG.

    One agent per (role, endpoint) pair. The first "shallow" endpoint
    becomes the "orchestrator" (does Turn 1 in the pipeline).
    """
    agents = []
    seen_ids: set[str] = set()

    def add_agent(agent_id: str, endpoint: str | None) -> None:
        if agent_id in seen_ids:
            return
        seen_ids.add(agent_id)
        client, model = get_actor_client(endpoint)
        agents.append(Agent(
            id=agent_id, model=model, client=client,
            max_tokens=16384, is_trainable=False,
        ))

    # Validate: no duplicate endpoints within a role
    for role, entries in model_config.items():
        ep_strs = [str(_parse_entry(e)[0]) for e in entries]
        if len(ep_strs) != len(set(ep_strs)):
            raise ValueError(f"Duplicate endpoints in '{role}': {entries}")

    # Orchestrator: always exists, uses first shallow endpoint (or None)
    shallow_entries = model_config.get("shallow", [None])
    orch_ep, _ = _parse_entry(shallow_entries[0])
    add_agent("orchestrator", orch_ep)

    # Extra shallow solvers (2nd endpoint onward)
    for entry in shallow_entries[1:]:
        ep, _ = _parse_entry(entry)
        add_agent(_actor_id_for("shallow", ep), ep)

    # Standard roles: one agent per endpoint
    for role in ("deep", "codegen", "image", "duo_pick", "judge", "hint"):
        for entry in model_config.get(role, []):
            ep, _ = _parse_entry(entry)
            add_agent(_actor_id_for(role, ep), ep)

    # Objects: 3 sub-agents per endpoint
    for entry in model_config.get("objects", []):
        ep, _ = _parse_entry(entry)
        suffix = _ep_suffix(ep)
        client, model = get_actor_client(ep)
        for sub_role in ("extractor", "transformer", "pipeline_solver"):
            aid = f"{sub_role}{suffix}"
            if aid not in seen_ids:
                seen_ids.add(aid)
                agents.append(Agent(
                    id=aid, model=model, client=client,
                    max_tokens=16384, is_trainable=False,
                ))

    return agents


def create_envs(
    pipeline_env: ArcPipelineEnv,
    model_config: ModelConfig,
) -> list:
    """
    Create all environments for the Registry routing table.

    Each env's name becomes a route: spawn({"task": env.name}) → that env.
    Only pipeline_env has a dataset — the rest are workers that run when spawned.
    Roles present in model_config get their envs created; missing roles are skipped.
    """
    envs: list = [pipeline_env]

    # Extra shallow solvers (2nd endpoint onward in "shallow")
    shallow_entries = model_config.get("shallow", [None])
    for entry in shallow_entries[1:]:
        ep, _ = _parse_entry(entry)
        aid = _actor_id_for("shallow", ep)
        envs.append(SingleSolverEnv(actor_id=aid))

    # Deep thinking: one DeepThinkingEnv per deep endpoint
    for entry in model_config.get("deep", []):
        ep, _ = _parse_entry(entry)
        aid = _actor_id_for("deep", ep)
        envs.append(DeepThinkingEnv(actor_id=aid))

    # Hint: extractor + one HintSolverEnv per hint endpoint
    if "hint" in model_config:
        hint_entries = model_config["hint"]
        # One extractor (uses first hint endpoint actor)
        first_ep, _ = _parse_entry(hint_entries[0])
        first_hint_aid = _actor_id_for("hint", first_ep)
        envs.append(HintExtractorEnv(
            actor_id=first_hint_aid,
            env_name=f"hint_extractor{_ep_suffix(first_ep)}",
        ))
        # One solver per endpoint
        for entry in hint_entries:
            ep, _ = _parse_entry(entry)
            aid = _actor_id_for("hint", ep)
            envs.append(HintSolverEnv(actor_id=aid))

    # Objects: one pipeline per endpoint
    if "objects" in model_config:
        for entry in model_config["objects"]:
            ep, _ = _parse_entry(entry)
            suffix = _ep_suffix(ep)
            envs.append(ObjectsPipelineEnv(
                actor_ids=[f"extractor{suffix}", f"transformer{suffix}", f"pipeline_solver{suffix}"],
                env_name=f"objects_pipeline{suffix}",
            ))

    # Codegen: v1b + v4 per codegen endpoint
    for entry in model_config.get("codegen", []):
        ep, _ = _parse_entry(entry)
        aid = _actor_id_for("codegen", ep)
        for version in ["v1b", "v4"]:
            envs.append(CodegenEnv(
                actor_id=aid, codegen_version=version,
            ))

    # Image: one ImageSolverEnv per image endpoint
    for entry in model_config.get("image", []):
        ep, _ = _parse_entry(entry)
        aid = _actor_id_for("image", ep)
        envs.append(ImageSolverEnv(actor_id=aid))

    # Judges (logic + consistency)
    if "judge" in model_config:
        for entry in model_config["judge"]:
            ep, _ = _parse_entry(entry)
            aid = _actor_id_for("judge", ep)
            envs.append(LogicJudgeEnv(actor_id=aid))
            envs.append(ConsistencyJudgeEnv(actor_id=aid))

    # Duo pick judges
    for entry in model_config.get("duo_pick", []):
        ep, _ = _parse_entry(entry)
        aid = _actor_id_for("duo_pick", ep)
        envs.append(DuoPickJudgeEnv(actor_id=aid))

    return envs


# =============================================================================
# Environment Loader
# =============================================================================

def load_environment(
    hf_dataset: str = HF_DATASET_REPO,
    split: str = "train",
    model_config: ModelConfig | None = None,
    enable_steps_3_4: bool = False,
) -> ArcPipelineEnv:
    """
    Entry point called by vf-eval. Loads data, builds the full pipeline, returns it.

    All args are passable from CLI via -a flag:
        -a '{"model_config": {"shallow": [null], "codegen": [null]}}'

    Args:
        hf_dataset: HuggingFace dataset repo. Default: "bhoy/arc-agi-2".
        split: Dataset split — "train" (1000 tasks) or "eval" (120 tasks).
        model_config: Unified config dict. Keys = active roles, values = endpoint
            lists. None in list = use -m default model. Missing key = role disabled.
            Defaults to MODULE-level MODEL_CONFIG if not provided.
        enable_steps_3_4: Run extended search Steps 3-4 (disabled by default).
    """
    mc = model_config or MODEL_CONFIG

    # 1. Load ARC tasks from HuggingFace → transform into vf-eval format
    raw_dataset = load_arc_dataset_hf(hf_dataset, split=split)
    dataset = prepare_dataset_for_eval(raw_dataset)

    # 2. Scoring: reward for every actor in config
    rubric = create_arc_rubric(mc)

    # 3. Agents: one per (role, endpoint) pair
    agents = create_agents(mc)

    # 4. Pipeline env: the only env with a dataset — framework iterates over it
    pipeline_env = ArcPipelineEnv(
        model_config=mc,
        enable_steps_3_4=enable_steps_3_4,
        rubric=rubric,
        dataset=dataset,
    )

    # 5. Worker envs + Registry: routing table so spawn({"task": name}) works
    envs = create_envs(pipeline_env, mc)
    Registry(agents=agents, envs=envs)

    return pipeline_env
