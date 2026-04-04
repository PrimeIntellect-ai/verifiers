"""Recipe executor for ARC curriculum generation.

Takes JSON recipes from the curriculum generator LoRA and produces
ARC tasks by calling existing generators and applying compositions.

The generator picks a difficulty level (A/B/C → 1/2/3). The seed
comes from the dataset (8 seeds per op). Grid sizes:
    A (level 1) = 5-8   (matches fixed L1 dataset)
    B (level 2) = 8-12
    C (level 3) = 12-16
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import sys
from typing import Any

logger = logging.getLogger(__name__)

Grid = list[list[int]]

# ─── Grid transforms (inlined to avoid cross-directory imports) ──────────────

def _copy_grid(grid: Grid) -> Grid:
    return [row[:] for row in grid]


def _rotate_grid(grid: Grid, degrees: int) -> Grid:
    rows, cols = len(grid), len(grid[0])
    if degrees == 90:
        return [[grid[rows - 1 - j][i] for j in range(rows)] for i in range(cols)]
    if degrees == 180:
        return [[grid[rows - 1 - r][cols - 1 - c] for c in range(cols)] for r in range(rows)]
    if degrees == 270:
        return [[grid[j][cols - 1 - i] for j in range(rows)] for i in range(cols)]
    return _copy_grid(grid)


def _reflect_grid(grid: Grid, axis: str) -> Grid:
    if axis == "horizontal":
        return [row[::-1] for row in grid]
    return grid[::-1]


def _apply_gravity(grid: Grid) -> Grid:
    rows, cols = len(grid), len(grid[0])
    out = [[0] * cols for _ in range(rows)]
    for c in range(cols):
        non_zero = [grid[r][c] for r in range(rows) if grid[r][c] != 0]
        for i, val in enumerate(non_zero):
            out[rows - len(non_zero) + i][c] = val
    return out


# ─── Post-transforms (applied to task outputs for composition) ───────────────

POST_TRANSFORMS: dict[str, Any] = {
    "rotate_90": lambda g: _rotate_grid(g, 90),
    "rotate_180": lambda g: _rotate_grid(g, 180),
    "rotate_270": lambda g: _rotate_grid(g, 270),
    "reflect_h": lambda g: _reflect_grid(g, "horizontal"),
    "reflect_v": lambda g: _reflect_grid(g, "vertical"),
    "gravity": _apply_gravity,
}

POST_OPS = list(POST_TRANSFORMS.keys())

# ─── Base ops (names of all 31 constructive generators) ──────────────────────

L1_OPS = [
    "color_replacement", "simple_rotation", "simple_reflection", "scaling_2x",
    "gravity_drop", "flood_fill", "boolean_and_or", "row_col_duplication",
    "solid_to_hollow", "histogram_rendering", "dimension_based_pattern",
    "connectivity_size_filter", "bounding_rect_fill",
]

L2_OPS = [
    "symmetry_completion", "border_drawing", "template_cloning", "shape_stamping",
    "object_extraction_recolor", "line_extension", "denoising", "interior_fill_multi",
    "crosshair_generation", "object_compaction", "fill_level_encoding",
    "expanding_diamond", "same_color_pair_bridging", "seam_marked_tiling",
    "value_proportional_expansion", "gap_and_beam", "missing_region_reconstruction",
    "noisy_grid_regularization",
]

BASE_OPS = L1_OPS + L2_OPS

ALL_OPS = BASE_OPS + POST_OPS

# ─── Curriculum grid sizes (override for the curriculum env) ─────────────────

def _curriculum_grid_dims(level, rng):
    """Grid sizes for curriculum levels. Monkey-patched onto generators at import."""
    if level == 1:
        return rng.randint(5, 8), rng.randint(5, 8)
    if level == 2:
        return rng.randint(8, 12), rng.randint(8, 12)
    return rng.randint(12, 16), rng.randint(12, 16)


# ─── Generator import ────────────────────────────────────────────────────────

_GENERATORS: dict | None = None


def _load_generators() -> dict:
    global _GENERATORS
    if _GENERATORS is not None:
        return _GENERATORS

    env_dir = os.path.dirname(os.path.abspath(__file__))
    gen_path = os.path.join(env_dir, "generate_arc_synthetic.py")
    if os.path.isfile(gen_path):
        sys.path.insert(0, env_dir)

    scripts_dir = os.path.join(env_dir, "..", "..", "scripts")
    if os.path.isdir(scripts_dir):
        sys.path.insert(0, os.path.abspath(scripts_dir))

    import generate_arc_synthetic
    generate_arc_synthetic.grid_dims_for_level = _curriculum_grid_dims

    _GENERATORS = generate_arc_synthetic.GENERATORS
    return _GENERATORS


# ─── Validation ──────────────────────────────────────────────────────────────

def validate_grid(grid: Grid) -> bool:
    if not isinstance(grid, list) or len(grid) == 0:
        return False
    for row in grid:
        if not isinstance(row, list) or len(row) == 0:
            return False
        for cell in row:
            if not isinstance(cell, int) or cell < 0 or cell > 9:
                return False
    rows, cols = len(grid), len(grid[0])
    return rows <= 30 and cols <= 30


def validate_task(task: dict) -> bool:
    grids = []
    for pair in task.get("train_pairs", []):
        grids.extend([pair.get("input"), pair.get("output")])
    grids.extend([task.get("test_input"), task.get("test_output")])
    return all(validate_grid(g) for g in grids if g is not None)


# ─── Recipe execution ────────────────────────────────────────────────────────

SEEDS = [1337, 42, 7890, 2024, 555, 9001, 314, 8675]


def execute_recipe(recipe: dict) -> dict | None:
    """Execute a recipe to produce an ARC task.

    Returns task dict with train_pairs, test_input, test_output, or None if invalid.
    """
    generators = _load_generators()

    base_op = recipe.get("base_op")
    if not base_op or base_op not in generators:
        return None

    level = recipe.get("level", 2)
    if level not in (1, 2, 3):
        level = 2
    seed = recipe.get("seed", SEEDS[0])
    post_ops = recipe.get("post_ops", [])

    for op in post_ops:
        if op not in POST_TRANSFORMS:
            return None

    gen_fn = generators[base_op][0]
    task = gen_fn(seed, level)

    for op in post_ops:
        fn = POST_TRANSFORMS[op]
        for pair in task["train_pairs"]:
            pair["output"] = fn(pair["output"])
        task["test_output"] = fn(task["test_output"])

    if not validate_task(task):
        return None

    task["recipe"] = recipe
    return task


def _normalize_level(raw) -> int | None:
    """Convert level value (A/B/C or 1/2/3) to integer 1-3."""
    if isinstance(raw, str) and raw.upper() in LEVEL_MAP:
        return LEVEL_MAP[raw.upper()]
    if isinstance(raw, int) and raw in (1, 2, 3):
        return raw
    return None


def parse_generator_output(response_text: str) -> dict | None:
    """Parse the generator LoRA's response text into a recipe dict.

    Accepts {"level": "A"}, {"level": 1}, or bare A/B/C.
    """
    match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(1))
            lvl = _normalize_level(parsed.get("level"))
            if lvl:
                return {"level": lvl}
        except json.JSONDecodeError:
            pass

    match = re.search(r"\{[^{}]*\}", response_text)
    if match:
        try:
            parsed = json.loads(match.group())
            lvl = _normalize_level(parsed.get("level"))
            if lvl:
                return {"level": lvl}
        except json.JSONDecodeError:
            pass

    # Fallback: bare letter or digit
    match = re.search(r"\b([ABCabc])\b", response_text)
    if match:
        return {"level": LEVEL_MAP[match.group(1).upper()]}

    match = re.search(r"\b([123])\b", response_text)
    if match:
        return {"level": int(match.group(1))}

    return None


# ─── Generator reward ────────────────────────────────────────────────────────

def generator_reward(solver_reward: float) -> float:
    """Reward for the curriculum generator.

    Binary: 1.0 if solver gets meaningful signal (reward >= 0.3),
    0.0 if solver can't engage with the task.

    GRPO compares 8 rollouts of the same base_op at different difficulties.
    The generator learns which difficulty settings produce learnable tasks.
    Signal comes from the contrast between solvable and unsolvable recipes
    within the same operation family.
    """
    if solver_reward >= 0.3:
        return 1.0
    return 0.0


# ─── Generator prompt building ───────────────────────────────────────────────

OP_DESCRIPTIONS = {
    "color_replacement": "Replace all cells of one color with another",
    "simple_rotation": "Rotate grid by 90/180/270 degrees",
    "simple_reflection": "Reflect grid horizontally or vertically",
    "scaling_2x": "Scale grid 2x (each cell becomes 2x2 block)",
    "gravity_drop": "Drop non-zero cells downward (gravity)",
    "flood_fill": "Fill enclosed region with a color",
    "boolean_and_or": "Boolean AND/OR of two grid halves",
    "row_col_duplication": "Duplicate specific rows or columns",
    "solid_to_hollow": "Convert solid rectangles to hollow outlines",
    "histogram_rendering": "Render bars proportional to values",
    "dimension_based_pattern": "Pattern based on grid dimensions",
    "connectivity_size_filter": "Filter connected components by size",
    "bounding_rect_fill": "Fill bounding rectangles of objects",
    "symmetry_completion": "Complete a pattern to make it symmetric",
    "border_drawing": "Draw borders around objects",
    "template_cloning": "Clone a template pattern across grid",
    "shape_stamping": "Stamp shapes at marked positions",
    "object_extraction_recolor": "Extract objects and recolor them",
    "line_extension": "Extend lines to grid edges",
    "denoising": "Remove noise from a regular pattern",
    "interior_fill_multi": "Fill interiors of multiple shapes",
    "crosshair_generation": "Generate crosshairs from marked cells",
    "object_compaction": "Compact objects (remove gaps)",
    "fill_level_encoding": "Encode values as fill levels in columns",
    "expanding_diamond": "Expand diamond shapes outward",
    "same_color_pair_bridging": "Bridge pairs of same-colored cells",
    "seam_marked_tiling": "Tile pattern along seam marks",
    "value_proportional_expansion": "Expand regions proportional to value",
    "gap_and_beam": "Fill gaps between beams",
    "missing_region_reconstruction": "Reconstruct missing region from pattern",
    "noisy_grid_regularization": "Regularize a noisy grid pattern",
    "rotate_90": "Rotate output 90 degrees clockwise",
    "rotate_180": "Rotate output 180 degrees",
    "rotate_270": "Rotate output 270 degrees clockwise",
    "reflect_h": "Flip output horizontally",
    "reflect_v": "Flip output vertically",
    "gravity": "Apply gravity (drop cells down) to output",
}


LEVEL_MAP = {"A": 1, "B": 2, "C": 3}


def build_generator_prompt(base_op: str) -> str:
    """Build the prompt for the curriculum generator LoRA.

    Each prompt specifies a single base_op. The generator chooses
    a level label (A, B, or C). Labels are opaque — the model
    must learn which works best through reward signal alone.
    """
    desc = OP_DESCRIPTIONS.get(base_op, base_op)

    lines = [
        f"Choose a setting for the {base_op} operation.",
        f"{base_op}: {desc}",
        "",
        'Output one of: {"level": "A"}, {"level": "B"}, or {"level": "C"}',
    ]

    return "\n".join(lines)
