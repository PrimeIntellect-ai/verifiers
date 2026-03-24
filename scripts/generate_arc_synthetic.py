"""Generate synthetic ARC-AGI-2 dataset and push to HuggingFace.

Level 1: 13 single-operation categories on small grids (5x5 to 8x8).

Usage:
    python scripts/generate_arc_synthetic.py
"""

import json
import random
from collections import deque

from datasets import Dataset


HF_REPO = "bhoy/arc-synthetic"
TASKS_PER_CATEGORY = 20

Grid = list[list[int]]
ARC_COLORS = list(range(1, 10))  # 1-9 (0 is background)


# =============================================================================
# Shared Utilities
# =============================================================================

def random_grid(rng: random.Random, rows: int, cols: int, colors: list[int],
                include_zero: bool = True) -> Grid:
    palette = ([0] + colors) if include_zero else colors
    return [[rng.choice(palette) for _ in range(cols)] for _ in range(rows)]


def random_sparse_grid(rng: random.Random, rows: int, cols: int,
                       colors: list[int], density: float = 0.3) -> Grid:
    grid = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if rng.random() < density:
                grid[r][c] = rng.choice(colors)
    return grid


def copy_grid(grid: Grid) -> Grid:
    return [row[:] for row in grid]


def pick_n_colors(rng: random.Random, n: int) -> list[int]:
    return rng.sample(ARC_COLORS, n)


def grid_dims_for_level(level: int, rng: random.Random) -> tuple[int, int]:
    if level == 1:
        return rng.randint(5, 8), rng.randint(5, 8)
    if level == 2:
        return rng.randint(8, 12), rng.randint(8, 12)
    return rng.randint(10, 15), rng.randint(10, 15)


def make_task(task_seed: int, level: int, rule_setup, pair_fn) -> dict:
    """Shared task generation pattern.

    rule_setup(rng, level) -> rule_params dict
    pair_fn(pair_rng, **rule_params) -> (input_grid, output_grid)
    """
    rng = random.Random(task_seed)
    params = rule_setup(rng, level)
    num_pairs = rng.randint(3, 4)

    train_pairs = []
    for i in range(num_pairs):
        pair_rng = random.Random(task_seed * 1000 + i)
        inp, out = pair_fn(pair_rng, **params)
        train_pairs.append({"input": inp, "output": out})

    test_rng = random.Random(task_seed * 1000 + num_pairs)
    test_inp, test_out = pair_fn(test_rng, **params)
    return {"train_pairs": train_pairs, "test_input": test_inp, "test_output": test_out}


# =============================================================================
# Grid Transforms
# =============================================================================

def rotate_grid(grid: Grid, degrees: int) -> Grid:
    rows, cols = len(grid), len(grid[0])
    if degrees == 90:
        return [[grid[rows - 1 - j][i] for j in range(rows)] for i in range(cols)]
    if degrees == 180:
        return [[grid[rows - 1 - r][cols - 1 - c] for c in range(cols)] for r in range(rows)]
    if degrees == 270:
        return [[grid[j][cols - 1 - i] for j in range(rows)] for i in range(cols)]
    return copy_grid(grid)


def reflect_grid(grid: Grid, axis: str) -> Grid:
    if axis == "horizontal":
        return [row[::-1] for row in grid]
    return grid[::-1]


def scale_2x(grid: Grid) -> Grid:
    out = []
    for row in grid:
        new_row = []
        for c in row:
            new_row.extend([c, c])
        out.append(new_row[:])
        out.append(new_row[:])
    return out


def apply_gravity(grid: Grid) -> Grid:
    rows, cols = len(grid), len(grid[0])
    out = [[0] * cols for _ in range(rows)]
    for c in range(cols):
        non_zero = [grid[r][c] for r in range(rows) if grid[r][c] != 0]
        for i, val in enumerate(non_zero):
            out[rows - len(non_zero) + i][c] = val
    return out


def find_components(grid: Grid) -> list[list[tuple[int, int]]]:
    rows, cols = len(grid), len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    components = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and not visited[r][c]:
                comp = []
                queue = deque([(r, c)])
                visited[r][c] = True
                while queue:
                    cr, cc = queue.popleft()
                    comp.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if not visited[nr][nc] and grid[nr][nc] != 0:
                                visited[nr][nc] = True
                                queue.append((nr, nc))
                components.append(comp)
    return components


# =============================================================================
# Generator 1: Color Replacement
# =============================================================================

def gen_color_replacement(task_seed: int, level: int) -> dict:
    def setup(rng, lvl):
        color_a, color_b = pick_n_colors(rng, 2)
        rows, cols = grid_dims_for_level(lvl, rng)
        return {"color_a": color_a, "color_b": color_b, "rows": rows, "cols": cols}

    def pair(pair_rng, color_a, color_b, rows, cols):
        inp = random_grid(pair_rng, rows, cols, [color_a, color_b])
        pos_r, pos_c = pair_rng.randint(0, rows - 1), pair_rng.randint(0, cols - 1)
        inp[pos_r][pos_c] = color_a
        out = copy_grid(inp)
        for r in range(rows):
            for c in range(cols):
                if out[r][c] == color_a:
                    out[r][c] = color_b
        return inp, out

    return make_task(task_seed, level, setup, pair)


# =============================================================================
# Generator 2: Simple Rotation
# =============================================================================

def gen_simple_rotation(task_seed: int, level: int) -> dict:
    def setup(rng, lvl):
        angle = rng.choice([90, 180, 270])
        size = rng.randint(5, 8) if lvl == 1 else rng.randint(8, 12)
        colors = pick_n_colors(rng, rng.randint(2, 4))
        return {"angle": angle, "size": size, "colors": colors}

    def pair(pair_rng, angle, size, colors):
        inp = random_grid(pair_rng, size, size, colors)
        out = rotate_grid(inp, angle)
        return inp, out

    return make_task(task_seed, level, setup, pair)


# =============================================================================
# Generator 3: Simple Reflection
# =============================================================================

def gen_simple_reflection(task_seed: int, level: int) -> dict:
    def setup(rng, lvl):
        axis = rng.choice(["horizontal", "vertical"])
        rows, cols = grid_dims_for_level(lvl, rng)
        colors = pick_n_colors(rng, rng.randint(2, 4))
        return {"axis": axis, "rows": rows, "cols": cols, "colors": colors}

    def pair(pair_rng, axis, rows, cols, colors):
        inp = random_grid(pair_rng, rows, cols, colors)
        out = reflect_grid(inp, axis)
        return inp, out

    return make_task(task_seed, level, setup, pair)


# =============================================================================
# Generator 4: Scaling 2x
# =============================================================================

def gen_scaling_2x(task_seed: int, level: int) -> dict:
    def setup(rng, lvl):
        in_rows = rng.randint(3, 4) if lvl == 1 else rng.randint(4, 6)
        in_cols = rng.randint(3, 4) if lvl == 1 else rng.randint(4, 6)
        colors = pick_n_colors(rng, rng.randint(2, 4))
        return {"in_rows": in_rows, "in_cols": in_cols, "colors": colors}

    def pair(pair_rng, in_rows, in_cols, colors):
        inp = random_grid(pair_rng, in_rows, in_cols, colors)
        out = scale_2x(inp)
        return inp, out

    return make_task(task_seed, level, setup, pair)


# =============================================================================
# Generator 5: Gravity Drop
# =============================================================================

def gen_gravity_drop(task_seed: int, level: int) -> dict:
    def setup(rng, lvl):
        rows, cols = grid_dims_for_level(lvl, rng)
        colors = pick_n_colors(rng, rng.randint(2, 3))
        return {"rows": rows, "cols": cols, "colors": colors}

    def pair(pair_rng, rows, cols, colors):
        inp = random_sparse_grid(pair_rng, rows, cols, colors, density=0.3)
        out = apply_gravity(inp)
        return inp, out

    return make_task(task_seed, level, setup, pair)


# =============================================================================
# Generator 6: Flood Fill
# =============================================================================

def gen_flood_fill(task_seed: int, level: int) -> dict:
    def setup(rng, lvl):
        rows, cols = grid_dims_for_level(lvl, rng)
        wall_color, fill_color = pick_n_colors(rng, 2)
        return {"rows": rows, "cols": cols, "wall_color": wall_color, "fill_color": fill_color}

    def pair(pair_rng, rows, cols, wall_color, fill_color):
        r1 = pair_rng.randint(0, max(0, rows // 3 - 1))
        r2 = pair_rng.randint(min(rows - 1, 2 * rows // 3 + 1), rows - 1)
        c1 = pair_rng.randint(0, max(0, cols // 3 - 1))
        c2 = pair_rng.randint(min(cols - 1, 2 * cols // 3 + 1), cols - 1)
        if r2 - r1 < 2:
            r2 = min(rows - 1, r1 + 2)
        if c2 - c1 < 2:
            c2 = min(cols - 1, c1 + 2)

        inp = [[0] * cols for _ in range(rows)]
        for r in range(r1, r2 + 1):
            inp[r][c1] = wall_color
            inp[r][c2] = wall_color
        for c in range(c1, c2 + 1):
            inp[r1][c] = wall_color
            inp[r2][c] = wall_color

        seed_r = pair_rng.randint(r1 + 1, r2 - 1)
        seed_c = pair_rng.randint(c1 + 1, c2 - 1)
        inp[seed_r][seed_c] = fill_color

        out = copy_grid(inp)
        for r in range(r1 + 1, r2):
            for c in range(c1 + 1, c2):
                if out[r][c] == 0:
                    out[r][c] = fill_color
        return inp, out

    return make_task(task_seed, level, setup, pair)


# =============================================================================
# Generator 7: Boolean AND/OR
# =============================================================================

def gen_boolean_and_or(task_seed: int, level: int) -> dict:
    def setup(rng, lvl):
        op = rng.choice(["and", "or"])
        half_rows = rng.randint(4, 6) if lvl == 1 else rng.randint(6, 8)
        half_cols = rng.randint(4, 6) if lvl == 1 else rng.randint(6, 8)
        color = pick_n_colors(rng, 1)[0]
        return {"op": op, "half_rows": half_rows, "half_cols": half_cols, "color": color}

    def pair(pair_rng, op, half_rows, half_cols, color):
        mask_a = [[pair_rng.random() < 0.4 for _ in range(half_cols)] for _ in range(half_rows)]
        mask_b = [[pair_rng.random() < 0.4 for _ in range(half_cols)] for _ in range(half_rows)]

        inp = [[0] * (half_cols * 2) for _ in range(half_rows)]
        for r in range(half_rows):
            for c in range(half_cols):
                if mask_a[r][c]:
                    inp[r][c] = color
                if mask_b[r][c]:
                    inp[r][half_cols + c] = color

        out = [[0] * half_cols for _ in range(half_rows)]
        for r in range(half_rows):
            for c in range(half_cols):
                if op == "and":
                    if mask_a[r][c] and mask_b[r][c]:
                        out[r][c] = color
                else:
                    if mask_a[r][c] or mask_b[r][c]:
                        out[r][c] = color
        return inp, out

    return make_task(task_seed, level, setup, pair)


# =============================================================================
# Generator 8: Row/Column Duplication
# =============================================================================

def gen_row_col_duplication(task_seed: int, level: int) -> dict:
    def setup(rng, lvl):
        mode = rng.choice(["row", "col"])
        in_rows = rng.randint(3, 5)
        in_cols = rng.randint(3, 5)
        colors = pick_n_colors(rng, rng.randint(2, 4))
        return {"mode": mode, "in_rows": in_rows, "in_cols": in_cols, "colors": colors}

    def pair(pair_rng, mode, in_rows, in_cols, colors):
        inp = random_grid(pair_rng, in_rows, in_cols, colors)
        if mode == "row":
            out = []
            for row in inp:
                out.append(row[:])
                out.append(row[:])
        else:
            out = []
            for row in inp:
                new_row = []
                for c in row:
                    new_row.extend([c, c])
                out.append(new_row)
        return inp, out

    return make_task(task_seed, level, setup, pair)


# =============================================================================
# Generator 9: Solid to Hollow
# =============================================================================

def gen_solid_to_hollow(task_seed: int, level: int) -> dict:
    def setup(rng, lvl):
        rows, cols = grid_dims_for_level(lvl, rng)
        rect_color = pick_n_colors(rng, 1)[0]
        return {"rows": rows, "cols": cols, "rect_color": rect_color}

    def pair(pair_rng, rows, cols, rect_color):
        r1 = pair_rng.randint(0, rows // 3)
        c1 = pair_rng.randint(0, cols // 3)
        r2 = pair_rng.randint(r1 + 2, rows - 1)
        c2 = pair_rng.randint(c1 + 2, cols - 1)

        inp = [[0] * cols for _ in range(rows)]
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                inp[r][c] = rect_color

        out = copy_grid(inp)
        for r in range(r1 + 1, r2):
            for c in range(c1 + 1, c2):
                out[r][c] = 0
        return inp, out

    return make_task(task_seed, level, setup, pair)


# =============================================================================
# Generator 10: Histogram Rendering
# =============================================================================

def gen_histogram_rendering(task_seed: int, level: int) -> dict:
    def setup(rng, lvl):
        width = rng.randint(5, 8)
        num_colors = rng.randint(2, 4)
        colors = pick_n_colors(rng, num_colors)
        return {"width": width, "colors": colors}

    def pair(pair_rng, width, colors):
        inp_row = [pair_rng.choice(colors) for _ in range(width)]
        inp = [inp_row]

        freq = {}
        for c in inp_row:
            freq[c] = freq.get(c, 0) + 1
        max_freq = max(freq.values())

        out = [[0] * width for _ in range(max_freq)]
        for col_idx in range(width):
            color = inp_row[col_idx]
            height = freq[color]
            for row_idx in range(max_freq - height, max_freq):
                out[row_idx][col_idx] = color
        return inp, out

    return make_task(task_seed, level, setup, pair)


# =============================================================================
# Generator 11: Dimension-Based Pattern
# =============================================================================

def gen_dimension_based_pattern(task_seed: int, level: int) -> dict:
    def setup(rng, lvl):
        pattern = rng.choice(["checkerboard", "horizontal_stripes", "vertical_stripes"])
        color_a, color_b = pick_n_colors(rng, 2)
        return {"pattern": pattern, "color_a": color_a, "color_b": color_b}

    def pair(pair_rng, pattern, color_a, color_b):
        rows = pair_rng.randint(3, 7)
        cols = pair_rng.randint(3, 7)
        inp = [[0] * cols for _ in range(rows)]

        out = [[0] * cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                if pattern == "checkerboard":
                    out[r][c] = color_a if (r + c) % 2 == 0 else color_b
                elif pattern == "horizontal_stripes":
                    out[r][c] = color_a if r % 2 == 0 else color_b
                else:
                    out[r][c] = color_a if c % 2 == 0 else color_b
        return inp, out

    return make_task(task_seed, level, setup, pair)


# =============================================================================
# Generator 12: Connectivity Size Filter
# =============================================================================

def gen_connectivity_size_filter(task_seed: int, level: int) -> dict:
    def setup(rng, lvl):
        rows, cols = grid_dims_for_level(lvl, rng)
        threshold = rng.randint(3, 4)
        color = pick_n_colors(rng, 1)[0]
        return {"rows": rows, "cols": cols, "threshold": threshold, "color": color}

    def pair(pair_rng, rows, cols, threshold, color):
        inp = [[0] * cols for _ in range(rows)]

        # Place 1-2 large blobs via random walk
        for _ in range(pair_rng.randint(1, 2)):
            sr = pair_rng.randint(1, rows - 2)
            sc = pair_rng.randint(1, cols - 2)
            cells = {(sr, sc)}
            candidates = [(sr, sc)]
            target_size = pair_rng.randint(threshold, threshold + 3)
            while len(cells) < target_size and candidates:
                cr, cc = pair_rng.choice(candidates)
                neighbors = [
                    (cr + dr, cc + dc)
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    if 0 <= cr + dr < rows and 0 <= cc + dc < cols
                    and (cr + dr, cc + dc) not in cells
                ]
                if neighbors:
                    nr, nc = pair_rng.choice(neighbors)
                    cells.add((nr, nc))
                    candidates.append((nr, nc))
                else:
                    candidates.remove((cr, cc))
            for r, c in cells:
                inp[r][c] = color

        # Place 2-4 isolated single pixels
        for _ in range(pair_rng.randint(2, 4)):
            for _attempt in range(20):
                r = pair_rng.randint(0, rows - 1)
                c = pair_rng.randint(0, cols - 1)
                if inp[r][c] == 0:
                    adjacent = any(
                        0 <= r + dr < rows and 0 <= c + dc < cols
                        and inp[r + dr][c + dc] != 0
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    )
                    if not adjacent:
                        inp[r][c] = color
                        break

        comps = find_components(inp)
        out = [[0] * cols for _ in range(rows)]
        for comp in comps:
            if len(comp) >= threshold:
                for r, c in comp:
                    out[r][c] = inp[r][c]
        return inp, out

    return make_task(task_seed, level, setup, pair)


# =============================================================================
# Generator 13: Bounding Rectangle Fill
# =============================================================================

def gen_bounding_rect_fill(task_seed: int, level: int) -> dict:
    def setup(rng, lvl):
        rows, cols = grid_dims_for_level(lvl, rng)
        color = pick_n_colors(rng, 1)[0]
        return {"rows": rows, "cols": cols, "color": color}

    def pair(pair_rng, rows, cols, color):
        r1 = pair_rng.randint(0, rows - 2)
        c1 = pair_rng.randint(0, cols - 2)
        r2 = pair_rng.randint(r1 + 1, rows - 1)
        c2 = pair_rng.randint(c1 + 1, cols - 1)

        inp = [[0] * cols for _ in range(rows)]
        inp[r1][c1] = color
        inp[r2][c2] = color

        out = [[0] * cols for _ in range(rows)]
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                out[r][c] = color
        return inp, out

    return make_task(task_seed, level, setup, pair)


# =============================================================================
# Registry & Main
# =============================================================================

GENERATORS: dict[str, tuple] = {
    "color_replacement":        (gen_color_replacement, [1]),
    "simple_rotation":          (gen_simple_rotation, [1]),
    "simple_reflection":        (gen_simple_reflection, [1]),
    "scaling_2x":               (gen_scaling_2x, [1]),
    "gravity_drop":             (gen_gravity_drop, [1]),
    "flood_fill":               (gen_flood_fill, [1]),
    "boolean_and_or":           (gen_boolean_and_or, [1]),
    "row_col_duplication":      (gen_row_col_duplication, [1]),
    "solid_to_hollow":          (gen_solid_to_hollow, [1]),
    "histogram_rendering":      (gen_histogram_rendering, [1]),
    "dimension_based_pattern":  (gen_dimension_based_pattern, [1]),
    "connectivity_size_filter": (gen_connectivity_size_filter, [1]),
    "bounding_rect_fill":       (gen_bounding_rect_fill, [1]),
}


def format_grid_preview(grid: Grid) -> str:
    return "\n".join(",".join(str(c) for c in row) for row in grid)


def main():
    records = []
    cat_names = list(GENERATORS.keys())

    for cat_idx, cat_name in enumerate(cat_names):
        gen_fn, levels = GENERATORS[cat_name]
        for level in levels:
            for task_idx in range(TASKS_PER_CATEGORY):
                task_seed = 10000 * level + cat_idx * 1000 + task_idx
                task = gen_fn(task_seed, level)
                records.append({
                    "train_pairs": json.dumps(task["train_pairs"]),
                    "test_input": json.dumps(task["test_input"]),
                    "test_output": json.dumps(task["test_output"]),
                    "task_id": f"{cat_name}_{level}_{task_idx}",
                    "task_type": cat_name,
                    "level": level,
                })

    print(f"Generated {len(records)} tasks across {len(cat_names)} categories")

    # Spot-check a few tasks
    for i in range(0, len(records), len(records) // 5):
        rec = records[i]
        test_in = json.loads(rec["test_input"])
        test_out = json.loads(rec["test_output"])
        print(f"\n--- {rec['task_id']} (type={rec['task_type']}, level={rec['level']}) ---")
        print(f"Input ({len(test_in)}x{len(test_in[0])}):")
        print(format_grid_preview(test_in))
        print(f"Output ({len(test_out)}x{len(test_out[0])}):")
        print(format_grid_preview(test_out))

    ds = Dataset.from_list(records)
    print(f"\nDataset: {ds}")

    print(f"\nPushing to {HF_REPO}...")
    ds.push_to_hub(HF_REPO, split="train")
    print("Done!")


if __name__ == "__main__":
    main()
