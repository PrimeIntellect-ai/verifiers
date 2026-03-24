"""Generate synthetic ARC-AGI-2 dataset and push to HuggingFace.

Level 1: 13 single-operation categories on small grids (5x5 to 8x8).
Level 2: 18 multi-operation categories on medium grids (8x8 to 12x12).

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
# Generator 14: Symmetry Completion (L2)
# =============================================================================

def gen_symmetry_completion(task_seed: int, level: int) -> dict:
    def setup(rng, lvl):
        rows, cols = grid_dims_for_level(lvl, rng)
        rows += rows % 2
        cols += cols % 2
        axis = rng.choice(["horizontal", "vertical"])
        colors = pick_n_colors(rng, rng.randint(2, 4))
        return {"rows": rows, "cols": cols, "axis": axis, "colors": colors}

    def pair(pair_rng, rows, cols, axis, colors):
        if axis == "vertical":
            half = rows // 2
            top = random_sparse_grid(pair_rng, half, cols, colors, density=0.4)
            full = [[0] * cols for _ in range(rows)]
            for r in range(half):
                for c in range(cols):
                    full[r][c] = top[r][c]
                    full[rows - 1 - r][c] = top[r][c]
            inp = copy_grid(full)
            for r in range(half, rows):
                for c in range(cols):
                    inp[r][c] = 0
        else:
            half = cols // 2
            left = random_sparse_grid(pair_rng, rows, half, colors, density=0.4)
            full = [[0] * cols for _ in range(rows)]
            for r in range(rows):
                for c in range(half):
                    full[r][c] = left[r][c]
                    full[r][cols - 1 - c] = left[r][c]
            inp = copy_grid(full)
            for r in range(rows):
                for c in range(half, cols):
                    inp[r][c] = 0
        return inp, full

    return make_task(task_seed, level, setup, pair)


# =============================================================================
# Generator 15: Border Drawing (L2)
# =============================================================================

def gen_border_drawing(task_seed: int, level: int) -> dict:
    def setup(rng, lvl):
        rows, cols = grid_dims_for_level(lvl, rng)
        shape_color, border_color = pick_n_colors(rng, 2)
        return {"rows": rows, "cols": cols, "shape_color": shape_color,
                "border_color": border_color}

    def pair(pair_rng, rows, cols, shape_color, border_color):
        inp = [[0] * cols for _ in range(rows)]
        for _ in range(pair_rng.randint(1, 2)):
            r1 = pair_rng.randint(1, rows - 4)
            c1 = pair_rng.randint(1, cols - 4)
            r2 = pair_rng.randint(r1 + 1, min(r1 + 4, rows - 2))
            c2 = pair_rng.randint(c1 + 1, min(c1 + 4, cols - 2))
            for r in range(r1, r2 + 1):
                for c in range(c1, c2 + 1):
                    inp[r][c] = shape_color

        out = copy_grid(inp)
        for r in range(rows):
            for c in range(cols):
                if inp[r][c] == 0:
                    if any(0 <= r + dr < rows and 0 <= c + dc < cols
                           and inp[r + dr][c + dc] == shape_color
                           for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]):
                        out[r][c] = border_color
        return inp, out

    return make_task(task_seed, level, setup, pair)


# =============================================================================
# Generator 16: Template Cloning (L2)
# =============================================================================

def gen_template_cloning(task_seed: int, level: int) -> dict:
    def setup(rng, lvl):
        tpl_size = rng.choice([2, 3])
        min_grid = tpl_size * 4
        rows = max(min_grid, rng.randint(8, 12))
        cols = max(min_grid, rng.randint(8, 12))
        tpl_colors = pick_n_colors(rng, 2)
        marker_color = rng.choice(ARC_COLORS)
        while marker_color in tpl_colors:
            marker_color = rng.choice(ARC_COLORS)
        return {"rows": rows, "cols": cols, "tpl_size": tpl_size,
                "tpl_colors": tpl_colors, "marker_color": marker_color}

    def pair(pair_rng, rows, cols, tpl_size, tpl_colors, marker_color):
        tpl = random_grid(pair_rng, tpl_size, tpl_size, tpl_colors, include_zero=False)
        inp = [[0] * cols for _ in range(rows)]
        for r in range(tpl_size):
            for c in range(tpl_size):
                inp[r][c] = tpl[r][c]

        markers = []
        for _ in range(pair_rng.randint(1, 2)):
            for _attempt in range(50):
                mr = pair_rng.randint(tpl_size + 1, rows - tpl_size)
                mc = pair_rng.randint(tpl_size + 1, cols - tpl_size)
                too_close = any(abs(mr - omr) <= tpl_size and abs(mc - omc) <= tpl_size
                               for omr, omc in markers)
                if not too_close:
                    markers.append((mr, mc))
                    break

        for mr, mc in markers:
            inp[mr][mc] = marker_color

        out = copy_grid(inp)
        offset = tpl_size // 2
        for mr, mc in markers:
            out[mr][mc] = 0
            for dr in range(tpl_size):
                for dc in range(tpl_size):
                    r, c = mr - offset + dr, mc - offset + dc
                    if 0 <= r < rows and 0 <= c < cols:
                        out[r][c] = tpl[dr][dc]
        return inp, out

    return make_task(task_seed, level, setup, pair)


# =============================================================================
# Generator 17: Shape Stamping (L2)
# =============================================================================

_STAMPS = [
    [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)],   # plus
    [(0, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)],  # X
    [(0, 0), (0, 1), (1, 0), (1, 1)],               # square
    [(0, 0), (-1, 0), (0, 1)],                       # L top-right
]


def gen_shape_stamping(task_seed: int, level: int) -> dict:
    def setup(rng, lvl):
        rows, cols = grid_dims_for_level(lvl, rng)
        stamp = rng.choice(_STAMPS)
        marker_color, stamp_color = pick_n_colors(rng, 2)
        return {"rows": rows, "cols": cols, "stamp": stamp,
                "marker_color": marker_color, "stamp_color": stamp_color}

    def pair(pair_rng, rows, cols, stamp, marker_color, stamp_color):
        markers = []
        for _ in range(pair_rng.randint(2, 4)):
            for _attempt in range(50):
                mr = pair_rng.randint(2, rows - 3)
                mc = pair_rng.randint(2, cols - 3)
                if not any(abs(mr - omr) < 4 and abs(mc - omc) < 4
                           for omr, omc in markers):
                    markers.append((mr, mc))
                    break

        inp = [[0] * cols for _ in range(rows)]
        for mr, mc in markers:
            inp[mr][mc] = marker_color

        out = [[0] * cols for _ in range(rows)]
        for mr, mc in markers:
            for dr, dc in stamp:
                r, c = mr + dr, mc + dc
                if 0 <= r < rows and 0 <= c < cols:
                    out[r][c] = stamp_color
        return inp, out

    return make_task(task_seed, level, setup, pair)


# =============================================================================
# Generator 18: Object Extraction Recolor (L2)
# =============================================================================

def gen_object_extraction_recolor(task_seed: int, level: int) -> dict:
    def setup(rng, lvl):
        rows, cols = grid_dims_for_level(lvl, rng)
        colors = pick_n_colors(rng, rng.randint(2, 3))
        return {"rows": rows, "cols": cols, "colors": colors}

    def pair(pair_rng, rows, cols, colors):
        inp = [[0] * cols for _ in range(rows)]
        blob_specs = [(pair_rng.randint(5, 10), pair_rng.choice(colors)),
                      (pair_rng.randint(2, 4), pair_rng.choice(colors)),
                      (pair_rng.randint(1, 3), pair_rng.choice(colors))]
        for target_size, color in blob_specs:
            sr, sc = 0, 0
            for _attempt in range(20):
                sr = pair_rng.randint(1, rows - 2)
                sc = pair_rng.randint(1, cols - 2)
                if inp[sr][sc] == 0:
                    break
            cells = {(sr, sc)}
            candidates = [(sr, sc)]
            while len(cells) < target_size and candidates:
                cr, cc = pair_rng.choice(candidates)
                neighbors = [(cr + dr, cc + dc) for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                             if 0 <= cr + dr < rows and 0 <= cc + dc < cols
                             and (cr + dr, cc + dc) not in cells and inp[cr + dr][cc + dc] == 0]
                if neighbors:
                    nr, nc = pair_rng.choice(neighbors)
                    cells.add((nr, nc))
                    candidates.append((nr, nc))
                else:
                    candidates.remove((cr, cc))
            for r, c in cells:
                inp[r][c] = color

        comps = find_components(inp)
        if not comps:
            return inp, copy_grid(inp)
        largest = max(comps, key=len)
        largest_set = set(largest)
        out = [[0] * cols for _ in range(rows)]
        for r, c in largest_set:
            out[r][c] = inp[r][c]
        return inp, out

    return make_task(task_seed, level, setup, pair)


# =============================================================================
# Generator 19: Line Extension (L2)
# =============================================================================

def gen_line_extension(task_seed: int, level: int) -> dict:
    def setup(rng, lvl):
        rows, cols = grid_dims_for_level(lvl, rng)
        direction = rng.choice(["horizontal", "vertical"])
        color = pick_n_colors(rng, 1)[0]
        return {"rows": rows, "cols": cols, "direction": direction, "color": color}

    def pair(pair_rng, rows, cols, direction, color):
        inp = [[0] * cols for _ in range(rows)]
        out = [[0] * cols for _ in range(rows)]
        used = set()

        for _ in range(pair_rng.randint(2, 4)):
            if direction == "horizontal":
                for _attempt in range(20):
                    r = pair_rng.randint(0, rows - 1)
                    if r not in used:
                        used.add(r)
                        break
                start = pair_rng.randint(1, cols // 3)
                end = pair_rng.randint(start + 1, min(start + 3, cols - 1))
                for c in range(start, end + 1):
                    inp[r][c] = color
                for c in range(cols):
                    out[r][c] = color
            else:
                for _attempt in range(20):
                    c = pair_rng.randint(0, cols - 1)
                    if c not in used:
                        used.add(c)
                        break
                start = pair_rng.randint(1, rows // 3)
                end = pair_rng.randint(start + 1, min(start + 3, rows - 1))
                for r in range(start, end + 1):
                    inp[r][c] = color
                for r in range(rows):
                    out[r][c] = color
        return inp, out

    return make_task(task_seed, level, setup, pair)


# =============================================================================
# Generator 20: Denoising (L2)
# =============================================================================

def gen_denoising(task_seed: int, level: int) -> dict:
    def setup(rng, lvl):
        tile_size = rng.randint(2, 3)
        grid_tiles = rng.randint(3, 4)
        colors = pick_n_colors(rng, 2)
        noise_count = rng.randint(3, 6)
        return {"tile_size": tile_size, "grid_tiles": grid_tiles,
                "colors": colors, "noise_count": noise_count}

    def pair(pair_rng, tile_size, grid_tiles, colors, noise_count):
        tile = [[pair_rng.choice(colors) for _ in range(tile_size)]
                for _ in range(tile_size)]
        total = tile_size * grid_tiles
        clean = [[tile[r % tile_size][c % tile_size]
                  for c in range(total)] for r in range(total)]
        noisy = copy_grid(clean)
        for _ in range(noise_count):
            r = pair_rng.randint(0, total - 1)
            c = pair_rng.randint(0, total - 1)
            noisy[r][c] = pair_rng.choice([cl for cl in colors if cl != clean[r][c]])
        return noisy, clean

    return make_task(task_seed, level, setup, pair)


# =============================================================================
# Generator 21: Interior Fill Multi (L2)
# =============================================================================

def gen_interior_fill_multi(task_seed: int, level: int) -> dict:
    def setup(rng, lvl):
        rows, cols = grid_dims_for_level(lvl, rng)
        colors = pick_n_colors(rng, rng.randint(2, 3))
        return {"rows": rows, "cols": cols, "colors": colors}

    def pair(pair_rng, rows, cols, colors):
        inp = [[0] * cols for _ in range(rows)]
        rects = []
        for _ in range(pair_rng.randint(1, 3)):
            for _attempt in range(30):
                r1 = pair_rng.randint(0, rows - 4)
                c1 = pair_rng.randint(0, cols - 4)
                r2 = pair_rng.randint(r1 + 2, min(r1 + 5, rows - 1))
                c2 = pair_rng.randint(c1 + 2, min(c1 + 5, cols - 1))
                overlap = any(not (r2 < er1 or r1 > er2 or c2 < ec1 or c1 > ec2)
                              for er1, ec1, er2, ec2, _ in rects)
                if not overlap:
                    rects.append((r1, c1, r2, c2, pair_rng.choice(colors)))
                    break

        for r1, c1, r2, c2, color in rects:
            for r in range(r1, r2 + 1):
                inp[r][c1] = color
                inp[r][c2] = color
            for c in range(c1, c2 + 1):
                inp[r1][c] = color
                inp[r2][c] = color

        out = copy_grid(inp)
        for r1, c1, r2, c2, color in rects:
            for r in range(r1 + 1, r2):
                for c in range(c1 + 1, c2):
                    out[r][c] = color
        return inp, out

    return make_task(task_seed, level, setup, pair)


# =============================================================================
# Generator 22: Crosshair Generation (L2)
# =============================================================================

def gen_crosshair_generation(task_seed: int, level: int) -> dict:
    def setup(rng, lvl):
        rows, cols = grid_dims_for_level(lvl, rng)
        marker_color = pick_n_colors(rng, 1)[0]
        return {"rows": rows, "cols": cols, "marker_color": marker_color}

    def pair(pair_rng, rows, cols, marker_color):
        markers = []
        for _ in range(pair_rng.randint(1, 3)):
            for _attempt in range(30):
                mr = pair_rng.randint(1, rows - 2)
                mc = pair_rng.randint(1, cols - 2)
                if (mr, mc) not in markers:
                    markers.append((mr, mc))
                    break

        inp = [[0] * cols for _ in range(rows)]
        for mr, mc in markers:
            inp[mr][mc] = marker_color

        out = copy_grid(inp)
        for mr, mc in markers:
            for c in range(cols):
                out[mr][c] = marker_color
            for r in range(rows):
                out[r][mc] = marker_color
        return inp, out

    return make_task(task_seed, level, setup, pair)


# =============================================================================
# Generator 23: Object Compaction (L2)
# =============================================================================

def gen_object_compaction(task_seed: int, level: int) -> dict:
    def setup(rng, lvl):
        rows, cols = grid_dims_for_level(lvl, rng)
        colors = pick_n_colors(rng, rng.randint(2, 3))
        return {"rows": rows, "cols": cols, "colors": colors}

    def pair(pair_rng, rows, cols, colors):
        inp = random_sparse_grid(pair_rng, rows, cols, colors, density=0.3)
        out = [[0] * cols for _ in range(rows)]
        for r in range(rows):
            non_zero = [inp[r][c] for c in range(cols) if inp[r][c] != 0]
            for i, val in enumerate(non_zero):
                out[r][i] = val
        return inp, out

    return make_task(task_seed, level, setup, pair)


# =============================================================================
# Generator 24: Fill-Level Encoding (L2)
# =============================================================================

def gen_fill_level_encoding(task_seed: int, level: int) -> dict:
    def setup(rng, lvl):
        num_cols = rng.randint(5, 9)
        max_height = rng.randint(5, 9)
        fill_color = pick_n_colors(rng, 1)[0]
        return {"num_cols": num_cols, "max_height": max_height,
                "fill_color": fill_color}

    def pair(pair_rng, num_cols, max_height, fill_color):
        heights = [pair_rng.randint(1, max_height) for _ in range(num_cols)]
        inp = [[0] * num_cols for _ in range(max_height)]
        for c in range(num_cols):
            for r in range(max_height - heights[c], max_height):
                inp[r][c] = fill_color
        out = [[heights[c] for c in range(num_cols)]]
        return inp, out

    return make_task(task_seed, level, setup, pair)


# =============================================================================
# Generator 25: Expanding Diamond (L2)
# =============================================================================

def gen_expanding_diamond(task_seed: int, level: int) -> dict:
    def setup(rng, lvl):
        size = rng.randint(9, 13) if lvl >= 2 else rng.randint(7, 9)
        num_rings = rng.randint(2, 4)
        colors = pick_n_colors(rng, num_rings)
        return {"size": size, "num_rings": num_rings, "colors": colors}

    def pair(pair_rng, size, num_rings, colors):
        sr = pair_rng.randint(2, size - 3)
        sc = pair_rng.randint(2, size - 3)
        inp = [[0] * size for _ in range(size)]
        inp[sr][sc] = colors[0]
        out = [[0] * size for _ in range(size)]
        for r in range(size):
            for c in range(size):
                d = abs(r - sr) + abs(c - sc)
                if d < num_rings:
                    out[r][c] = colors[d]
        return inp, out

    return make_task(task_seed, level, setup, pair)


# =============================================================================
# Generator 26: Same-Color Pair Bridging (L2)
# =============================================================================

def gen_same_color_pair_bridging(task_seed: int, level: int) -> dict:
    def setup(rng, lvl):
        rows, cols = grid_dims_for_level(lvl, rng)
        colors = pick_n_colors(rng, rng.randint(2, 3))
        return {"rows": rows, "cols": cols, "colors": colors}

    def pair(pair_rng, rows, cols, colors):
        inp = [[0] * cols for _ in range(rows)]
        pairs_list = []
        for _ in range(pair_rng.randint(2, 3)):
            color = pair_rng.choice(colors)
            if pair_rng.random() < 0.5:
                for _attempt in range(30):
                    r = pair_rng.randint(0, rows - 1)
                    c1 = pair_rng.randint(0, cols // 3)
                    c2 = pair_rng.randint(c1 + 2, cols - 1)
                    if inp[r][c1] == 0 and inp[r][c2] == 0:
                        pairs_list.append(("h", r, c1, c2, color))
                        inp[r][c1] = color
                        inp[r][c2] = color
                        break
            else:
                for _attempt in range(30):
                    c = pair_rng.randint(0, cols - 1)
                    r1 = pair_rng.randint(0, rows // 3)
                    r2 = pair_rng.randint(r1 + 2, rows - 1)
                    if inp[r1][c] == 0 and inp[r2][c] == 0:
                        pairs_list.append(("v", c, r1, r2, color))
                        inp[r1][c] = color
                        inp[r2][c] = color
                        break

        out = copy_grid(inp)
        for p in pairs_list:
            if p[0] == "h":
                _, r, c1, c2, color = p
                for c in range(c1, c2 + 1):
                    out[r][c] = color
            else:
                _, c, r1, r2, color = p
                for r in range(r1, r2 + 1):
                    out[r][c] = color
        return inp, out

    return make_task(task_seed, level, setup, pair)


# =============================================================================
# Generator 27: Seam-Marked Tiling (L2)
# =============================================================================

def gen_seam_marked_tiling(task_seed: int, level: int) -> dict:
    def setup(rng, lvl):
        tile_h = rng.randint(3, 4)
        tile_w = rng.randint(3, 4)
        reps_r = rng.randint(2, 3)
        reps_c = rng.randint(2, 3)
        tile_colors = pick_n_colors(rng, 2)
        seam_color = rng.choice(ARC_COLORS)
        while seam_color in tile_colors:
            seam_color = rng.choice(ARC_COLORS)
        return {"tile_h": tile_h, "tile_w": tile_w, "reps_r": reps_r,
                "reps_c": reps_c, "tile_colors": tile_colors, "seam_color": seam_color}

    def pair(pair_rng, tile_h, tile_w, reps_r, reps_c, tile_colors, seam_color):
        tile = [[pair_rng.choice(tile_colors) for _ in range(tile_w)]
                for _ in range(tile_h)]
        inp = [row[:] for row in tile]
        total_h = tile_h * reps_r
        total_w = tile_w * reps_c
        out = [[0] * total_w for _ in range(total_h)]
        for r in range(total_h):
            for c in range(total_w):
                val = tile[r % tile_h][c % tile_w]
                at_seam = (r % tile_h == 0 and r > 0) or (c % tile_w == 0 and c > 0)
                if val != 0 and at_seam:
                    out[r][c] = seam_color
                else:
                    out[r][c] = val
        return inp, out

    return make_task(task_seed, level, setup, pair)


# =============================================================================
# Generator 28: Value-Proportional Expansion (L2)
# =============================================================================

def gen_value_proportional_expansion(task_seed: int, level: int) -> dict:
    def setup(rng, lvl):
        in_rows = rng.randint(2, 4)
        in_cols = rng.randint(3, 6)
        max_val = rng.randint(2, 4)
        return {"in_rows": in_rows, "in_cols": in_cols, "max_val": max_val}

    def pair(pair_rng, in_rows, in_cols, max_val):
        col_vals = [pair_rng.randint(1, max_val) for _ in range(in_cols)]
        inp = [[col_vals[c] for c in range(in_cols)] for _ in range(in_rows)]
        out = []
        for r in range(in_rows):
            out_row = []
            for c in range(in_cols):
                out_row.extend([col_vals[c]] * col_vals[c])
            out.append(out_row)
        return inp, out

    return make_task(task_seed, level, setup, pair)


# =============================================================================
# Generator 29: Gap and Beam (L2)
# =============================================================================

def gen_gap_and_beam(task_seed: int, level: int) -> dict:
    def setup(rng, lvl):
        rows, cols = grid_dims_for_level(lvl, rng)
        wall_color, fill_color = pick_n_colors(rng, 2)
        gap_side = rng.choice(["top", "bottom", "left", "right"])
        return {"rows": rows, "cols": cols, "wall_color": wall_color,
                "fill_color": fill_color, "gap_side": gap_side}

    def pair(pair_rng, rows, cols, wall_color, fill_color, gap_side):
        r1 = pair_rng.randint(2, rows // 3)
        c1 = pair_rng.randint(2, cols // 3)
        r2 = pair_rng.randint(2 * rows // 3, rows - 3)
        c2 = pair_rng.randint(2 * cols // 3, cols - 3)
        if r2 - r1 < 3:
            r2 = min(r1 + 3, rows - 2)
        if c2 - c1 < 3:
            c2 = min(c1 + 3, cols - 2)

        inp = [[0] * cols for _ in range(rows)]
        for r in range(r1, r2 + 1):
            inp[r][c1] = wall_color
            inp[r][c2] = wall_color
        for c in range(c1, c2 + 1):
            inp[r1][c] = wall_color
            inp[r2][c] = wall_color

        if gap_side == "top":
            gp = pair_rng.randint(c1 + 1, c2 - 1)
            inp[r1][gp] = 0
        elif gap_side == "bottom":
            gp = pair_rng.randint(c1 + 1, c2 - 1)
            inp[r2][gp] = 0
        elif gap_side == "left":
            gp = pair_rng.randint(r1 + 1, r2 - 1)
            inp[gp][c1] = 0
        else:
            gp = pair_rng.randint(r1 + 1, r2 - 1)
            inp[gp][c2] = 0

        out = copy_grid(inp)
        for r in range(r1 + 1, r2):
            for c in range(c1 + 1, c2):
                out[r][c] = fill_color

        if gap_side == "top":
            for r in range(0, r1):
                out[r][gp] = fill_color
        elif gap_side == "bottom":
            for r in range(r2 + 1, rows):
                out[r][gp] = fill_color
        elif gap_side == "left":
            for c in range(0, c1):
                out[gp][c] = fill_color
        else:
            for c in range(c2 + 1, cols):
                out[gp][c] = fill_color
        return inp, out

    return make_task(task_seed, level, setup, pair)


# =============================================================================
# Generator 30: Missing Region Reconstruction (L2)
# =============================================================================

def gen_missing_region_reconstruction(task_seed: int, level: int) -> dict:
    def setup(rng, lvl):
        half = rng.randint(4, 6) if lvl >= 2 else rng.randint(3, 4)
        colors = pick_n_colors(rng, rng.randint(2, 3))
        erase_quadrant = rng.choice(["tl", "tr", "bl", "br"])
        return {"half": half, "colors": colors, "erase_quadrant": erase_quadrant}

    def pair(pair_rng, half, colors, erase_quadrant):
        q = random_sparse_grid(pair_rng, half, half, colors, density=0.4)
        size = half * 2
        full = [[0] * size for _ in range(size)]
        for r in range(half):
            for c in range(half):
                val = q[r][c]
                full[r][c] = val
                full[r][size - 1 - c] = val
                full[size - 1 - r][c] = val
                full[size - 1 - r][size - 1 - c] = val

        inp = copy_grid(full)
        if erase_quadrant == "tl":
            for r in range(half):
                for c in range(half):
                    inp[r][c] = 0
        elif erase_quadrant == "tr":
            for r in range(half):
                for c in range(half, size):
                    inp[r][c] = 0
        elif erase_quadrant == "bl":
            for r in range(half, size):
                for c in range(half):
                    inp[r][c] = 0
        else:
            for r in range(half, size):
                for c in range(half, size):
                    inp[r][c] = 0
        return inp, full

    return make_task(task_seed, level, setup, pair)


# =============================================================================
# Generator 31: Noisy Grid Regularization (L2)
# =============================================================================

def gen_noisy_grid_regularization(task_seed: int, level: int) -> dict:
    def setup(rng, lvl):
        rows, cols = grid_dims_for_level(lvl, rng)
        pattern = rng.choice(["checkerboard", "horizontal_stripes", "vertical_stripes"])
        color_a, color_b = pick_n_colors(rng, 2)
        return {"rows": rows, "cols": cols, "pattern": pattern,
                "color_a": color_a, "color_b": color_b}

    def pair(pair_rng, rows, cols, pattern, color_a, color_b):
        clean = [[0] * cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                if pattern == "checkerboard":
                    clean[r][c] = color_a if (r + c) % 2 == 0 else color_b
                elif pattern == "horizontal_stripes":
                    clean[r][c] = color_a if r % 2 == 0 else color_b
                else:
                    clean[r][c] = color_a if c % 2 == 0 else color_b

        noisy = copy_grid(clean)
        num_corrupted = max(2, int(rows * cols * 0.08))
        for _ in range(num_corrupted):
            r = pair_rng.randint(0, rows - 1)
            c = pair_rng.randint(0, cols - 1)
            noisy[r][c] = color_b if clean[r][c] == color_a else color_a
        return noisy, clean

    return make_task(task_seed, level, setup, pair)


# =============================================================================
# Registry & Main
# =============================================================================

GENERATORS: dict[str, tuple] = {
    # Level 1
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
    # Level 2
    "symmetry_completion":          (gen_symmetry_completion, [2]),
    "border_drawing":               (gen_border_drawing, [2]),
    "template_cloning":             (gen_template_cloning, [2]),
    "shape_stamping":               (gen_shape_stamping, [2]),
    "object_extraction_recolor":    (gen_object_extraction_recolor, [2]),
    "line_extension":               (gen_line_extension, [2]),
    "denoising":                    (gen_denoising, [2]),
    "interior_fill_multi":          (gen_interior_fill_multi, [2]),
    "crosshair_generation":         (gen_crosshair_generation, [2]),
    "object_compaction":            (gen_object_compaction, [2]),
    "fill_level_encoding":          (gen_fill_level_encoding, [2]),
    "expanding_diamond":            (gen_expanding_diamond, [2]),
    "same_color_pair_bridging":     (gen_same_color_pair_bridging, [2]),
    "seam_marked_tiling":           (gen_seam_marked_tiling, [2]),
    "value_proportional_expansion": (gen_value_proportional_expansion, [2]),
    "gap_and_beam":                 (gen_gap_and_beam, [2]),
    "missing_region_reconstruction": (gen_missing_region_reconstruction, [2]),
    "noisy_grid_regularization":    (gen_noisy_grid_regularization, [2]),
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
