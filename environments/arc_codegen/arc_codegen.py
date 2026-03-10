"""
ARC-AGI Codegen Training Environment.

Single-actor environment that trains a model to write Python solver(input_grid)
functions for ARC-AGI tasks. Uses shaped rewards from code quality through
correctness to give GRPO signal at every skill level.

Usage:
    prime env install arc-codegen
    prime eval run arc-codegen -m Qwen/Qwen3-4B-Instruct-2507 -n 10 -r 4
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import subprocess
import sys
import tempfile
from typing import Any, List

from datasets import Dataset

import verifiers as vf
from verifiers.envs.actor import Actor
from verifiers.envs.multiagent_env import MultiAgentEnv
from verifiers.envs.protocol import Protocol
from verifiers.rubrics.multiagent_rubric import MultiAgentRubric
from verifiers.types import Messages, State

logger = logging.getLogger(__name__)

HF_DATASET_REPO = "bhoy/arc-agi-2"
Grid = List[List[int]]


# =============================================================================
# Grid Utilities (self-contained copies from arc_multiagent)
# =============================================================================

def format_grid(grid: Grid) -> str:
    """Format grid as CSV (comma-separated rows)."""
    if grid is None:
        return ""
    return "\n".join(",".join(str(c) for c in row) for row in grid)


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


def dims_match(predicted: Grid | None, expected: Grid | None) -> bool:
    """Check if predicted grid has same dimensions as expected."""
    if predicted is None or expected is None:
        return False
    if len(predicted) != len(expected):
        return False
    return all(len(pr) == len(er) for pr, er in zip(predicted, expected))


# =============================================================================
# Code Extraction (self-contained copy from arc_multiagent)
# =============================================================================

def _extract_solver_code(llm_response: str) -> str | None:
    """
    Extract Python code containing def solver() from LLM response.
    Multi-stage extraction for robustness.
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
# Sandbox Execution (self-contained copy from arc_multiagent)
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

    driver_fd, driver_path = tempfile.mkstemp(suffix=".py", prefix="arc_sandbox_")
    try:
        with os.fdopen(driver_fd, "w", encoding="utf-8") as f:
            f.write(_SANDBOX_DRIVER)

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

        result = json.loads(stdout_data)

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
# Prompt Building
# =============================================================================

def build_codegen_prompt(
    train_pairs: list[dict],
    test_input: Grid,
    version: str = "v1b",
) -> str:
    """Build codegen prompt asking model to write a solver() function."""
    if version == "v4":
        return _build_codegen_prompt_v4(train_pairs, test_input)
    return _build_codegen_prompt_v1b(train_pairs, test_input)


def _build_codegen_prompt_v1b(
    train_pairs: list[dict],
    test_input: Grid,
) -> str:
    lines = [
        "Below is an ARC AGI task. You're given the training input/output pairs. "
        "Write a Python function `def solver(input_grid)` that transforms any "
        "input to its correct output. The function should take a 2D list of "
        "integers and return a 2D list of integers.",
        "",
        "Training pairs:",
    ]

    for idx, pair in enumerate(train_pairs, start=1):
        lines.append(f"Example {idx}:")
        lines.append("input:")
        lines.append(format_grid(pair["input"]))
        lines.append("output:")
        lines.append(format_grid(pair["output"]))
        lines.append("")

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
        "Training Pairs:",
    ]

    for idx, pair in enumerate(train_pairs, start=1):
        lines.append(f"Pair {idx}:")
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

    lines.extend([
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
        "2. Identify objects, symmetries, and transformations.",
        "3. Hypothesize the transformation rule.",
        "",
        "### PHASE 2: VERIFICATION",
        "4. Test your hypothesis against ALL training pairs.",
        "5. If any pair fails, revise and re-test.",
        "",
        "### PHASE 3: FINAL SOLUTION",
        "6. Write the final `solver(input_grid)` function.",
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


# =============================================================================
# Response Helpers
# =============================================================================

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
# Shaped Reward Evaluation
# =============================================================================

def evaluate_codegen_response(
    response_text: str,
    train_pairs: list[dict],
    test_input: Grid,
    test_output: Grid,
) -> dict[str, Any]:
    """
    Run the full shaped-reward evaluation pipeline on a codegen response.

    Returns a dict of extras to store in state for the reward function.
    """
    extras: dict[str, Any] = {}

    # Step 1: Extract code
    code = _extract_solver_code(response_text)
    extras["code_extracted"] = code is not None
    if code is None:
        return extras

    # Step 2: Check if code compiles
    try:
        compile(code, "<solver>", "exec")
        extras["code_compiles"] = True
    except SyntaxError:
        extras["code_compiles"] = False
        return extras

    # Step 3: Run against each training pair
    train_passed = 0
    train_total = len(train_pairs)
    train_ran = 0
    first_train_dims_correct = False

    for i, pair in enumerate(train_pairs):
        result = run_solver_in_sandbox(code, pair["input"], timeout_s=10.0)
        if result is not None:
            train_ran += 1
            if grids_equal(result, pair["output"]):
                train_passed += 1
            elif i == 0:
                first_train_dims_correct = dims_match(result, pair["output"])
        elif i == 0:
            first_train_dims_correct = False

    extras["train_results_count"] = train_ran
    extras["train_pairs_passed"] = train_passed
    extras["train_pairs_total"] = train_total
    extras["first_train_dims_correct"] = first_train_dims_correct

    # Step 4: Only run test if ALL training pairs pass
    if train_passed < train_total:
        extras["test_ran"] = False
        extras["test_correct"] = False
        extras["test_similarity"] = 0.0
        return extras

    test_result = run_solver_in_sandbox(code, test_input, timeout_s=10.0)
    extras["test_ran"] = test_result is not None

    if test_result is not None:
        extras["test_correct"] = grids_equal(test_result, test_output)
        extras["test_similarity"] = grid_similarity(test_result, test_output)
    else:
        extras["test_correct"] = False
        extras["test_similarity"] = 0.0

    return extras


def codegen_reward(state: State, **kwargs) -> float:
    """Shaped reward from code quality through correctness."""
    extras = state.get("extras", {})

    if not extras.get("code_extracted"):
        return 0.0

    if not extras.get("code_compiles"):
        return 0.05

    if extras.get("train_results_count", 0) == 0:
        return 0.1

    train_passed = extras.get("train_pairs_passed", 0)
    train_total = extras.get("train_pairs_total", 1)

    if train_passed == 0:
        if extras.get("first_train_dims_correct"):
            return 0.15
        return 0.1

    # Passed some training pairs (0.2 to 0.4 range)
    if train_passed < train_total:
        return 0.2 + 0.2 * (train_passed / train_total)

    # Passed ALL training pairs
    if not extras.get("test_ran"):
        return 0.4

    if extras.get("test_correct"):
        return 1.0

    # Partial test credit (0.5 to 0.9 range)
    similarity = extras.get("test_similarity", 0.0)
    return 0.5 + 0.4 * similarity


# =============================================================================
# Metric Functions (zero-weight, for logging only)
# =============================================================================

def code_extracted_metric(state: State, **kwargs) -> float:
    return 1.0 if state.get("extras", {}).get("code_extracted") else 0.0

def code_compiles_metric(state: State, **kwargs) -> float:
    return 1.0 if state.get("extras", {}).get("code_compiles") else 0.0

def train_pass_rate_metric(state: State, **kwargs) -> float:
    extras = state.get("extras", {})
    total = extras.get("train_pairs_total", 0)
    if total == 0:
        return 0.0
    return extras.get("train_pairs_passed", 0) / total

def all_train_pass_metric(state: State, **kwargs) -> float:
    extras = state.get("extras", {})
    total = extras.get("train_pairs_total", 0)
    passed = extras.get("train_pairs_passed", 0)
    return 1.0 if total > 0 and passed == total else 0.0

def test_correct_metric(state: State, **kwargs) -> float:
    return 1.0 if state.get("extras", {}).get("test_correct") else 0.0

def test_similarity_metric(state: State, **kwargs) -> float:
    return state.get("extras", {}).get("test_similarity", 0.0)


# =============================================================================
# CodegenSolverEnv
# =============================================================================

class CodegenSolverEnv(MultiAgentEnv):
    """
    Single-actor, single-turn environment for ARC codegen training.

    The model receives an ARC task and must write a solver() function.
    on_turn_complete runs the full evaluation pipeline and stores results
    in state["extras"] for the shaped reward function.
    """

    def __init__(self, actor_id: str = "codegen", **kwargs):
        self._actor_id = actor_id
        self.actors = [actor_id]
        self.name = "arc_codegen"
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
        """Extract code, run evaluation pipeline, store results in extras."""
        response_text = _get_last_response_text(state)
        if not response_text:
            return

        info = state.get("info", {})
        if isinstance(info, str):
            info = json.loads(info)

        train_pairs = json.loads(info["train_pairs"])
        test_input = json.loads(info["test_input"])
        test_output = json.loads(info.get("test_output", "[]"))

        eval_results = evaluate_codegen_response(
            response_text, train_pairs, test_input, test_output
        )
        state["extras"].update(eval_results)
        state["extras"]["full_response"] = response_text

    @vf.stop
    async def single_turn_done(self, state: State) -> bool:
        """Stop after 1 turn."""
        return len(state.get("trajectory", [])) >= 1


# =============================================================================
# Dataset Preparation
# =============================================================================

def max_grid_dim(grid: Grid) -> int:
    """Return the max dimension (rows or cols) of a grid."""
    if not grid:
        return 0
    return max(len(grid), max((len(r) for r in grid), default=0))


def load_and_prepare_dataset(
    hf_dataset: str = HF_DATASET_REPO,
    split: str = "train",
    num_examples: int | None = None,
    prompt_version: str = "v1b",
    max_dim: int = 30,
) -> Dataset:
    """
    Load ARC tasks from HuggingFace and prepare for vf-eval.

    Filters tasks by max grid dimension and formats into the standard
    dataset columns: prompt, answer, info, task, example_id.
    """
    from datasets import load_dataset

    raw = load_dataset(hf_dataset, split=split)
    logger.info(f"Loaded {len(raw)} tasks from {hf_dataset} (split={split})")

    records = []
    for i in range(len(raw)):
        row = raw[i]
        train_pairs = json.loads(row["train_pairs"])
        test_input = json.loads(row["test_input"])
        test_output_str = row.get("test_output", "") or ""

        # Filter by max grid dimension
        all_grids = [p["input"] for p in train_pairs] + [p["output"] for p in train_pairs] + [test_input]
        if test_output_str:
            all_grids.append(json.loads(test_output_str))

        if any(max_grid_dim(g) > max_dim for g in all_grids):
            continue

        prompt_text = build_codegen_prompt(train_pairs, test_input, version=prompt_version)

        records.append({
            "prompt": [{"role": "user", "content": prompt_text}],
            "answer": test_output_str,
            "info": json.dumps({
                "train_pairs": row["train_pairs"],
                "test_input": row["test_input"],
                "test_output": test_output_str,
                "task_id": row["task_id"],
            }),
            "task": "arc_codegen",
            "example_id": i,
        })

    if num_examples and len(records) > num_examples:
        records = records[:num_examples]

    logger.info(f"Prepared {len(records)} tasks (filtered from {len(raw)}, max_dim={max_dim})")
    return Dataset.from_list(records)


# =============================================================================
# Rubric
# =============================================================================

def create_codegen_rubric() -> MultiAgentRubric:
    """Create rubric with shaped reward + logging metrics for codegen actor."""
    rubric = MultiAgentRubric()

    rubric.add_actor_reward_func("codegen", codegen_reward, weight=1.0)

    rubric.add_actor_metric("codegen", code_extracted_metric)
    rubric.add_actor_metric("codegen", code_compiles_metric)
    rubric.add_actor_metric("codegen", train_pass_rate_metric)
    rubric.add_actor_metric("codegen", all_train_pass_metric)
    rubric.add_actor_metric("codegen", test_correct_metric)
    rubric.add_actor_metric("codegen", test_similarity_metric)

    return rubric


# =============================================================================
# Entry Point
# =============================================================================

def load_environment(
    hf_dataset: str = HF_DATASET_REPO,
    split: str = "train",
    num_examples: int | None = None,
    prompt_version: str = "v1b",
    max_dim: int = 30,
) -> CodegenSolverEnv:
    """
    Entry point called by vf-eval / prime-rl.

    Args:
        hf_dataset: HuggingFace dataset repo.
        split: Dataset split — "train" or "eval".
        num_examples: Max tasks to include (None = all).
        prompt_version: "v1b" (simple) or "v4" (expert).
        max_dim: Filter out tasks with grids larger than this.
    """
    dataset = load_and_prepare_dataset(
        hf_dataset, split, num_examples, prompt_version, max_dim
    )
    rubric = create_codegen_rubric()

    codegen_actor = Actor(
        id="codegen",
        system_prompt=(
            "You are an expert Python programmer specializing in ARC-AGI puzzles. "
            "Write a solver(input_grid) function that implements the transformation."
        ),
        is_trainable=True,
    )

    env = CodegenSolverEnv(
        actor_id="codegen",
        rubric=rubric,
        dataset=dataset,
    )

    Protocol(actors=[codegen_actor], envs=[env])

    return env
