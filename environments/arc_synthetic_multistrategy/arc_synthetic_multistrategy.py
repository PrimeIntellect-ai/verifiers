"""
ARC Synthetic Multi-Strategy Training Environment.

Trains two codegen strategies on synthetic ARC-AGI-2 tasks:
- codegen_v1b: Write solver code (minimal prompt)
- codegen_v4: Write solver code (structured analysis prompt)

Supports filtering by difficulty level and task type.

Usage:
    prime env install arc-synthetic-multistrategy
    prime eval run arc-synthetic-multistrategy -m Qwen/Qwen3-4B-Instruct-2507 -n 10 -r 2
"""

from __future__ import annotations

import asyncio
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
from verifiers.envs.agent import Agent
from verifiers.envs.multiagent_env import MultiAgentEnv
from verifiers.envs.registry import Registry
from verifiers.rubrics.multiagent_rubric import MultiAgentRubric
from verifiers.types import Messages, State

logger = logging.getLogger(__name__)

HF_DATASET_REPO = "bhoy/arc-synthetic"
Grid = List[List[int]]


# =============================================================================
# Grid Utilities
# =============================================================================

def format_grid(grid: Grid) -> str:
    if grid is None:
        return ""
    return "\n".join(",".join(str(c) for c in row) for row in grid)


def grids_equal(a: Grid | None, b: Grid | None) -> bool:
    if a is None or b is None:
        return False
    if len(a) != len(b):
        return False
    return all(row_a == row_b for row_a, row_b in zip(a, b))


def grid_similarity(predicted: Grid | None, expected: Grid | None) -> float:
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
    if predicted is None or expected is None:
        return False
    if len(predicted) != len(expected):
        return False
    return all(len(pr) == len(er) for pr, er in zip(predicted, expected))


# =============================================================================
# Code Extraction
# =============================================================================

def _extract_solver_code(llm_response: str) -> str | None:
    code_search_area = None

    if "### FINAL SOLUTION ###" in llm_response:
        parts = llm_response.split("### FINAL SOLUTION ###")
        code_search_area = parts[-1]

    pattern = r"```python(.*?)```"
    if not code_search_area:
        blocks = re.findall(pattern, llm_response, re.DOTALL)
        for block in reversed(blocks):
            if "def solver" in block:
                return block.strip()

    if code_search_area:
        match = re.search(pattern, code_search_area, re.DOTALL)
        if match:
            return match.group(1).strip()
        if "def solver" in code_search_area:
            lines = code_search_area.splitlines()
            for i, line in enumerate(lines):
                if "def solver" in line:
                    return "\n".join(lines[i:])

    if "def solver" in llm_response:
        lines = llm_response.splitlines()
        for i, line in enumerate(lines):
            if "def solver" in line:
                return "\n".join(lines[i:])

    return None


# =============================================================================
# Sandbox Execution
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

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        json.dump({"ok": False, "error": f"{type(e).__name__}: {str(e)}"}, sys.stdout)
"""


def run_solver_in_sandbox(code: str, input_grid: list, timeout_s: float = 10.0) -> list | None:
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
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, **kwargs,
        )
        stdout_data, _ = p.communicate(input=json.dumps(payload), timeout=timeout_s)

        if p.returncode != 0 or not stdout_data.strip():
            return None

        result = json.loads(stdout_data)
        if result.get("ok"):
            out = result["output"]
            if isinstance(out, list) and len(out) > 0 and isinstance(out[0], list):
                return out
        return None
    except (subprocess.TimeoutExpired, Exception):
        return None
    finally:
        try:
            os.remove(driver_path)
        except OSError:
            pass


# =============================================================================
# Prompt Building
# =============================================================================

def build_codegen_prompt(train_pairs: list[dict], test_input: Grid, version: str = "v1b") -> str:
    if version == "v4":
        return _build_codegen_prompt_v4(train_pairs, test_input)
    return _build_codegen_prompt_v1b(train_pairs, test_input)


def _build_codegen_prompt_v1b(train_pairs: list[dict], test_input: Grid) -> str:
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


def _build_codegen_prompt_v4(train_pairs: list[dict], test_input: Grid) -> str:
    lines = ["[ARC TASK DATA START]", "", "Training Pairs:"]
    for idx, pair in enumerate(train_pairs, start=1):
        lines.append(f"Pair {idx}:")
        lines.append("input:")
        lines.append(format_grid(pair["input"]))
        lines.append("output:")
        lines.append(format_grid(pair["output"]))
        lines.append("")
    lines.extend([
        "Input-only training data (Probe Inputs):",
        "Probe 1:", "input:", format_grid(test_input), "",
        "[ARC TASK DATA END]", "",
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
    trajectory = state.get("trajectory", [])
    if not trajectory:
        return ""
    last_completion = trajectory[-1].get("completion", [])
    if not last_completion:
        return ""
    return last_completion[-1].get("content", "")


def _get_train_pairs(state: State) -> list[dict]:
    info = state.get("info", {})
    if isinstance(info, str):
        info = json.loads(info)
    return json.loads(info["train_pairs"])


def _get_test_input(state: State) -> Grid:
    info = state.get("info", {})
    if isinstance(info, str):
        info = json.loads(info)
    return json.loads(info["test_input"])


def _get_test_output(state: State) -> Grid | None:
    info = state.get("info", {})
    if isinstance(info, str):
        info = json.loads(info)
    raw = info.get("test_output", "")
    if raw:
        return json.loads(raw)
    return None


# =============================================================================
# Codegen Evaluation
# =============================================================================

_SANDBOX_SEMAPHORE: asyncio.Semaphore | None = None
MAX_CONCURRENT_SANDBOXES = 32


def _get_sandbox_semaphore() -> asyncio.Semaphore:
    global _SANDBOX_SEMAPHORE
    if _SANDBOX_SEMAPHORE is None:
        _SANDBOX_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_SANDBOXES)
    return _SANDBOX_SEMAPHORE


async def evaluate_codegen_response(
    response_text: str,
    train_pairs: list[dict],
    test_input: Grid,
    test_output: Grid,
    task_id: str = "",
) -> dict[str, Any]:
    sem = _get_sandbox_semaphore()
    extras: dict[str, Any] = {}

    code = _extract_solver_code(response_text)
    extras["code_extracted"] = code is not None
    if code is None:
        return extras

    compile(code, "<solver>", "exec")
    extras["code_compiles"] = True

    train_total = len(train_pairs)

    async def run_pair(i: int, pair: dict) -> tuple[int, list | None]:
        async with sem:
            result = await asyncio.to_thread(run_solver_in_sandbox, code, pair["input"], 10.0)
        return i, result

    pair_results = await asyncio.gather(*[run_pair(i, p) for i, p in enumerate(train_pairs)])

    train_passed = 0
    train_ran = 0
    first_train_dims_correct = False

    for i, result in sorted(pair_results, key=lambda x: x[0]):
        if result is not None:
            train_ran += 1
            if grids_equal(result, train_pairs[i]["output"]):
                train_passed += 1
            elif i == 0:
                first_train_dims_correct = dims_match(result, train_pairs[i]["output"])
        elif i == 0:
            first_train_dims_correct = False

    extras["train_results_count"] = train_ran
    extras["train_pairs_passed"] = train_passed
    extras["train_pairs_total"] = train_total
    extras["first_train_dims_correct"] = first_train_dims_correct

    if train_passed < train_total:
        extras["test_ran"] = False
        extras["test_correct"] = False
        extras["test_similarity"] = 0.0
        return extras

    async with sem:
        test_result = await asyncio.to_thread(run_solver_in_sandbox, code, test_input, 10.0)
    extras["test_ran"] = test_result is not None

    if test_result is not None:
        extras["test_correct"] = grids_equal(test_result, test_output)
        extras["test_similarity"] = grid_similarity(test_result, test_output)
        extras["predicted_grid"] = test_result
        if extras["test_correct"]:
            print(f"[arc_synth] [{task_id}] codegen *** SOLVED! ***", flush=True)
    else:
        extras["test_correct"] = False
        extras["test_similarity"] = 0.0

    return extras


# =============================================================================
# Reward Functions
# =============================================================================

def codegen_reward(state: State, **kwargs) -> float:
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

    if train_passed < train_total:
        return 0.2 + 0.2 * (train_passed / train_total)

    if not extras.get("test_ran"):
        return 0.4

    if extras.get("test_correct"):
        return 1.0

    similarity = extras.get("test_similarity", 0.0)
    return 0.5 + 0.4 * similarity


# =============================================================================
# Metric Functions (zero-weight, for logging)
# =============================================================================

def code_extracted_metric(state: State, **kwargs) -> float:
    return 1.0 if state.get("extras", {}).get("code_extracted") else 0.0

def code_compiles_metric(state: State, **kwargs) -> float:
    return 1.0 if state.get("extras", {}).get("code_compiles") else 0.0

def train_pass_rate_metric(state: State, **kwargs) -> float:
    extras = state.get("extras", {})
    total = extras.get("train_pairs_total", 0)
    return extras.get("train_pairs_passed", 0) / total if total else 0.0

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
# Candidate Collection
# =============================================================================

def collect_candidates(
    candidates: dict[tuple, dict],
    reasoning_store: dict[str, str],
    states: list[State],
) -> None:
    for state in states:
        extras = state.get("extras", {})
        predicted = extras.get("predicted_grid")
        if predicted is None:
            continue

        grid_tuple = tuple(tuple(row) for row in predicted)
        actor_id = extras.get("current_actor_id", "unknown")
        model = state.get("model", "")
        run_id = f"{actor_id}_{model}"

        if grid_tuple not in candidates:
            expected = _get_test_output(state)
            candidates[grid_tuple] = {
                "grid": predicted,
                "count": 0,
                "models": [],
                "is_correct": grids_equal(predicted, expected) if expected else None,
            }

        candidates[grid_tuple]["count"] += 1
        candidates[grid_tuple]["models"].append(run_id)

        full_response = extras.get("full_response", "")
        if full_response:
            reasoning_store[run_id] = full_response


# =============================================================================
# Sub-Environments
# =============================================================================

class CodegenEnv(MultiAgentEnv):
    """Single-turn codegen solver (used for v1b and v4 strategies)."""

    def __init__(self, actor_id: str, env_name: str, **kwargs):
        self._actor_id = actor_id
        self.actors = [actor_id]
        self.name = env_name
        super().__init__(max_turns=1, **kwargs)

    def get_initial_actor(self, state: State) -> str:
        return self._actor_id

    def get_next_actor(self, state: State) -> str:
        return self._actor_id

    async def build_actor_prompt(self, actor_id: str, state: State) -> Messages:
        messages: Messages = []
        for msg in state.get("prompt", []):
            if msg.get("role") != "system":
                messages.append(msg)
        return messages

    async def on_turn_complete(self, state: State) -> None:
        response_text = _get_last_response_text(state)
        if not response_text:
            return

        train_pairs = _get_train_pairs(state)
        test_input = _get_test_input(state)
        test_output = _get_test_output(state) or []
        info = state.get("info", {})
        if isinstance(info, str):
            info = json.loads(info)
        task_id = info.get("task_id", "unknown")

        eval_results = await evaluate_codegen_response(
            response_text, train_pairs, test_input, test_output, task_id=task_id,
        )

        strategy = "codegen_v4" if "v4" in self.name else "codegen_v1b"
        reward = codegen_reward({"extras": eval_results})
        print(f"[arc_synth] [{task_id}] {strategy} reward={reward:.2f}", flush=True)

        state["extras"].update(eval_results)
        state["extras"]["full_response"] = response_text
        state["extras"]["strategy"] = strategy

    @vf.stop
    async def codegen_done(self, state: State) -> bool:
        return len(state.get("trajectory", [])) >= 1


# =============================================================================
# Pipeline Environment
# =============================================================================

class ArcSyntheticMultistrategyEnv(MultiAgentEnv):
    """
    Multi-strategy pipeline: codegen_v1b (parent) + codegen_v4 (child).
    """

    def __init__(self, **kwargs):
        self._actor_id = "codegen_v1b"
        self.actors = ["codegen_v1b", "codegen_v4"]
        self.name = "arc_synthetic_multistrategy"
        super().__init__(max_turns=1, **kwargs)

    def get_initial_actor(self, state: State) -> str:
        return self._actor_id

    def get_next_actor(self, state: State) -> str:
        return self._actor_id

    async def build_actor_prompt(self, actor_id: str, state: State) -> Messages:
        messages: Messages = []
        for msg in state.get("prompt", []):
            if msg.get("role") != "system":
                messages.append(msg)
        return messages

    def create_actor_states(self, state, actor_ids=None):
        if actor_ids is None:
            actor_ids = self.actors
        result = []
        if "codegen_v1b" in actor_ids:
            result.extend(super().create_actor_states(state, actor_ids=["codegen_v1b"]))
        for child_state in state.get("child_states", []):
            child_actor_id = child_state.get("extras", {}).get("current_actor_id")
            if child_actor_id and child_actor_id in actor_ids:
                result.append(child_state)
        return result

    async def on_turn_complete(self, state: State) -> None:
        train_pairs = _get_train_pairs(state)
        test_input = _get_test_input(state)
        test_output = _get_test_output(state) or []
        answer_str = state.get("answer", "")
        info_dict = state.get("info", {})
        if isinstance(info_dict, str):
            info_dict = json.loads(info_dict)
        task_id = info_dict.get("task_id", "unknown")
        idx = state.get("example_id", 0)

        candidates: dict[tuple, dict] = {}
        reasoning_store: dict[str, str] = {}

        # --- Evaluate parent (codegen_v1b) ---
        response_text = _get_last_response_text(state)
        if response_text:
            eval_results = await evaluate_codegen_response(
                response_text, train_pairs, test_input, test_output, task_id=task_id,
            )
            state["extras"].update(eval_results)
            state["extras"]["full_response"] = response_text
            state["extras"]["strategy"] = "codegen_v1b"

            reward = codegen_reward({"extras": eval_results})
            print(f"[arc_synth] [{task_id}] codegen_v1b reward={reward:.2f}", flush=True)

            predicted = eval_results.get("predicted_grid") if eval_results.get("test_ran") else None
            if predicted:
                grid_tuple = tuple(tuple(row) for row in predicted)
                candidates[grid_tuple] = {
                    "grid": predicted, "count": 1, "models": ["codegen_v1b"],
                    "is_correct": grids_equal(predicted, test_output) if test_output else None,
                }
                reasoning_store["codegen_v1b"] = response_text

        # --- Spawn child: codegen_v4 ---
        v4_prompt = build_codegen_prompt(train_pairs, test_input, version="v4")
        child_inputs = [{
            "task": "codegen_v4",
            "prompt": [{"role": "user", "content": v4_prompt}],
            "answer": answer_str, "example_id": idx, "info": info_dict,
        }]

        child_states = await self.registry.spawn(
            inputs=child_inputs, client=state["client"],
            model=state["model"], sampling_args=state.get("sampling_args"),
            score=False,
        )
        collect_candidates(candidates, reasoning_store, child_states)
        for s in child_states:
            state["child_states"].append(s)

        num_candidates = len(candidates)
        total_votes = sum(c["count"] for c in candidates.values())
        print(
            f"[arc_synth] [{task_id}] {num_candidates} candidates, "
            f"{total_votes} total votes",
            flush=True,
        )

        correct_candidates = [c for c in candidates.values() if c.get("is_correct")]
        state["extras"]["num_candidates"] = num_candidates
        state["extras"]["has_correct_candidate"] = len(correct_candidates) > 0

    @vf.stop
    async def pipeline_done(self, state: State) -> bool:
        return len(state.get("trajectory", [])) >= 1


# =============================================================================
# Dataset Preparation
# =============================================================================

def max_grid_dim(grid: Grid) -> int:
    if not grid:
        return 0
    return max(len(grid), max((len(r) for r in grid), default=0))


def load_and_prepare_dataset(
    hf_dataset: str = HF_DATASET_REPO,
    split: str = "train",
    num_examples: int | None = None,
    max_dim: int = 100,
    sort_by_size: bool = False,
    levels: list[int] | None = None,
    task_types: list[str] | None = None,
) -> Dataset:
    from datasets import load_dataset

    raw = load_dataset(hf_dataset, split=split)
    logger.info(f"Loaded {len(raw)} tasks from {hf_dataset} (split={split})")

    if levels:
        raw = raw.filter(lambda x: x["level"] in levels)
        logger.info(f"Filtered to levels {levels}: {len(raw)} tasks")

    if task_types:
        raw = raw.filter(lambda x: x["task_type"] in task_types)
        logger.info(f"Filtered to task_types {task_types}: {len(raw)} tasks")

    records = []
    for i in range(len(raw)):
        row = raw[i]
        train_pairs = json.loads(row["train_pairs"])
        test_input = json.loads(row["test_input"])
        test_output_str = row.get("test_output", "") or ""

        all_grids = [p["input"] for p in train_pairs] + [p["output"] for p in train_pairs] + [test_input]
        if test_output_str:
            all_grids.append(json.loads(test_output_str))

        task_max_dim = max(max_grid_dim(g) for g in all_grids)
        if task_max_dim > max_dim:
            continue

        prompt_text = build_codegen_prompt(train_pairs, test_input, version="v1b")

        records.append({
            "prompt": [{"role": "user", "content": prompt_text}],
            "answer": test_output_str,
            "info": json.dumps({
                "train_pairs": row["train_pairs"],
                "test_input": row["test_input"],
                "test_output": test_output_str,
                "task_id": row["task_id"],
            }),
            "task": "arc_synthetic_multistrategy",
            "example_id": i,
            "_max_dim": task_max_dim,
        })

    if sort_by_size:
        records.sort(key=lambda r: (r["_max_dim"], r["example_id"]))

    if num_examples and len(records) > num_examples:
        records = records[:num_examples]

    for r in records:
        del r["_max_dim"]

    logger.info(f"Prepared {len(records)} tasks (filtered from {len(raw)}, max_dim={max_dim}, sort_by_size={sort_by_size})")
    return Dataset.from_list(records)


# =============================================================================
# Rubric
# =============================================================================

def create_rubric() -> MultiAgentRubric:
    rubric = MultiAgentRubric()

    rubric.add_actor_reward_func("codegen_v1b", codegen_reward, weight=1.0)
    rubric.add_actor_metric("codegen_v1b", code_extracted_metric)
    rubric.add_actor_metric("codegen_v1b", code_compiles_metric)
    rubric.add_actor_metric("codegen_v1b", train_pass_rate_metric)
    rubric.add_actor_metric("codegen_v1b", all_train_pass_metric)
    rubric.add_actor_metric("codegen_v1b", test_correct_metric)
    rubric.add_actor_metric("codegen_v1b", test_similarity_metric)

    rubric.add_actor_reward_func("codegen_v4", codegen_reward, weight=1.0)
    rubric.add_actor_metric("codegen_v4", code_extracted_metric)
    rubric.add_actor_metric("codegen_v4", code_compiles_metric)
    rubric.add_actor_metric("codegen_v4", train_pass_rate_metric)
    rubric.add_actor_metric("codegen_v4", all_train_pass_metric)
    rubric.add_actor_metric("codegen_v4", test_correct_metric)
    rubric.add_actor_metric("codegen_v4", test_similarity_metric)

    return rubric


# =============================================================================
# Entry Point
# =============================================================================

CODEGEN_SYSTEM_PROMPT = (
    "You write Python code to solve ARC-AGI puzzles. "
    "Given training input/output pairs, write a concise `def solver(input_grid)` function. "
    "Output ONLY a single python code block. No explanation, no comments, no analysis. "
    "The function takes a 2D list of integers and must return a 2D list of integers. "
    "You may use numpy, scipy, and cv2."
)


def load_environment(
    hf_dataset: str = HF_DATASET_REPO,
    split: str = "train",
    num_examples: int | None = None,
    max_dim: int = 100,
    sort_by_size: bool = False,
    levels: list[int] | None = None,
    task_types: list[str] | None = None,
    actor_endpoints: dict[str, str] | None = None,
    **kwargs,
) -> ArcSyntheticMultistrategyEnv:
    """Entry point for vf-eval / prime-rl."""
    dataset = load_and_prepare_dataset(
        hf_dataset, split, num_examples, max_dim, sort_by_size,
        levels=levels, task_types=task_types,
    )
    rubric = create_rubric()

    actor_endpoints = actor_endpoints or {}
    codegen_v1b_url = actor_endpoints.get("codegen_v1b")
    codegen_v4_url = actor_endpoints.get("codegen_v4")

    codegen_v1b_agent = Agent(
        id="codegen_v1b", system_prompt=CODEGEN_SYSTEM_PROMPT, is_trainable=True,
        model="Qwen/Qwen3-4B-Instruct-2507",
    )
    codegen_v4_agent = Agent(
        id="codegen_v4", system_prompt=CODEGEN_SYSTEM_PROMPT, is_trainable=True,
        model="Qwen/Qwen3-4B-Instruct-2507",
    )

    if codegen_v1b_url:
        from openai import AsyncOpenAI
        codegen_v1b_agent.client = AsyncOpenAI(base_url=codegen_v1b_url, api_key="EMPTY")
    if codegen_v4_url:
        from openai import AsyncOpenAI
        codegen_v4_agent.client = AsyncOpenAI(base_url=codegen_v4_url, api_key="EMPTY")

    pipeline_env = ArcSyntheticMultistrategyEnv(rubric=rubric, dataset=dataset)
    codegen_v4_env = CodegenEnv(actor_id="codegen_v4", env_name="codegen_v4")

    Registry(
        agents=[codegen_v1b_agent, codegen_v4_agent],
        envs=[pipeline_env, codegen_v4_env],
    )

    return pipeline_env
