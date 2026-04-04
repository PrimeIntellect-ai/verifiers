"""
ARC Curriculum: 2-LoRA Self-Play Environment.

Each dataset example specifies a base_op. The curriculum generator LoRA
receives the op and outputs a single JSON recipe choosing difficulty
settings (level, post_ops, seed). GRPO's rollouts_per_example=8 naturally
produces 8 difficulty variations per base_op.

The solver LoRA attempts each generated task. Generator is rewarded 1.0 if
the solver gets meaningful signal (reward >= 0.3), 0.0 otherwise. GRPO
compares the 8 difficulty variations: generator learns which settings
produce learnable tasks for each operation.

Actors:
    - curriculum_generator: outputs JSON recipe for a given base_op
    - codegen_v1b: solver (writes Python code)

Usage:
    prime env install arc-curriculum
    prime eval run arc-curriculum -m Qwen/Qwen3-4B-Instruct-2507 -n 10 -r 2
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

from collections import deque

from recipe_executor import (
    L1_OPS, L2_OPS, BASE_OPS, POST_OPS, ALL_OPS, SEEDS,
    execute_recipe, parse_generator_output, generator_reward,
    build_generator_prompt, validate_task, OP_DESCRIPTIONS,
)

logger = logging.getLogger(__name__)

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

def build_codegen_prompt(train_pairs: list[dict], test_input: Grid) -> str:
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

    try:
        compile(code, "<solver>", "exec")
    except SyntaxError:
        extras["code_compiles"] = False
        return extras
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
            print(f"[arc_curriculum] [{task_id}] *** SOLVED! ***", flush=True)
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


def curriculum_generator_reward(state: State, **kwargs) -> float:
    """Generator reward with mastery override.

    Returns 0.0 if the chosen level is mastered (computed in on_turn_complete).
    Otherwise normal binary reward based on solver performance.
    """
    extras = state.get("extras", {})

    if not extras.get("recipe_valid"):
        return 0.0

    if extras.get("level_mastered"):
        return 0.0

    solver_reward = extras.get("solver_reward")
    if solver_reward is None:
        return 0.0

    return generator_reward(solver_reward)


# =============================================================================
# Metric Functions
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

def recipe_valid_metric(state: State, **kwargs) -> float:
    return 1.0 if state.get("extras", {}).get("recipe_valid") else 0.0

def solver_reward_metric(state: State, **kwargs) -> float:
    return state.get("extras", {}).get("solver_reward", 0.0)


# =============================================================================
# Sub-Environments
# =============================================================================

class CodegenEnv(MultiAgentEnv):
    """Single-turn codegen solver."""

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

        info = state.get("info", {})
        if isinstance(info, str):
            info = json.loads(info)

        train_pairs = json.loads(info["train_pairs"])
        test_input = json.loads(info["test_input"])
        test_output = json.loads(info.get("test_output", "[]")) or []
        task_id = info.get("task_id", "unknown")

        eval_results = await evaluate_codegen_response(
            response_text, train_pairs, test_input, test_output, task_id=task_id,
        )

        reward = codegen_reward({"extras": eval_results})
        print(f"[arc_curriculum] [{task_id}] codegen_v1b reward={reward:.2f}", flush=True)

        state["extras"].update(eval_results)
        state["extras"]["full_response"] = response_text
        state["extras"]["strategy"] = "codegen_v1b"

    @vf.stop
    async def codegen_done(self, state: State) -> bool:
        return len(state.get("trajectory", [])) >= 1


# =============================================================================
# Pipeline Environment
# =============================================================================

class ArcCurriculumEnv(MultiAgentEnv):
    """
    2-LoRA self-play pipeline with mastery-based curriculum progression.

    Generator picks level A/B/C. Once the solver's rolling average on a
    level exceeds mastery_threshold, the generator gets 0 reward for that
    level, pushing it toward harder unexplored levels.
    """

    def __init__(self, mastery_threshold=0.7, mastery_window=40, **kwargs):
        self._actor_id = "curriculum_generator"
        self.actors = ["curriculum_generator", "codegen_v1b"]
        self.name = "arc_curriculum"
        self._mastery_threshold = mastery_threshold
        self._mastery_window = mastery_window
        # Keyed by (op, level) → deque of solver rewards
        self._mastery_history: dict[tuple[str, int], deque] = {}
        super().__init__(max_turns=1, **kwargs)

    def _is_mastered(self, op: str, level: int) -> bool:
        history = self._mastery_history.get((op, level))
        if not history or len(history) < 8:
            return False
        return sum(history) / len(history) >= self._mastery_threshold

    def _record_result(self, op: str, level: int, reward: float):
        key = (op, level)
        if key not in self._mastery_history:
            self._mastery_history[key] = deque(maxlen=self._mastery_window)
        self._mastery_history[key].append(reward)

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
        if "curriculum_generator" in actor_ids:
            result.extend(super().create_actor_states(state, actor_ids=["curriculum_generator"]))
        for child_state in state.get("child_states", []):
            child_actor_id = child_state.get("extras", {}).get("current_actor_id")
            if child_actor_id and child_actor_id in actor_ids:
                result.append(child_state)
        return result

    async def on_turn_complete(self, state: State) -> None:
        response_text = _get_last_response_text(state)
        if not response_text:
            state["extras"]["recipe_valid"] = False
            return

        # Get the base_op this example is targeting
        info = state.get("info", {})
        if isinstance(info, str):
            info = json.loads(info)
        target_base_op = info.get("base_op", "")

        recipe = parse_generator_output(response_text)
        if recipe is None:
            state["extras"]["recipe_valid"] = False
            print(f"[arc_curriculum] generator: invalid JSON for {target_base_op}", flush=True)
            return

        # Force base_op and seed from dataset
        recipe["base_op"] = target_base_op
        recipe["seed"] = info.get("seed", SEEDS[0])
        level = recipe.get("level", 2)

        task = execute_recipe(recipe)
        if task is None:
            state["extras"]["recipe_valid"] = False
            print(f"[arc_curriculum] generator: recipe failed ({target_base_op})", flush=True)
            return

        state["extras"]["recipe_valid"] = True
        state["extras"]["recipe"] = recipe
        state["extras"]["generated_task_type"] = target_base_op
        task_id = f"gen_{target_base_op}_L{level}_s{recipe['seed']}"

        print(
            f"[arc_curriculum] generator: recipe={json.dumps(recipe)}",
            flush=True,
        )

        # Spawn one solver child
        child_info = {
            "train_pairs": json.dumps(task["train_pairs"]),
            "test_input": json.dumps(task["test_input"]),
            "test_output": json.dumps(task["test_output"]),
            "task_id": task_id,
        }
        answer_str = json.dumps(task["test_output"])

        v1b_prompt = build_codegen_prompt(task["train_pairs"], task["test_input"])
        v1b_inputs = [{
            "task": "codegen_v1b",
            "prompt": [{"role": "user", "content": v1b_prompt}],
            "answer": answer_str,
            "example_id": state.get("example_id", 0),
            "info": child_info,
        }]

        v1b_states = await self.registry.spawn(
            inputs=v1b_inputs,
            client=state["client"],
            model=state["model"],
            sampling_args=state.get("sampling_args"),
            score=False,
        )

        solver_reward = 0.0
        for s in v1b_states:
            state["child_states"].append(s)
            solver_reward = codegen_reward(s)

        # Track per-(op, level) solver performance
        self._record_result(target_base_op, level, solver_reward)
        mastered = self._is_mastered(target_base_op, level)

        # Generator reward: 0 if level is mastered (push to harder levels)
        if mastered:
            gen_r = 0.0
        else:
            gen_r = generator_reward(solver_reward)

        state["extras"]["solver_reward"] = solver_reward
        state["extras"]["level_mastered"] = mastered

        history = self._mastery_history.get((target_base_op, level))
        avg = sum(history) / len(history) if history else 0.0
        print(
            f"[arc_curriculum] [{task_id}] solver={solver_reward:.2f} gen={gen_r:.1f} {target_base_op}:L{level}_avg={avg:.2f}{' MASTERED' if mastered else ''}",
            flush=True,
        )

    @vf.stop
    async def pipeline_done(self, state: State) -> bool:
        return len(state.get("trajectory", [])) >= 1


# =============================================================================
# Dataset Preparation
# =============================================================================

def load_and_prepare_dataset(
    num_examples: int | None = None,
    levels: list[int] | None = None,
) -> Dataset:
    """Build a dataset with one entry per (base_op, seed) combination.

    13 ops × 8 seeds = 104 entries. Each gets its own GRPO rollout group.

    Args:
        num_examples: Limit total entries (for quick testing).
        levels: Which op tiers to include. [1] = 13 L1 ops, [2] = 18 L2 ops,
                [1,2] = all 31. Defaults to [1].
    """
    levels = levels or [1]
    pool = []
    if 1 in levels:
        pool.extend(L1_OPS)
    if 2 in levels:
        pool.extend(L2_OPS)
    records = []
    for i, base_op in enumerate(pool):
        prompt_text = build_generator_prompt(base_op)
        for seed in SEEDS:
            records.append({
                "prompt": [{"role": "user", "content": prompt_text}],
                "answer": "",
                "info": json.dumps({"base_op": base_op, "seed": seed}),
                "task": "arc_curriculum",
                "example_id": len(records),
            })
    if num_examples and num_examples < len(records):
        records = records[:num_examples]

    logger.info(f"Prepared {len(records)} curriculum examples ({len(pool)} ops × {len(SEEDS)} seeds)")
    return Dataset.from_list(records)


# =============================================================================
# Rubric
# =============================================================================

def create_rubric() -> MultiAgentRubric:
    rubric = MultiAgentRubric()

    rubric.add_actor_reward_func("curriculum_generator", curriculum_generator_reward, weight=1.0)
    rubric.add_actor_metric("curriculum_generator", recipe_valid_metric)
    rubric.add_actor_metric("curriculum_generator", solver_reward_metric)

    rubric.add_actor_reward_func("codegen_v1b", codegen_reward, weight=1.0)
    rubric.add_actor_metric("codegen_v1b", code_extracted_metric)
    rubric.add_actor_metric("codegen_v1b", code_compiles_metric)
    rubric.add_actor_metric("codegen_v1b", train_pass_rate_metric)
    rubric.add_actor_metric("codegen_v1b", all_train_pass_metric)
    rubric.add_actor_metric("codegen_v1b", test_correct_metric)
    rubric.add_actor_metric("codegen_v1b", test_similarity_metric)

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

GENERATOR_SYSTEM_PROMPT = (
    "You are a curriculum designer for an ARC-AGI puzzle solver. "
    "Choose a setting for the given operation. "
    'Output: {"level": "A"}, {"level": "B"}, or {"level": "C"}'
)


def load_environment(
    num_examples: int | None = None,
    levels: list[int] | None = None,
    mastery_threshold: float = 0.7,
    mastery_window: int = 40,
    actor_endpoints: dict[str, str] | None = None,
    **kwargs,
) -> ArcCurriculumEnv:
    """Entry point for vf-eval / prime-rl."""
    dataset = load_and_prepare_dataset(
        num_examples=num_examples,
        levels=levels,
    )
    rubric = create_rubric()

    actor_endpoints = actor_endpoints or {}
    gen_url = actor_endpoints.get("curriculum_generator")
    v1b_url = actor_endpoints.get("codegen_v1b")

    gen_agent = Agent(
        id="curriculum_generator",
        system_prompt=GENERATOR_SYSTEM_PROMPT,
        is_trainable=True,
        model="Qwen/Qwen3-4B-Instruct-2507",
        sampling_args={"temperature": 1.2},
    )
    codegen_v1b_agent = Agent(
        id="codegen_v1b",
        system_prompt=CODEGEN_SYSTEM_PROMPT,
        is_trainable=True,
        model="Qwen/Qwen3-4B-Instruct-2507",
    )

    if gen_url:
        from openai import AsyncOpenAI
        gen_agent.client = AsyncOpenAI(base_url=gen_url, api_key="EMPTY")
    if v1b_url:
        from openai import AsyncOpenAI
        codegen_v1b_agent.client = AsyncOpenAI(base_url=v1b_url, api_key="EMPTY")

    pipeline_env = ArcCurriculumEnv(
        mastery_threshold=mastery_threshold,
        mastery_window=mastery_window,
        rubric=rubric,
        dataset=dataset,
        **kwargs,
    )
    codegen_v1b_env = CodegenEnv(actor_id="codegen_v1b", env_name="codegen_v1b")

    Registry(
        agents=[gen_agent, codegen_v1b_agent],
        envs=[pipeline_env, codegen_v1b_env],
    )

    return pipeline_env
