import ast
import json
import re
from textwrap import dedent

from datasets import Dataset, load_dataset

import verifiers as vf

DEFAULT_PROMPT_TEMPLATE = """PROBLEM DESCRIPTION:
You will be provided with a scientific research coding subproblem. Generate the disciplinary knowledge necessary for solving the next step, then develop a Python solution focused on this step.

PREVIOUS STEPS DESCRIPTION:
{problem_steps_str}

NEXT STEP - PROBLEM DESCRIPTION AND FUNCTION HEADER:
{next_step_str}

DEPENDENCIES:
Use only the following dependencies in your solution. Do not include these dependencies at the beginning of your code.
{dependencies}

RESPONSE GUIDELINES:
1. Start with the scientific background required for the next step, formatted as a Python comment beginning with `# Background:`.
2. Then write the complete Python implementation for this next step in one fenced ```python code block.
3. Follow the supplied function/class header exactly.
4. Do not include previous function code, example usage, or test code.
"""

FALLBACK_PROBLEMS = [
    {
        "problem_name": "Periodic boundary wrapping",
        "problem_id": "fallback-1",
        "required_dependencies": "import numpy as np",
        "sub_steps": [
            {
                "step_number": "fallback-1.1",
                "step_description_prompt": "Wrap coordinates into a periodic cubic simulation box.",
                "step_background": "Use modulo arithmetic so each coordinate lies in [0, L).",
                "function_header": dedent(
                    '''
                    def wrap(r, L):
                        ''' + '"""Apply periodic boundary conditions to coordinates r for a cubic box of size L."""'
                ).strip(),
                "return_line": "    return coord",
            }
        ],
    }
]


def extract_function_name(function_header: str) -> str:
    match = re.search(r"\b(?:def|class)\s+(\w+)\s*[(:]", function_header)
    if not match:
        raise ValueError("Function or class name not found in header")
    return match.group(1)


def extract_python_code(completion: str) -> str:
    if "```" not in completion:
        return completion.strip()
    python_block = re.search(r"```python\s*(.*?)```", completion, re.DOTALL | re.IGNORECASE)
    if python_block:
        return python_block.group(1).strip()
    generic_block = re.search(r"```\s*(.*?)```", completion, re.DOTALL)
    return generic_block.group(1).strip() if generic_block else completion.strip()


def strip_imports(code: str) -> str:
    lines = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def process_problem_code(problem: dict, step_index: int) -> str:
    sub_step = problem["sub_steps"][step_index]
    return f'{sub_step["function_header"]}\n\n{sub_step["return_line"]}'


def build_prompt(problem: dict, step_index: int, with_background: bool = True) -> str:
    sub_steps = problem["sub_steps"]
    previous_chunks = []
    for previous in sub_steps[:step_index]:
        description = previous["step_description_prompt"]
        if with_background and previous.get("step_background"):
            description = f'{description}\n{previous["step_background"]}'
        previous_chunks.append(description)
        previous_chunks.append(previous["function_header"])
        previous_chunks.append("------")

    current = sub_steps[step_index]
    next_step = current["step_description_prompt"]
    if with_background and current.get("step_background"):
        next_step = f'{next_step}\n{current["step_background"]}'
    next_step = f"{next_step}\n\n{process_problem_code(problem, step_index)}"

    return DEFAULT_PROMPT_TEMPLATE.format(
        problem_steps_str="\n\n".join(previous_chunks[:-1]),
        next_step_str=next_step,
        dependencies=problem.get("required_dependencies", ""),
    )


def normalize_problem(problem: dict, max_steps_per_problem: int | None = None) -> list[dict]:
    rows = []
    sub_steps = problem.get("sub_steps") or []
    if max_steps_per_problem is not None:
        sub_steps = sub_steps[:max_steps_per_problem]
    for step_index, sub_step in enumerate(sub_steps):
        try:
            function_name = extract_function_name(sub_step["function_header"])
        except ValueError:
            continue
        rows.append(
            {
                "question": build_prompt(problem, step_index),
                "answer": function_name,
                "info": {
                    "problem_id": problem.get("problem_id"),
                    "problem_name": problem.get("problem_name"),
                    "step_number": sub_step.get("step_number"),
                    "function_header": sub_step.get("function_header"),
                    "return_line": sub_step.get("return_line"),
                    "required_dependencies": problem.get("required_dependencies", ""),
                },
            }
        )
    return rows


def build_dataset(split: str = "validation", max_problems: int | None = 15, max_steps_per_problem: int | None = None) -> Dataset:
    try:
        raw = list(load_dataset("SciCode1/SciCode", split=split))
    except Exception:
        raw = FALLBACK_PROBLEMS

    if max_problems is not None:
        raw = raw[:max_problems]

    rows = []
    for problem in raw:
        rows.extend(normalize_problem(problem, max_steps_per_problem=max_steps_per_problem))
    return Dataset.from_list(rows)


def syntax_reward(completion: str, answer: str, **kwargs) -> float:
    _ = answer, kwargs
    code = strip_imports(extract_python_code(completion))
    try:
        ast.parse(code)
        return 1.0
    except SyntaxError:
        return 0.0


def function_name_reward(completion: str, answer: str, **kwargs) -> float:
    _ = kwargs
    code = strip_imports(extract_python_code(completion))
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0.0
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) and node.name == answer:
            return 1.0
    return 0.0


def code_block_reward(completion: str, answer: str, **kwargs) -> float:
    _ = answer, kwargs
    return 1.0 if re.search(r"```python\s+.*?```", completion, re.DOTALL | re.IGNORECASE) else 0.0


def no_top_level_test_reward(completion: str, answer: str, **kwargs) -> float:
    _ = answer, kwargs
    code = strip_imports(extract_python_code(completion))
    banned = [r"\bassert\b", r"if\s+__name__\s*==", r"print\s*\(", r"pytest", r"unittest"]
    return 0.0 if any(re.search(pattern, code) for pattern in banned) else 1.0


def background_comment_reward(completion: str, answer: str, **kwargs) -> float:
    _ = answer, kwargs
    code = extract_python_code(completion)
    return 1.0 if re.search(r"^\s*#\s*Background\s*:", code, re.IGNORECASE | re.MULTILINE) else 0.0


def return_statement_reward(completion: str, answer: str, **kwargs) -> float:
    _ = answer, kwargs
    code = strip_imports(extract_python_code(completion))
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0.0
    return 1.0 if any(isinstance(node, ast.Return) for node in ast.walk(tree)) else 0.0


def load_environment(split: str = "validation", max_problems: int | None = 15) -> vf.Environment:
    rubric = vf.Rubric(
        funcs=[
            syntax_reward,
            function_name_reward,
            code_block_reward,
            no_top_level_test_reward,
            background_comment_reward,
            return_statement_reward,
        ],
        weights=[1.0, 2.0, 1.0, 1.0, 0.5, 0.5],
    )
    return vf.SingleTurnEnv(
        dataset=build_dataset(split=split, max_problems=max_problems),
        system_prompt="You are a careful scientific Python programmer. Return only one fenced Python code block.",
        parser=vf.Parser(),
        rubric=rubric,
    )


def main() -> None:
    dataset = build_dataset(max_problems=1)
    _ = load_environment(max_problems=1)
    print(json.dumps({"environment": "scicode", "num_tasks": len(dataset)}))


if __name__ == "__main__":
    main()
