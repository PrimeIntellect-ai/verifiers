import json
import re
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path

from datasets import Dataset

import verifiers as vf

SYSTEM_PROMPT = """You are a pragmatic Rust programmer who practices test-driven development.
Given the prompt, write one self-contained Rust solution using only the standard library.

Requirements:
- Respond with exactly one ```rust code block.
- Include the requested function implementation.
- Include a #[cfg(test)] mod tests block using super::*.
- Include multiple assert!/assert_eq! statements.
- Do not add a main function."""

TASKS: list[dict[str, str]] = [
    {
        "task_id": "sum_even_numbers",
        "rust_prompt": "Write fn sum_even_numbers(nums: &[i32]) -> i32 that returns the sum of the even integers in nums.",
        "required_function": "sum_even_numbers",
    },
    {
        "task_id": "is_palindrome",
        "rust_prompt": "Write fn is_palindrome(s: &str) -> bool that returns true when s is a palindrome ignoring case and non-alphanumeric characters.",
        "required_function": "is_palindrome",
    },
    {
        "task_id": "dedup_sorted",
        "rust_prompt": "Write fn dedup_sorted(nums: Vec<i32>) -> Vec<i32> that removes duplicate values from an already-sorted vector while preserving order.",
        "required_function": "dedup_sorted",
    },
    {
        "task_id": "word_count",
        "rust_prompt": "Write fn word_count(text: &str) -> usize that counts whitespace-separated words.",
        "required_function": "word_count",
    },
    {
        "task_id": "fibonacci",
        "rust_prompt": "Write fn fibonacci(n: u32) -> u64 that returns the nth Fibonacci number with fibonacci(0)=0 and fibonacci(1)=1.",
        "required_function": "fibonacci",
    },
]


def build_dataset() -> Dataset:
    return Dataset.from_list(
        [
            {
                "question": task["rust_prompt"],
                "answer": task["required_function"],
                "info": {"task_id": task["task_id"]},
            }
            for task in TASKS
        ]
    )


def extract_rust_code(response: str) -> str:
    match = re.search(r"```rust\s*(.*?)\s*```", response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return response.strip()


def extract_test_module(code: str) -> str | None:
    match = re.search(r"#\[cfg\(test\)\]\s*mod\s+tests\s*\{", code)
    if not match:
        return None
    return code[match.start() :]


def count_assertions(code: str) -> int:
    return len(re.findall(r"\bassert(?:_eq|_ne)?!\s*\(", code))


def has_single_rust_block(response: str) -> bool:
    blocks = re.findall(r"```rust\s*.*?\s*```", response, re.DOTALL | re.IGNORECASE)
    return len(blocks) == 1


def rust_project_files(code: str) -> dict[str, str]:
    return {
        "Cargo.toml": textwrap.dedent(
            """
            [package]
            name = "rust-cargo-verifier"
            version = "0.1.0"
            edition = "2021"

            [dependencies]
            """
        ).strip()
        + "\n",
        "src/lib.rs": textwrap.dedent(
            f"""
            #![allow(dead_code)]

            {code}
            """
        ).strip()
        + "\n",
    }


def run_cargo_tool(code: str, tool: str, timeout: int = 20) -> tuple[bool, str]:
    if shutil.which("cargo") is None:
        return False, "cargo executable not found"

    with tempfile.TemporaryDirectory(prefix="vf-rust-cargo-") as tmpdir:
        root = Path(tmpdir)
        for relative_path, content in rust_project_files(code).items():
            path = root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)

        command = ["cargo", tool, "--quiet"]
        if tool == "clippy":
            command.extend(["--", "-D", "warnings"])
        try:
            result = subprocess.run(
                command,
                cwd=root,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except Exception as exc:
            return False, str(exc)

    output = (result.stdout + "\n" + result.stderr).strip()
    return result.returncode == 0, output


def format_reward(completion: str, answer: str, **kwargs) -> float:
    _ = answer, kwargs
    code = extract_rust_code(completion)
    score = 0.0
    if has_single_rust_block(completion):
        score += 0.35
    if "fn " in code:
        score += 0.20
    if "fn main" not in code:
        score += 0.15
    if extract_test_module(code) is not None:
        score += 0.20
    if count_assertions(code) >= 2:
        score += 0.10
    return score


def required_function_reward(completion: str, answer: str, **kwargs) -> float:
    _ = kwargs
    code = extract_rust_code(completion)
    return 1.0 if re.search(rf"\bfn\s+{re.escape(answer)}\s*\(", code) else 0.0


def assertions_reward(completion: str, answer: str, **kwargs) -> float:
    _ = answer, kwargs
    code = extract_rust_code(completion)
    assertions = count_assertions(code)
    if assertions >= 4:
        return 1.0
    return 0.25 * assertions


def cargo_build_reward(completion: str, answer: str, **kwargs) -> float:
    _ = answer, kwargs
    passed, _output = run_cargo_tool(extract_rust_code(completion), "build")
    return 1.0 if passed else 0.0


def cargo_clippy_reward(completion: str, answer: str, **kwargs) -> float:
    _ = answer, kwargs
    passed, _output = run_cargo_tool(extract_rust_code(completion), "clippy")
    return 1.0 if passed else 0.0


def cargo_test_reward(completion: str, answer: str, **kwargs) -> float:
    _ = answer, kwargs
    code = extract_rust_code(completion)
    if extract_test_module(code) is None:
        return 0.0
    passed, _output = run_cargo_tool(code, "test")
    return 1.0 if passed else 0.0


def load_environment() -> vf.Environment:
    rubric = vf.Rubric(
        funcs=[
            format_reward,
            required_function_reward,
            assertions_reward,
            cargo_build_reward,
            cargo_clippy_reward,
            cargo_test_reward,
        ],
        weights=[0.5, 0.5, 0.5, 1.0, 1.0, 2.0],
    )
    return vf.SingleTurnEnv(
        dataset=build_dataset,
        system_prompt=SYSTEM_PROMPT,
        parser=vf.Parser(),
        rubric=rubric,
    )


def main() -> None:
    _ = load_environment()
    print(json.dumps({"environment": "rust-cargo", "num_tasks": len(build_dataset())}))


if __name__ == "__main__":
    main()
