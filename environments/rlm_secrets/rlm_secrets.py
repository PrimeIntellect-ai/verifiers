"""
RLM Secrets Environment.

A puzzle environment designed to thoroughly test RLM functionality:
- Root-level tools
- Sub-LLM tools
- Sub-LLM calls (llm_batch)
- File operations (listing, reading, deleting)

The puzzle:
1. Files with random names contain random UUIDs
2. Model must discover the correct ordering of files
3. Model must determine which file to keep
4. Model must delete all other files and report the answer

Flow:
1. Root-LLM lists files via bash
2. For each file: read content, call sub-LLM to get decryption code
3. Sub-LLM calls get_code_from_file_data tool, returns code
4. Root-LLM calls decrypt_position to learn file's position
5. Once all positions known, call unveil_file_number with sorted filenames
6. Delete all files except the correct one
7. Set final answer to the position of the kept file
"""

import contextvars
import os
import random
import shutil
import string
import tempfile
from pathlib import Path
from typing import Any

from datasets import Dataset

import verifiers as vf
from verifiers.envs.experimental.rlm_env import RLMEnv
from verifiers.types import State


def generate_random_filename(length: int = 8) -> str:
    """Generate a random lowercase filename with .txt extension."""
    return "".join(random.choices(string.ascii_lowercase, k=length)) + ".txt"


def generate_random_code(length: int = 16) -> str:
    """Generate a random alphanumeric code string."""
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


def generate_random_uuid() -> str:
    """Generate a UUID-format string using seeded random (reproducible)."""
    hex_chars = "0123456789abcdef"
    parts = [
        "".join(random.choices(hex_chars, k=8)),
        "".join(random.choices(hex_chars, k=4)),
        "".join(random.choices(hex_chars, k=4)),
        "".join(random.choices(hex_chars, k=4)),
        "".join(random.choices(hex_chars, k=12)),
    ]
    return "-".join(parts)


def generate_puzzle(num_files: int = 4) -> dict[str, Any]:
    """
    Generate a puzzle configuration.

    Returns a dict with:
    - filenames: list of random filenames
    - contents: list of UUID strings (file contents)
    - correct_order: list where correct_order[i] = sorted position (0-indexed) of filenames[i]
    - correct_position: 1-indexed position of the file to keep
    - codes: dict mapping filename -> decryption code
    - num_files: number of files in the puzzle
    """
    filenames = [generate_random_filename() for _ in range(num_files)]
    contents = [generate_random_uuid() for _ in range(num_files)]

    # correct_order[i] = the sorted position (0-indexed) of filenames[i]
    positions = list(range(num_files))
    random.shuffle(positions)
    correct_order = positions

    # The 1-indexed position of the file to keep
    correct_position = random.randint(1, num_files)

    # Generate unique decryption codes for each file
    codes = {fn: generate_random_code() for fn in filenames}

    return {
        "filenames": filenames,
        "contents": contents,
        "correct_order": correct_order,
        "correct_position": correct_position,
        "codes": codes,
        "num_files": num_files,
    }


class RLMSecretsEnv(RLMEnv):
    """
    RLM environment for testing recursive LLM capabilities via a file puzzle.

    This environment tests:
    - Root-level tool calling (decrypt_position, unveil_file_number)
    - Sub-LLM tool calling (get_code_from_file_data)
    - Sub-LLM invocation via llm_batch
    - File system operations (ls, cat, rm)

    Args:
        num_files: Number of files in each puzzle (default: 4)
        repl_language: REPL language to use, "bash" or "python" (default: "bash")
        **kwargs: Additional arguments passed to RLMEnv
    """

    def __init__(
        self,
        num_files: int = 4,
        repl_language: str = "bash",
        **kwargs,
    ):
        self.num_files = num_files

        # Context var for passing state to sub-tools during sub-LLM execution
        self._sub_tool_state_var: contextvars.ContextVar[State | None] = (
            contextvars.ContextVar("rlm_secrets_sub_tool_state", default=None)
        )

        # Build tools before calling super().__init__
        root_tools = self._build_secrets_root_tools()
        sub_tools = self._build_secrets_sub_tools()

        super().__init__(
            root_tools=root_tools,
            sub_tools=sub_tools,
            repl_language=repl_language,
            retain_filesystem_after_rollout=True,
            **kwargs,
        )

    def _get_puzzle_from_state(self, state: State | None) -> dict[str, Any]:
        """Extract puzzle configuration from state."""
        if state is None:
            return {}
        # First check state["puzzle"] (set in setup_state)
        puzzle = state.get("puzzle")
        if puzzle:
            return puzzle
        # Fallback to info
        info = state.get("info", {})
        return info.get("puzzle", {})

    def _build_secrets_root_tools(self) -> list:
        """Build root-level tools that the main RLM can call."""
        env = self

        async def decrypt_position(file_name: str, code: str) -> str:
            """
            Decrypt the position of a file using its code.

            You must first obtain the code by having a sub-LLM call
            get_code_from_file_data with the filename and its content.

            Args:
                file_name: The exact filename (e.g., "abcdefgh.txt")
                code: The decryption code obtained from a sub-LLM

            Returns:
                The 1-indexed position of this file in the correct order,
                or an error message if the code is invalid.
            """
            context = env._root_tool_context_var.get()
            state = context.get("state") if context else None
            puzzle = env._get_puzzle_from_state(state)

            codes = puzzle.get("codes", {})
            if file_name not in codes:
                return f"Error: File '{file_name}' not found in puzzle"

            if code != codes[file_name]:
                return f"Error: Invalid code '{code}' for file '{file_name}'"

            filenames = puzzle.get("filenames", [])
            correct_order = puzzle.get("correct_order", [])
            try:
                idx = filenames.index(file_name)
                position = correct_order[idx] + 1  # Convert to 1-indexed
                return str(position)
            except (ValueError, IndexError):
                return f"Error: Could not determine position for '{file_name}'"

        async def unveil_file_number(sorted_filenames: list[str]) -> str:
            """
            Reveal which file to keep, given filenames in their correct sorted order.

            Call this after you have determined the position of each file using
            decrypt_position. Pass all filenames sorted by their positions
            (position 1 first, position 2 second, etc.).

            Args:
                sorted_filenames: List of all filenames in ascending position order

            Returns:
                The 1-indexed position of the file to keep, or an error message
                if the order is incorrect.
            """
            context = env._root_tool_context_var.get()
            state = context.get("state") if context else None
            puzzle = env._get_puzzle_from_state(state)

            filenames = puzzle.get("filenames", [])
            correct_order = puzzle.get("correct_order", [])

            if len(sorted_filenames) != len(filenames):
                return (
                    f"Error: Expected {len(filenames)} filenames, "
                    f"got {len(sorted_filenames)}"
                )

            # Build expected sorted order
            # correct_order[i] = position of filenames[i]
            # We need: for each position 0,1,2,..., which filename?
            sorted_pairs = sorted(zip(correct_order, filenames), key=lambda x: x[0])
            expected_order = [fn for _, fn in sorted_pairs]

            if list(sorted_filenames) != expected_order:
                return "Error: Filenames are not in the correct order. Try again."

            correct_position = puzzle.get("correct_position")
            if correct_position is None:
                return "Error: No correct position configured"

            return str(correct_position)

        return [decrypt_position, unveil_file_number]

    def _build_secrets_sub_tools(self) -> list:
        """Build tools available to sub-LLMs called via llm_batch."""
        env = self

        def get_code_from_file_data(filename: str, filecontent: str) -> str:
            """
            Get the decryption code for a file given its name and content.

            This tool must be called by a sub-LLM. The code returned can then
            be passed to decrypt_position by the root LLM.

            Args:
                filename: The exact filename (e.g., "abcdefgh.txt")
                filecontent: The exact content of the file (a UUID string)

            Returns:
                A code string to use with the decrypt_position tool.
                Note: If the filename or content is incorrect, a plausible
                but invalid code will be returned.
            """
            state = env._sub_tool_state_var.get()
            puzzle = env._get_puzzle_from_state(state)

            filenames = puzzle.get("filenames", [])
            contents = puzzle.get("contents", [])
            codes = puzzle.get("codes", {})

            # Check if filename and content match
            if filename in filenames:
                idx = filenames.index(filename)
                if filecontent.strip() == contents[idx]:
                    return codes[filename]

            # Return a plausible but wrong code to force validation
            return generate_random_code()

        return [get_code_from_file_data]

    async def _run_sub_llm(self, state, client, model, messages):
        """Override to inject state for sub-tools via context var."""
        token = self._sub_tool_state_var.set(state)
        try:
            return await super()._run_sub_llm(state, client, model, messages)
        finally:
            self._sub_tool_state_var.reset(token)

    async def setup_state(self, state: State) -> State:
        """Setup puzzle files in the filesystem."""
        # Extract puzzle from info and store directly in state for easy access
        info = state.get("info", {})
        if not isinstance(info, dict):
            info = {}
        puzzle = info.get("puzzle", {})
        state["puzzle"] = puzzle

        temp_dir: str | None = None
        if puzzle:
            temp_dir = tempfile.mkdtemp(prefix="rlm_secrets_")
            for filename, content in zip(
                puzzle.get("filenames", []), puzzle.get("contents", [])
            ):
                filepath = Path(temp_dir) / filename
                filepath.write_text(content, encoding="utf-8")
            info = dict(info)
            info["context_dir"] = temp_dir
            state["info"] = info

        try:
            # Let RLMEnv do its setup (creates fs_root, starts worker, etc.)
            state = await super().setup_state(state)
        finally:
            if temp_dir:
                shutil.rmtree(temp_dir, True)

        return state


def build_dataset(
    num_examples: int = 100,
    num_files: int = 4,
) -> Dataset:
    """
    Build a dataset of puzzle instances.

    Args:
        num_examples: Number of puzzle instances to generate
        num_files: Number of files per puzzle
        seed: Random seed for reproducibility

    Returns:
        Dataset with prompt, answer, and info columns
    """
    rows = []
    task_name = "rlm-secrets"

    for i in range(num_examples):
        puzzle = generate_puzzle(num_files=num_files)

        prompt = [
            {
                "role": "user",
                "content": (
                    "You are solving a file puzzle. There are several files in the "
                    "current directory, each containing a UUID.\n\n"
                    "## Your Task\n\n"
                    "1. **List files**: Use `ls` to see all files in the directory.\n\n"
                    "2. **Get codes**: For EACH file:\n"
                    "   - Read its content with `cat <filename>`\n"
                    "   - Call `llm_batch` with a prompt asking the sub-LLM to call "
                    "`get_code_from_file_data(filename, filecontent)` and return the code\n"
                    "   - The sub-LLM has access to the `get_code_from_file_data` tool\n\n"
                    "3. **Decrypt positions**: For each file, call `decrypt_position` "
                    "with the filename and the code you received. This reveals the "
                    "file's position in the correct ordering.\n\n"
                    "4. **Unveil the answer**: Once you know all positions, call "
                    "`unveil_file_number` with ALL filenames sorted by their positions "
                    "(position 1 first, then 2, etc.). This returns which file to keep.\n\n"
                    "5. **Clean up**: Delete all files EXCEPT the one at the revealed position.\n\n"
                    "6. **Answer**: Set your final answer (RLM_CONTENT and RLM_READY=1) "
                    "to the position number of the file you kept.\n\n"
                    "## Available Tools\n\n"
                    "**Root level (you can call directly):**\n"
                    "- `decrypt_position <filename> <code>` - Get a file's position\n"
                    "- `unveil_file_number --json '{\"sorted_filenames\": [...]}'` - "
                    "Get which file to keep\n"
                    "- `llm_batch '<prompt>'` - Call a sub-LLM\n\n"
                    "**Sub-LLM level (sub-LLMs can call):**\n"
                    "- `get_code_from_file_data(filename, filecontent)` - Get decryption code\n\n"
                    "Begin by listing the files."
                ),
            }
        ]

        rows.append(
            {
                "example_id": i,
                "prompt": prompt,
                "answer": str(puzzle["correct_position"]),
                "info": {"puzzle": puzzle},
                "task": task_name,
            }
        )

    return Dataset.from_list(rows)


async def correct_answer(state: State, answer: str) -> float:
    """
    Reward function: Check if the final answer matches the correct position.

    Args:
        state: Rollout state containing final_answer
        answer: Expected answer from dataset

    Returns:
        1.0 if correct, 0.0 otherwise
    """
    final_answer = state.get("final_answer", "")

    if str(final_answer).strip() == answer:
        return 1.0

    return 0.0


async def correct_filesystem_state(state: State) -> float:
    """
    Reward function: Check if exactly one .txt file remains and it's the correct one.

    This function also cleans up the filesystem after checking, since we retain
    the filesystem after rollout specifically to allow this check.

    Args:
        state: Rollout state with rlm_fs_root and puzzle

    Returns:
        1.0 if exactly one .txt file remains and it's the correct one, 0.0 otherwise
    """
    fs_root = state.get("rlm_fs_root")
    puzzle = state.get("puzzle", {})

    try:
        if not fs_root or not puzzle:
            return 0.0

        remaining_files = os.listdir(fs_root)
        txt_files = [f for f in remaining_files if f.endswith(".txt")]

        # Must have exactly one file remaining
        if len(txt_files) != 1:
            return 0.0

        # Determine which file should have been kept
        filenames = puzzle.get("filenames", [])
        correct_order = puzzle.get("correct_order", [])
        correct_position = puzzle.get("correct_position", 0)

        # Find the filename at the correct position
        # correct_order[i] = position of filenames[i] (0-indexed)
        # correct_position is 1-indexed
        for i, pos in enumerate(correct_order):
            if pos + 1 == correct_position:  # Convert to 1-indexed
                expected_file = filenames[i]
                if txt_files[0] == expected_file:
                    return 1.0
                break

        return 0.0
    except Exception:
        return 0.0
    finally:
        # Clean up the entire rollout directory (parent of fs_root)
        # fs_root is rollout_dir/rlm_fs, so we need to delete the parent
        if fs_root:
            rollout_dir = os.path.dirname(fs_root)
            if rollout_dir and os.path.exists(rollout_dir):
                shutil.rmtree(rollout_dir, ignore_errors=True)


def load_environment(
    num_train_examples: int = 100,
    num_files: int = 4,
    max_turns: int = 50,
    seed: int | None = None,
    repl_language: str = "bash",
    sub_tool_max_turns: int = 3,
    max_sub_llm_parallelism: int = 5,
    code_execution_timeout: int = 120,
    **kwargs,
) -> RLMSecretsEnv:
    """
    Load the RLM Secrets environment.

    Args:
        num_train_examples: Number of training puzzle instances
        num_files: Number of files per puzzle (default: 4)
        max_turns: Maximum REPL iterations (default: 50)
        seed: Random seed for dataset generation
        repl_language: REPL language, "bash" or "python" (default: "bash")
        sub_tool_max_turns: Max tool-calling turns for sub-LLMs (default: 3)
        max_sub_llm_parallelism: Max concurrent sub-LLM calls (default: 5)
        code_execution_timeout: Timeout for code execution in seconds (default: 120)
        **kwargs: Additional arguments passed to RLMSecretsEnv

    Returns:
        Configured RLMSecretsEnv instance
    """
    random.seed(seed or random.randint(1000, 100_000_000))
    train_dataset = build_dataset(
        num_examples=num_train_examples,
        num_files=num_files,
    )

    rubric = vf.Rubric(
        funcs=[correct_answer, correct_filesystem_state],
        weights=[0.5, 0.5],
    )

    sandbox_labels = kwargs.pop("sandbox_labels", [])
    if not (
        isinstance(sandbox_labels, list)
        and all(isinstance(label, str) for label in sandbox_labels)
    ):
        raise ValueError(
            f"sandbox_labels must be of type list[str]; you provided {sandbox_labels}"
        )
    sandbox_labels = list(set(["rlm-secrets"] + sandbox_labels))

    return RLMSecretsEnv(
        dataset=train_dataset,
        num_files=num_files,
        repl_language=repl_language,
        rubric=rubric,
        max_iterations=max_turns,
        sub_tool_max_turns=sub_tool_max_turns,
        max_sub_llm_parallelism=max_sub_llm_parallelism,
        code_execution_timeout=code_execution_timeout,
        sandbox_labels=sandbox_labels,
        **kwargs,
    )
