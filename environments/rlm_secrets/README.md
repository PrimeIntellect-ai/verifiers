# RLM Secrets

A puzzle environment designed to thoroughly test RLM (Recursive Language Model) functionality.

## Overview

This environment tests all major RLM components through a file-based puzzle:

- **Root-level tools**: `decrypt_position`, `unveil_file_number`
- **Sub-LLM tools**: `get_code_from_file_data`
- **Sub-LLM calls**: via `llm_batch`
- **File operations**: `ls`, `cat`, `rm`

## The Puzzle

1. Several files with random names exist in the working directory
2. Each file contains a random UUID as content
3. Files have a hidden "correct order" (positions 1, 2, 3, ...)
4. One position is designated as the "file to keep"

### Solution Flow

1. **List files**: Root LLM runs `ls` to discover files
2. **Get codes**: For each file:
   - Read content with `cat`
   - Call sub-LLM via `llm_batch` asking it to use `get_code_from_file_data`
   - Sub-LLM calls the tool and returns the code
3. **Decrypt positions**: Root LLM calls `decrypt_position(filename, code)` to learn each file's position
4. **Unveil answer**: Root LLM calls `unveil_file_number([sorted_filenames])` to learn which position to keep
5. **Clean up**: Delete all files except the one at the revealed position
6. **Answer**: Set `RLM_CONTENT` to the kept file's position, `RLM_READY=1`

## Tools

### Root-Level (called directly by root LLM)

```bash
decrypt_position <filename> <code>
```

Returns the 1-indexed position if the code is valid, error message otherwise.

```bash
unveil_file_number --json '{"sorted_filenames": ["file1.txt", "file2.txt", ...]}'
```

Returns which position's file to keep if order is correct, error message otherwise.

### Sub-LLM Level (called by sub-LLMs via tool use)

```python
get_code_from_file_data(filename: str, filecontent: str) -> str
```

Returns the decryption code if filename and content match, a fake code otherwise.

## Usage

```bash
uv run vf-eval rlm-secrets
```

## Reward Functions

Both reward functions have equal weight (0.5 each):

- **correct_answer**: 1.0 if final answer matches correct position
- **correct_filesystem_state**: 1.0 if exactly one .txt file remains AND it's the correct one

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_train_examples` | 100 | Training puzzles |
| `num_eval_examples` | 20 | Evaluation puzzles |
| `num_files` | 4 | Files per puzzle |
| `max_turns` | 50 | Max REPL iterations |
| `sub_tool_max_turns` | 3 | Max tool turns for sub-LLMs |
| `max_sub_llm_parallelism` | 5 | Concurrent sub-LLM calls |
| `code_execution_timeout` | 120 | Bash execution timeout (seconds) |
| `**kwargs` | - | Passed on `RLMEnv.__init__` |

## Why This Environment?

This environment is specifically designed to test RLM capabilities:

1. **Forces root-LLM usage**: The correct order can only be obtained by calling the root-level tools
2. **Forces sub-LLM usage**: The code can only be obtained by having a sub-LLM call `get_code_from_file_data`
3. **Forces sub-LLM tool use**: Sub-LLMs must use their tool to get the code
4. **Tests file operations**: Model must list, read, and delete files
5. **Tests information flow**: Data must flow: file → sub-LLM → root-LLM → tool → answer

The puzzle is simple enough that models should be able to solve it, while being complex enough to exercise all RLM components.
