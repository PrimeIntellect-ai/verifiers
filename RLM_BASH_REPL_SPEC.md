# RLM Bash REPL Spec

## Goals
- Add a Bash-based REPL option to RLMEnv that feels like a real terminal.
- Make Bash the default REPL language while preserving Python mode behavior unchanged.
- Share code between Bash and Python modes where it reduces risk and maintenance burden.
- Keep existing sub-LLM interception, root tools, metrics, and rollout flow intact.
- Allow large context usage without forcing the model to paste it into prompts.

## Non-Goals
- No hard OS-level sandboxing guarantees for Bash in this phase.
- No change to backend tool protocols (HTTP + pickle for root tools).
- No redesign of RLM sub-LLM batching semantics.

## Summary of Decisions
- Default REPL language: bash.
- Tool names are explicit: call_bash_repl and call_python_repl.
- Bash REPL output is raw terminal output only (no Out[n], no stderr labels, no timing footer).
- Answer readiness and content are read from environment variables RLM_READY and RLM_CONTENT.
- Non-zero exit codes are not automatically surfaced; the model can check with echo $?.
- Root tools are exposed to Bash as shell functions only (not executables).
- llm_batch in Bash supports positional prompts and line-delimited stdin; output is plain text.

## High-Level Architecture
- Keep the LocalRLMExecutor and FIFO-based worker protocol.
- Introduce a second worker template for Bash mode.
- The Bash worker is a Python process that:
  - Spawns a persistent bash process under a PTY.
  - For each tool call, feeds the command to bash and captures raw output.
  - Reads RLM_READY and RLM_CONTENT from the bash environment.
  - Returns a JSON response via the existing FIFO response channel.
- Python worker remains unchanged for python mode.

## Configuration and API
### New RLMEnv Parameter
- repl_language: Literal["bash", "python"] = "bash"
  - Validates input; unknown values raise ValueError.

### Tool Names
- If repl_language == "bash": add tool call_bash_repl.
- If repl_language == "python": add tool call_python_repl (existing behavior).
- Both tools still use StatefulToolEnv update_tool_args to inject state.

### System Prompt
- Introduce a Bash-specific system prompt template:
  - Contains filesystem summary.
  - Shows bash snippets (pwd, ls) instead of Python.
  - Explains RLM_READY and RLM_CONTENT.
  - Lists root tools and sub-tools as commands.
- Python prompt remains unchanged and uses _RLM_SYSTEM_PROMPT.

## Bash REPL Behavior (Terminal-First)
- Use a persistent bash session so state is preserved across calls.
- Return raw output exactly as a terminal would display it:
  - Combined stdout/stderr (interleaved).
  - No extra prefixes or postfixes.
  - No execution time footer.
- Output is returned as-is; no normalization or trimming except removal of internal markers.

## Answer via Environment Variables
- RLM_READY maps to answer["ready"].
  - Truthy values: 1, true, yes, y, on (case-insensitive).
- RLM_CONTENT maps to answer["content"].
- The worker reads these after every command and returns them in the JSON response.

## Root Tools in Bash
### Exposure Model
- Root tools are exposed as bash functions (not executables) for terminal-like feel.
- Each function calls a helper script (Python) in the control directory:
  - The helper posts to the root tool HTTP endpoint using stdlib urllib (no venv).
  - The helper decodes the pickle response.

### Output Formatting
- If result is a string, print as-is.
- If result is JSON-serializable, print JSON.
- Otherwise, print repr().

### llm_batch Command
- Supported usage:
  - Positional prompts:
    - llm_batch "prompt 1" "prompt 2"
  - Line-delimited prompts from stdin:
    - llm_batch --lines < prompts.txt
- Output is plain text:
  - Each response is printed sequentially.
  - A single newline separates responses (no JSON array output).
  - No metadata summary lines by default.

## Worker Protocol (Shared)
- Keep FIFO-based request/response.
- Request payload:
  - {"code": <command>, "seq": <int>}
- Response payload:
  - {"status": "ok"|"error", "stdout": <raw>, "stderr": <raw>, "result": <optional>,
     "execution_count": <int>, "seq": <int>, "answer": {"ready": bool, "content": str}}

### Bash Worker Internals
- The bash worker is a Python script that:
  - Creates a PTY and runs bash --noprofile --norc in fs_root.
  - Defines root tool functions in the bash session (llm_batch + root tools).
  - For each command, emits a hidden end-marker and a hidden env-marker,
    then reads the PTY until the markers arrive.
  - Strips markers before returning output.

### Exit Codes
- Non-zero exit codes do not automatically change the output.
- The model can check exit status via `echo $?` like in a real terminal.
- Internally, worker may set status="error" only if it cannot execute the command
  or the session is broken.

## Filesystem and Safety
- Python mode retains FilesystemJail best-effort restrictions.
- Bash mode cannot enforce equivalent restrictions at the process level.
- Best-effort measures in Bash mode:
  - Start in fs_root.
  - Restrict PATH to known system dirs + tool helpers.
  - Optional rbash is not enabled by default; cross-platform support is inconsistent.
- Document: Bash mode is not a security sandbox.

## Code Sharing Strategy
Shared:
- Interception server and sub-LLM plumbing.
- Root tool HTTP endpoint and pickle serialization.
- Rollout lifecycle, metrics, and trajectory handling.
- Filesystem context creation and copying.

Split:
- Worker template (Python vs Bash).
- System prompt template.
- Tool name and REPL semantics.

## Compatibility
- Python mode remains unchanged:
  - Same REPL semantics, same output formatting, same worker script.
  - Existing tests must continue to pass.
- Default becomes Bash mode for new instantiations.

## Testing Plan
- Update and/or add tests to cover:
  - Default repl_language == "bash".
  - Python mode still uses call_python_repl and current behavior.
  - Bash mode uses call_bash_repl and includes bash prompt instructions.
  - llm_batch wrapper accepts positional args and --lines.
  - Bash mode does not append execution-time footer.
  - Answer parsing from RLM_READY/RLM_CONTENT.

## Open Risks
- PTY handling differs across platforms; ensure macOS/Linux compatibility.
- Bash output capture and marker stripping must handle edge cases without
  corrupting output.
