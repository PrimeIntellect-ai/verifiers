"""
Recursive Language Model (RLM) Environment.

Implements the RLM inference strategy where language models can decompose and
recursively interact with input context of unbounded length through REPL environments.

Based on: https://www.alexzhang.dev/blog/recursive-language-models

Key features:
- Works with any dataset that has a normal prompt
- Optional large context can be provided in info["context"]
- Root model only sees query, not full context (unless it peeks via code)
- Model can make recursive sub-LLM calls via llm() function
- Final answer returned via answer variable
- Call types (root vs sub) are tagged for logging/analysis
"""

import base64
import json
import logging
import textwrap
from typing import Any

from verifiers.envs.cli_agent_env import CliAgentEnv
from verifiers.types import Messages, ModelResponse, State

logger = logging.getLogger(__name__)


# Agent script template that runs inside the sandbox
# This implements the REPL loop for RLM
_RLM_AGENT_SCRIPT_TEMPLATE = '''
import os
import sys
import re
import json
import time
import asyncio
import traceback
from io import StringIO
from pathlib import Path

# Configuration from environment
BASE_URL = os.environ.get("OPENAI_BASE_URL")
ROOT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
SUB_MODEL = os.environ.get("RLM_SUB_MODEL", ROOT_MODEL)
MAX_ITERATIONS = int(os.environ.get("RLM_MAX_ITERATIONS", "50"))
MAX_OUTPUT_LENGTH = int(os.environ.get("RLM_MAX_OUTPUT_LENGTH", "8192"))
MAX_SUB_LLM_PARALLELISM = int(os.environ.get("RLM_MAX_SUB_LLM_PARALLELISM", "10"))
CONTEXT_FILE = "/tmp/rlm_context.json"
MESSAGES_FILE = "/tmp/rlm_messages.json"
ANSWER_FILE = "/tmp/rlm_answer.json"

# Wait for required files to be created by setup_state
# This handles the race condition where the sandbox starts before files are written
REQUIRED_FILES = [CONTEXT_FILE, MESSAGES_FILE]
MAX_WAIT_SECONDS = 120
WAIT_INTERVAL = 0.5

print("[RLM] Waiting for setup files...", flush=True)
for _ in range(int(MAX_WAIT_SECONDS / WAIT_INTERVAL)):
    if all(Path(f).exists() for f in REQUIRED_FILES):
        print("[RLM] Setup files found", flush=True)
        break
    time.sleep(WAIT_INTERVAL)
else:
    print(f"ERROR: Required files not found after {MAX_WAIT_SECONDS}s: {REQUIRED_FILES}", file=sys.stderr)
    sys.exit(1)

if not BASE_URL:
    print("ERROR: OPENAI_BASE_URL not set", file=sys.stderr)
    sys.exit(1)

# Import OpenAI after file check (gives pip install more time if needed)
from openai import OpenAI, AsyncOpenAI

# Sync client for root model calls in the REPL loop
client = OpenAI(base_url=BASE_URL, api_key="dummy-key")

# Async client for sub-LLM calls (used by the llm() function)
async_client = AsyncOpenAI(base_url=BASE_URL, api_key="dummy-key")

# Semaphore to control parallelism of sub-LLM calls
_sub_llm_semaphore = asyncio.Semaphore(MAX_SUB_LLM_PARALLELISM)

# Load context from file (may be empty if no context provided)
with open(CONTEXT_FILE, "r") as f:
    context = json.load(f)

# Load initial messages (the user's prompt)
with open(MESSAGES_FILE, "r") as f:
    messages = json.load(f)

# Initialize answer structure
answer = {"ready": False, "content": ""}


async def llm(prompt: str, **kwargs) -> str:
    """
    Make an async sub-LLM call. Use with asyncio.gather() for parallel execution.
    
    Parallelism is controlled by a semaphore (configurable via max_sub_llm_parallelism).
    
    Args:
        prompt: The prompt/query for the sub-LLM
        **kwargs: Additional arguments (e.g., max_tokens)
    
    Returns:
        The sub-LLM's response as a string
    """
    sub_messages = [{"role": "user", "content": prompt}]
    
    api_kwargs = {
        "model": SUB_MODEL,
        "messages": sub_messages,
        "extra_body": {"rlm_call_type": "sub"},  # Tag for logging
    }
    
    # Merge additional kwargs (but not model/messages/extra_body)
    for k, v in kwargs.items():
        if k not in ("model", "messages", "extra_body"):
            api_kwargs[k] = v
    
    try:
        async with _sub_llm_semaphore:
            response = await async_client.chat.completions.create(**api_kwargs)
        return response.choices[0].message.content or ""
    except Exception as e:
        return f"Error in sub-LLM call: {e}"


def execute_code(code: str) -> str:
    """Execute Python code and capture output."""
    global answer
    
    exec_globals = {
        "context": context,
        "answer": answer,
        "llm": llm,
        "print": print,
        "__builtins__": __builtins__,
    }
    
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()
    
    result = ""
    try:
        exec(code, exec_globals)
        result = captured_output.getvalue()
        # Update answer from exec namespace
        answer = exec_globals.get("answer", answer)
    except Exception as e:
        result = f"Error: {type(e).__name__}: {e}\\n{traceback.format_exc()}"
    finally:
        sys.stdout = old_stdout
    
    if len(result) > MAX_OUTPUT_LENGTH:
        result = result[:MAX_OUTPUT_LENGTH] + "\\n... [output truncated]"
    
    return result or "(no output)"


def extract_code_blocks(text: str) -> list:
    """Extract Python code blocks from model output."""
    pattern = r"```(?:python)?\\s*\\n(.*?)\\n```"
    return re.findall(pattern, text, re.DOTALL)


print(f"[RLM] Starting REPL", flush=True)

# Main REPL loop
for iteration in range(MAX_ITERATIONS):
    print(f"[RLM] Iteration {iteration + 1}/{MAX_ITERATIONS}", flush=True)
    
    try:
        # Call the root model (tagged as "root" for logging)
        response = client.chat.completions.create(
            model=ROOT_MODEL,
            messages=messages,
            extra_body={"rlm_call_type": "root"},  # Tag for logging
        )
        
        assistant_content = response.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": assistant_content})
        
        # Extract and execute code blocks
        code_blocks = extract_code_blocks(assistant_content)
        
        if not code_blocks:
            # No code blocks - prompt for action
            if "answer" in assistant_content.lower() and "ready" in assistant_content.lower():
                messages.append({
                    "role": "user",
                    "content": "Please set your answer using a Python code block:\\n```python\\nanswer[\\"ready\\"] = True\\nanswer[\\"content\\"] = \\"your answer\\"\\n```"
                })
            else:
                messages.append({
                    "role": "user",
                    "content": "Please provide Python code to explore the context or set your answer."
                })
            continue
        
        # Execute all code blocks
        all_outputs = []
        for i, code in enumerate(code_blocks):
            output = execute_code(code)
            all_outputs.append(f"[Code block {i + 1}]\\n{output}")
            print(f"[RLM] Executed code block {i + 1}: {output[:200]}...", flush=True)
        
        combined_output = "\\n\\n".join(all_outputs)
        
        # Check if answer is ready
        if answer.get("ready", False):
            print(f"[RLM] Answer ready: {str(answer.get('content', ''))[:200]}...", flush=True)
            break
        
        # Add execution results
        messages.append({
            "role": "user", 
            "content": f"Execution output:\\n{combined_output}"
        })
        
    except Exception as e:
        print(f"[RLM] Error in iteration {iteration + 1}: {e}", file=sys.stderr, flush=True)
        messages.append({
            "role": "user",
            "content": f"Error occurred: {e}\\nPlease try again."
        })

# Save final answer
with open(ANSWER_FILE, "w") as f:
    json.dump(answer, f)

# Signal completion
with open("/tmp/vf_complete", "w") as f:
    f.write("done")

print("[RLM] REPL complete", flush=True)
'''


# System prompt for RLM - added to user's messages
_RLM_SYSTEM_PROMPT_TEMPLATE = """You are operating in a Recursive Language Model (RLM) environment - an iterative Python REPL where you explore data step by step.

## Critical: This is an ITERATIVE environment

You will write code, see its output, then write more code based on what you learned. **Do NOT try to solve everything in one code block.** Each code block executes and returns output before you continue.

**You MUST wrap all code in markdown code blocks** using triple backticks with `python`:
```python
# your code here
```
Only code inside these blocks will be executed. Text outside code blocks is for your reasoning.

## Available Variables and Functions

- `context`: A dictionary containing:
  - `context["input_data_metadata"]`: Metadata about the input (type, size, structure, etc.)
  - `context["input_data"]`: The actual input data

- `answer`: A dictionary for your final answer:
  - `answer["content"]`: Your answer (string) - write and update this throughout execution
  - `answer["ready"]`: Set to `True` to finish - **this immediately terminates execution**

- `llm(prompt, **kwargs)`: Async function to call a sub-LLM for help with subtasks
  - Returns the response as a string
  - Useful for semantic understanding, summarization, complex reasoning
  - Include any context you need directly in the prompt string
  - **This is an async function** - use with `asyncio.run()`:
    ```python
    import asyncio
    # Single call
    result = asyncio.run(llm("What is the main theme of this text?"))
    ```
  - Use `asyncio.gather()` to execute multiple calls in parallel:
    ```python
    import asyncio
    results = asyncio.run(asyncio.gather(
        llm("Summarize section 1: " + section1),
        llm("Summarize section 2: " + section2),
        llm("Summarize section 3: " + section3),
    ))
    # results is a list of responses in the same order as the calls
    ```
  - Parallelism is automatically rate-limited to prevent API overload

## Workflow

**Step 1: Inspect metadata first**
```python
print(context["input_data_metadata"])
```
Wait for output. This tells you the data type, size, and structure before you look at the actual data.

**Step 2: Explore the data based on what you learned**
```python
# Look at a sample of the actual data
data = context["input_data"]
print(type(data))
print(data[:500] if isinstance(data, str) else data[:3])  # First part only
```
Wait for output. Now you know the actual format.

**Step 3: Process and build your answer**
```python
# Based on what you've seen, write code to solve the task
# ...
answer["content"] = "your current best answer"
```
You can update `answer["content"]` multiple times as you refine your solution.

**Step 4: Verify and finalize (only after reviewing output)**
```python
print(f"My answer: {{answer['content']}}")
# Only after confirming this looks correct:
answer["ready"] = True
```

## Important Rules

1. **Always use markdown code blocks** - wrap code in triple backticks with `python`, or it won't execute
2. **NEVER set `answer["ready"] = True` until you have seen execution output** - you need feedback first
3. **Start with metadata** - always run `print(context["input_data_metadata"])` before accessing `input_data`
4. **One step at a time** - write small code blocks, see output, then continue
5. **Use `llm()` for semantic tasks** - summarization, understanding text, classification, etc.
6. **`llm()` is async** - always use `asyncio.run(llm(...))` or `asyncio.run(asyncio.gather(llm(...), ...))`
7. You can think in natural language between code blocks - reasoning and planning are encouraged

The environment executes your code and shows you the output. Use that feedback to iterate toward the correct answer.
"""


_START_COMMAND_TEMPLATE = textwrap.dedent(
    """
    bash -lc '
    set -euo pipefail
    pip install -q openai || true
    agent_path="/tmp/rlm_agent.py"
    python - <<'PY'
import base64
from pathlib import Path
Path("{agent_path}").write_bytes(base64.b64decode("{agent_b64}"))
PY
    python -u "$agent_path" &
    tail -f /dev/null
    '
    """
)


class RLMEnv(CliAgentEnv):
    """
    Recursive Language Model Environment.

    Extends CliAgentEnv to provide a Python REPL environment where the model can:
    - Interact with large context stored as a variable
    - Make recursive sub-LLM calls
    - Return final answers via an answer variable

    Works with any dataset that has a normal prompt. Context can optionally
    be provided in info[context_key] for large data that shouldn't be in the prompt.

    Args:
        sub_model: Model to use for sub-LLM calls (defaults to same as root model)
        max_iterations: Maximum REPL iterations before stopping
        max_output_length: Maximum length of code execution output
        max_sub_llm_parallelism: Maximum number of concurrent sub-LLM calls (default: 10)
        context_key: Key in info containing optional context data (default: "context")
        system_prompt: Custom system prompt (default: RLM standard prompt)
        **kwargs: Additional arguments passed to CliAgentEnv
    """

    def __init__(
        self,
        sub_model: str | None = None,
        max_iterations: int = 50,
        max_output_length: int = 8192,
        max_sub_llm_parallelism: int = 5,
        context_key: str = "context",
        system_prompt: str | None = None,
        **kwargs,
    ):
        self.sub_model = sub_model
        self.max_iterations = max_iterations
        self.max_output_length = max_output_length
        self.max_sub_llm_parallelism = max_sub_llm_parallelism
        self.context_key = context_key
        self.custom_system_prompt = system_prompt

        # Generate the agent script
        agent_b64 = base64.b64encode(_RLM_AGENT_SCRIPT_TEMPLATE.encode("utf-8")).decode(
            "utf-8"
        )

        start_command = _START_COMMAND_TEMPLATE.format(
            agent_path="/tmp/rlm_agent.py",
            agent_b64=agent_b64,
        )

        # Build environment variables
        env_vars = kwargs.pop("environment_vars", None) or {}
        env_vars.update(
            {
                "RLM_MAX_ITERATIONS": str(max_iterations),
                "RLM_MAX_OUTPUT_LENGTH": str(max_output_length),
                "RLM_MAX_SUB_LLM_PARALLELISM": str(max_sub_llm_parallelism),
            }
        )
        if sub_model:
            env_vars["RLM_SUB_MODEL"] = sub_model

        super().__init__(
            start_command=start_command,
            environment_vars=env_vars,
            **kwargs,
        )

    async def _write_file_to_sandbox(
        self, sandbox_client: Any, sandbox_id: str, filepath: str, content: str
    ) -> None:
        """
        Write content to a file in the sandbox, handling large content safely.
        
        Uses base64 encoding and chunked writing to avoid command-line length limits.
        """
        content_bytes = content.encode("utf-8")
        content_b64 = base64.b64encode(content_bytes).decode("ascii")
        
        # For small content, write directly
        # Command length limit is typically around 128KB, but we use a conservative threshold
        # to account for the Python wrapper overhead
        chunk_size = 50000  # ~50KB chunks (base64 encoded)
        
        if len(content_b64) <= chunk_size:
            # Small content: write in one command
            cmd = f"""python3 -c "
import base64
from pathlib import Path
Path('{filepath}').write_bytes(base64.b64decode('{content_b64}'))
" """
            await sandbox_client.execute_command(sandbox_id, cmd)
        else:
            # Large content: write in chunks
            # First, create an empty file
            await sandbox_client.execute_command(
                sandbox_id, f"python3 -c \"from pathlib import Path; Path('{filepath}').write_text('')\""
            )
            
            # Write chunks
            for i in range(0, len(content_b64), chunk_size):
                chunk = content_b64[i:i + chunk_size]
                cmd = f"""python3 -c "
import base64
from pathlib import Path
chunk = base64.b64decode('{chunk}')
with open('{filepath}', 'ab') as f:
    f.write(chunk)
" """
                await sandbox_client.execute_command(sandbox_id, cmd)

    async def setup_state(self, state: State) -> State:
        """Setup sandbox with messages, context, and answer files."""
        state = await super().setup_state(state)

        sandbox_id = state.get("sandbox_id")
        if not sandbox_id:
            raise RuntimeError("Sandbox ID not set")

        from prime_sandboxes import AsyncSandboxClient

        sandbox_client = AsyncSandboxClient()
        await sandbox_client.wait_for_creation(sandbox_id)

        # Get optional context from info
        info = state.get("info", {})
        context_data = info.get(self.context_key, None)

        # Build context dict
        if context_data is not None:
            context_dict = {
                "input_data": context_data,
                "input_data_metadata": self._build_context_metadata(context_data),
            }
        else:
            context_dict = {
                "input_data": None,
                "input_data_metadata": {"type": "none", "size": 0},
            }

        # Build messages from state["prompt"]
        prompt = state.get("prompt", [])
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]

        # Build system prompt
        if self.custom_system_prompt:
            system_prompt = self.custom_system_prompt
        else:
            system_prompt = _RLM_SYSTEM_PROMPT_TEMPLATE

        # Prepend system prompt to messages if not already present
        messages = list(prompt)  # Copy
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": system_prompt})

        # Write files to sandbox using safe chunked method
        messages_json = json.dumps(messages)
        await self._write_file_to_sandbox(
            sandbox_client, sandbox_id, "/tmp/rlm_messages.json", messages_json
        )

        context_json = json.dumps(context_dict)
        await self._write_file_to_sandbox(
            sandbox_client, sandbox_id, "/tmp/rlm_context.json", context_json
        )

        answer_json = json.dumps({"ready": False, "content": ""})
        await self._write_file_to_sandbox(
            sandbox_client, sandbox_id, "/tmp/rlm_answer.json", answer_json
        )

        state["rlm_context"] = context_dict

        return state

    def _build_context_metadata(self, context_data: Any) -> dict[str, Any]:
        """Build minimal metadata dictionary for the context."""
        metadata: dict[str, Any] = {}

        # Expressive type-name; can e.g. distinguish between pandas and polars DataFrames
        metadata["type"] = str(type(context_data))

        # Simple size estimation.
        # If __len__ is not defined, the type will tell the model how to approach the data.
        if context_data is None:
            metadata["size"] = 0
        elif hasattr(context_data, "__len__"):
            metadata["size"] = len(context_data)
        else:
            metadata["size"] = "unknown"

        return metadata

    async def add_model_response(
        self,
        state: State,
        prompt_messages: Messages,
        response: ModelResponse,
    ):
        """
        Override to skip adding sub-LLM calls to the trajectory.
        
        Sub-LLM calls are internal helper calls that should not be part of
        the visible conversation trajectory. They are identified by
        rlm_call_type: "sub" in the intercepted request.
        """
        # Check if this is a sub-LLM call
        request_id = state.get("current_request_id")
        call_type = "unknown"
        if request_id and request_id in self.intercepts:
            request_body = self.intercepts[request_id].get("request_body", {})
            call_type = request_body.get("rlm_call_type") or "unknown"
        
        if call_type == "sub":
            # Skip adding sub-LLM calls to trajectory
            logger.debug(f"Skipping sub-LLM call {request_id}")
            if request_id in self.intercepts:
                del self.intercepts[request_id]
            return
        
        # For root calls, add to trajectory normally
        await super().add_model_response(state, prompt_messages, response)

    async def post_rollout(self, state: State):
        """
        Read final answer from sandbox before destruction.

        Sets state["final_answer"] to the model's answer string.
        Empty string if no answer was provided.
        """
        sandbox_id = state.get("sandbox_id")
        if not sandbox_id:
            state["final_answer"] = ""
            return

        try:
            from prime_sandboxes import AsyncSandboxClient

            sandbox_client = AsyncSandboxClient()

            result = await sandbox_client.execute_command(
                sandbox_id,
                'cat /tmp/rlm_answer.json 2>/dev/null || echo \'{"content": ""}\'',
            )

            answer = json.loads(result.stdout.strip())
            state["final_answer"] = answer.get("content", "")

        except Exception as e:
            logger.warning(f"Failed to read RLM answer: {e}")
            state["final_answer"] = ""

# TODO: Improve system prompt
# TODO: Add logging for sub-LLM calls
# TODO: Add support for additional tools, usable by the sub-LLMs
# TODO: Experiment with putting the user query inside the `context`
# TODO: Simplify usage of the `llm` function for the RLM agent
