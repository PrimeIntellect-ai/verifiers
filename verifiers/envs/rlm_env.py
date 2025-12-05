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
from verifiers.types import State

logger = logging.getLogger(__name__)


# Agent script template that runs inside the sandbox
# This implements the REPL loop for RLM
_RLM_AGENT_SCRIPT_TEMPLATE = '''
import os
import sys
import re
import json
import time
import traceback
from io import StringIO
from pathlib import Path

# Configuration from environment
BASE_URL = os.environ.get("OPENAI_BASE_URL")
ROOT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
SUB_MODEL = os.environ.get("RLM_SUB_MODEL", ROOT_MODEL)
MAX_ITERATIONS = int(os.environ.get("RLM_MAX_ITERATIONS", "50"))
MAX_OUTPUT_LENGTH = int(os.environ.get("RLM_MAX_OUTPUT_LENGTH", "8192"))
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
from openai import OpenAI

client = OpenAI(base_url=BASE_URL, api_key="dummy-key")

# Load context from file (may be empty if no context provided)
with open(CONTEXT_FILE, "r") as f:
    context = json.load(f)

# Load initial messages (the user's prompt)
with open(MESSAGES_FILE, "r") as f:
    messages = json.load(f)

# Initialize answer structure
answer = {"ready": False, "content": ""}


def llm(prompt: str, **kwargs) -> str:
    """
    Make a sub-LLM call. Available in the REPL for recursive queries.
    
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
        response = client.chat.completions.create(**api_kwargs)
        return response.choices[0].message.content or ""
    except Exception as e:
        return f"Error in sub-LLM call: {e}"


def rlm(prompt: str, sub_context=None, **kwargs) -> str:
    """
    Make a recursive RLM call with optional sub-context.
    
    Args:
        prompt: The query for the recursive RLM
        sub_context: Context to pass (prepended to prompt)
        **kwargs: Additional arguments
    
    Returns:
        The response as a string
    """
    if sub_context is not None:
        context_str = str(sub_context)
        if len(context_str) > MAX_OUTPUT_LENGTH:
            context_str = context_str[:MAX_OUTPUT_LENGTH] + "... [truncated]"
        full_prompt = f"Context:\\n{context_str}\\n\\nQuery: {prompt}"
    else:
        full_prompt = prompt
    return llm(full_prompt, **kwargs)


def execute_code(code: str) -> str:
    """Execute Python code and capture output."""
    global answer
    
    exec_globals = {
        "context": context,
        "answer": answer,
        "llm": llm,
        "rlm": rlm,
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
_RLM_SYSTEM_PROMPT_TEMPLATE = """You are operating in a Recursive Language Model (RLM) environment.

You have access to a Python REPL with the following:

## Available Variables and Functions:

- `context`: A dictionary containing:
  - `context["input_data"]`: {context_description}
  - `context["input_data_metadata"]`: Metadata about the input (type, size, etc.)

- `answer`: A dictionary you must modify to return your final answer:
  - `answer["ready"]`: Set to `True` when you have the final answer
  - `answer["content"]`: Your final answer (as a string)

- `llm(prompt, **kwargs)`: Call a sub-LLM for help with subtasks
  - Returns the sub-LLM's response as a string
  - Use for semantic understanding, classification, summarization, etc.

- `rlm(prompt, sub_context=None, **kwargs)`: Make a recursive RLM call
  - Can pass a subset of context to the sub-RLM
  - Use for recursively processing chunks of data

## How to Respond:

1. Output Python code blocks to interact with the environment:
```python
# Your code here
```

2. The code will be executed and you'll see the output.

3. When you have the final answer, set:
```python
answer["ready"] = True
answer["content"] = "Your final answer here"
```

## Tips:

- Start by peeking at the context to understand its structure
- Use `len()`, `type()`, slicing to explore the data
- For large contexts, partition and use `llm()` for sub-queries
- Build up your answer incrementally
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
        context_key: Key in info containing optional context data (default: "context")
        system_prompt: Custom system prompt (default: RLM standard prompt)
        **kwargs: Additional arguments passed to CliAgentEnv
    """
    
    def __init__(
        self,
        sub_model: str | None = None,
        max_iterations: int = 50,
        max_output_length: int = 8192,
        context_key: str = "context",
        system_prompt: str | None = None,
        **kwargs,
    ):
        self.sub_model = sub_model
        self.max_iterations = max_iterations
        self.max_output_length = max_output_length
        self.context_key = context_key
        self.custom_system_prompt = system_prompt
        
        # Generate the agent script
        agent_b64 = base64.b64encode(
            _RLM_AGENT_SCRIPT_TEMPLATE.encode("utf-8")
        ).decode("utf-8")
        
        start_command = _START_COMMAND_TEMPLATE.format(
            agent_path="/tmp/rlm_agent.py",
            agent_b64=agent_b64,
        )
        
        # Build environment variables
        env_vars = kwargs.pop("environment_vars", None) or {}
        env_vars.update({
            "RLM_MAX_ITERATIONS": str(max_iterations),
            "RLM_MAX_OUTPUT_LENGTH": str(max_output_length),
        })
        if sub_model:
            env_vars["RLM_SUB_MODEL"] = sub_model
        
        super().__init__(
            start_command=start_command,
            environment_vars=env_vars,
            **kwargs,
        )
    
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
            context_description = f"The input data (type: {context_dict['input_data_metadata'].get('type', 'unknown')}, size: {context_dict['input_data_metadata'].get('size', 'unknown')})"
        else:
            context_dict = {
                "input_data": None,
                "input_data_metadata": {"type": "none", "size": 0},
            }
            context_description = "No additional context provided"
        
        # Build messages from state["prompt"]
        prompt = state.get("prompt", [])
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]
        
        # Build system prompt
        if self.custom_system_prompt:
            system_prompt = self.custom_system_prompt
        else:
            system_prompt = _RLM_SYSTEM_PROMPT_TEMPLATE.format(
                context_description=context_description,
            )
        
        # Prepend system prompt to messages if not already present
        messages = list(prompt)  # Copy
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": system_prompt})
        
        # Write messages file
        messages_json = json.dumps(messages)
        await sandbox_client.execute_command(
            sandbox_id,
            f"cat > /tmp/rlm_messages.json << 'MESSAGES_EOF'\n{messages_json}\nMESSAGES_EOF"
        )
        
        # Write context file
        context_json = json.dumps(context_dict)
        await sandbox_client.execute_command(
            sandbox_id,
            f"cat > /tmp/rlm_context.json << 'CONTEXT_EOF'\n{context_json}\nCONTEXT_EOF"
        )
        
        # Write initial answer file
        answer_json = json.dumps({"ready": False, "content": ""})
        await sandbox_client.execute_command(
            sandbox_id,
            f"echo '{answer_json}' > /tmp/rlm_answer.json"
        )
        
        state["rlm_context"] = context_dict
        
        return state
    
    def _build_context_metadata(self, context_data: Any) -> dict[str, Any]:
        """Build metadata dictionary for the context."""
        metadata: dict[str, Any] = {}
        
        if context_data is None:
            metadata["type"] = "none"
            metadata["size"] = 0
        elif isinstance(context_data, str):
            metadata["type"] = "string"
            metadata["size"] = len(context_data)
            metadata["num_chars"] = len(context_data)
            metadata["num_lines"] = context_data.count("\n") + 1
            metadata["estimated_tokens"] = len(context_data) // 4
        elif isinstance(context_data, list):
            metadata["type"] = "list"
            metadata["size"] = len(context_data)
            metadata["num_items"] = len(context_data)
            if context_data and isinstance(context_data[0], dict):
                metadata["item_type"] = "dict"
                metadata["sample_keys"] = list(context_data[0].keys())[:5]
        elif isinstance(context_data, dict):
            metadata["type"] = "dict"
            metadata["size"] = len(context_data)
            metadata["keys"] = list(context_data.keys())[:10]
        else:
            metadata["type"] = type(context_data).__name__
            metadata["size"] = len(str(context_data))
        
        return metadata
    
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
                "cat /tmp/rlm_answer.json 2>/dev/null || echo '{\"content\": \"\"}'"
            )
            
            answer = json.loads(result.stdout.strip())
            state["final_answer"] = answer.get("content", "")
            
        except Exception as e:
            logger.warning(f"Failed to read RLM answer: {e}")
            state["final_answer"] = ""
