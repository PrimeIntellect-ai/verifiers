# Scaffold PoC Notes

This document captures findings and observations from implementing and testing the Scaffold abstraction.

## Overview

The Scaffold abstraction separates "how an agent interacts" from "what task is being solved":

- **Scaffold**: Wraps LLM client, manages tools, handles tool loops
- **Environment**: Defines tasks, state transitions, rewards

## Implementation

### Files Added

- `verifiers/scaffolds/scaffold.py` - Core scaffold classes
- `verifiers/scaffolds/__init__.py` - Module exports
- `examples/scaffold_poc.py` - Basic comparison of bare vs tool scaffolds
- `examples/scaffold_integration_test.py` - Comprehensive test suite
- `examples/scaffold_mcp_test.py` - MCP server integration test

### Classes

1. **`Scaffold`** - Base class, no tools
   - Wraps `AsyncOpenAI` client
   - Simple pass-through to LLM

2. **`ToolScaffold`** - Native Python tools
   - Handles tool loop internally
   - Executes tools and re-prompts until no more tool calls

3. **`MCPScaffold`** - MCP server integration
   - Connects to MCP servers via stdio
   - Dynamically discovers and registers tools
   - Reuses MCP infrastructure from existing `MCPEnv`

## Test Results

### Integration Tests (9/9 passing)

| Test | Status | Notes |
|------|--------|-------|
| `test_bare_scaffold_basic` | PASS | Simple LLM call works |
| `test_tool_scaffold_single_tool` | PASS | Single tool call works |
| `test_tool_scaffold_multiple_tools` | PASS | Multiple tool types work |
| `test_tool_scaffold_no_tool_needed` | PASS | Doesn't force tool use |
| `test_tool_scaffold_tool_with_params` | PASS | Multi-param tools work |
| `test_scaffold_parallel_execution` | PASS | Concurrent requests work |
| `test_scaffold_with_environment` | PASS | Works alongside env infra |
| `test_tool_error_handling` | PASS | Graceful error handling |
| `test_max_tool_turns` | PASS | Respects turn limits |

### MCP Test

- Successfully connected to `mcp-server-fetch`
- Tool discovery worked (registered `fetch` tool)
- Tool execution worked (fetched httpbin.org/json)
- Cleanup/teardown worked

## Observations & Peculiarities

### 1. Tool Call Parsing

vLLM with `--tool-call-parser hermes` works well for Qwen models. The tool calls are properly formatted and parsed.

### 2. MCP Connection Lifecycle

The MCP connection runs in a background thread (inherited from `MCPEnv` implementation). This works but could be simplified if we move to a service-based MCP backend.

### 3. ScaffoldResult Design

The `ScaffoldResult` includes:
- `response`: The final `ChatCompletion`
- `messages`: Full conversation history including tool calls
- `tool_calls_made`: Count for metrics
- `metadata`: Arbitrary dict for extensions

This allows the environment to see the complete interaction history, which is important for:
- Trajectory recording
- Debugging
- Metrics

### 4. Integration with Environments

Currently, scaffolds work *alongside* environments but aren't integrated into `Environment.generate()`. The full integration would require:

1. Update `Environment.generate()` to accept scaffold instead of `(client, model, sampling_args)`
2. Update `get_model_response()` to delegate to scaffold
3. Handle the expanded message history from tool scaffolds

For this PoC, we tested scaffolds independently and verified they work with environment rubrics/datasets.

### 5. Differences from ToolEnv

| Aspect | ToolEnv | ToolScaffold |
|--------|---------|--------------|
| Tool loop location | MultiTurnEnv rollout loop | Internal to scaffold |
| Trajectory | One step per tool turn | Single step (final response) |
| State management | Environment owns state | Scaffold is stateless |
| Stop conditions | Environment decorators | max_tool_turns parameter |

For benchmarking scaffolds, the ToolScaffold approach is cleaner because:
- Same environment can be used with different scaffolds
- Tool interaction is encapsulated
- Easier to compare with/without tools

### 6. GPU Memory

During testing, had to use GPU 1 because GPU 0 was occupied by another process. The vLLM server needs ~15GB for Qwen2.5-7B-Instruct.

## ToolEnv vs ToolScaffold Comparison

Ran a direct comparison using the `tool_test` environment task (calling specific tools):

| Approach | Accuracy |
|----------|----------|
| ToolEnv | 100% |
| ToolScaffold | 100% |

Both approaches produce identical results, validating that the scaffold implementation is consistent with the existing ToolEnv behavior.

## Next Steps

1. **Full Environment Integration**: Modify `Environment.generate()` to accept scaffolds
2. **MCP Backend Service**: Move MCP execution to a separate service for load balancing
3. **Scaffold Metrics**: Add rubric integration for scaffold-level metrics (tool call counts, etc.)
4. **Context Management**: Add support for context truncation/summarization in scaffold

## Usage Example

```python
from openai import AsyncOpenAI
import verifiers as vf
from verifiers.scaffolds import Scaffold, ToolScaffold, MCPScaffold

client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

# Bare scaffold (no tools)
bare = Scaffold(client, "Qwen/Qwen2.5-7B-Instruct")

# Native Python tools
tool_scaffold = ToolScaffold(
    client, "Qwen/Qwen2.5-7B-Instruct",
    tools=[my_calculator, my_search],
    max_tool_turns=5,
)

# MCP tools
mcp_scaffold = MCPScaffold(
    client, "Qwen/Qwen2.5-7B-Instruct",
    mcp_servers=[{"name": "fetch", "command": "uvx", "args": ["mcp-server-fetch"]}],
)
await mcp_scaffold.setup()

# Use with any environment
messages = [{"role": "user", "content": "Calculate 123 * 456"}]
result = await tool_scaffold.generate(messages, state={})
print(result.messages[-1]["content"])
print(f"Tool calls made: {result.tool_calls_made}")
```
