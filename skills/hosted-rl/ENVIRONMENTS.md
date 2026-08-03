# Environments Reference

## Structure

```
environments/my_env/
├── my_env.py           # Must expose load_environment(**kwargs) -> vf.Environment
├── pyproject.toml      # Dependencies ([project] dependencies = ["verifiers>=0.1.8"])
└── README.md           # Document env vars, dataset, usage
```

`pyproject.toml` supports `[tool.verifiers.eval]` for default eval params and `[tool.hatch.build]` for include files.

## Dataset Schema

Datasets are HuggingFace `Dataset` objects with two prompt options:

- `prompt` column (list of chat messages): `[{"role": "user", "content": "..."}]` — used directly
- `question` column (string): auto-wrapped in user message; ignored if `prompt` exists

Optional columns: `answer` (ground truth), `info` (dict or JSON for metadata).

## Environment Types

| Type | Use Case | Key Features |
|------|----------|--------------|
| `SingleTurnEnv` | Q&A, one-shot tasks | Simplest option |
| `MultiTurnEnv` | Games, simulations | Override `env_response()`, `@vf.stop` conditions |
| `ToolEnv` | Stateless tool use | `tools=[fn1, fn2]`, auto schema extraction |
| `StatefulToolEnv` | Per-rollout state tools | `setup_state()`, `update_tool_args()`, `args_to_skip` |
| `SandboxEnv` | Containerized bash shell | Uses Prime Sandboxes |
| `PythonEnv` | Python REPL | Extends SandboxEnv with persistent ipython |

**Stable**: SingleTurnEnv, MultiTurnEnv, ToolEnv, StatefulToolEnv, SandboxEnv, PythonEnv

## Lifecycle Hooks

- `setup_state(state)` — init per-rollout state
- `@vf.cleanup` — after each rollout
- `@vf.teardown` — at env shutdown
- `@vf.stop` — custom stop conditions (return boolean)

All support `priority=` (higher runs first). End rollout from `env_response` by setting `state["final_env_response"]`.

## Tools

Tools are plain Python functions with type hints and docstrings. Verifiers extracts OpenAI tool schemas automatically.

```python
class MyToolEnv(vf.ToolEnv):
    def __init__(self):
        super().__init__(self)
        self.add_tool(self.search, weight=1.0)

    async def search(self, search_term: str) -> str:
        """Search from a search thing.

        Args:
            search_term: A term to search for

        Returns:
            The result.
        """
        result = search_for_thing(search_term)
        return result
```

## StatefulToolEnv Pattern

```python
class MyEnv(vf.StatefulToolEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_tool(self.run_code, args_to_skip=["session_id"])

    async def setup_state(self, state, **kwargs):
        state["session_id"] = await create_session()
        return await super().setup_state(state, **kwargs)

    def update_tool_args(self, tool_name, tool_args, messages, state, **kwargs):
        if tool_name == "run_code":
            tool_args["session_id"] = state["session_id"]
        return tool_args
```

## Error Hierarchy

`vf.Error` base → `vf.ModelError` (`InvalidModelResponseError`, `EmptyModelResponseError`) | `OverlongPromptError` | `ToolError` (`ToolParseError`, `ToolCallError`) | `InfraError` (`SandboxError`).

Errors in `stop_errors` end rollout immediately; others return as tool response messages.

## Recommended Examples by Pattern

Check `verifiers/envs/` in the repo for canonical implementations:
