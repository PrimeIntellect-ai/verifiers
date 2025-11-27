# Verifiers v0.1.8 Release Notes

**Post-release update:**
- Fix for eval log ordering (post0).

*Date:* 11/19/2025

Verifiers v0.1.8 introduces a major refactor of the rollout system to use trajectory-based tracking, where each LLM request/response pair is recorded as an independent step. This enables cleaner horizontal training workflows (e.g. truncated thinking, branching rollouts, sub-agents, self-summarization) and eliminates retokenization brittleness by preserving vLLM's native token IDs and logprobs.

## Highlights

- **Trajectory-based rollouts**: All rollouts now track trajectory steps automatically. Each step represents one complete LLM API call with its prompt, completion, tokens, and logprobs.
- **Input-first API**: `rollout()` now takes `input: RolloutInput` as the primary parameter, matching dataset structure more closely.
- **Decorator-based termination**: New `@stop`, `@cleanup`, and `@teardown` decorators for declarative rollout lifecycle management.
- **State structure improvements**: Clear separation between `init_state()` (environment-agnostic) and `setup_state()` (environment-specific configuration).
- **Horizontal training support**: Each trajectory step can be processed as an independent training example without retokenization.
- **Training integration**: Both `RLTrainer` and `prime-rl` now use trajectory-based rollouts for training. `prime-rl` support is available via the `will/trajectories` branch, which is automatically pinned when using `vf-setup`.
- **Group reward functions**: Reward functions can now operate on groups of rollouts simultaneously by accepting plural parameters (`states`, `prompts`, `completions`, etc.), enabling relative scoring (e.g. pairwise, tournament). 
- **Simplified scoring logic**: Scoring is now performed at the group level via `score_group()` by default, parallelizing across rollouts for any rollout-based (non-group) reward functions.
- **Completion rendering**: Internal `_render_completion()` method automatically renders completion from trajectory for output saving, ensuring consistent formatting.

## Breaking Changes

### `is_completed()` Signature Change

**⚠️ Important**: The `is_completed()` signature has changed. **Do not override `is_completed()`**—use the `@stop` decorator instead.

**Old pattern (deprecated):**
```python
async def is_completed(self, messages: Messages, state: State, **kwargs) -> bool:
    return len(state["responses"]) == 1
```

**New pattern (recommended):**
```python
import verifiers as vf

class MyEnv(MultiTurnEnv):
    @vf.stop
    async def custom_condition(self, state: State) -> bool:
        return len(state["trajectory"]) == 1
```

**Key changes:**
- The `messages` parameter has been removed from `is_completed()` (use `state["trajectory"]` to reconstruct conversation if needed)
- **Do not override `is_completed()`**—use `@stop` decorators for all stop conditions
- Use `state["trajectory"]` instead of `state["responses"]` for trajectory-based logic
- Multiple `@stop` decorated methods are automatically checked; the first one returning `True` terminates the rollout

**If you must override `is_completed()`** (not recommended), you must:
- Remove the `messages` parameter
- Call `super().is_completed(state)` first to check `@stop` decorated methods
- Only then check your custom conditions

### State Structure Changes

- `state["trajectory"]` is now always present (never `None`) and contains a list of `TrajectoryStep` objects
- `state["responses"]` is deprecated in favor of `state["trajectory"]`
- `state["completion"]` is now rendered from the trajectory when rollout completes

### Rollout API Changes

**⚠️ Breaking for environments that don't override `MultiTurnEnv`**: The `rollout()` and `init_state()` method signatures have changed.

**Old `rollout()` signature (deprecated):**
```python
async def rollout(
    self,
    client: AsyncOpenAI,
    model: str,
    prompt: Messages,
    completion: Messages | None = None,
    answer: str = "",
    state: State = {},
    task: str = "default",
    info: Info | None = None,
    example_id: int = 0,
    sampling_args: SamplingArgs | None = None,
    **kwargs,
) -> tuple[Messages, State]:
    # Returns tuple of (completion, state)
    # ...
```

**New `rollout()` signature:**
```python
async def rollout(
    self,
    input: RolloutInput,
    client: AsyncOpenAI,
    model: str,
    sampling_args: SamplingArgs | None = None,
) -> State:
    state = await self.init_state(input, client, model, sampling_args)
    state = await self.setup_state(state)
    # ... rollout logic ...
    return state
```

**Old `init_state()` signature (deprecated):**
```python
async def init_state(
    self,
    prompt: Messages,
    completion: Messages,
    answer: str,
    task: str,
    info: Info,
    example_id: int,
    **kwargs,
) -> State:
    # Takes individual parameters, returns state dict
    # ...
```

**New `init_state()` signature:**
```python
async def init_state(
    self,
    input: RolloutInput,
    client: AsyncOpenAI,
    model: str,
    sampling_args: SamplingArgs | None = None,
) -> State:
    # Creates state from input, sets up trajectory list, timing, etc.
    # ...
```

**Key changes:**
- `rollout()` now takes `input: RolloutInput` as the first parameter instead of many individual parameters (`prompt`, `completion`, `answer`, `task`, `info`, `example_id`, etc.)
- `rollout()` now returns `State` instead of `tuple[Messages, State]`
- `init_state()` now takes `input: RolloutInput, client: AsyncOpenAI, model: str, sampling_args: SamplingArgs | None = None` instead of individual parameters
- `setup_state()` is now abstract in `Environment` and **must be implemented** by environments that inherit directly from `Environment`. `MultiTurnEnv` implements it as a no-op (returns state unchanged), so environments inheriting from `MultiTurnEnv` can override it as needed but don't need to implement it
- State is created internally via `init_state()` and `setup_state()` within `rollout()`

**Impact:**
- Environments that inherit from `MultiTurnEnv` (including `SingleTurnEnv`, `ToolEnv`, etc.) are unaffected as `MultiTurnEnv` handles these changes
- Environments that inherit directly from `Environment` (e.g., custom environments) **must** update their `rollout()` and `init_state()` signatures and implement `setup_state()`

### Scoring Changes

- **Group-level scoring**: Scoring is now always performed at the group level via `rubric.score_group()`. The `interleave_scoring` flag is deprecated and no longer has any effect.
- **Group reward functions**: Reward functions can now accept plural parameters (`states`, `prompts`, `completions`, `answers`, `tasks`, `infos`) to score multiple rollouts together. Rubrics automatically detect group functions by signature inspection.

## Migration Guide

### For Environment Developers

1. **Update `rollout()` and `init_state()` signatures** (if inheriting directly from `Environment`): If your environment doesn't inherit from `MultiTurnEnv`, you must update these method signatures:
   ```python
   async def init_state(
       self,
       input: RolloutInput,
       client: AsyncOpenAI,
       model: str,
       sampling_args: SamplingArgs | None = None,
   ) -> State:
       # Create state from input
       state = await super().init_state(input, client, model, sampling_args)
       # ... any custom initialization ...
       return state
   
   async def rollout(
       self,
       input: RolloutInput,
       client: AsyncOpenAI,
       model: str,
       sampling_args: SamplingArgs | None = None,
   ) -> State:
       state = await self.init_state(input, client, model, sampling_args)
       state = await self.setup_state(state)
       # ... your rollout logic ...
       return state
   ```

2. **Implement `setup_state()`** (if inheriting directly from `Environment`): `setup_state()` is abstract in `Environment` and must be implemented. Note that `MultiTurnEnv` implements it as a no-op, so environments inheriting from `MultiTurnEnv` can override it as needed but don't need to implement it:
   ```python
   async def setup_state(self, state: State) -> State:
       # Environment-specific setup (tools, sandbox IDs, etc.)
       return state
   ```

3. **Replace `is_completed()` overrides with `@stop` decorators**: Do not override `is_completed()`. Instead, define stop conditions as methods decorated with `@stop`:
   ```python
   from verifiers import stop
   
   class MyEnv(MultiTurnEnv):
       @stop
       async def max_turns_reached(self, state: State) -> bool:
           return len(state["trajectory"]) >= self.max_turns
   ```

4. **Update state access**: Use `state["trajectory"]` instead of `state["responses"]`

### For Users

- Most existing environments continue to work without changes due to backward compatibility via state forwarding (e.g. any SingleTurnEnv, ToolEnv, TextArenaEnv, etc.) will work without changes. Environments that override `is_completed()` should be updated to use `@stop` decorators instead.
- Trajectory data is now available in all rollouts via `state["trajectory"]`
- Each trajectory step contains `prompt`, `completion`, `response`, `tokens`, `reward`, and `advantage` fields

## Technical Details

- Trajectory steps preserve vLLM's native token IDs and logprobs, eliminating retokenization
- Token extraction requires vLLM configuration with `return_tokens_as_token_ids=True` and `return_token_ids=True` in `sampling_args` (trainers must pass these flags to enable training)
- Only completion logprobs are stored (prompt logprobs are not included)
- `_render_completion()` is called automatically when rollouts complete to render `state["completion"]` from the trajectory for backward compatibility and output saving
- Group reward functions are detected automatically by checking for plural parameter names or list return types

**Full Changelog**: https://github.com/PrimeIntellect-ai/verifiers/compare/v0.1.7...v0.1.8

