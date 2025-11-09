# Trajectory Implementation

## Core Philosophy

**A trajectory is a sequence of LLM request/response pairs**, not token-level segments. Each trajectory step represents one complete API call to the model with its returned tokens and logprobs, eliminating retokenization brittleness and enabling clean horizontal training.

## Implementation Status

### 1. Trajectory-Based Rollouts ✅

**Solution**: Trajectories are always enabled. Each rollout tracks every LLM request/response as an independent step in `state["trajectory"]`.

```python
TrajectoryStepTokens = TypedDict("TrajectoryStepTokens", {
    "prompt_ids": list[int],
    "prompt_mask": list[int],
    "completion_ids": list[int],
    "completion_mask": list[int],
    "completion_logprobs": list[float],
})

TrajectoryStep = TypedDict("TrajectoryStep", {
    "prompt": Messages,           # Exact prompt sent to LLM for THIS request
    "completion": Messages,       # Exact completion returned from THIS request
    "response": ModelResponse,    # Raw response object with token_ids/logprobs from vLLM
    "tokens": TrajectoryStepTokens | None,  # Extracted token IDs, masks, and logprobs (None if unavailable)
    "reward": float | None,       # Reward for this step (None during rollout, filled by Rubric)
    "advantage": float | None,    # Advantage for this step (for RL training)
    "extras": dict[str, Any],     # Additional step metadata
})

BaseRolloutInput = TypedDict("BaseRolloutInput", {
    "prompt": Messages,
    "example_id": int,
    "task": str,
})

RolloutInput(BaseRolloutInput, total=False):  # answer and info are optional
    "answer": str,
    "info": Info,
```

### State Structure

The `State` object is a dict subclass that tracks rollout information throughout an interaction:

```python
class State(dict):
    """
    Dict subclass with forwarding for INPUT_FIELDS.
    
    Accessing state["prompt"] forwards to state["input"].prompt if "input" exists,
    otherwise accesses directly from dict (backward compat).
    """
    INPUT_FIELDS = ["prompt", "answer", "task", "info", "example_id"]
    
    # Required: input fields (always in "input" RolloutInput)
    input: RolloutInput
    
    # Created during rollout
    is_completed: bool
    stop_condition: str | None
    oai_tools: list[ChatCompletionToolParam]
    trajectory: list[TrajectoryStep]  # Always a list (never None)
    completion: Messages | None       # Full completion (rendered from trajectory)
    reward: float | None               # Final reward for rollout
    advantage: float | None            # Final advantage for rollout
    metrics: dict[str, float] | None   # Additional metrics
    timing: RolloutTiming | None       # Timing info
    
    # Custom fields (env-specific, added by subclasses)
    # Examples:
    # "prompt_too_long": bool,         # For max length handling
    # "sandbox_id": str,                # For SandboxEnv
```

**Key Points**:
- State forwards access to `INPUT_FIELDS` (prompt, answer, task, info, example_id) from `state["input"]` for backward compatibility
- `trajectory` is always a list (never None) - trajectories are always enabled
- `completion` is rendered from trajectory when rollout completes
- Custom fields can be added by environment subclasses

**Key Benefits**:
- Each trajectory step = one vLLM request with native token_ids/logprobs (no retokenization)
- Prompts can be completely independent (no prefix-sharing requirement)
- Enables horizontal training: each step becomes an independent training example
- Supports truncated thinking naturally (just another step with new prompt)
- Clean foundation for step-based APIs

### 2. Input-First API ✅

**Solution**: `rollout()` takes `input: RolloutInput` as the primary parameter. State is created internally via `init_state()` and `setup_state()`.

```python
async def rollout(
    self,
    input: RolloutInput,
    client: AsyncOpenAI,
    model: str,
    sampling_args: SamplingArgs | None = None,
) -> State:
    """Execute rollout starting from input. Returns state with trajectory."""
    state = await self.init_state(input)
    state = await self.setup_state(state)
    # ... rollout logic ...
    return state
```

**Key Points**:
- `rollout()` is called with `input: RolloutInput` (dataset row fields)
- `init_state()` creates initial state from input
- `setup_state()` injects environment-specific configuration
- State is returned with populated trajectory

### 3. Completion Rendering ✅

**Solution**: Internal `_render_completion()` method renders completion from trajectory for display/scoring.

```python
async def _render_completion(self, state: State):
    """Render completion from last trajectory step."""
    last_prompt = state["trajectory"][-1]["prompt"]
    last_completion = state["trajectory"][-1]["completion"]
    full_conversation = concat_messages([last_prompt, last_completion])
    state["completion"] = full_conversation[len(state["prompt"]) :]
```

**Impact**:
- Called automatically when rollout completes (via `is_completed()`)
- Sets `state["completion"]` from trajectory for backward compatibility
- Used by evaluation tools and result saving
- Environments can override `_render_completion()` for custom rendering

### 4. init_state() and setup_state() Separation ✅

**Solution**: Clear separation of concerns:

- **`init_state()`**: Environment-agnostic. Takes `input: RolloutInput` and creates initial state. Sets up trajectory list, timing, and basic fields.
- **`setup_state()`**: Environment-specific. Injects env configuration (tools, sandbox IDs, etc.) before rollout begins.

```python
async def init_state(
    self,
    input: RolloutInput,
    **kwargs,
) -> State:
    """
    Create initial state from dataset row.
    Environment-agnostic - just stores the data.
    
    Creates State with input fields in "input" RolloutInput for structured access,
    but State's forwarding behavior allows backward-compatible direct access.
    """
    state_input = deepcopy(input)
    if "info" in state_input and isinstance(state_input["info"], str):
        state_input["info"] = json.loads(state_input["info"])
    if "task" not in state_input:
        state_input["task"] = self.env_id or "default"
    state = State(input=RolloutInput(**state_input))
    state["is_completed"] = False
    state["oai_tools"] = None
    # Resolve oai_tools from info or env
    if "info" in state and hasattr(state["info"], "oai_tools"):
        state["oai_tools"] = state["info"]["oai_tools"]
    elif hasattr(self, "oai_tools"):
        state["oai_tools"] = self.oai_tools
    else:
        state["oai_tools"] = []
    state["trajectory"] = []  # Always enabled
    state["reward"] = None
    state["metrics"] = None
    state["timing"] = RolloutTiming(
        generation_ms=0.0,
        scoring_ms=0.0,
        total_ms=0.0,
        start_time=time.time(),
    )
    return state

async def setup_state(self, state: State) -> State:
    """
    Inject environment-specific configuration.
    Called at start of rollout, before first LLM call.
    
    Examples:
    - ToolEnv: inject oai_tools if not already present
    - SandboxEnv: start sandbox provisioning (don't await)
    - Custom envs: any env-specific setup
    """
    return state
```

**Backward Compatibility**: Existing environments that put `oai_tools` in dataset's `info` column will continue to work (init_state checks if already present).

### 5. Termination Patterns with Decorators ✅

**Solution**: Decorator-based pattern for stop conditions, cleanup, and teardown handlers. Methods decorated with `@stop`, `@cleanup`, or `@teardown` are automatically discovered and called.

**Stop Conditions** (`@stop`):
- Methods decorated with `@stop` define when a rollout should terminate
- All stop conditions are checked by `is_completed()` each turn
- First condition that returns `True` terminates the rollout
- Sets `state["is_completed"] = True` and `state["stop_condition"]` to the method name

```python
from verifiers import stop

class MultiTurnEnv(Environment):
    @stop
    async def max_turns_reached(self, state: State) -> bool:
        """Check if the maximum number of turns has been reached."""
        return len(state["trajectory"]) >= self.max_turns and self.max_turns > 0
    
    @stop
    async def prompt_too_long(self, state: State) -> bool:
        return state.get("prompt_too_long", False)

async def is_completed(self, state: State, **kwargs) -> bool:
    """Check all stop conditions. Sets state.is_completed=True if any condition is met."""
    for condition in self._stop_conditions:  # Auto-discovered via @stop decorator
        if await self._render_stop(state, condition):
            await self._render_timing(state)
            await self._render_completion(state)
            return True
    return False
```

**Rollout Cleanup** (`@cleanup`):
- Methods decorated with `@cleanup` are called after each rollout completes
- Used for per-rollout resource cleanup (e.g., closing sandbox connections)
- Called via `_cleanup()` after rollout finishes

```python
from verifiers import cleanup

class SandboxEnv(MultiTurnEnv):
    @cleanup
    async def release_sandbox(self, state: State):
        """Release sandbox resources after rollout."""
        if "sandbox_id" in state:
            await self.sandbox_client.release(state["sandbox_id"])
```

**Environment Teardown** (`@teardown`):
- Methods decorated with `@teardown` are called when the environment is destroyed
- Used for global resource cleanup (e.g., deleting all sandboxes, closing connections)
- Automatically registered with `atexit` and signal handlers

```python
from verifiers import teardown

class SandboxEnv(MultiTurnEnv):
    @teardown
    async def cleanup_sandboxes(self):
        """Delete all active sandboxes on environment destruction."""
        for sandbox_id in self.active_sandboxes:
            await self.sandbox_client.delete(sandbox_id)
```

**Key Benefits**:
- Declarative: Just add decorators, no manual registration needed
- Automatic discovery: All decorated methods are found via `__post_init__()`
- Clean separation: Stop conditions vs cleanup vs teardown
- State tracking: `stop_condition` field records which condition terminated the rollout

**⚠️ Breaking Change**: Environments that override `is_completed()` need to be updated:

1. **Signature changed**: `is_completed(messages, state, **kwargs)` → `is_completed(state, **kwargs)` (messages parameter removed)
2. **Must call super()**: Custom `is_completed()` implementations must call `super().is_completed(state)` to check `@stop` decorated methods, otherwise stop conditions won't be checked

**Migration Example**:
```python
# OLD (broken):
async def is_completed(self, messages: Messages, state: State, **kwargs) -> bool:
    return len(state["responses"]) == 1

# NEW (fixed):
async def is_completed(self, state: State, **kwargs) -> bool:
    # Always check base stop conditions first (max_turns, prompt_too_long, etc.)
    if await super().is_completed(state, **kwargs):
        return True
    # Then check custom conditions
    return len(state["trajectory"]) == 1  # Use trajectory instead of responses
```

**Note**: The `messages` parameter was removed because the current conversation can be reconstructed from `state["trajectory"]` if needed. Use `state["trajectory"]` instead of `state["responses"]` for trajectory-based logic.

### 6. Horizontal Training Processing

**Solution**: Trajectory-based rollouts enable clean horizontal processing where each trajectory step becomes an independent training example:

```python
def process_trajectories_for_training(
    states: list[State],
    mask_prompts: bool = True,
) -> list[TrainingExample]:
    """
    Convert trajectory-based states to training examples.
    
    Each trajectory step becomes ONE training example with:
    - token_ids: extracted directly from step["tokens"] (already tokenized by vLLM)
    - logprobs: extracted directly from step["tokens"]
    - mask: simple prompt vs completion masking
    - reward: from step["reward"]
    
    No retokenization, no prefix reconstruction, no brittle manipulation.
    """
    examples = []
    for state in states:
        if state.get("trajectory"):
            for step in state["trajectory"]:
                if step["tokens"] is None:
                    continue  # Skip steps without token data
                # Extract token_ids and logprobs directly from trajectory step tokens
                prompt_ids = step["tokens"]["prompt_ids"]
                completion_ids = step["tokens"]["completion_ids"]
                completion_logprobs = step["tokens"]["completion_logprobs"]
                
                # Combine prompt and completion
                token_ids = prompt_ids + completion_ids
                logprobs = [0.0] * len(prompt_ids) + completion_logprobs  # No prompt logprobs
                
                # Use masks from trajectory step
                if mask_prompts:
                    mask = step["tokens"]["prompt_mask"] + step["tokens"]["completion_mask"]
                else:
                    mask = [1] * len(token_ids)
                
                examples.append(TrainingExample(
                    token_ids=token_ids,
                    logprobs=logprobs,
                    mask=mask,
                    reward=step.get("reward", 0.0),
                ))
    
    return examples
```

**Key Insight**: Trainer sees each trajectory step as a separate training example, processed horizontally, not as interleaved turns in one vertical sequence.

### 7. MultiTurnEnv Trajectory Support ✅

**How it works**: Each LLM call adds a trajectory step via `add_model_response()`:

```python
async def add_model_response(
    self,
    state: State,
    prompt_messages: Messages,
    response: ModelResponse,
):
    """Add a model response as a trajectory step."""
    if response is not None and response.id == "overlong-prompt":
        state["prompt_too_long"] = True
        return
    completion_messages = await parse_response_messages(response, self.message_type)
    tokens = await parse_response_tokens(response, self.message_type)
    trajectory_step = TrajectoryStep(
        prompt=prompt_messages,
        completion=completion_messages,
        response=response,
        tokens=tokens,
        reward=None,
        advantage=None,
        extras={},
    )
    state["trajectory"].append(trajectory_step)

async def rollout(
    self,
    input: RolloutInput,
    client: AsyncOpenAI,
    model: str,
    sampling_args: SamplingArgs | None = None,
) -> State:
    """Generate a multi-turn rollout with the environment."""
    state = await self.init_state(input)
    state = await self.setup_state(state)
    while not await self.is_completed(state):
        prompt_messages = await self.get_prompt_messages(state)
        response = await self.get_model_response(
            client,
            model,
            prompt_messages,
            oai_tools=state["oai_tools"],
            sampling_args=sampling_args,
            message_type=self.message_type,
        )
        await self.add_model_response(state, prompt_messages, response)
    return state
```

**Key point**: Each step is a completely independent LLM request. No prefix requirement, no retokenization.

## Implementation Details

### TrajectoryStep Structure

```python
TrajectoryStepTokens = TypedDict("TrajectoryStepTokens", {
    "prompt_ids": list[int],
    "prompt_mask": list[int],
    "completion_ids": list[int],
    "completion_mask": list[int],
    "completion_logprobs": list[float],
})

TrajectoryStep = TypedDict("TrajectoryStep", {
    "prompt": Messages,                    # Exact prompt sent to LLM for this request
    "completion": Messages,                 # Exact completion returned from this request
    "response": ModelResponse,             # Raw response (ChatCompletion or Completion) with token_ids/logprobs
    "tokens": TrajectoryStepTokens | None, # Extracted token IDs, masks, and logprobs (None if unavailable)
    "reward": float | None,                # Reward for this step (None during rollout, filled by scoring)
    "advantage": float | None,             # Advantage for this step (for RL training)
    "extras": dict[str, Any],               # Additional step metadata
})
```

**Note**: 
- `tokens` field contains extracted token IDs and logprobs from the response, enabling direct use for training without retokenization.
- `tokens` can be `None` if token data is unavailable (e.g., when logprobs not requested).
- Only `completion_logprobs` are stored (no prompt logprobs).
- `extras` allows environments to add custom metadata per step.

### Token Extraction

Token extraction is handled by `parse_response_tokens()` in `verifiers/utils/response_utils.py`:

```python
async def parse_response_tokens(
    response: ModelResponse, message_type: MessageType
) -> TrajectoryStepTokens | None:
    """Extract token IDs, masks, and logprobs from vLLM response."""
    if message_type == "chat":
        assert isinstance(response, ChatCompletion)
        # Extract from response.prompt_token_ids and response.choices[0].token_ids
        # Extract logprobs from response.choices[0].logprobs.content
    elif message_type == "completion":
        assert isinstance(response, Completion)
        # Extract from response.choices[0].prompt_token_ids and token_ids
        # Extract logprobs from response.choices[0].logprobs.token_logprobs
    return None  # If token data unavailable
```

**vLLM Configuration**: To ensure token_ids are included in responses, vLLM must be configured with the following flags in `sampling_args`:

```python
sampling_args = {
    "logprobs": 1,  # Enable logprobs
    "extra_body": {
        "return_tokens_as_token_ids": True,  # Return tokens as token IDs
        "return_token_ids": True,             # Include token_ids in response
        "prompt_logprobs": 1,                 # Optional: include prompt logprobs
    },
}
```

**Key insight**: Token IDs and logprobs come directly from vLLM response. No retokenization needed. If these flags are not set, `step["tokens"]` will be `None` and that trajectory step will be skipped during training data extraction.

### Environment Setup Pattern

```python
class ToolEnv(MultiTurnEnv):
    def __init__(self, tools: list[Callable], **kwargs):
        super().__init__(**kwargs)
        # Convert tools to OpenAI format and store at env level
        self.oai_tools = [convert_tool_to_oai_format(t) for t in tools]
    
    async def setup_state(self, state: State) -> State:
        """Inject oai_tools into state if not already present."""
        # Respect dataset-level tools if present (backward compat)
        # oai_tools already set in init_state() from info or self.oai_tools
        return state
```

**Pattern**: Environment-level config stored in `__init__`, resolved in `init_state()`, can be overridden by dataset-level info.

## Examples

### Example 1: Single-Turn Evaluation

```python
import verifiers as vf
from openai import AsyncOpenAI

# Load environment (works as before)
env = vf.load_environment("gsm8k")

# Run evaluation (no changes needed)
client = AsyncOpenAI()
results = await env.evaluate(client, model="gpt-4o-mini", num_examples=100)
print(f"Average reward: {results.metadata.avg_reward}")

# Access trajectory from results
for state in results.state:
    print(f"Trajectory steps: {len(state['trajectory'])}")
    for step in state["trajectory"]:
        print(f"  Prompt: {step['prompt']}")
        print(f"  Completion: {step['completion']}")
```

### Example 2: Multi-Turn with Trajectories

```python
# Multi-turn environment automatically tracks trajectories
env = vf.load_environment("wordle")

# Create input from dataset row
row = env.dataset[0]
input = {
    "prompt": row["prompt"],
    "answer": row.get("answer", ""),
    "task": row.get("task", "default"),
    "info": row.get("info", {}),
    "example_id": row["example_id"],
}

# Run rollout
state = await env.rollout(input, client, model="gpt-4o-mini")

# Access trajectory steps
for i, step in enumerate(state["trajectory"]):
    print(f"Step {i}:")
    print(f"  Prompt length: {len(step['prompt'])}")
    print(f"  Completion: {step['completion']}")
    print(f"  Reward: {step['reward']}")
    if step["tokens"]:
        print(f"  Token IDs: {step['tokens']['completion_ids']}")
```

### Example 3: Training with Trajectories

```python
# Generate rollouts (trajectories always enabled)
env = vf.load_environment("math")
results = await env.generate(
    dataset,
    client=client,
    model="model-checkpoint",
    score_rollouts=True,
)

# Process for training (horizontal processing)
training_examples = process_trajectories_for_training(
    states=results.state,
    mask_prompts=True,
)

# Each example has token_ids, logprobs, mask, reward
for example in training_examples[:5]:
    print(f"Sequence length: {len(example.token_ids)}")
    print(f"Reward: {example.reward}")
```

## Design Decisions

### 1. Trajectories = LLM Request/Response Pairs ✅
**Decision**: Each trajectory step stores the complete prompt/completion/response from one LLM API call.
**Rationale**: Eliminates retokenization brittleness, keeps vLLM's native token_ids/logprobs pristine, enables horizontal training.

### 2. init_state() vs setup_state() Separation ✅
**Decision**: 
- `init_state()`: Environment-agnostic, takes `input: RolloutInput`, creates state with trajectory list
- `setup_state()`: Environment-specific, injects config like oai_tools
**Rationale**: Clear separation of concerns, backward compatible with existing datasets that include `info["oai_tools"]`.

### 3. Trajectories Always Enabled ✅
**Decision**: `state["trajectory"] = []` is always set in `init_state()`, not opt-in.
**Rationale**: Simpler API, all rollouts have trajectory data available. No need for legacy mode.

### 4. Input-First API ✅
**Decision**: `rollout()` takes `input: RolloutInput`, creates state internally.
**Rationale**: Clean API that matches dataset structure. State is internal implementation detail.

### 5. Horizontal Training Processing ✅
**Decision**: Each trajectory step becomes an independent training example.
**Rationale**: Simpler than vertical interleaving, avoids retokenization issues, more flexible for trainers.

### 6. Token Extraction ✅
**Decision**: Only `completion_logprobs` stored, not prompt logprobs. `tokens` can be `None` if unavailable.
**Rationale**: Prompt logprobs rarely needed for training. Graceful handling when token data unavailable.

### 7. Decorator-Based Termination ✅
**Decision**: Use `@stop`, `@cleanup`, `@teardown` decorators for lifecycle management instead of manual registration.
**Rationale**: Declarative pattern, automatic discovery via `__post_init__()`, clean separation of concerns, easier to extend.

## Summary

This implementation achieves:

1. **Clean trajectory support** - Each step is one LLM request/response
2. **No retokenization brittleness** - Use vLLM's native token_ids/logprobs
3. **Horizontal training** - Each trajectory step = independent training example
4. **Backward compatibility** - Most existing environments work unchanged (via state forwarding)
5. **Setup pattern clarity** - `init_state()` for data, `setup_state()` for config
6. **Always-on trajectories** - All rollouts track trajectory data
7. **Input-first API** - Clean interface matching dataset structure
8. **Decorator-based termination** - `@stop`, `@cleanup`, `@teardown` for declarative lifecycle management

The implementation focuses on **not breaking existing environments** while providing trajectory data for all rollouts, enabling new trajectory-based patterns for training and advanced use cases.
