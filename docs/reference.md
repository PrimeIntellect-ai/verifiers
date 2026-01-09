# API Reference

## Table of Contents

- [Type Aliases](#type-aliases)
- [Data Types](#data-types)
- [Classes](#classes)
  - [Environment Classes](#environment-classes)
  - [Parser Classes](#parser-classes)
  - [Rubric Classes](#rubric-classes)
- [Configuration Types](#configuration-types)
- [Decorators](#decorators)
- [Utility Functions](#utility-functions)

---

## Type Aliases

### Messages

```python
Messages = str | list[ChatMessage]
```

The primary message type. Either a plain string (completion mode) or a list of chat messages (chat mode).

### ChatMessage

```python
ChatMessage = ChatCompletionMessageParam  # from openai.types.chat
```

OpenAI's chat message type with `role`, `content`, and optional `tool_calls` / `tool_call_id` fields.

### Info

```python
Info = dict[str, Any]
```

Arbitrary metadata dictionary from dataset rows.

### SamplingArgs

```python
SamplingArgs = dict[str, Any]
```

Generation parameters passed to the inference server (e.g., `temperature`, `top_p`, `max_tokens`).

### RewardFunc

```python
IndividualRewardFunc = Callable[..., float | Awaitable[float]]
GroupRewardFunc = Callable[..., list[float] | Awaitable[list[float]]]
RewardFunc = IndividualRewardFunc | GroupRewardFunc
```

Individual reward functions operate on single rollouts. Group reward functions operate on all rollouts for an example together (useful for relative scoring).

### ModelResponse

```python
ModelResponse = Completion | ChatCompletion | None
```

Raw response from the OpenAI API.

---

## Data Types

### State

```python
class State(dict):
    INPUT_FIELDS = ["prompt", "answer", "task", "info", "example_id"]
```

A `dict` subclass that tracks rollout information. Accessing keys in `INPUT_FIELDS` automatically forwards to the nested `input` object.

**Fields set during initialization:**

| Field | Type | Description |
|-------|------|-------------|
| `input` | `RolloutInput` | Nested input data |
| `client` | `AsyncOpenAI` | OpenAI client |
| `model` | `str` | Model name |
| `sampling_args` | `SamplingArgs \| None` | Generation parameters |
| `is_completed` | `bool` | Whether rollout has ended |
| `is_truncated` | `bool` | Whether generation was truncated |
| `oai_tools` | `list[ChatCompletionToolParam]` | Available tools |
| `trajectory` | `list[TrajectoryStep]` | Multi-turn trajectory |
| `trajectory_id` | `str` | UUID for this rollout |
| `timing` | `RolloutTiming` | Timing information |

**Fields set after scoring:**

| Field | Type | Description |
|-------|------|-------------|
| `completion` | `Messages \| None` | Final completion |
| `reward` | `float \| None` | Final reward |
| `advantage` | `float \| None` | Advantage over group mean |
| `metrics` | `dict[str, float] \| None` | Per-function metrics |
| `stop_condition` | `str \| None` | Name of triggered stop condition |
| `error` | `Error \| None` | Error if rollout failed |

### RolloutInput

```python
class RolloutInput(TypedDict):
    prompt: Messages        # Required
    example_id: int         # Required
    task: str               # Required
    answer: str             # Optional
    info: Info              # Optional
```

### TrajectoryStep

```python
class TrajectoryStep(TypedDict):
    prompt: Messages
    completion: Messages
    response: ModelResponse
    tokens: TrajectoryStepTokens | None
    reward: float | None
    advantage: float | None
    is_truncated: bool
    trajectory_id: str
    extras: dict[str, Any]
```

A single turn in a multi-turn rollout.

### TrajectoryStepTokens

```python
class TrajectoryStepTokens(TypedDict):
    prompt_ids: list[int]
    prompt_mask: list[int]
    completion_ids: list[int]
    completion_mask: list[int]
    completion_logprobs: list[float]
    overlong_prompt: bool
    is_truncated: bool
```

Token-level data for training.

### RolloutTiming

```python
class RolloutTiming(TypedDict, total=False):
    start_time: float
    generation_ms: float
    scoring_ms: float
    total_ms: float
```

### GenerateOutputs

```python
class GenerateOutputs(TypedDict):
    prompt: list[Messages]
    completion: list[Messages]
    answer: list[str]
    state: list[State]
    task: list[str]
    info: list[Info]
    example_id: list[int]
    reward: list[float]
    metrics: dict[str, list[float]]
    stop_conditions: list[str | None]
    is_truncated: list[bool]
    metadata: GenerateMetadata
```

Output from `Environment.generate()`.

### GenerateMetadata

```python
class GenerateMetadata(TypedDict):
    env_id: str
    env_args: dict
    model: str
    base_url: str
    num_examples: int
    rollouts_per_example: int
    sampling_args: SamplingArgs
    date: str
    time_ms: float
    avg_reward: float
    avg_metrics: dict[str, float]
    state_columns: list[str]
    path_to_save: Path
```

### RolloutScore / RolloutScores

```python
class RolloutScore(TypedDict):
    reward: float
    metrics: dict[str, float]

class RolloutScores(TypedDict):
    reward: list[float]
    metrics: dict[str, list[float]]
```

### ProcessedOutputs

```python
class ProcessedOutputs(TypedDict):
    prompt_ids: list[list[int]]
    prompt_mask: list[list[int]]
    completion_ids: list[list[int]]
    completion_mask: list[list[int]]
    completion_logprobs: list[list[float]]
    rewards: list[float]
    is_truncated: list[bool]
```

Tokenized outputs for training.

---

## Classes

### Environment Classes

#### Environment

```python
class Environment(ABC):
    def __init__(
        self,
        dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        system_prompt: str | None = None,
        few_shot: list[ChatMessage] | None = None,
        parser: Parser | None = None,
        rubric: Rubric | None = None,
        sampling_args: SamplingArgs | None = None,
        message_type: MessageType = "chat",
        oai_tools: list[ChatCompletionToolParam] | None = None,
        max_workers: int = 512,
        env_id: str | None = None,
        env_args: dict | None = None,
        max_seq_len: int | None = None,
        **kwargs,
    ): ...
```

Abstract base class for all environments.

**Generation methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `generate(inputs, client, model, ...)` | `GenerateOutputs` | Run rollouts asynchronously |
| `generate_sync(inputs, client, ...)` | `GenerateOutputs` | Synchronous wrapper |
| `evaluate(client, model, ...)` | `GenerateOutputs` | Evaluate on eval_dataset |
| `evaluate_sync(client, model, ...)` | `GenerateOutputs` | Synchronous evaluation |

**Dataset methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `get_dataset(n=-1, seed=None)` | `Dataset` | Get training dataset (optionally first n, shuffled) |
| `get_eval_dataset(n=-1, seed=None)` | `Dataset` | Get evaluation dataset |
| `make_dataset(...)` | `Dataset` | Static method to create dataset from inputs |

**Rollout methods (used internally or by subclasses):**

| Method | Returns | Description |
|--------|---------|-------------|
| `rollout(input, client, model, sampling_args)` | `State` | Abstract: run single rollout |
| `init_state(input, client, model, sampling_args)` | `State` | Create initial state from input |
| `get_model_response(state, prompt, ...)` | `ModelResponse` | Get model response for prompt |
| `is_completed(state)` | `bool` | Check all stop conditions |
| `run_rollout(sem, input, client, model, sampling_args)` | `State` | Run rollout with semaphore |
| `run_group(group_inputs, client, model, ...)` | `list[State]` | Generate and score one group |

**Configuration methods:**

| Method | Description |
|--------|-------------|
| `set_kwargs(**kwargs)` | Set attributes using setter methods when available |
| `add_rubric(rubric)` | Add or merge rubric |
| `set_max_seq_len(max_seq_len)` | Set maximum sequence length |
| `set_interleaved_rollouts(bool)` | Enable/disable interleaved rollouts |
| `set_score_rollouts(bool)` | Enable/disable scoring |

#### SingleTurnEnv

Single-response Q&A tasks. Inherits from `Environment`.

#### MultiTurnEnv

```python
class MultiTurnEnv(Environment):
    def __init__(self, max_turns: int = -1, **kwargs): ...
```

Multi-turn interactions. Subclasses must implement `env_response`.

**Abstract method:**

```python
async def env_response(self, messages: Messages, state: State, **kwargs) -> Messages:
    """Generate environment feedback after model turn."""
```

**Built-in stop conditions:** `has_error`, `prompt_too_long`, `max_turns_reached`, `has_final_env_response`

**Hooks:**

| Method | Description |
|--------|-------------|
| `setup_state(state)` | Initialize per-rollout state |
| `get_prompt_messages(state)` | Customize prompt construction |
| `render_completion(state)` | Customize completion rendering |
| `add_trajectory_step(state, step)` | Customize trajectory handling |

#### ToolEnv

```python
class ToolEnv(MultiTurnEnv):
    def __init__(
        self,
        tools: list[Callable] | None = None,
        max_turns: int = 10,
        error_formatter: Callable[[Exception], str] = lambda e: f"{e}",
        stop_errors: list[type[Exception]] | None = None,
        **kwargs,
    ): ...
```

Tool calling with stateless Python functions. Automatically converts functions to OpenAI tool format.

**Built-in stop condition:** `no_tools_called` (ends when model responds without tool calls)

**Methods:**

| Method | Description |
|--------|-------------|
| `add_tool(tool)` | Add a tool at runtime |
| `remove_tool(tool)` | Remove a tool at runtime |
| `call_tool(name, args, id)` | Override to customize tool execution |

#### StatefulToolEnv

Tools requiring per-rollout state. Override `setup_state` and `update_tool_args` to inject state.

#### SandboxEnv

Sandboxed container execution using `prime` sandboxes.

#### PythonEnv

Persistent Python REPL in sandbox. Extends `SandboxEnv`.

#### EnvGroup

```python
env_group = vf.EnvGroup(
    envs=[env1, env2, env3],
    names=["math", "code", "qa"]  # optional
)
```

Combines multiple environments for mixed-task training.

---

### Parser Classes

#### Parser

```python
class Parser:
    def __init__(self, extract_fn: Callable[[str], str] = lambda x: x): ...
    
    def parse(self, text: str) -> Any: ...
    def parse_answer(self, completion: Messages) -> str | None: ...
    def get_format_reward_func(self) -> Callable: ...
```

Base parser. Default behavior returns text as-is.

#### XMLParser

```python
class XMLParser(Parser):
    def __init__(
        self,
        fields: list[str | tuple[str, ...]],
        answer_field: str = "answer",
        extract_fn: Callable[[str], str] = lambda x: x,
    ): ...
```

Extracts structured fields from XML-tagged output.

```python
parser = vf.XMLParser(fields=["reasoning", "answer"])
# Parses: <reasoning>...</reasoning><answer>...</answer>

# With alternatives:
parser = vf.XMLParser(fields=["reasoning", ("code", "answer")])
# Accepts either <code> or <answer> for second field
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `parse(text)` | `SimpleNamespace` | Parse XML into object with field attributes |
| `parse_answer(completion)` | `str \| None` | Extract answer field from completion |
| `get_format_str()` | `str` | Get format description string |
| `get_fields()` | `list[str]` | Get canonical field names |
| `format(**kwargs)` | `str` | Format kwargs into XML string |

#### ThinkParser

```python
class ThinkParser(Parser):
    def __init__(self, extract_fn: Callable[[str], str] = lambda x: x): ...
```

Extracts content after `</think>` tag. For models that always include `<think>` tags but don't parse them automatically.

#### MaybeThinkParser

Handles optional `<think>` tags (for models that may or may not think).

---

### Rubric Classes

#### Rubric

```python
class Rubric:
    def __init__(
        self,
        funcs: list[RewardFunc] | None = None,
        weights: list[float] | None = None,
        parser: Parser | None = None,
    ): ...
```

Combines multiple reward functions with weights. Default weight is `1.0`. Functions with `weight=0.0` are tracked as metrics only.

**Methods:**

| Method | Description |
|--------|-------------|
| `add_reward_func(func, weight=1.0)` | Add a reward function |
| `add_metric(func, weight=0.0)` | Add a metric (no reward contribution) |
| `add_class_object(name, obj)` | Add object accessible in reward functions |

**Reward function signature:**

```python
def my_reward(
    completion: Messages,
    answer: str = "",
    prompt: Messages | None = None,
    state: State | None = None,
    parser: Parser | None = None,  # if rubric has parser
    task: str = "",
    info: Info | None = None,
    **kwargs
) -> float:
    ...
```

**Group reward function signature:**

```python
def my_group_reward(
    completions: list[Messages],
    answers: list[str],
    states: list[State],
    # ... plural versions of individual args
    **kwargs
) -> list[float]:
    ...
```

#### JudgeRubric

LLM-as-judge evaluation.

#### MathRubric

Math-specific evaluation using `math-verify`.

#### RubricGroup

Combines rubrics for `EnvGroup`.

---

## Configuration Types

### ClientConfig

```python
class ClientConfig(BaseModel):
    api_key_var: str = "PRIME_API_KEY"
    api_base_url: str = "https://api.pinference.ai/api/v1"
    timeout: float = 3600.0
    max_connections: int = 28000
    max_keepalive_connections: int = 28000
    max_retries: int = 10
    extra_headers: dict[str, str] | None = None
```

### EvalConfig

```python
class EvalConfig(BaseModel):
    env_id: str
    env_args: dict
    env_dir_path: str
    model: str
    client_config: ClientConfig
    sampling_args: SamplingArgs
    num_examples: int
    rollouts_per_example: int
    max_concurrent: int
    max_concurrent_generation: int | None = None
    max_concurrent_scoring: int | None = None
    extra_env_kwargs: dict = {}
    print_results: bool = False
    verbose: bool = False
    state_columns: list[str] | None = None
    save_results: bool = False
    save_every: int = -1
    save_to_hf_hub: bool = False
    hf_hub_dataset_name: str | None = None
```

### Endpoint

```python
Endpoint = TypedDict("Endpoint", {"key": str, "url": str, "model": str})
Endpoints = dict[str, Endpoint]
```

---

## Decorators

### @vf.stop

```python
@vf.stop
async def my_condition(self, state: State) -> bool:
    """Return True to end the rollout."""
    ...

@vf.stop(priority=10)  # Higher priority runs first
async def early_check(self, state: State) -> bool:
    ...
```

Mark a method as a stop condition. All stop conditions are checked by `is_completed()`.

### @vf.cleanup

```python
@vf.cleanup
async def my_cleanup(self, state: State) -> None:
    """Called after each rollout completes."""
    ...

@vf.cleanup(priority=10)
async def early_cleanup(self, state: State) -> None:
    ...
```

Mark a method as a rollout cleanup handler.

### @vf.teardown

```python
@vf.teardown
async def my_teardown(self) -> None:
    """Called when environment is destroyed."""
    ...

@vf.teardown(priority=10)
async def early_teardown(self) -> None:
    ...
```

Mark a method as an environment teardown handler.

---

## Utility Functions

### Data Utilities

```python
vf.load_example_dataset(name: str) -> Dataset
```

Load a built-in example dataset.

```python
vf.extract_boxed_answer(text: str) -> str | None
```

Extract answer from LaTeX `\boxed{}` format.

```python
vf.extract_hash_answer(text: str) -> str | None
```

Extract answer after `####` marker (GSM8K format).

### Environment Utilities

```python
vf.load_environment(env_id: str, **kwargs) -> Environment
```

Load an environment by ID (e.g., `"primeintellect/gsm8k"`).

### Logging Utilities

```python
vf.print_prompt_completions_sample(outputs: GenerateOutputs, n: int = 3)
```

Pretty-print sample rollouts.

```python
vf.setup_logging(level: str = "INFO")
```

Configure verifiers logging. Set `VF_LOG_LEVEL` env var to change default.
