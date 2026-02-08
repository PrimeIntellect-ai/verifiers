# environments/AGENTS.md

<!-- Generated for repository development workflows. Do not edit directly. -->

This file mirrors the "Environments" documentation page.

---

This guide walks through building environments in Verifiers, from simple single-turn tasks to complex multi-turn agents with tools. See [Overview](overview.md) for how to initialize a new environment template.

## Table of Contents
- [Your First Environment](#your-first-environment)
- [Datasets](#datasets)
  - [Building the Prompt](#building-the-prompt)
  - [Evaluation Datasets](#evaluation-datasets)
  - [Lazy Loading with DatasetBuilder](#lazy-loading-with-datasetbuilder)
- [Rubrics](#rubrics)
  - [Reward Functions](#reward-functions)
  - [Multiple Reward Functions](#multiple-reward-functions)
  - [Execution Order and State](#execution-order-and-state)
  - [Group-Based Reward Functions](#group-based-reward-functions)
  - [Shared Objects](#shared-objects)
  - [Rubric Groups](#rubric-groups)
  - [Metrics and Monitor Rubrics](#metrics-and-monitor-rubrics)
- [Tool Environments](#tool-environments)
  - [MCP Tool Environments](#mcp-tool-environments)
  - [Stateful Tool Environments](#stateful-tool-environments)
- [Custom Multi-Turn Environments](#custom-multi-turn-environments)
  - [The Rollout Loop](#the-rollout-loop)
  - [Stop Conditions](#stop-conditions)
  - [Error Handling](#error-handling)
  - [State Initialization](#state-initialization)
  - [Cleanup and Teardown](#cleanup-and-teardown)
  - [Signaling Early Termination](#signaling-early-termination)
- [Developing Environments](#developing-environments)
  - [pyproject.toml](#pyprojecttoml)
  - [Managing Dependencies](#managing-dependencies)
  - [Installation](#installation)
- [Environment Groups](#environment-groups)
- [Integrations and Experimental Environments](#integrations-and-experimental-environments)

## Your First Environment

The simplest single-turn environments need only a dataset of tasks and a reward function for scoring responses:

```python
import verifiers as vf
from datasets import Dataset

def load_environment():
    # Your task data
    dataset = Dataset.from_list([
        {"prompt": [{"role": "user", "content": "What is 2+2?"}], "answer": "4"},
        {"prompt": [{"role": "user", "content": "What is 3*5?"}], "answer": "15"},
    ])
    
    # Your reward function
    async def correct_answer(completion, answer) -> float:
        response = completion[-1]["content"]
        return 1.0 if answer in response else 0.0
    
    rubric = vf.Rubric(funcs=[correct_answer])
    
    return vf.SingleTurnEnv(dataset=dataset, rubric=rubric)
```

When running this environment, each row in the dataset becomes a **rollout**:

1. The `prompt` is sent to the model
2. The model generates a response, which becomes the `completion`
3. The reward function scores the result

In `SingleTurnEnv`, the simplest environment type, just a single model response occurs per rollout. More complex environment types will allow us to add tool use or other custom interaction protocols.

## Datasets

Environments use the `datasets` library from Hugging Face for loading and manipulating datasets. Each row typically has a `prompt` column, containing a list of initial messages to send to the model. Additionally, there are optional columns for scoring:

- `answer` — a simple string for ground truth comparisons
- `info` — structured metadata (dict or JSON string)

Depending on what your environment needs, you can include `answer`, `info`, both, or neither.

When using `info`, prefer using JSON strings if rows may have different schemas, e.g. different fields or nested structures:

```python
dataset = Dataset.from_list([
    {"prompt": [...], "info": '{"type": "math", "difficulty": 3}'},
    {"prompt": [...], "info": '{"type": "code", "language": "python"}'},
])
```

These are parsed into a `dict` by the environment when running rollouts. 


### Building the Prompt

The examples above use `prompt` directly, providing a list of messages ready to send to the model. Alternatively, you can provide a `question` column containing a string, and the environment will wrap it in a user message:

```python
dataset = Dataset.from_list([
    {"question": "What is 2+2?", "answer": "4"},
])
```

You can also pass a `system_prompt` to the environment, which prepends a system message:

```python
return vf.SingleTurnEnv(
    dataset=dataset,
    system_prompt="You are a helpful math tutor.",
    rubric=rubric,
)
```

Together, these construct the full prompt:
```python
[
    {"role": "system", "content": "You are a helpful math tutor."},
    {"role": "user", "content": "What is 2+2?"}
]
```

If your dataset already has a `prompt` column, `question` is ignored. However, if a `system_prompt` is provided, it will be prepended to existing prompts that don't already start with a system message.

### Evaluation Datasets

Environments can be initialized with a separate `eval_dataset` for evaluation, distinct from the training dataset:

```python
return vf.SingleTurnEnv(
    dataset=train_dataset,
    eval_dataset=eval_dataset,
    rubric=rubric,
)
```

When running `prime eval run`, the evaluation dataset is used by default. If no `eval_dataset` is provided, evaluation falls back to the training dataset.

### Lazy Loading with DatasetBuilder

For large datasets or when running multiple environment replicas, you can defer dataset loading using a `DatasetBuilder`—a callable that returns a `Dataset` when invoked:

```python
def get_dataset_builder(split: str = "train", seed: int = 42) -> vf.DatasetBuilder:
    """Returns a builder that lazily loads the dataset."""
    def build() -> Dataset:
        ds = load_dataset("my-dataset", split=split)
        ds = ds.shuffle(seed=seed)
        return ds
    return build

def load_environment():
    dataset_builder = get_dataset_builder(split="train")
    eval_builder = get_dataset_builder(split="test")
    
    return vf.SingleTurnEnv(
        dataset=dataset_builder,      # built on first access
        eval_dataset=eval_builder,    # built on first access
        rubric=rubric,
    )
```

The builder pattern is useful when:
- Dataset loading is expensive (e.g., downloading from Hugging Face)
- Multiple environment replicas don't all need to own the dataset
- You want to parameterize dataset creation without loading it immediately

When a raw `Dataset` is passed directly (the default pattern), it is loaded eagerly during environment initialization for backwards compatibility.

## Rubrics

Each environment has a `Rubric` that manages scoring. The rubric holds reward functions, combines their outputs into a final reward score, and tracks metrics for observability.

### Reward Functions

Reward functions evaluate rollouts and return floats, typically between 0.0 and 1.0. They can request data from the rollout by naming arguments directly:

```python
async def correct_answer(completion, answer) -> float:
    response = completion[-1]["content"]
    return 1.0 if answer in response else 0.0
```

The basic available arguments, if present, are:
- `completion` — the model's output (list of messages)
- `prompt` — the input messages
- `answer` — from dataset
- `info` — from dataset
- `state` — the full rollout state (used in more complex environments)

This reference pattern extends to additional objects that the rubric provides in more advanced use cases.

### Multiple Reward Functions

Rubrics can combine multiple reward functions with custom weights:

```python
async def check_keywords(completion, info) -> float:
    response = completion[-1]["content"]
    keywords = info["required_keywords"]
    found = sum(1 for kw in keywords if kw.lower() in response.lower())
    return found / len(keywords)

async def length_reward(completion) -> float:
    response = completion[-1]["content"]
    return 1.0 if len(response) < 500 else 0.5

rubric = vf.Rubric(
    funcs=[check_keywords, length_reward],
    weights=[1.0, 0.1]
)
```

The final rollout reward is computed as the weighted sum of all reward function scores.

Reward functions can also be added to a rubric after initialization:
```python
rubric = vf.Rubric()
rubric.add_reward_func(check_keywords, weight=1.0)
rubric.add_reward_func(length_reward, weight=0.1)
```

Beyond the final score, reward functions can be used to track metrics for observability by setting `weight=0`:

```python
async def response_length(completion) -> float:
    return float(len(completion[-1]["content"]))
rubric.add_metric(response_length)  # shorthand for weight=0
```

All reward functions (weighted or not) appear in the rollout metrics.

### Execution Order and State

Reward functions execute in the order they are added to the rubric. Since `state` is mutable and shared across all reward functions, earlier functions can store computed values for later functions to use:

```python
async def similarity_score(completion, answer, state) -> float:
    response = completion[-1]["content"]
    score = compute_similarity(response, answer)  # continuous 0-1
    state["similarity"] = score
    return score

async def similarity_threshold(state) -> float:
    return 1.0 if state["similarity"] > 0.8 else 0.0

rubric = vf.Rubric(
    funcs=[similarity_score, similarity_threshold],
    weights=[0.0, 1.0]  # log similarity, but only reward threshold
)
```

This avoids redundant computation when multiple reward functions need access to the same derived value.

### Group-Based Reward Functions

During evaluation and RL training, rollouts are organized into **groups** of rollouts from the same input example. When evaluating, group structure enables per-example aggregate statistics (e.g., pass@k). When training with RL, groups are used for advantage computation relative to other rollouts for the same example. For a dataset with 100 example rows, running 4 rollouts per example yields 100 groups of 4 rollouts each.

In some cases, it is useful for reward functions to operate at the group level, such as to measure diversity or compute relative rankings. To define a group reward function, use plural argument names (`completions`, `prompts`, `answers`, `infos`) and return a list of scores:

```python
async def diversity_bonus(completions) -> list[float]:
    """Reward unique responses within a group."""
    responses = [c[-1]["content"] for c in completions]
    unique = set(responses)
    # Higher reward if this response is unique
    return [0.2 if responses.count(r) == 1 else 0.0 for r in responses]

rubric = vf.Rubric(funcs=[correct_answer, diversity_bonus])
```

### Shared Objects

Beyond rollout data, reward functions can request static objects that live within the Rubric class. These are stored in the Rubric's `class_objects` dictionary, and can be added after initialization via `add_class_object()`:

```python
rubric = vf.Rubric(funcs=[my_reward_func])
rubric.add_class_object("my_helper", some_helper_object)

async def my_reward_func(completion, my_helper) -> float:
    # my_helper is now available by name
    return await my_helper.score(completion)
```

Two common types of shared objects are **parsers** and **judges**.

Parsers encapsulate logic for extracting structured content from model responses. When passed to a rubric, the parser is automatically available to reward functions:

```python
parser = vf.XMLParser(["reasoning", "answer"])
rubric = vf.Rubric(funcs=[my_reward_func], parser=parser)

async def my_reward_func(completion, parser) -> float:
    parsed = parser.parse_answer(completion)
    # parsed.reasoning, parsed.answer available
    ...
```

Parsers can also be passed to environments, where they are often used during rollouts to validate or extract content. This allows parsing logic to be shared between the environment's interaction loop and the rubric's reward functions.

Judges are used for tasks where deterministic evaluation is impractical, and an LLM is used to score responses. **JudgeRubric** is a built-in class which stores an LLM client inside the rubric, and provides a `judge` callable to reward functions for scoring responses:

```python
judge_rubric = vf.JudgeRubric(
    judge_model="gpt-4.1-mini",
)

async def judge_correctness(prompt, completion, answer, judge) -> float:
    verdict = await judge(prompt, completion, answer)
    return 1.0 if "yes" in verdict.lower() else 0.0

judge_rubric.add_reward_func(judge_correctness)
```

The `judge` callable formats a prompt comparing the model's response to the ground truth and returns the judge model's verdict.

For more control, JudgeRubric accepts a custom `judge_prompt` template and exposes its internals (`judge_client`, `judge_model`, `judge_prompt`, `judge_sampling_args`) as class objects:

```python
judge_rubric = vf.JudgeRubric(
    judge_model="gpt-4.1-mini",
    judge_prompt="""Rate the writing quality of this response from 0-10.
Response: {response}
Score:"""
)

async def quality_score(completion, judge_client, judge_model, judge_prompt, parser) -> float:
    response = parser.parse_answer(completion)
    filled_prompt = judge_prompt.format(response=response)
    result = await judge_client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": filled_prompt}],
    )
    # parse numeric score from result
    ...
    return score
```

### Rubric Groups

Environments can include multiple rubrics by combining them into a `RubricGroup` (which itself behaves as a single rubric), aggregating all rewards and metrics from constituent rubrics. This is particularly useful for conjoining multiple rubrics of different types.

For example, `MathRubric` is a built-in rubric that uses symbolic verification to check mathematical correctness:

```python
math_rubric = vf.MathRubric()
```

MathRubric includes a `correct_answer` reward function that parses `\boxed{}` answers and uses the `math-verify` library for symbolic equivalence checking. To add LLM-based evaluation alongside it:

```python
math_rubric = vf.MathRubric()
judge_rubric = vf.JudgeRubric(judge_model="gpt-4.1-mini")
judge_rubric.add_reward_func(judge_correctness, weight=0.5)

rubric = vf.RubricGroup([math_rubric, judge_rubric])
```

All rubrics in a group are executed in parallel, and the final reward is the sum of all rubric rewards. Metrics from all rubrics are collected together.

### Metrics and Monitor Rubrics

For simple cases, metrics can be added directly to a rubric via `add_metric()` as shown above. Monitor rubrics extend this pattern by packaging metrics into separate rubrics that are combined via `add_rubric()`. This allows each environment type in a class hierarchy to contribute its own metrics automatically.

Many environment types automatically include a monitor rubric that tracks metrics specific to their level of the environment class hierarchy:

| Environment | Tracked Metrics |
|-------------|-----------------|
| `MultiTurnEnv` | `num_turns` |
| `ToolEnv` | `total_tool_calls`, per-tool counts |
| `SandboxEnv` | `sandbox_ready_wait_time`, `sandbox_command_execution_time` |
| `PythonEnv` | `python_ready_wait_time` |

These metrics appear automatically in rollout results alongside any custom reward functions.

To add custom metrics to an environment, define a monitor rubric class and add it via `add_rubric()`:

```python
class MyMonitorRubric(vf.Rubric):
    def __init__(self):
        super().__init__()
        self.add_metric(self.custom_metric)
    
    async def custom_metric(self, state: vf.State) -> float:
        return len(state["trajectory"])

env = vf.ToolEnv(dataset=dataset, tools=tools, rubric=rubric)
env.add_rubric(MyMonitorRubric())
```

The environment automatically wraps rubrics in a `RubricGroup` as needed, so monitor rubrics stack up the class hierarchy—`PythonEnv` inherits metrics from both `SandboxEnv` and `ToolEnv`.

## Tool Environments

All currently-supported environment types in Verifiers are built on `MultiTurnEnv`, which implements the core single-agent rollout loop (even `SingleTurnEnv` is simply a `MultiTurnEnv` with `max_turns=1` and a placeholder `env_response` method). `ToolEnv` adds tool calling to this foundation.

Tools are defined as Python functions. Verifiers extracts tool schemas from function signatures and docstrings for use with OpenAI-compatible tool calling:

```python
async def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.
    
    Args:
        expression: A mathematical expression to evaluate (e.g. "2 + 2 * 3")
    
    Returns:
        The result of the evaluation.
    """
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

async def lookup(term: str) -> str:
    """Look up a term in the knowledge base.
    
    Args:
        term: The term to search for.
    
    Returns:
        Information about the term.
    """
    # your lookup logic here
    ...
```

The function name becomes the tool name, type hints define the parameter types, and the docstring provides both the tool description and individual parameter descriptions (via the Args section). Tools can be sync or async, though we always recommend using async for performance to avoid blocking the main thread.

To create a tool environment, pass the tools to `ToolEnv` directly:

```python
vf_env = vf.ToolEnv(
    dataset=dataset,
    tools=[calculate, lookup],
    rubric=rubric,
    max_turns=10,
)
```

During rollouts, the model can call tools, receive results, and continue reasoning until it produces a response without tool calls (or hits `max_turns`). Each turn consists of a model response followed by the environment's tool execution. Tool call counts are tracked automatically via monitor rubrics (see above).

### MCP Tool Environments

For tools implemented as MCP (Model Context Protocol) servers, `MCPEnv` extends `ToolEnv` to provide an integration that automatically connects to MCP servers and exposes their tools to the model:

```python
mcp_servers = [
    {
        "name": "fetch",
        "command": "uvx",
        "args": ["mcp-server-fetch"],
    },
]

vf_env = vf.MCPEnv(
    mcp_servers=mcp_servers,
    dataset=dataset,
    rubric=rubric,
)
```

### Stateful Tool Environments

`ToolEnv` and `MCPEnv` are designed for stateless, read-only tools where no session state needs to persist across calls within a rollout. For tools that require per-rollout state—such as a sandbox container, database connection, or session ID—use `StatefulToolEnv`.

The `setup_state` method is called at the beginning of each rollout for all environments which extend `MultiTurnEnv`, but is a no-op by default (including in `ToolEnv`). 

`StatefulToolEnv` overrides this to initialize per-rollout resources, and introduces two additional concepts:

1. **Hidden arguments**: Tool functions can have parameters that are injected by the environment but hidden from the model's tool schema (via `args_to_skip`)
2. **`update_tool_args`**: An abstract method you implement to inject state into tool calls at runtime

```python
class MySandboxEnv(vf.StatefulToolEnv):
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
    
    async def run_code(self, code: str, session_id: str) -> str:
        """Execute code in the sandbox."""
        return await execute_in_session(session_id, code)
```

The model sees `run_code(code: str)` in its tool schema, but the environment injects `session_id` from rollout state before each call.

Verifiers includes several built-in stateful environment classes: `SandboxEnv` provides a containerized bash shell, and `PythonEnv` extends it with a persistent Python REPL (both of which are configured for use with Prime Intellect's [Sandboxes](https://docs.primeintellect.ai/sandboxes/overview)). These handle sandbox lifecycle management automatically.

Both `SandboxEnv` and `CliAgentEnv` accept a `labels` parameter for tagging sandboxes:

```python
env = vf.SandboxEnv(
    dataset=dataset,
    rubric=rubric,
    labels=["experiment-1", "math-tasks"],  # optional labels for sandbox categorization
)
```

Labels are passed to the Prime Sandboxes API and can be used for organizing, filtering, and managing sandboxes across experiments or training runs.

Stateful environments often define methods decorated with `@vf.cleanup` (called after each rollout) or `@vf.teardown` (called once at environment shutdown) for resource management. These decorators, along with `@vf.stop` for custom stop conditions (boolean functions checked after each turn), are powerful tools for rollout lifecycle control in custom `MultiTurnEnv` subclasses.

## Custom Multi-Turn Environments

For interaction patterns beyond tool calling—games, simulations, or other custom protocols—`MultiTurnEnv` can be subclassed directly, exposing full control over the rollout loop's behavior.

### The Rollout Loop

Each rollout follows this structure:

1. **Initialize state** — `setup_state(state)` is called to prepare per-rollout resources
2. **Loop until done:**
   - Get prompt messages (initial prompt, or previous conversation + environment response)
   - Get model response
   - Check stop conditions — if any `@vf.stop` method returns `True`, exit loop
3. **Render completion** — final conversation is assembled into `state["completion"]`
4. **Cleanup** — all `@vf.cleanup` methods are called

The `env_response` method is an abstract method that must be overridden by all `MultiTurnEnv` subclasses, and defines how the environment responds after each model turn:

```python
class MyGameEnv(vf.MultiTurnEnv):
    async def env_response(self, messages: vf.Messages, state: vf.State) -> vf.Messages:
        """Generate the environment's response after each model turn."""
        parsed = self.parser.parse(messages)
        action = parsed.action
        feedback = process_action(action)
        return [{"role": "user", "content": feedback}]


async def correct_action(parser, completion, answer) -> float:
    parsed = parser.parse(completion)
    return 1.0 if parsed.action == answer else 0.0


def load_environment():
    parser = vf.XMLParser(fields=["action"])
    rubric = vf.Rubric(funcs=[correct_action], parser=parser)
    return MyGameEnv(dataset=dataset, rubric=rubric, parser=parser)
```

`env_response` receives the full conversation history thus far (and `state`) and returns a list of *new* messages to append. When a parser is passed to the environment, it becomes available as `self.parser`. Passing the same parser to the rubric makes it available to reward functions by name. For tool environments, `env_response` typically executes tool calls and returns results. For games or other custom protocols, this might involve parsing structured output (as above) and returning state updates or feedback.

Several other methods can optionally be overridden for more control in complex custom environments:

- `setup_state(state)` — add environment-specific state fields at rollout start
- `get_prompt_messages(state)` — customize how messages are assembled (e.g. for non-linear conversations)
- `render_completion(state)` — customize how the final completion is assembled
- `add_trajectory_step(state, step)` — set intermediate rewards, advantages, or extra metadata per turn

### Stop Conditions

Rollouts continue until a stop condition is met, checked after each model response. Custom stop conditions are defined with the `@vf.stop` decorator:

```python
class MyGameEnv(vf.MultiTurnEnv):
    @vf.stop
    async def game_won(self, state: vf.State) -> bool:
        return state.get("won", False)
    
    @vf.stop
    async def game_lost(self, state: vf.State) -> bool:
        return state.get("lives", 1) <= 0
```

`MultiTurnEnv` includes built-in stop conditions for errors, prompt length limits, and `max_turns` by default.

Execution order can be controlled with `priority` (higher runs first). This is useful for checking cheap conditions before expensive ones:

```python
@vf.stop(priority=10)  # cheap keyword check runs first
async def answer_submitted(self, state: vf.State) -> bool:
    completion = state.get("completion", [])
    if not completion:
        return False
    return "FINAL ANSWER:" in completion[-1].get("content", "")

@vf.stop(priority=-10)  # expensive validation runs last
async def answer_detected(self, state: vf.State) -> bool:
    # only runs if cheap checks didn't already stop
    return await self.validator_client.check_for_answer(state)
```

### Error Handling

Verifiers defines a hierarchy of error types under `vf.Error`:

- `vf.ModelError` — errors from model interactions (e.g., `vf.EmptyModelResponseError`)
- `vf.OverlongPromptError` — prompt exceeds model context length
- `vf.ToolError` — tool-related errors (`vf.ToolParseError`, `vf.ToolCallError`)
- `vf.InfraError` — infrastructure errors (e.g., `vf.SandboxError`)

When a `vf.Error` is raised during a rollout, it is automatically caught and stored in `state["error"]`, triggering the built-in `has_error` stop condition at the next check. This allows rollouts to terminate gracefully rather than crashing.

For tool environments, you can configure which errors should stop the rollout immediately via `stop_errors`:

```python
vf_env = vf.ToolEnv(
    tools=[my_tool],
    stop_errors=[vf.ToolParseError],  # stop on parse errors, but continue on other tool errors
    ...
)
```

Errors not in `stop_errors` are caught and returned as tool response messages, providing the model a chance to recover.

### State Initialization

Override `setup_state` to initialize per-rollout state:

```python
class MyGameEnv(vf.MultiTurnEnv):
    async def setup_state(self, state: vf.State) -> vf.State:
        state["board"] = initialize_board()
        state["score"] = 0
        return await super().setup_state(state)
```

### Cleanup and Teardown

For resource management, use `@vf.cleanup` (per-rollout) and `@vf.teardown` (at environment shutdown):

```python
class MyGameEnv(vf.MultiTurnEnv):
    @vf.cleanup
    async def save_game_log(self, state: vf.State):
        await log_game_result(state["game_id"], state["score"])

    @vf.teardown
    async def close_connections(self):
        await self.db_connection.close()
```

> **Important:** Cleanup methods should be **idempotent**—safe to call multiple times—and handle errors gracefully. This ensures correct behavior when rollouts are cancelled or interrupted, and that cleanup completes even when resources are in unexpected states.

### Signaling Early Termination

To end a rollout from within `env_response` (e.g., when the game ends), set `state["final_env_response"]`:

```python
async def env_response(self, messages: vf.Messages, state: vf.State) -> vf.Messages:
    if check_game_over(state):
        final_message = [{"role": "user", "content": "Game over! Final score: " + str(state["score"])}]
        state["final_env_response"] = final_message
        return final_message
    # ... normal response logic
```
This bypasses the normal model response loop and immediately terminates the rollout, which is useful when the environment response itself signals completion (e.g. a game is won, an answer is submitted) or is required for reward computation (e.g. final feedback or tool results).

## Developing Environments

Environments are packaged as installable Python projects. We recommend developing environments in a workspace with `environments/` and `configs/` folders. The `vf-setup` command initializes this structure:

```bash
vf-setup
```

The `vf-init` command initializes a new environment project:

```bash
vf-init my-env
```

This creates the following structure:

```
environments/my_env/
├── my_env.py          # environment implementation
├── pyproject.toml     # package metadata and dependencies
└── README.md          # documentation template
```

The environment file must export a `load_environment()` function that returns a `vf.Environment`. Explicitly declare any arguments your environment accepts:

```python
import verifiers as vf

def load_environment(difficulty: str = "easy", num_examples: int = -1) -> vf.Environment:
    # build dataset, rubric, etc.
    return vf.SingleTurnEnv(dataset=dataset, rubric=rubric)
```

### pyproject.toml

The `pyproject.toml` defines package metadata, dependencies, and evaluation defaults:

```toml
[project]
name = "my-env"
description = "My custom environment"
tags = ["single-turn", "math", "train", "eval"]
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "verifiers>=0.1.8",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build]
include = ["my_env.py", "pyproject.toml"]

[tool.verifiers.eval]
num_examples = 20
rollouts_per_example = 5
```

Key `pyproject.toml` sections:

- **`[project]`** — Package name (used by `prime env install` and `prime eval run`), description, version, and dependencies. The `tags` field is optional metadata for categorizing environments.
- **`[build-system]`** — Hatchling is used as the build backend for the Environments Hub.
- **`[tool.hatch.build]`** — Lists files to include in the package. Always include `pyproject.toml` alongside your environment file to ensure that environment metadata is available when the environment is installed. Add any additional source files here.
- **`[tool.verifiers.eval]`** — Default parameters for `prime eval run` when flags aren't provided.

### Managing Dependencies

All packages your environment needs must be declared in the `dependencies` array. Always include `verifiers` with a minimum version. If your environment uses additional libraries, add them here—they will be installed automatically when the environment is installed:

```toml
dependencies = [
    "verifiers>=0.1.8",
    "chromadb",
    "nltk>=3.9.2",
]
```

### Required API Keys

Environments that require external API keys (e.g., for judge models or external services) should validate them early in `load_environment()` using `vf.ensure_keys()`:

```python
import verifiers as vf

def load_environment(api_key_var: str = "OPENAI_API_KEY") -> vf.Environment:
    vf.ensure_keys([api_key_var])
    # now safe to use os.environ[api_key_var]
    ...
```

This raises `MissingKeyError` with a clear message listing all missing keys and instructions for setting them:

- **Environments Hub CI**: Add secrets on the environment's Settings page
- **Hosted Training**: Set `env_file` in your config (e.g., `env_file = ["secrets.env"]`)
- **Local**: Export in your shell (e.g., `export OPENAI_API_KEY=...`)

Document required variables in your README under a "Required Environment Variables" section.

### Installation

Install a local environment with `prime env install`:

```bash
prime env install my-env                    # from ./environments/my_env
prime env install my-env -p /path/to/environments   # custom path
```

This runs `uv pip install -e` for local environments, making them importable by `prime eval run` and other integrations.

## Environment Groups

`EnvGroup` combines multiple environments into a single environment class, enabling multi-task evaluation and training across heterogeneous environments from a unified entrypoint. Each sub-environment maintains its own dataset, rubric, and rollout logic, while the group handles routing and metric aggregation:

```python
math_env = load_math_environment()
code_env = load_code_environment()
reasoning_env = load_reasoning_environment()

combined = vf.EnvGroup(
    envs=[math_env, code_env, reasoning_env],
    env_names=["math", "code", "reasoning"],
)
```

The group concatenates all sub-environment datasets, tagging each row with a `task` column that routes rollouts to the appropriate environment for generation and scoring. Metrics from all environments are tracked together. 

## Integrations and Experimental Environments

Beyond the core environment types, Verifiers includes integrations with several third-party environment libraries, as well as a few newer and more experimental environment classes (which are less stable and more subject to frequent changes).

Supported third-party environment integrations include:

- **`TextArenaEnv`** — wraps [TextArena](https://github.com/LeonGuertler/TextArena) text-based game environments
- **`ReasoningGymEnv`** — wraps [reasoning-gym](https://github.com/open-thought/reasoning-gym) procedural datasets
- **`BrowserEnv`** — unified browser automation via [Browserbase](https://browserbase.com) with DOM and CUA modes
- **`OpenEnvEnv`** — wraps OpenEnv gym and MCP contracts using Prime Sandboxes with prebuilt images referenced from `.build.json`

These require additional dependencies installed via extras (e.g., `uv add 'verifiers[ta]'` for TextArena, `uv add 'verifiers[browser]'` for BrowserEnv, `uv add 'verifiers[openenv]'` for OpenEnvEnv). For OpenEnv environments, build the bundled project image with `uv run vf-build <env-id>` before evaluation or training.

Newer and more experimental environment classes include:

- **`GymEnv`** — universal runner for Gym-compatible environments (OpenAI Gym / Gymnasium API)
- **`CliAgentEnv`** — runs custom agent code inside sandboxes, intercepting API requests. Accepts sandbox configuration parameters including `docker_image`, `cpu_cores`, `memory_gb`, `disk_size_gb`, `gpu_count`, `timeout_minutes`, `environment_vars`, and `labels` for sandbox categorization. Also accepts retry tuning (like `max_retries`) and connection pooling ( like `sandbox_client_max_workers`) parameters via `SandboxMixin`
- **`HarborEnv`** — loads Harbor-format agent benchmark tasks
- **`RLMEnv`** — implements Recursive Language Models for unbounded context processing. Execution supports both local and sandbox backends via `execution_backend` (`"local"` default, `"sandbox"` to run the REPL inside a Prime Sandbox). Context is still filesystem-based: a provided `context_dir` is copied into the working directory, or legacy JSON-serializable `context` data is written to `context.json`/`context.txt`. The RLM scaffolding prompt (filesystem availability note, REPL workflow, tool docs) is injected into the first user message wrapped in `<RLM_SCAFFOLDING>...</RLM_SCAFFOLDING>`, preserving any external system prompt; the model-visible prompt is stored in `state["prompt"]`, while the original input prompt is preserved in `state["raw_prompt"]`. The REPL language is configurable via `repl_language` (default: `bash`); use `repl_language="python"` to retain the Python REPL. Bash mode uses `call_bash_repl` and behaves like a terminal; Python mode uses `call_python_repl`. Sub-LLM and root-tool interception for sandboxes is routed through a Prime Tunnel unless `interception_url` is provided. Tooling can be split via `tools` (shared), `root_tools` (REPL-only), and `sub_tools` (sub-LLM tools). Fixed root tools like `llm_batch` are always present and cannot be overridden. Tool ordering is fixed tools → shared tools → role-specific tools, with per-list deduplication by name. Root tools are callable only inside the REPL; sub-LLM tools use standard tool-calling. When using the sandbox backend, the sandbox and worker are started eagerly during `setup_state`, and package installs are skipped when the package is already importable in the image. Environments can pre-set `state["rlm_fs_root_remote"]` (and optionally `state["rlm_control_dir_remote"]`) before calling `super().setup_state` to point the worker at an existing filesystem path in the sandbox. For further customization, override `get_sandbox_request`, `on_sandbox_ready`, or `customize_worker_script` on `RLMEnv`.
