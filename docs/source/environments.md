# Environments

This guide covers how to create, develop, and use environments in Verifiers.

## Creating a New Environment

The recommended approach is to create an environment *module*, i.e. a self-contained package that can be installed and reused.

### Initialize from Template

```bash
vf-init my-math-env
```

This creates:
```
environments/my_math_env/
├── my_math_env.py      # Main implementation
├── pyproject.toml      # Dependencies and metadata
└── README.md           # Documentation
```

### Basic Environment Structure

Every environment module must export a `load_environment` function:

```python
# my_math_env.py
import verifiers as vf

def load_environment(**kwargs):
    """Load and configure the environment."""
    # 1. Load dataset
    dataset = vf.load_example_dataset("gsm8k", split="train")
    
    # 2. Configure parser
    parser = vf.ThinkParser()
    
    # 3. Define reward functions -- can automatically reference:
    # - parser, prompt, completion, answer, state , task, info 
    def correct_answer(parser, completion, answer):
        response = parser.parse_answer(completion) or ''
        return 1.0 if response.strip() == answer.strip() else 0.0
    
    # 4. Create rubric
    rubric = vf.Rubric(
        funcs=[correct_answer, parser.get_format_reward_func()],
        weights=[1.0, 0.2]
    )
    
    # 5. Return configured environment
    return vf.SingleTurnEnv(
        dataset=dataset,
        system_prompt="Think step-by-step, then give your answer.",
        parser=parser,
        rubric=rubric,
        **kwargs  # Pass through additional arguments
    )
```

### Adding Dependencies

Specify environment-specific dependencies in `pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "my_math_env"
description = "Single-turn math environment"
tags = ["math", "verifiable-reward"]
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "verifiers",
    "sympy",  # For symbolic math
]
```

## Development Workflow

### 1. Install Your Environment

During development, install your environment locally:

```bash
vf-install my-math-env # wraps 'uv pip install -e ...'
```

This installs the module and its dependencies in your Python environment.

### 2. Test Your Environment

Use the CLI to quickly test:

```bash
vf-eval my-math-env -m gpt-4.1-mini -n 5 # runs a small batch of rollouts; use -h to see options
```

Or test programmatically:

```python
import verifiers as vf
from openai import OpenAI

# Load your environment
env = vf.load_environment("my-math-env")

# Test with a model
client = OpenAI()
results = env.evaluate_sync(
    client=client,
    model="gpt-4.1-mini",
    num_examples=5,
    rollouts_per_example=2,
    max_concurrent=32,
    save_every=10,
)
print(results)
```

Prefer `AsyncOpenAI` + `await env.evaluate(...)` for fully async workflows; the sync helper is ideal when integrating into existing blocking scripts. Intermediate saving via `save_every` requires the default interleaved scoring pipeline.

### 3. Iterate on Design

Common iterations:
- Adjust system prompts for better performance
- Refine parser logic for edge cases
- Add new reward functions to the rubric
- Configure dataset filtering or sampling

## Using Prime CLI and the Environments Hub

Prime Intellect provides a hosted [Environments Hub](https://docs.primeintellect.ai/tutorials-environments/environments) for discovering, installing, and sharing verifiers packages. The [`prime` CLI](https://github.com/PrimeIntellect-ai/prime-cli) wraps the same templates and packaging helpers exposed here, so you can publish environments once and re-use them from local machines, CI pipelines, or Prime pods.

### Install and authenticate

```bash
uv tool install prime
prime login  # stores an API token used for hub operations
```

If you collaborate through a team account, run `prime config use <team>` (or set `--team` flags in later commands) so pushes end up in the shared namespace.

### Bootstrap templates from the CLI

`prime env init <name>` uses the same generator as `vf-init`, but it pre-populates the directory inside `./environments/` and prints the follow-up commands for publishing. Use this when you're starting a project that you plan to ship through the Hub:

```bash
prime env init vf-math-demo
cd environments/vf_math_demo
# edit vf_math_demo.py, pyproject.toml, README.md, etc.
```

For existing environments you created earlier with `vf-init`, no migration is required—`prime env push` operates on any directory that contains a valid `pyproject.toml` and `load_environment` implementation.

### Publish versions to the Hub

Once your package is ready, build and upload it with:

```bash
prime env push --visibility PUBLIC  # or PRIVATE for internal distributions
```

The command builds a wheel (using `uv build` when available), computes a deterministic content hash, and uploads the artifact. Add `--auto-bump` to increment the patch version before publishing, or pass `--team <slug>` to publish under a team namespace. Successful pushes print a dashboard link plus a one-line install command.

You can manage published artifacts directly from the CLI:

- `prime env version list owner/name` shows version history and hashes.
- `prime env version delete owner/name <content_hash>` removes a specific build.
- `prime env delete owner/name` deletes the environment entirely.

### Discover, install, and inspect environments

The CLI also helps consumers find and install verifiers:

```bash
prime env list --owner my-team           # browse available environments
prime env info my-team/vf-math-demo      # show install commands and metadata
prime env install my-team/vf-math-demo   # install latest release with uv
prime env install owner/env@0.1.2 --with pip  # pin version & use pip instead
prime env pull owner/env@latest --target ./tmp-env  # download source tarball
```

`prime env install` prefers installing from the Hub's simple index (so upgrades work with `uv add`/`pip install` too), and falls back to direct wheel URLs for older releases. After installation the package becomes available to `verifiers.load_environment` just like any other module:

```python
from verifiers import load_environment

env = load_environment("vf-math-demo")
```

When you run workloads on Prime pods provisioned via `prime pods create`, include these install commands in your startup scripts so the same environment definitions are available remotely.

## Working with Rubrics

Rubrics are central to defining what makes a good response in your environment. Here's how to use them effectively:

### Basic Reward Functions

A reward function takes the full context and returns a score (typically 0.0 to 1.0):

```python
def exact_match(prompt, completion, answer, state):
    """Reward exact matches."""
    response = completion[-1]['content']
    return 1.0 if response.strip() == answer.strip() else 0.0

def partial_credit(prompt, completion, answer, state):
    """Give partial credit for containing key terms."""
    key_terms = answer.lower().split()
    response = completion[-1]['content']
    found = sum(1 for term in key_terms if term in response.lower())
    return found / len(key_terms) if key_terms else 0.0
```

### Creating Rubrics

Combine multiple reward functions with weights:

```python
# Single criterion
rubric = vf.Rubric(funcs=[exact_match])

# Multi-criteria with weights
rubric = vf.Rubric(
    funcs=[exact_match, partial_credit, length_penalty],
    weights=[1.0, 0.5, 0.1]  # Relative importance
)
```

### Using Parser Format Rewards

Parsers often provide format reward functions:

```python
parser = vf.ThinkParser(extract_fn=extract_boxed_answer)

def correct_answer(parser, completion, answer):
    parsed = parser.parse_answer(completion) # applies extract_fn to final message
    return 1.0 if parsed == answer else 0.0

rubric = vf.Rubric(
    funcs=[
        correct_answer,
        parser.get_format_reward_func()  # Rewards proper <think> format
    ],
    weights=[1.0, 0.2]
)
```

### Stateful Reward Functions

Access environment state for complex evaluation:

```python
def efficiency_reward(prompt, response, answer, state):
    """Reward based on number of steps taken."""
    max_steps = 10
    steps_taken = state.get("turn", 0)
    return max(0, (max_steps - steps_taken) / max_steps)
```

## Environment Types

Choose the appropriate base class for your task:

### SingleTurnEnv

For one-shot tasks with clear input/output:

```python
def load_environment(**kwargs):
    return vf.SingleTurnEnv(
        dataset=dataset,
        system_prompt="Answer the question.", # only used if dataset has 'question' (str) and not 'prompt'
        parser=parser,
        rubric=rubric,
        **kwargs
    )
```

### MultiTurnEnv

For interactive tasks requiring multiple steps:

```python
from verifiers.types import Messages, State
from typing import Tuple

class MyGameEnv(vf.MultiTurnEnv):

    async def env_response(self, messages: Messages, state: State) -> Tuple[Messages, State]:
        """Define how the environment responds."""
        last_msg = messages[-1]
        if last_msg["role"] != "assistant":
            return [], state

        player_action = last_msg["content"]
        if self.is_game_over(state):
            state["done"] = True
            return [{"role": "user", "content": "Game over!"}], state

        state = self.update_state(state, player_action)
        feedback = self.get_game_feedback(state)
        return [{"role": "user", "content": feedback}], state

    async def is_completed(self, messages: Messages, state: State) -> bool:
        if await super().is_completed(messages, state):
            return True
        return state.get("solved", False) or state.get("failed", False)
```

### ToolEnv

For tasks requiring external tools:

```python
def calculate(expression: str) -> float:
    """Calculate a mathematical expression."""
    return eval(expression)  # Simplified example

def load_environment(**kwargs):
    return vf.ToolEnv(
        dataset=dataset,
        tools=[calculate],  # Automatically converted to tool schemas
        parser=parser,
        rubric=rubric,
        **kwargs
    )
```

Tool functions should be deterministic and free of hidden side effects. A rollout ends when the model produces an assistant turn with no tool calls, so store per-session context in `state` rather than globals. Use `StatefulToolEnv` if you need to inject extra arguments or secrets into each tool invocation.

#### StatefulToolEnv

For stateful workflows, extend `vf.StatefulToolEnv`:

```python
class SandboxAwareEnv(vf.StatefulToolEnv):
    async def setup_state(self, state: State, **kwargs) -> State:
        state = await super().setup_state(state, **kwargs)
        state["sandbox_id"] = await provision_sandbox_async()
        return state

    def update_tool_args(self, tool_name: str, tool_args: dict, messages: Messages, state: State, **kwargs) -> dict:
        return {**tool_args, "sandbox_id": state["sandbox_id"]}
```

Keep heavy provisioning asynchronous and defer expensive await calls until a tool actually needs the resource. Remove sensitive args from the tool schema via `add_tool(..., args_to_skip=[...])` so the model never sees them.

#### SandboxEnv & PythonEnv

`vf.SandboxEnv` packages the pattern above for Prime sandboxes: it kicks off container startup in `setup_state` and injects handles into tool calls once ready. `vf.PythonEnv` is the canonical example—review it when building similar sandboxes or remote runtimes.

## Advanced Patterns

### Configurable Environments

Accept parameters to customize behavior:

```python
def load_environment(
    dataset_name="gsm8k",
    num_examples=None,
    difficulty="all",
    use_calculator=False,
    **kwargs
):
    # Load dataset with filtering
    dataset = vf.load_example_dataset(dataset_name)
    if difficulty != "all":
        dataset = dataset.filter(lambda x: x["difficulty"] == difficulty)
    if num_examples:
        dataset = dataset.select(range(num_examples))
    
    # Conditionally add tools
    tools = [calculate] if use_calculator else []
    
    # Return appropriate environment type
    if tools:
        return vf.ToolEnv(dataset=dataset, tools=tools, **kwargs)
    else:
        return vf.SingleTurnEnv(dataset=dataset, **kwargs)
```

### Custom Datasets

Load datasets from various sources:

```python
def load_environment(dataset_path=None, **kwargs):
    if dataset_path:
        # Load from file
        dataset = Dataset.from_json(dataset_path)
    else:
        # Load from Hugging Face
        dataset = load_dataset("owner/dataset-name", split="train")
    
    # Ensure required columns
    assert "prompt" in dataset.column_names
    assert "answer" in dataset.column_names or "info" in dataset.column_names
    
    return vf.SingleTurnEnv(dataset=dataset, **kwargs)
```

### Composition with EnvGroup

Combine multiple environments for training on diverse tasks:

```python
def load_environment(**kwargs):
    # Environment 1: GSM8K
    gsm8k_dataset = vf.load_example_dataset("gsm8k")
    gsm8k_env = vf.SingleTurnEnv(
        dataset=gsm8k_dataset,
        parser=parser,
        rubric=gsm8k_rubric
    )
    
    # Environment 2: MATH
    math_dataset = vf.load_example_dataset("math")
    math_env = vf.SingleTurnEnv(
        dataset=math_dataset,
        parser=parser,
        rubric=math_rubric
    )
    
    # Create grouped environment
    return vf.EnvGroup(
        envs=[gsm8k_env, math_env],
        env_names=["gsm8k", "math"] # stored as "task" column
    )
```

**How EnvGroup Works:**
- **Dataset Concatenation**: Combines datasets from all environments with task labels
- **Automatic Routing**: Routes rollouts to the correct environment based on the `task` column
- **Unified Scoring**: Aggregates scores across all environments

This is particularly useful for:
- Training on multiple task types simultaneously
- Evaluating general capabilities across domains
- Creating curriculum learning setups

## Installing from Repository

Install environments from the verifiers repository:

```bash
# Install specific environment
vf-install math-python --from-repo

# Install from branch
vf-install wordle --from-repo -b dev

# List available environments
vf-install --list
```

## Best Practices

1. **Start Simple**: Begin with SingleTurnEnv and basic reward functions
2. **Test Early**: Use `vf-eval` to test your environment during development
3. **Document Well**: Include clear README with examples and expected behavior
4. **Handle Errors**: Ensure parsers and reward functions handle edge cases
5. **Version Dependencies**: Pin specific versions in pyproject.toml


## Next Steps

- See [Components](components.md) for advanced rubrics, tools, parsers, and practical examples
- Explore [Training](training.md) to use your environment for model improvement