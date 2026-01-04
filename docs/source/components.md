# Components

This guide covers the advanced components available in Verifiers: Rubrics, Tools, and Parsers. Each section includes practical examples of how to use these components in real-world scenarios.

## Advanced Rubrics

Beyond basic reward functions, Verifiers provides specialized rubric types for complex evaluation scenarios.

### JudgeRubric: LLM-Based Evaluation

Use language models to evaluate responses when rule-based scoring is insufficient:

```python
# Basic usage with default prompt
judge_rubric = vf.JudgeRubric()

# Custom evaluation criteria
judge_rubric = vf.JudgeRubric(
    judge_prompt="""Evaluate the response based on:
    1. Accuracy of the solution
    2. Clarity of explanation
    3. Appropriate use of mathematical notation
    
    Rate from 0.0 to 1.0."""
)
```

**Note on Concurrency and Caching**

When multiple reward functions rely on the same judge call, `JudgeRubric` avoids making redundant API requests by caching the judge's response within the `state` dictionary for each rollout.

To make this caching effective, `JudgeRubric` defaults to `parallelize_scoring=False`. This forces its reward functions to run sequentially, ensuring that the first function makes the API call and populates the cache, while subsequent functions get an instant cache hit.

**Example: Multi-Step Math with Judge Evaluation**

```python
def load_environment(**kwargs):
    # Base rubric for correctness
    def check_answer(prompt, response, answer, state):
        final_answer = extract_number(response)
        return 1.0 if abs(final_answer - float(answer)) < 0.01 else 0.0
    
    base_rubric = vf.Rubric(funcs=[check_answer])
    
    # Add judge for solution quality
    judge = vf.JudgeRubric(
        judge_prompt="Evaluate the mathematical reasoning: Is each step justified? Are there logical errors?"
    )
    
    # Combine with RubricGroup
    return vf.SingleTurnEnv(
        dataset=dataset,
        rubric=vf.RubricGroup([base_rubric, judge]),
        **kwargs
    )
```

### RubricGroup: Combining Multiple Rubrics

Aggregate scores from different rubrics:

```python
# Combine different evaluation approaches
group = vf.RubricGroup([
    correctness_rubric,  # Weight: 1.0 (default)
    style_rubric,        # Weight: 1.0
    efficiency_rubric    # Weight: 1.0
])

# With custom weights: set weights inside each Rubric, then group them
correctness = vf.Rubric(funcs=[check_answer], weights=[2.0])
style = vf.Rubric(funcs=[style_score], weights=[1.0])

group = vf.RubricGroup([correctness, style])
```

**Example: Multi-Criteria Code Evaluation**

```python
class CodeEvalEnv(vf.MultiTurnEnv):
    def __init__(self, **kwargs):
        # Rubric 1: Correctness
        correctness = vf.Rubric(funcs=[self.test_correctness])
        
        # Rubric 2: Performance
        performance = vf.Rubric(funcs=[self.measure_performance])
        
        # Rubric 3: Style (via judge)
        style_judge = vf.JudgeRubric(
            judge_prompt="Rate code style: readability, naming, structure (0-1)"
        )
        
        # Combine all rubrics
        super().__init__(
            rubric=vf.RubricGroup([correctness, performance, style_judge]),
            **kwargs
        )
```

### ToolRubric: Tracking Tool Usage

Count total and per-tool calls during a rollout. Pass your tool functions to enable per-tool counters. By default, counts are added as metrics with zero weight; adjust `reward_weights` if you want the counts to affect reward.

```python
# Define tools (type hints + docstrings omitted for brevity)
def calculate(expr: str) -> float: ...
def search_web(query: str, max_results: int = 5) -> list[dict]: ...

# Initialize with tools to track
tool_rubric = vf.ToolRubric(tools=[calculate, search_web])

# Metrics exposed (names):
# - total_tool_calls
# - calculate_calls
# - search_web_calls

# Optional: turn counts into rewards by setting weights
# Index 0 corresponds to total_tool_calls; subsequent indices follow the tools order
tool_rubric.reward_weights[0] = -0.1   # penalize excessive tool calls
tool_rubric.reward_weights[2] = 0.2    # reward using search_web specifically
```

### Rubric.class_objects: Passing Objects to Reward Functions

The `class_objects` pattern allows you to pass class instances or other objects directly to your reward functions. This is especially useful when your reward functions need access to parsers, clients, or other stateful objects.

**How it works:**

When a rubric calls a reward function, it automatically merges `self.class_objects` with the standard arguments (`prompt`, `completion`, `answer`, `state`, etc.). Your reward functions can then accept these objects as parameters.

**Basic Example:**

```python
class CustomRubric(vf.Rubric):
    def __init__(self, api_client, custom_parser, **kwargs):
        super().__init__(**kwargs)
        self.api_client = api_client
        self.custom_parser = custom_parser
        
        # Make objects available to reward functions
        self.class_objects = {
            "parser": self.parser,           # Standard parser (automatically added)
            "api_client": self.api_client,   # Custom API client
            "custom_parser": self.custom_parser,  # Additional parser
        }
        
        # Add reward functions that use these objects
        self.reward_funcs = [self.api_based_reward, self.parsing_reward]

    def api_based_reward(self, completion, answer, api_client, **kwargs):
        """Reward function that uses the API client."""
        # Extract answer from completion
        extracted = self.parser.parse_answer(completion)
        
        # Use API client to validate answer
        validation_result = api_client.validate(extracted, answer)
        return 1.0 if validation_result.is_correct else 0.0
    
    def parsing_reward(self, completion, custom_parser, **kwargs):
        """Reward function that uses a custom parser."""
        # Use the custom parser passed via class_objects
        parsed_data = custom_parser.parse(completion)
        return 1.0 if parsed_data.is_well_formed else 0.0
```

**Real-world Example: JudgeRubric**

The built-in `JudgeRubric` demonstrates this pattern:

```python
class JudgeRubric(vf.Rubric):
    def __init__(self, judge_client=None, judge_model="gpt-4o-mini", **kwargs):
        super().__init__(**kwargs)
        self.judge_client = judge_client or AsyncOpenAI()
        self.judge_model = judge_model
        
        # Expose objects to reward functions
        self.class_objects = {
            "parser": self.parser,
            "judge": self.judge,  # The judge method itself
            "judge_client": self.judge_client,
            "judge_model": self.judge_model,
        }

    async def judge(self, prompt, completion, answer, state, **kwargs):
        """Judge method that can be called by reward functions."""
        # Implementation uses self.judge_client to evaluate responses
        ...
```

**Usage in Reward Functions:**

```python
# Your reward function can access any object from class_objects
def quality_reward(completion, answer, judge, **kwargs):
    """Use the judge method to evaluate response quality."""
    judgment = await judge(prompt, completion, answer, state)
    return 1.0 if "excellent" in judgment.lower() else 0.5

def format_reward(completion, parser, **kwargs):
    """Use the parser to check formatting."""
    parsed = parser.parse_answer(completion)
    return 1.0 if parsed else 0.0
```

**Best Practices:**

1. **Include all necessary objects**: Add any parsers, clients, or helpers your reward functions need
2. **Use descriptive names**: Choose clear names for your class_objects keys
3. **Document dependencies**: Make it clear which reward functions use which objects
4. **Handle missing objects gracefully**: Use `**kwargs` and check for object availability

**Advanced Pattern: Shared State**

```python
class StatefulRubric(vf.Rubric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.shared_state = {"call_count": 0}
        
        self.class_objects = {
            "parser": self.parser,
            "shared_state": self.shared_state,
        }
        
        self.reward_funcs = [self.counting_reward]
    
    def counting_reward(self, completion, shared_state, **kwargs):
        """Track calls across reward function invocations."""
        shared_state["call_count"] += 1
        return 1.0 if shared_state["call_count"] <= 5 else 0.5
```

## Tools

Verifiers provides native support for tool calling, leveraging models' built-in function calling capabilities.

### Defining Tools

Tools are simple Python functions with type hints, and can be either sync or async:

```python
def calculate(expression: str) -> float:
    """Evaluate a mathematical expression safely."""
    # Use a safe math parser in production
    import ast
    return eval(expression, {"__builtins__": {}}, {})

async def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Search the web for information.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        List of search results with title, snippet, and url
    """
    # Implementation here
    return results
```

### Using ToolEnv

ToolEnv automatically converts Python functions to tool schemas and handles tool calling:

```python
def load_environment(**kwargs):
    return vf.ToolEnv(
        dataset=dataset,
        tools=[calculate, search_web],  # Just pass the functions
        max_turns=10,
        rubric=rubric,
        **kwargs
    )
```

**Note**: ToolEnv uses the model's native tool calling format via the tokenizer's chat template. It automatically injects tool schemas into request payloads and treats `role: tool` messages as tool outputs. It does NOT impose any XML structure or require hardcoded patterns.

### Tool Design Best Practices

1. **Clear Signatures**: Use descriptive names and type hints
2. **Comprehensive Docstrings**: Models use these to understand tool purpose
3. **Error Handling**: Return helpful error messages, don't raise exceptions
4. **Timeouts**: Add timeouts for long-running operations
5. **Input Validation**: Validate and sanitize inputs

**Example: Wiki Search Environment**

```python
def wiki_search(query: str) -> str:
    """Search Wikipedia for information."""
    try:
        # Add timeout
        with timeout(5.0):
            results = wikipedia.search(query, results=3)
            if results:
                page = wikipedia.page(results[0])
                return f"Title: {page.title}\n\n{page.summary[:500]}..."
            return "No results found."
    except Exception as e:
        return f"Search error: {str(e)}"

def wiki_get_page(title: str) -> str:
    """Get full Wikipedia page content."""
    try:
        with timeout(5.0):
            page = wikipedia.page(title)
            return page.content[:2000]  # Limit length
    except Exception as e:
        return f"Page error: {str(e)}"

def load_environment(**kwargs):
    dataset = load_qa_dataset()  # Questions requiring research
    
    # Rubric rewards correct answers and efficient tool use
    rubric = vf.Rubric(
        funcs=[check_answer, efficiency_bonus],
        weights=[1.0, 0.2]
    )
    
    return vf.ToolEnv(
        dataset=dataset,
        tools=[wiki_search, wiki_get_page],
        max_turns=8,
        rubric=rubric,
        **kwargs
    )
```

### Complex Tool Examples

For more sophisticated tool setups, see the `wiki_search` environment in the repository, which demonstrates:
- Multiple interdependent tools
- State management across tool calls
- Sophisticated error handling
- Tool usage optimization

## Parsers

Parsers extract structured information from model outputs. While many tasks work with raw text, parsers help when you need specific formats.

### Built-in Parsers

#### XMLParser

Extract XML-tagged content:

```python
# Define which tags are expected in the output
# Strings define fixed tags; tuples define canonical name + allowed aliases
parser = vf.XMLParser(
    fields=["think", ("answer", "code")],
    answer_field="answer",
)

# In practice
response = "<think>Let me calculate...</think>\n<answer>42</answer>"
answer = parser.parse_answer(response)  # => "42"
```

#### ThinkParser

Separate reasoning from final answers:

```python
# Strip any content before </think>, then apply extract_fn

def extract_number(text: str) -> str:
    import re
    m = re.search(r"[-+]?\d*\.?\d+", text)
    return m.group() if m else ""

parser = vf.ThinkParser(extract_fn=extract_number)
```

### Custom Parser Patterns

Create domain-specific parsers by extending the base class:

**Example: Code Block Parser**

```python
class CodeParser(vf.Parser):
    """Extract and validate code blocks from responses."""
    
    def parse_answer(self, response: str) -> str:
        # Extract code between triple backticks
        import re
        code_blocks = re.findall(r'```(?:python)?\n(.*?)```', response, re.DOTALL)
        
        if not code_blocks:
            return ""
        
        # Return the last code block (usually the final solution)
        code = code_blocks[-1].strip()
        
        # Basic validation
        try:
            compile(code, '<string>', 'exec')
            return code
        except SyntaxError:
            return ""  # Invalid Python code
```

**Example: Math Step Parser**

```python
class MathStepParser(vf.Parser):
    """Parse step-by-step math solutions."""
    
    def parse_answer(self, response: str) -> str:
        lines = response.strip().split('\n')
        
        # Look for final answer patterns
        for line in reversed(lines):
            if any(marker in line.lower() for marker in ['therefore', 'answer:', '=']):
                # Extract number from this line
                import re
                match = re.search(r'[-+]?\d*\.?\d+', line)
                if match:
                    return match.group()
        
        return ""
    
    def get_format_reward_func(self):
        def reward_steps(prompt, response, answer, state):
            # Reward showing work
            steps = response.count('\n')
            return min(1.0, steps / 5)  # Expect ~5 steps
        return reward_steps
```

### Parser Integration

Parsers integrate seamlessly with environments and rubrics:

```python
def load_environment(**kwargs):
    parser = CodeParser()
    
    def code_runs(prompt, response, answer, state):
        code = parser.parse_answer(response)
        if not code:
            return 0.0
        try:
            exec(code)
            return 1.0
        except:
            return 0.0
    
    rubric = vf.Rubric(
        funcs=[code_runs, parser.get_format_reward_func()],
        weights=[1.0, 0.1]
    )
    
    return vf.SingleTurnEnv(
        dataset=dataset,
        parser=parser,
        rubric=rubric,
        **kwargs
    )
```

## Practical Examples

### Interactive Game Environment

Build a Wordle-like game with multi-turn interaction:

```python
from verifiers.types import Messages, State

class WordleEnv(vf.MultiTurnEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_guesses = 6
    
    async def env_response(self, messages: Messages, state: State) -> Messages:
        if len(state["trajectory"]) == 0:
            # First turn: initialize
            state["target"] = state["answer"]
            state["guesses"] = []
            return [{"role": "user", "content": "Guess a 5-letter word. You have 6 attempts."}]
        
        # Get the last assistant message
        last_msg = messages[-1]
        if last_msg["role"] != "assistant":
            return []  # No response if not assistant message
            
        guess = last_msg["content"].strip().upper()
        target = state["target"]
        
        # Validate guess
        if len(guess) != 5 or not guess.isalpha():
            return [{"role": "user", "content": "Please guess a 5-letter word."}]
        
        # Generate feedback
        feedback = self.get_feedback(guess, target)
        state["guesses"].append(guess)
        
        if guess == target:
            state["solved"] = True
            return [{"role": "user", "content": f"Correct! The word was {target}."}]
        elif len(state["guesses"]) >= self.max_guesses:
            state["failed"] = True
            return [{"role": "user", "content": f"Out of guesses. The word was {target}."}]
        else:
            remaining = self.max_guesses - len(state["guesses"])
            return [{"role": "user", "content": f"{feedback}\n{remaining} guesses remaining."}]
    
    @vf.stop
    async def game_over(self, state: State) -> bool:
        return state.get("solved", False) or state.get("failed", False)
```

### Training Data Generation

Generate training data using environment rollouts:

```python
import asyncio

async def generate_training_data(env, client, model, num_samples=1000):
    """Generate diverse solutions for training."""
    results = []
    score_sem = asyncio.Semaphore(1)  # Semaphore for scoring
    
    for i in range(num_samples):
        # Get a random prompt
        prompt = env.dataset[i]["prompt"]
        answer = env.dataset[i]["answer"]
        
        # Generate multiple solutions
        for temp in [0.3, 0.7, 1.0]:
            input = {"prompt": prompt, "answer": answer,
                    "task": "default", "example_id": i}
            state = await env.rollout(input=input, client=client, model=model, sampling_args={"temperature": temp, "max_tokens": 1000})
            
            # Score the solution
            await env.rubric.score_rollout(state, score_sem)
            
            # Save high-quality solutions
            if state.get("reward") and state["reward"] > 0.8:
                results.append({
                    "prompt": prompt,
                    "completion": state["completion"],
                    "score": state["reward"]
                })
    
    return Dataset.from_list(results)
```

### Environment Composition

Build complex environments from simpler ones:

```python
def load_math_suite(**kwargs):
    """Comprehensive math environment covering multiple domains."""
    
    # Shared components
    parser = vf.ThinkParser(extract_fn=extract_boxed_answer)
    
    # Basic arithmetic
    arithmetic_env = vf.SingleTurnEnv(
        dataset=load_arithmetic_dataset(),
        parser=parser,
        rubric=vf.Rubric(funcs=[exact_match]),
        system_prompt="Solve the arithmetic problem."
    )
    
    # Algebra with tools
    algebra_env = vf.ToolEnv(
        dataset=load_algebra_dataset(),
        tools=[solve_equation, factor_polynomial],
        parser=parser,
        rubric=vf.Rubric(funcs=[check_algebra, tool_efficiency])
    )
    
    # Geometry with judge
    geometry_env = vf.SingleTurnEnv(
        dataset=load_geometry_dataset(),
        parser=parser,
        rubric=vf.RubricGroup([
            vf.Rubric(funcs=[check_geometry]),
            vf.JudgeRubric(judge_prompt="Rate the geometric reasoning and diagram interpretation.")
        ])
    )
    
    # Combine all
    return vf.EnvGroup(
        envs=[arithmetic_env, algebra_env, geometry_env],
        env_names=["arithmetic", "algebra", "geometry"],
        **kwargs
    )
```

## Best Practices

### For Rubrics
- Start simple with basic reward functions
- Use JudgeRubric when rule-based evaluation is insufficient
- Combine rubrics with RubricGroup for multi-faceted evaluation
- Test reward functions thoroughly with edge cases

### For Tools
- Keep tool functions simple and focused
- Use clear names and comprehensive docstrings
- Handle errors gracefully - return messages, don't raise
- Add timeouts for external operations
- Let the model's chat template handle tool calling format

### For Parsers
- Use built-in parsers when they fit your needs
- Create custom parsers for domain-specific formats
- Always handle parsing failures gracefully
- Consider providing format rewards to guide model output

## Next Steps

- Build your own environments using these components in [Environments](environments.md)
- Train models with your environments in [Training](training.md)
- Understand the type system in [Type Reference](api_reference.md)