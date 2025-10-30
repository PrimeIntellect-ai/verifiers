# prml

Pattern Recognition and Machine Learning (PRML) environment with SymPy symbolic mathematics tool support for solving exercises from Christopher Bishop's PRML textbook.

This environment features a high-quality dataset covering 340 out of 420 exercises (81%) from the textbook, with 2500+ equations cleaned and scraped.

### Overview
- **Environment ID**: `prml`
- **Short description**: Mathematical problem-solving environment for PRML exercises with SymPy tool calling for symbolic computation, simplification, and verification
- **Tags**: mathematics, machine-learning, prml, tools, eval

### Datasets
- **Primary dataset(s)**: `Vivek/prml-exercises` - High-quality curated dataset from Christopher Bishop's PRML textbook
- **Coverage**: 340 exercises out of 420 total questions from the textbook (~81% coverage)
- **Quality enhancements**: 
  - All 2500+ equations cleaned, scraped, and embedded directly into questions and answers
  - Equation numbers replaced with actual LaTeX expressions for complete context
  - Solutions gathered from various sources including github and official answers. 
- **Source links**: [HuggingFace Dataset](https://huggingface.co/datasets/Vivek/prml-exercises)

### Dataset Quality

This is one of the most comprehensive and high-quality PRML exercise datasets available:

- **Comprehensive Coverage**: 340/420 exercises from Bishop's PRML textbook (81% coverage)
- **Self-Contained**: All 2500+ equations from the textbook were scraped, cleaned, and embedded directly into questions and answers
- **No External References**: Equation numbers (e.g., "Equation 3.14") replaced with actual LaTeX expressions
- **Complete Context**: Each exercise contains all necessary mathematical definitions and equations inline
- **Verified Solutions**: Curated from the official solution manual.
- **Production Ready**: Suitable for training and evaluating LLMs on complex mathematical reasoning

### Task
- **Type**: Multi-turn tool use (ToolEnv)
- **Parser**: ThinkParser (extracts content from `<think>...</think>` and `<answer>...</answer>` tags)
- **Rubric overview**: 
  - **Correctness** (weight 2.0): Binary score for arriving at correct final result
  - **Stepwise Validity** (weight 2.0): Logical soundness of derivation steps (0.0-1.0)
  - **Readability** (weight 1.0): Clarity and structure of explanation (0.0-1.0)
  - **Completeness** (weight 1.0): Coverage of essential elements (0.0-1.0)
  - **Similarity** (weight 3.0): Semantic similarity to ground truth using embeddings
  - All rewards are scaled by difficulty multiplier (easy: 0.7x, medium: 1.0x, hard: 1.5x)

### Quickstart


Run an evaluation with default settings:
```bash
uv run vf-eval prml -m gpt-4.1-mini
```

Run with custom settings and save results:
```bash
uv run vf-eval prml -m gpt-4.1-mini -n 10 -r 1 -s
```

Configure sampling parameters:
```bash
uv run vf-eval prml \
  -m gpt-4.1-mini \
  -n 20 -r 3 \
  -t 1024 -T 0.7 \
  -s -v
```

Notes:
- Use `-s` / `--save-results` to save evaluation outputs to disk
- Use `-v` / `--verbose` for detailed logging
- Results are saved to `outputs/evals/prml--<model>/<uuid>/`

### Environment Arguments

Pass environment-specific configuration using `-a` / `--env-args` as JSON:

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `dataset_name` | str | `"Vivek/prml-exercises"` | HuggingFace dataset to load |
| `judge_model` | str | `"gpt-4.1-mini"` | Model used for grading responses |
| `judge_base_url` | str | `"https://api.openai.com/v1"` | API base URL for judge |
| `judge_api_key_var` | str | `"OPENAI_API_KEY"` | Environment variable for judge API key |
| `embed_model` | str | `"text-embedding-3-small"` | Model for similarity embeddings |
| `embed_base_url` | str | `"https://api.openai.com/v1"` | API base URL for embeddings |
| `embed_api_key_var` | str | `"OPENAI_API_KEY"` | Environment variable for embeddings API key |
| `use_similarity_reward` | bool | `true` | Enable semantic similarity reward |
| `use_tools` | bool | `true` | Enable SymPy tool calling (uses ToolEnv if true) |

Example with custom arguments:
```bash
uv run vf-eval prml -m gpt-4.1-mini \
  -a '{"judge_model": "gpt-4o", "use_similarity_reward": false}'
```

### Tools Available

The environment provides SymPy-based mathematical tools:

1. **`sympy_simplify_expression(expr_latex: str)`**: Simplifies mathematical expressions using SymPy
   - Input: LaTeX expression (without `$$` delimiters)
   - Output: Simplified LaTeX expression
   - Example: `x^2 + 2*x + 1` → `(x + 1)^2`

### Metrics

| Metric | Range | Meaning |
| ------ | ----- | ------- |
| `reward` | 0.0 - ~11.5 | Weighted sum of all criteria (scaled by difficulty) |
| `grade_reward_correctness` | 0.0 - 1.5 | Binary correctness × difficulty multiplier |
| `grade_reward_clarity_structure` | 0.0 - 1.5 | Stepwise validity score × difficulty multiplier |
| `grade_reward_readability` | 0.0 - 1.5 | Readability score × difficulty multiplier |
| `grade_reward_completeness` | 0.0 - 1.5 | Completeness score × difficulty multiplier |
| `similarity_reward_wrapper` | 0.0 - 4.5 | Embedding similarity × difficulty multiplier |

### Example Output

Models are expected to format responses as:

```
<think>
Step-by-step mathematical derivation goes here...
$$E = mc^2$$
Using the property of...
</think>

<answer>
The final result is: $$E = mc^2$$
</answer>
```

### Requirements

- OpenAI API key in environment variable `OPENAI_API_KEY`
- Python 3.11+
- Dependencies: verifiers, datasets, openai, numpy, sympy

