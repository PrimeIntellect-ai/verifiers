# Environments

This folder contains installable example environments that showcase common usage patterns in Verifiers. Each module exposes a `load_environment(...)` function that returns a ready-to-use `vf.Environment` object.

## Quick start

- **Install an environment from this GitHub repo**: `vf-install math-python --from-repo`
- **Evaluate**: `vf-eval math-python` (defaults to gpt-4.1-mini, small sample)

## Common usage patterns and examples

### SingleTurnEnv (prompt → single response)
- **gsm8k**: Classic QA with exact-match reward; toggles `ThinkParser` vs `Parser` and format reward.
- **math**: Hendrycks MATH dataset with `MathRubric` reward (using HuggingFace's `math-verify` scorer).
- **reverse_text**: XML formatting with non-binary LCS reward + format reward.
- **gpqa**: Multiple-choice; demonstrates optional judge-based secondary scoring via `RubricGroup`.
- **simpleqa**: Judge-graded A/B/C classification using `JudgeRubric` rewards.
- **summarize_text**: Multiple rewards (length/format + similarity) combined in one `Rubric`.
- **continuation_quality**: Completion-style generation (`message_type="completion"`) judged for prose quality with `JudgeRubric`.
- **mmmu**: Multimodal inputs (image + text) packed in chat content; single-turn boxed-answer check.

### SingleTurnEnv subclass (custom dataset/scoring wrappers)
- **reasoning_gym_env**: Wraps `reasoning_gym` procedural datasets, converts to HF datasets, uses `XMLParser` and task-specific scoring.

### MultiTurnEnv (custom interaction protocols)
- **doublecheck**: Simple follow-up turn ("Are you sure?") with math rewards; minimal `is_completed`/`env_response` implementation.
- **sentence_repeater**: Multi-turn Q/A over a paragraph; rewards compare assistant messages to expected answers.
- **wordle**: Game-style interaction via `TextArenaEnv`; multiple rewards (correctness, partial credit, few-turn bonus) and XML formatting.

### Tool use
- **ToolEnv (native function-calling)**
  - **tool_test**: Validates parallel tool calls and checks exact tool usage via `ToolRubric` + custom reward.
  - **wiki_search**: Multi-tool retrieval (search/view/read) with `ToolEnv`; final judgment combined via `RubricGroup` with a `JudgeRubric`.

- **XML tool calling (roll-your-own on MultiTurnEnv)**
  - **xml_tool_env**: Parses `<tool>{...}</tool>` commands with `XMLParser`, executes Python functions, and returns `<result>...</result>` via `env_response`.
  - **xlam_function_calling**: Single-turn XML tool-call verification (no execution) that checks called tools match the ground truth list.
  - **smolagents_math_tools**: Integrates Smolagents `Tool` objects and a custom parser for tool/answer XML; demonstrates external tool frameworks.

### Sandboxes
- **PythonEnv (ipython-style REPL)**
  - **math_python**: Solve math problems using Python in a sandbox environment.

### Composition
- **EnvGroup**
  - **math_group**: Groups two `SingleTurnEnv` tasks (GSM8K + Math) into one environment with shared interface.

- **RubricGroup**
  - **math_python**: `ToolRubric` (tool adherence) + `MathRubric` (answer correctness).
  - **gpqa**: Adds a `JudgeRubric` alongside base rubric for auxiliary scoring.
  - **wiki_search**: Merges judge scoring with the tool-use rubric.

### Judge-based evaluation (LLM-as-judge)
- **simpleqa**: Judge rubric maps graded letters to reward.
- **continuation_quality**: Judge rubric extracts `<grade>` and maps A–F to a continuous score.
- **toxicity_explanation**: Judge rubric returns 0–10 normalized score for both classification correctness and explanation quality.
- **self_reward**: pattern for `SingleTurnEnv` with only a `JudgeRubric` over a dataset that supplies `question`/`answer`; intended for online RL where model acts as its own judge.

### Parsers and formatting
- **ThinkParser**: Used in `gsm8k`, `wiki_search` to separate reasoning from final answers.
- **XMLParser**: Used in `reverse_text`, `wordle`, `summarize_text`, `reasoning_gym_env`, `xml_tool_env`, `xlam_function_calling` to enforce structured outputs and enable format rewards.
- **Custom parsers**: `smolagents_math_tools` defines a bespoke parser to interoperate with external tool schemas.

### Multimodal inputs
- **mmmu**: Demonstrates passing images via chat `content` items with `{type: "image_url", image_url: {url: ...}}` and standard answer parsing.

## What to look at for each pattern
- **Minimal SingleTurnEnv**: `reverse_text`, `gsm8k`
- **JudgeRubric end-to-end**: `simpleqa`, `continuation_quality`, `toxicity_explanation`, `self_reward`
- **ToolEnv with real tools**: `wiki_search`, `math_python`
- **Custom MultiTurnEnv**: `doublecheck`, `sentence_repeater`, `wordle`
- **XML tools without native function-calling**: `xml_tool_env`, `xlam_function_calling`
- **Environment and rubric composition**: `math_group`, `math_python`, `gpqa`, `wiki_search`
- **Procedural datasets**: `reasoning_gym_env`
- **Multimodal**: `mmmu`

## Running examples
All environments export `load_environment(...)`. 

In-line usage:
```python
import verifiers as vf
from openai import AsyncOpenAI
vf_env = vf.load_environment("reverse-text")
results = vf_env.evaluate(client=AsyncOpenAI(), model="gpt-4.1-mini", num_examples=25)
```

CLI usage:
```bash
vf-install reverse-text --from-repo
vf-eval reverse-text -n 50 -r 1
```

If you are building a new environment, prefer starting from `vf-init` and consult the top-level README and docs for dataset format, parser/rubric design, and rollout constraints.
