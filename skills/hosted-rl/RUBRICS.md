# Rubrics Reference

## Reward Functions

Functions can be sync or async. Arguments received by name from: `completion`, `prompt`, `answer`, `info`, `state`, `task`, `parser`, plus any `class_objects`.

```python
async def correct_answer(completion, answer, **kwargs) -> float:
    return 1.0 if answer in completion[-1]["content"] else 0.0

rubric = vf.Rubric(funcs=[correct_answer, format_fn], weights=[1.0, 0.2])
```

- Functions execute in order; `state` is mutable and shared
- Use `weight=0.0` (or `rubric.add_metric(fn)`) for metrics that don't affect training
- Add shared objects via `rubric.add_class_object("name", obj)`

## Group Reward Functions

Use plural argument names (`completions`, `prompts`, `answers`, `infos`, `states`, `tasks`) and return `list[float]`:

```python
async def diversity_bonus(completions) -> list[float]:
    responses = [c[-1]["content"] for c in completions]
    return [0.2 if responses.count(r) == 1 else 0.0 for r in responses]
```

## Built-in Rubrics

### vf.Rubric

Core rubric combining weighted reward functions.

- `add_reward_func(fn, weight)`
- `add_metric(fn)` — weight=0
- `add_class_object(name, obj)`

### vf.JudgeRubric

LLM-as-judge evaluation. Provides a `judge(prompt, completion, answer, state)` callable.

```python
judge_rubric = vf.JudgeRubric(judge_model="gpt-4.1-mini", judge_prompt="...")

async def score(judge, prompt, completion, answer, state) -> float:
    verdict = await judge(prompt, completion, answer, state)
    return 1.0 if "yes" in verdict.lower() else 0.0

judge_rubric.add_reward_func(score)
```

Exposes: `judge_client`, `judge_model`, `judge_prompt`, `judge_sampling_args` as class objects.

## Parsers

### vf.Parser(extract_fn=...)

Base parser. `parse_answer()` returns last assistant message content. `get_format_reward_func()` always returns 1.0.

### vf.ThinkParser(extract_fn=...)

Strips `<think>...</think>` and returns content after.

**Warning:** Do NOT use with Qwen3 or DeepSeek-R1 — their chat templates auto-remove `<think>` tags.

### vf.MaybeThinkParser(extract_fn=...)

Like ThinkParser but thinking is optional.

### vf.XMLParser(fields, answer_field, extract_fn)

Structured XML output. Fields are strings or tuples (canonical name + aliases).

```python
parser = vf.XMLParser(fields=["think", ("answer", "code")], answer_field="answer")
# Parses: <think>...</think>\n<answer>...</answer>

parsed = parser.parse(text)     # -> SimpleNamespace(think=..., answer=...)
parser.get_format_str()         # format description string
parser.get_fields()             # ["think", "answer"] (canonical names)
parser.format(think="...", answer="...")  # produce XML string
parser.get_format_reward_func() # checks field presence, spacing, alignment
```
