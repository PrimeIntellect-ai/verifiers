# Experimental Rubrics

## OracleRubric

OracleRubric scores generations using an external oracle (API server, ML model, simulator, or any callable). It is a direct parallel of JudgeRubric: same ergonomics, same injection pattern, just pointing at your backend instead of an LLM judge.

### Arguments

| Argument | Required | Role |
| --- | --- | --- |
| oracle | optional | Backend client or callable (analogous to `judge_client` in JudgeRubric). Passed as `oracle` kwarg to `oracle_fn`. |
| parser | optional | Parses completion into response text. |
| funcs / add_reward_func | optional | Reward function registration — same as JudgeRubric. |
| oracle_fn | optional | Calls the backend. Receives `oracle` (backend), `prompt`, `completion`, `answer`, `state`, `response`. If omitted, calls the backend directly with the parsed response. |
| cache_measurements | optional | Cache oracle outputs within a rollout. Default True. |

### Pattern

Instantiate with just the oracle backend, optionally supply `oracle_fn` to handle how the backend is called, then register reward functions with `add_reward_func`. Reward functions receive `oracle` injected automatically — they call it directly, exactly like `judge` in JudgeRubric:

```python
async def my_score(oracle, prompt, completion, answer, state, **kwargs):
    result = await oracle(prompt, completion, answer, state)
    threshold = answer.get("threshold", 0) if isinstance(answer, dict) else 0
    return 1.0 if float(result) >= threshold else 0.0

rubric = OracleRubric(
    oracle=my_backend,
    oracle_fn=call_backend,  # optional: defaults to calling oracle directly
)
rubric.add_reward_func(my_score)
```

### oracle vs oracle_fn

- `oracle` — the raw backend object (SolubilityPredictClient, sklearn model, HTTP client, …)
- `oracle_fn(oracle, prompt, completion, answer, state, response, **kwargs)` — handles calling the backend and returning its output. Receives `oracle` as the raw backend. If omitted, the backend is called with just the parsed response text.
- In reward functions, `oracle` is `self.oracle` (the bound method on OracleRubric) — calling `await oracle(prompt, completion, answer, state)` triggers the full inference + caching pipeline.
