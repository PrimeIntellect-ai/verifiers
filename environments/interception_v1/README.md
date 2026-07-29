# interception-v1

An example v1 environment that defines deterministic and judge-backed guards directly
with `@vf.intercept`.

The task class installs two response handlers in priority order:

1. `deterministic_guard` checks `exchange.message` with an exact rule.
2. `judge_guard` calls `exchange.judge()` only when the deterministic rule did not
   rewrite the candidate.

Both handlers receive one `vf.ModelExchange`: the current prompt, typed candidate, trace,
and direction. `exchange.replace()` returns an inert message of the same kind, while
`exchange.judge()` builds and parses the guard prompt and records the ordinary judge call
and its usage on the trace.

```python
@vf.intercept()
async def judge_guard(self, exchange: vf.ModelExchange[vf.AssistantMessage]):
    if await exchange.judge(JUDGE_RUBRIC) == "BLOCK":
        return exchange.replace("Blocked by the judge guard.")
```

The higher-priority deterministic handler always runs first. The lower-priority judge
sees any replacement it produced, so it classifies the cleaned result rather than the
original candidate. The reward checks `trace.interceptions` to show which decorated
guard rewrote the exchange.

## Develop

Install the package and evaluate both examples:

```bash
uv pip install -e environments/interception_v1
uv run eval interception-v1 -n 2
```

The example uses `vf.Judge()` and its default model. A handler can select another model
by constructing the judge with a config:

```python
judge = vf.Judge(vf.JudgeConfig(model="openai/gpt-5.4-nano"))
verdict = await exchange.judge(JUDGE_RUBRIC, judge=judge)
```
