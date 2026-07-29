# interception-v1

An example v1 environment that defines deterministic and judge-backed guards directly
with `@vf.intercept`.

The task class installs two handlers in priority order:

1. `deterministic_guard` checks `trace.last_message` with an exact rule.
2. `judge_guard` then passes `trace.messages` to an ordinary `vf.Judge`.

Both handlers receive only the trace. During interception, `trace.messages` is the full
exchange and `trace.last_message` is the candidate crossing the model boundary.
`trace.replace()` returns an inert message of the same kind. Passing the trace to
`Judge.evaluate()` records the judge call and its usage on that trace.

```python
@vf.intercept()
async def judge_guard(self, trace: vf.Trace):
    verdict = await JUDGE.evaluate(trace=trace, exchange=trace.messages)
    if verdict.text.strip() == "BLOCK":
        return trace.replace("Blocked by the judge guard.")
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

The example uses `vf.Judge()` and its default model. Select another model in the judge
configuration:

```python
judge = vf.Judge(vf.JudgeConfig(model="openai/gpt-5.4-nano", prompt=JUDGE_PROMPT))
verdict = await judge.evaluate(trace=trace, exchange=trace.messages)
```
