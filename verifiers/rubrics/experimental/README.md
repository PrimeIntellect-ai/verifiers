# Experimental Rubrics

## OracleRubric

OracleRubric is intended for reward pipelines where model output must be evaluated via an external oracle (API, model server, simulator, tool backend).

### Intended Argument Roles

| Argument | Required | Intended Role |
| --- | --- | --- |
| oracle | optional | Backend object or callable used for inference. |
| parser | optional | Parses completion into response text used by oracle_input_fn. |
| oracle_fn (or backend_caller) | optional | Connects to inference endpoint/backend and returns oracle output. |
| oracle_input_fn | optional | Builds oracle input payload from response, prompt, completion, answer, and state. |
| property_extractor | optional | Dictionary-output adapter. Use this only when oracle_fn returns a dictionary and you want to extract a single property first. |
| score_function (or comparator) | optional | Final reward computation. Receives property_value plus context (prompt/completion/answer/state/task/info kwargs). |
| target_extractor | optional | Optional helper to precompute target from answer and pass it as target into score_function. |
| threshold_extractor | optional | Optional helper to precompute threshold from answer and pass it as threshold into score_function. |
| expose_oracle_property_metric | optional | If true, logs oracle_property as a metric. Default is false (internal-only). |
| cache_measurements | optional | Enables cache for oracle outputs within rollout state. |

### Pipeline Contract

1. oracle_input_fn prepares backend input.
2. oracle_fn (or backend predict/callable path) executes inference.
3. If oracle output is a dictionary and property_extractor is set, extractor is applied.
4. Otherwise, oracle output is passed through directly as property_value.
5. score_function computes the final reward for the rollout.
6. target_extractor and threshold_extractor are optional convenience hooks.

### Notes

- score_function is the canonical reward entry.
- Metric naming for the reward follows the score_function callable name when provided.
- oracle_property is available as an optional diagnostic metric, disabled by default.
- Oracle outputs and derived values are still stored in state for debugging and reuse.

### Recommended Simplified Mode

For most tasks, use only:

- oracle
- oracle_fn
- oracle_input_fn
- score_function

In this mode, score_function can read answer/state/task/info directly, so property_extractor/target_extractor/threshold_extractor are often unnecessary.

### Current TODO

- Harden cache-key construction for multi-prediction servers by including normalized oracle_input (and optionally caller/extractor discriminator) to avoid cache collisions.
