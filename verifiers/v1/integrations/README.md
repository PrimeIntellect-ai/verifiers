# Braintrust episode export

`log_episode` exports a completed native v1 `Episode` to a caller-owned Braintrust
logger. It also accepts a saved episode loaded with `WireEpisode`. There is no
runtime monkey-patching, environment-variable auto-enablement, CLI configuration,
or required Braintrust dependency in verifiers.

Install the SDK separately with `uv pip install braintrust` and authenticate it
using Braintrust's normal mechanism. Project selection and authentication stay
outside the episode and eval configuration.

```python
import braintrust

from verifiers.v1.integrations.braintrust import log_episode

logger = braintrust.init_logger(project="verifiers", set_current=False)
try:
    episode = await env.run(task)
    log_episode(episode, logger)
finally:
    logger.flush()
```

For an async consumer, call `await asyncio.to_thread(log_episode, episode, logger)`
after saving the episode locally. Export failures propagate; the caller decides
whether they should fail the run or be reported separately. Flush once after a
batch, not after each message. SDK delivery/retry behavior remains the SDK's
responsibility.

## Inspect an existing run

```python
from pathlib import Path

import braintrust

from verifiers.v1.episode import WireEpisode
from verifiers.v1.integrations.braintrust import log_episode

logger = braintrust.init_logger(project="verifiers", set_current=False)
try:
    with Path("outputs/my-run/traces.jsonl").open() as records:
        for line in records:
            if line.strip():
                log_episode(WireEpisode.model_validate_json(line), logger)
finally:
    logger.flush()
```

Only call this when you intend to send these records to Braintrust. Each invocation
creates a new export, so retrying a batch can create duplicate traces. This is an
explicit post-completion export, not live token/turn streaming or an automatic
`prime eval` integration.

## Recorded structure

- One episode root, one child per agent trace.
- One point span per native message node. `parent_node` and `semantic_parents`
  preserve graph edges without expanding shared history into every branch.
- One LLM span per actual model call, including unsuccessful calls without a
  committed node. Recorded start/end times are preserved. Message timestamps are
  commit instants, **not tool execution durations**.
- Provider usage is logged only on model-call spans to avoid double counting.
  Braintrust prompt tokens include cache reads; reasoning tokens remain a subset
  of completion tokens. Judge/off-graph usage is separate trace metadata.
- The weighted reward is `metrics.verifiers_reward`; named numeric metrics use a
  `verifiers/` prefix. Raw reward scores and weights remain trace metadata because
  verifiers rewards are not restricted to Braintrust's `[0, 1]` score range.
- Episodes without traces become export-time points, labelled `timing_source`.
  Missing call starts fall back to the trace start and are labelled likewise.

## Content and size

By default, exports include identifiers, model names, graph edges, reward/metric
names and values, error **types**, and timings, but no message bodies, tool
arguments/results, error messages, task data, agent configs, or `trace.info`.
These metadata fields can still be sensitive; this is not an anonymization layer.

`log_episode(episode, logger, include_content=True)` explicitly enables message
JSON excerpts and error messages. Each excerpt is capped at 16,384 characters;
message metadata marks `content_truncated`. Truncated JSON is an excerpt, not a
parseable document. Content may contain credentials, personal information, or
multimodal URLs, so review/redact it before opting in. Provider replay state,
token arrays/tensors, task data, agent configs, and `trace.info` are never exported.

Span count and traversal are linear in nodes plus calls, not total expanded branch
length. There is no attachment upload or full-conversation replay per model call.
This keeps individual content fields bounded, but does not bound the SDK's batch
queue or total export size; control concurrency at the consumer.
