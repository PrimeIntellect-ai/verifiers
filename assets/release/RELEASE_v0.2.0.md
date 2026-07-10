# Verifiers v0.2.0 Release Notes

*Date:* 07/10/2026

## Highlights since v0.1.14

- **A task-centric v1 API.** Verifiers v1 now separates serializable per-row `TaskData` from the `Task` class that owns lifecycle hooks, tools, metrics, rewards, and judges. A `Taskset` loads typed tasks, a `Harness` defines how the model runs, and the two compose into an environment without mixing v1 and legacy APIs.
- **Portable agent execution.** The same taskset can run across built-in and custom harnesses on local subprocesses, Docker, or remote Prime and Modal sandboxes. Model traffic passes through the interception layer, which adapts the harness's provider dialect, records the trace as it happens, applies sampling configuration, and supports response and tool-result interception.
- **Typed authoring, tools, and scoring.** Strict Pydantic configs cover tasksets, tasks, harnesses, runtimes, and evaluation. Environment authors can scaffold packages with `uv run init`, expose taskset- or task-scoped MCP tools, attach user simulators, and score traces with rewards, metrics, group rewards, reusable scoring helpers, and configurable LLM judges.
- **A complete v1 evaluation workflow.** `uv run eval` supports TOML configs, dotted CLI overrides, dry runs, concurrency controls, rich progress and paging, resumable runs, and platform upload. Native `validate`, `debug`, and `replay` entrypoints cover environment checks, interactive diagnosis, and offline re-scoring of saved traces.
- **First-class traces and observability.** Traces capture the message graph, task identity and data, rewards, metrics, errors, token and timing information, runtime metadata, and per-message creation timestamps. Evaluation artifacts now use `traces.jsonl`, with improved streaming, memory use, and dashboard reporting for long or highly concurrent runs.
- **Harbor and production environment support.** v1 includes a Harbor taskset base, reusable built-in harnesses and tasksets, container-aware runtime configuration, and a companion catalog of production benchmarks in `research-environments`.

## Selected changes included in v0.2.0

### v1 framework and authoring

- Introduce the complete Verifiers v1 taskset, harness, runtime, trace, and CLI stack (#1825)
- Move per-task behavior from `Taskset` onto typed `Task` objects (#1948)
- Add native v1 `debug` and `validate` workflows (#1905)
- Add reusable scoring helpers and configurable reference/rubric judges (#1907, #1918)
- Add built-in Harbor support and reusable Lean taskset foundations (#1877, #1882, #1883)
- Refresh the v1 user documentation (#1916)

### Evaluation, traces, and reliability

- Add offline trace replay and platform upload for eval runs (#1936, #1947)
- Rename the v1 evaluation artifact to `traces.jsonl` (#1956)
- Add trace record export, token-length attribution, runtime metadata, and message timestamps (#1844, #1875, #1959, #1960)
- Improve the rich dashboard with phase markers, usage totals, pending-rollout visibility, and paging (#1866, #1869, #1912, #1913)
- Make resumed runs restore their saved traces and complete only missing or errored work (#1892, #1962)
- Improve streaming, concurrency, and memory behavior for large v1 evaluations (#1793, #1807, #1815, #1838)

**Full Changelog**: https://github.com/PrimeIntellect-ai/verifiers/compare/v0.1.14...v0.2.0
