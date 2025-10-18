## Rollout Orchestration Refactor – Design

### Objectives
- Interleave rollouts and scoring by default to reduce end-to-end latency.
- Separate semaphores for generation and scoring (`max_concurrent_generation`, `max_concurrent_scoring`), while `max_concurrent` still applies to both by default.
- Preserve current behavior with a boolean flag to disable interleaving.
- Track an `id` per example and add it to each rollout `state`.
- Track generation, scoring, and total time per rollout (exposed via `state`).
- Restore robust `make_dataset` support suitable for `vf-eval`/HF pushes.
  - Centralize sanitation of tool calls and message normalization in one utility.
  - Keep essential dataset logic discoverable from `Environment`, with thin utility extraction to reduce file size.
- Lay groundwork for future type improvements (e.g., structured `state`, column-to-`info` compilation) without breaking current users.

---

### Current Architecture (as of this design)
- Generation orchestration happens in `Environment.a_generate` and `Environment.run_rollouts`.
  - `a_generate` normalizes inputs, runs `run_rollouts` (optionally with a single `max_concurrent` semaphore), then calls `Rubric.score_rollouts` (also with the same `max_concurrent`).
  - Scoring waits for all rollouts to complete; there is no interleaving between generation and scoring.
- `Rubric.score_rollouts` evaluates rollouts with optional parallelism (per-rollout), gathering results at the end.
- `MultiTurnEnv.rollout` performs per-turn interaction and accumulates `state["responses"]` (+ `responses_start_idx` for completion-style) and `completion`.
- `Environment.make_dataset` exists and sanitizes `completion` with `sanitize_tool_calls`, while the CLI `vf-eval` currently assembles a dataset independently and also sanitizes.
- Concurrency:
  - One `max_concurrent` governs rollout generation (`run_rollouts`) and the same value is passed to scoring.
  - No distinct semaphore for scoring.

Key references:
- `verifiers/envs/environment.py`: `a_generate`, `run_rollouts`, `make_dataset`
- `verifiers/envs/multiturn_env.py`: rollout loop
- `verifiers/rubrics/rubric.py`: scoring
- `verifiers/scripts/eval.py`: CLI dataset construction and saving
- `verifiers/utils/message_utils.py`: `sanitize_tool_calls`, `cleanup_messages`

---

### Proposed Architecture

#### New orchestration in `Environment.a_generate`
- Add interleaved pipeline with two independent concurrency controls:
  - Generation stage (rollout) limited by `max_concurrent_generation`.
  - Scoring stage limited by `max_concurrent_scoring`.
- Default behavior:
  - `interleave_scoring=True`.
  - If `max_concurrent_generation`/`max_concurrent_scoring` are not set, they inherit `max_concurrent` (maintaining a simple one-knob experience).
- Fallback behavior:
  - `interleave_scoring=False` preserves the current two-phase flow: generate all (with `max_concurrent`), then score all (with `max_concurrent`).

Implementation sketch (internal only):
- Prepare pre-sized containers for `completion`, `state`, `reward`, and `metrics`, indexed by rollout index to preserve ordering.
- Launch generation tasks with a generation semaphore (or `as_completed` with bounded concurrency). For each finished generation:
  - Record per-rollout generation time in `state["timing"]["generation_ms"]`.
  - Enqueue a scoring task (single-rollout `rubric.score_rollout(...)`) guarded by the scoring semaphore.
  - When scoring completes, write `reward` and `metrics` for that index and set `state["timing"]["scoring_ms"]`, `state["timing"]["total_ms"]`.
- Await completion of both stages (queues drained / tasks joined), then return `GenerateOutputs` with all fields filled.

Notes:
- `Rubric.score_rollout` is already capable of single-rollout scoring; no rubric API change is required.
- For batched/complex scoring strategies, we can keep `Rubric.score_rollouts` in the non-interleaved path.

#### API Additions (non-breaking)
- `Environment.a_generate(...,
  interleave_scoring: bool = True,
  max_concurrent_generation: int | None = None,
  max_concurrent_scoring: int | None = None,
  ...)`.
- `Environment.generate(...)` and `Environment.evaluate(...)` pass-through these flags with sensible defaults.
- `AsyncBatchGenerator` continues to call `a_generate` and benefits from interleaving by default without changes.

#### Example ID Tracking
- Inputs:
  - If an `example_id` column is present in `inputs`/`Dataset`, carry it through to `state["example_id"]` per rollout.
  - If not present, synthesize a simple 0..N-1 `example_id` by example (before repeat), and propagate to each rollout.
- For `evaluate(..., rollouts_per_example>1)`: replicate rollouts using `Dataset.repeat`; map `example_id` as `i // rollouts_per_example` if IDs are synthesized. If an `example_id` column exists, repeat it directly.
- Do not change `GenerateOutputs` schema; expose `example_id` via `state` and surface it in `make_dataset`.

#### Timing Tracking
- Add `state["timing"] = {"generation_ms": float, "scoring_ms": float, "total_ms": float}`.
- Timestamps are measured internally in `a_generate` around rollout and scoring calls.
- No change to external types; timing is opt-in to expose through `make_dataset` via `state_columns`.

---

### Dataset Construction and Sanitization

Goals:
- Single source of truth for message sanitation and dataset assembly usable by both `Environment.make_dataset` and `vf-eval`.

Plan:
- Create `verifiers/utils/dataset_utils.py` with:
  - `sanitize_messages(messages: Messages) -> Messages` that wraps `message_utils.sanitize_tool_calls` for both `prompt` and `completion` and normalizes content consistently.
  - `build_dataset_from_outputs(results: GenerateOutputs, *, include_ids: bool = True, state_columns: list[str] | None = None) -> Dataset` that:
    - Assembles `prompt`, `completion`, `answer` (if non-empty), `task`, `reward`, per-metric columns, `info` (if non-empty), and `id` (if present via `state`).
    - Applies sanitation to both `prompt` and `completion`.
    - Optionally extracts specific `state` fields (e.g., `timing`) into columns.
- Update `Environment.make_dataset` to delegate to `dataset_utils.build_dataset_from_outputs`, preserving the current public API and adding support for `id` and `state_columns`.
- Update `vf-eval` to use `env.make_dataset(results, ...)` rather than reassembling the dataset manually (no change to CLI flags; behavior becomes consistent with library logic).

Notes:
- `message_utils.cleanup_messages` remains for input normalization; sanitation for persistence should flow through `dataset_utils` consistently.

---

### Types and Future-Proofing (no immediate breaking changes)
- Keep `State` as `dict[str, Any]` for now; document the `state["id"]` and `state["timing"]` keys.
- Later, consider a `TypedState` Pydantic model that is backwards-compatible (e.g., via duck typing or optional casting) and supports:
  - Stable well-known fields (prompt/completion/answer/task/info/id/timing/responses...)
  - User-provided passthrough fields.
- Consider allowing users to declare extra input columns that are folded into `info` during `a_generate` preprocessing (opt-in, e.g., `fold_columns_to_info=[...]`).

---

### Backward Compatibility
- Default `interleave_scoring=True` changes performance characteristics but not outputs; disabling it restores the exact two-phase behavior.
- `Rubric` API unchanged.
- `GenerateOutputs` unchanged; `id` and timings live in `state`.
- `Environment.make_dataset` remains stable, augmented to include optional `id` and `state_columns`.
- CLI continues to work; its dataset formatting becomes consistent with `Environment.make_dataset`.

---

### Testing Plan
- Unit tests for interleaved orchestration:
  - Verify outputs match non-interleaved results for deterministic reward functions.
  - Ensure list ordering is preserved and aligned across `prompt`, `completion`, `reward`, and `metrics`.
  - Verify separate generation and scoring semaphores are respected (cap concurrency independently).
- `state` enrichment:
  - `id` propagation: synthesized and pre-existing `id` columns.
  - Timing fields present and reasonably sized (>0).
- `make_dataset`:
  - Includes `id` when available.
  - Applies sanitation to both `prompt` and `completion`.
  - Extracts requested `state_columns` (e.g., `timing`).
- CLI integration:
  - `vf-eval -s` produces identical schema via `env.make_dataset`.

---

### Migration Steps (Implementation Order)
1) Introduce new `a_generate` flags and internal interleaved orchestration (default on) with separate semaphores and timing capture.
2) Add `id` propagation logic in `a_generate` preprocessing and evaluation repeat logic.
3) Create `utils/dataset_utils.py` and switch `Environment.make_dataset` to delegate.
4) Update `vf-eval` to use `env.make_dataset` for saving/pushing datasets.
5) Add tests outlined above.

All steps are additive and non-breaking; a single PR can land all, or split across two PRs (orchestration first, dataset/CLI second).

---

### Open Questions
- Naming:
  - Approve `interleave_scoring`, `max_concurrent_generation`, `max_concurrent_scoring`?
- Defaults:
  - Should `interleave_scoring=True` be default, or keep `False` until a major/minor release note?
  - If only `max_concurrent` is provided, confirm both generation and scoring should default to the same value.
- Dataset:
  - Should `id` be included in datasets by default when present in `state`, or only when explicitly requested via `state_columns`? Proposed: include by default.
  - Include timing columns by default or only via `state_columns`? Proposed: opt-in via `state_columns=["timing"]` or specific fields like `"timing.total_ms"`.
- Utilities placement:
  - Is `verifiers/utils/dataset_utils.py` acceptable for dataset assembly and sanitation, while leaving `Environment.make_dataset` as a thin wrapper?
  - Should parts of `process_env_results_vllm` be moved into a `utils/rollout_processing.py` module to shrink `Environment` without changing its public surface?
- Any preferences on how to fold extra columns into `info` in the future (opt-in list, regex, or callable)?

---

### Acceptance Criteria
- Interleaved orchestration reduces wall-clock latency on mixed latency workloads (e.g., slow rubric functions) without changing outputs.
- Independent semaphores correctly bound concurrent generations vs. scorings.
- `id` and timing are recorded per rollout and are available for inclusion in datasets.
- `vf-eval -s` and HF pushes use a single, centralized dataset construction path with consistent sanitation.
