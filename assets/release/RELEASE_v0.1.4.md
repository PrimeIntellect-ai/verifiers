# Verifiers v0.1.4 Release Notes

*Date:* 9/23/25

Verifiers v0.1.4 focuses on getting evaluation results back to you faster while strengthening orchestration safety nets and
modernizing our contributor tooling.

## Highlights

- **Interleaved rollout orchestration** streams generation and scoring concurrently with independent concurrency limits,
  dramatically reducing end-to-end evaluation latency for large jobs and documenting the upgrade for custom integrators.
- **Stronger guardrails** land repository-wide type checking with `ty`, import regression tests, async tool call fixes, and more
  resilient error messaging throughout the evaluation stack.
- **Sharper maintainer ergonomics** arrive via richer logging, saner CLI defaults, and refreshed contributor documentation plus
  automation that keeps workflows discoverable.

## Pull request catalog

### Orchestration and scoring
- [#324](https://github.com/willccbb/verifiers/pull/324) overhauled rollout orchestration to interleave generation and scoring
  with independent concurrency controls, upgraded timing telemetry, and refreshed the public docs that explain the new flow.
- [#357](https://github.com/willccbb/verifiers/pull/357) taught `RubricGroup.score_rollout` to accept parser-provided state so
  interleaved scoring remains stable across rubric compositions.

### Evaluation experience
- [#319](https://github.com/willccbb/verifiers/pull/319) normalized `vf-eval`'s `-n` handling so requests larger than the dataset
  fall back gracefully without negative sampling assertions.
- [#326](https://github.com/willccbb/verifiers/pull/326) fixed async tool invocation inside `StatefulToolEnv` by properly
  awaiting tool calls, restoring parity with the stateless tool runner.
- [#309](https://github.com/willccbb/verifiers/pull/309) instrumented environment loading with detailed logging around import
  failures, default argument usage, and instantiation success.
- [#335](https://github.com/willccbb/verifiers/pull/335) tightened the error surfaced when a candidate environment omits the
  required `load_environment` hook so CLI diagnostics stay actionable.
- [#291](https://github.com/willccbb/verifiers/pull/291) wrapped judge model calls with targeted exception handling, turning
  OpenAI API failures into actionable error messages instead of silent crashes.

### Training and typing
- [#320](https://github.com/willccbb/verifiers/pull/320) fixed GRPO trainer `num_iteration` handling to respect configured
  iteration counts during policy updates.
- [#347](https://github.com/willccbb/verifiers/pull/347) added the `ty` type checker to pre-commit, locking in static analysis
  alongside `ruff` for every patch.
- [#353](https://github.com/willccbb/verifiers/pull/353) introduced import regression tests that traverse `verifiers.__all__`,
  preventing accidental export gaps.
- [#355](https://github.com/willccbb/verifiers/pull/355) completed the typing cleanup by pruning unused math utilities,
  tightening trainer types, and reorganizing dependency declarations to keep `ty` green in CI.

### Documentation and contributor experience
- [#346](https://github.com/willccbb/verifiers/pull/346) published a consolidated `AGENTS.md`, refreshed parser tests, and
  aligned contributor expectations across the repo.
- [#344](https://github.com/willccbb/verifiers/pull/344) expanded environment hub documentation with concrete publishing and
  consumption workflows.
- [#323](https://github.com/willccbb/verifiers/pull/323) unified the style workflow and introduced an automated
  `publish-environments` action for teams distributing shared tasks.

## Contributors

Huge thanks to everyone who shipped improvements in this cycle:
- @cdreetz
- @fsndzomga
- @srthkdev
- @ZhichenRen
- @bdsaglam
- @willccbb

**Full Changelog**: https://github.com/willccbb/verifiers/compare/v0.1.3...v0.1.4
