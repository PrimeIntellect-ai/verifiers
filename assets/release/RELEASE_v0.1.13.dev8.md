# Verifiers v0.1.13.dev8 Release Notes

*Date:* 04/28/2026

## Highlights since v0.1.13.dev7

- **Per-rollout wall-clock timeout** for `MultiTurnEnv`. New `timeout_seconds: float | None` kwarg bounds total rollout via `asyncio.wait_for`; on fire, `mark_timed_out(state)` writes `timed_out=True`, `is_completed=True`, `stop_condition="timeout_reached"`. New `vf-eval --timeout SECONDS` CLI flag injects `timeout_seconds` into the env constructor (and recognizes `timeout = N` at the top of `[[eval]]` TOML tables); wins over `--extra-env-kwargs`. `CliAgentEnv` derives sandbox `timeout_minutes` from this (rollout deadline + 60min scoring buffer, clamped to a 24h SDK ceiling) and exposes `compute_sandbox_timeout_minutes` on `SandboxMixin` for taskset overrides via `SandboxSpec(timeout_minutes=None)`.
- Several smaller `CliAgentEnv` / composable / RLM fixes.

## Changes included in v0.1.13.dev8 (since v0.1.13.dev7)

### Features and enhancements

- feat: rollout timeout (#1258)
- Harness.keep_trajectory_step; rlm: rlm_max_depth, include_sub_rlm_trajectories (#1253)
- TITO: treat content='' / None as equal in prefix-match + warn on MITO fallback past turn 1 (#1259)
- Harness.environment_vars: per-rollout callable, rlm: rand threshold (#1248)
- TaskSet: accept DatasetBuilder for lazy dataset construction (#1251)
- cli_agent_env: bump default poll_interval from 1s to 5s (#1255)

### Fixes and maintenance

- fix: avoid TOCTOU port race in CliAgentEnv interception server (#1264)
- rlm harness: remove sandbox-side git shim (replaced by rlm tool-level block, rlm#70) (#1262)
- fix: narrow math_verify BaseException catch to specific TimeoutException (#1197)
- fix: handle dict word_list in TextArenaEnv.ta_to_hf() (#1214)
- fix: correct timing accumulation in RubricGroup score_rollout and score_group (#1215)
- composable_env: skip caches/.git/.venv when tarring upload dirs (#1257)
- swe tasksets: default ds_num_proc to None for all SWE tasksets (#1256)
- git_checkout_cache: hold per-process in-use lock so concurrent resolves don't nuke active worktrees (#1252)
- fix: prepend vllm/ to slashless OPENAI_MODEL in composable opencode harness (#1250)
- rlm harness: stage git-refusal shim into $HOME/.local/bin so scoring can run git (#1244)
- rlm harness: remove dead RLM_KERNEL_PYTHON detection block (#1242)
- SWEBenchTaskSet.setup: symlink venv at /testbed/.venv matching WORKDIR (#1241)

**Full Changelog**: https://github.com/PrimeIntellect-ai/verifiers/compare/v0.1.13.dev7...v0.1.13.dev8
