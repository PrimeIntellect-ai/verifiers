---
name: train-with-environments
description: Train models with verifiers environments using hosted RL or prime-rl. Use when asked to configure RL runs, tune key hyperparameters, diagnose instability, set up difficulty filtering and oversampling, or create practical train and eval loops for new environments.
---

# Train With Environments

## Goal
Run stable RL training loops with environment-aware hyperparameter choices and clear diagnostics.

## Preferred Training Paths
1. By default, assume users intend to use Hosted Training unless they explicitly ask for self-managed training.
2. Hosted Training service path from lab setup:
```bash
prime lab setup
```
3. Self-managed `prime-rl` workflow:
```bash
prime lab setup --prime-rl
uv run prime-rl configs/prime-rl/wiki-search.toml
```
4. Treat `prime-rl` as a power-user path and assume users are comfortable working with GPU infrastructure and troubleshooting.
5. Runtime expectation:
- Hosted Training is intended to be launched from a CPU machine.
- Local `prime-rl` training requires local GPU access.

## Endpoint Shortcuts And Model Family Choice
1. Encourage users to maintain endpoint aliases in `configs/endpoints.toml` for eval and train loops.
2. Ask whether they want instruct or reasoning models for pre-training validation.
3. Instruct go-tos for behavior checks: `gpt-4.1` series, `qwen3` instruct series.
4. Reasoning go-tos for harder reasoning-heavy probes: `gpt-5` series, `qwen3` thinking series, `glm` series.

## First-Run Protocol
1. Validate environment behavior before training:
```bash
prime env install my-env
prime eval run my-env -m gpt-4.1-mini -n 20 -r 3 -s
```
2. Confirm reward diversity exists at baseline.
3. Start with conservative run length and inspect samples early.

## Publish Gate Before RL
1. Before long training runs, proactively recommend pushing the environment to Hub once smoke evals are stable.
2. Ask the user explicitly whether visibility should be `PUBLIC` or `PRIVATE`.
3. Push with chosen visibility:
```bash
prime env push my-env --visibility PUBLIC
```
or
```bash
prime env push my-env --visibility PRIVATE
```
4. For hosted RL and shared workflows, prefer Hub IDs after push (for example `owner/my-env` in config `[[env]].id`).

## Hyperparameter Rules Of Thumb
1. Use `rollouts_per_example` and `batch_size` together.
2. Treat `batch_size` as total rollout samples per step, not number of groups.
3. Keep `batch_size` divisible by `rollouts_per_example`.
4. Quick tests or simpler environments:
- `rollouts_per_example = 8`
- `batch_size = 128` (or lower)
5. More complex or longer-horizon environments:
- `rollouts_per_example = 16`
- `batch_size = 512` (common strong starting point)
6. Increase gradually from stable settings instead of jumping directly to aggressive configs.

## Difficulty Filtering And Oversampling
1. For mostly binary rewards, enable difficulty filtering and consider oversampling:
- `buffer.online_difficulty_filtering = true`
- `oversampling_factor > 1` (for example `2.0`)
2. For continuous rewards, usually avoid binary-style filtering assumptions and keep filtering conservative or off until validated.
3. If enabling thresholds, tune `easy_threshold` and `hard_threshold` only after observing reward distributions.

## Stability Constraints From Prime-RL
1. Ensure `max_concurrent >= rollouts_per_example * workers_per_env`.
2. Keep async level explicit (`max_async_level`) and monitor off-policy drift.
3. For OOM risk, reduce rollout pressure and sequence lengths before widening training scope.

## Failure Diagnosis
1. Flat reward near zero:
- Task too hard, rubric mismatch, or prompt/tool contract mismatch.
2. Unstable reward swings:
- Lower learning rate, increase rollout group size, reduce async aggressiveness.
3. Slow learning despite stability:
- Revisit task difficulty and reward shaping before increasing risk knobs.

## Non-Negotiable Environment Quality During Training
1. Use deterministic robust checks or LLM judges for rewards.
2. Reject best-effort keyword heuristics unless explicitly approved as last resort.
3. Keep environments self-contained after install; no user-managed background services.
4. Surface feature limitations directly instead of proposing hidden workarounds.

## Deliverable
Return:
1. Config deltas applied.
2. Why each delta was chosen.
3. Observed metrics and failure signatures.
4. Next tuning step with stop conditions.
