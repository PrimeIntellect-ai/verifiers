# Verifiers v0.1.2.post1 – Release Notes

Incremental update focused on a new stateful tool environment, environment folder cleanup/renaming, math verification robustness, reporting improvements, and bug fixes.

## Highlights
- Stateful tools: add a stateful tool environment and move tool JSON loading into environment responses (PR #224).
- Environments: consolidation/renames for clarity and new environment tags (PR #222 and related changes).
- Lazy imports: training-related libraries are only imported when accessed
- Verification: more robust default math verification (PR #213).
- RL support: enable base-model RL with `message_type="completions"` (PR #201), plus Prime-RL integration and docs (PR #204) and GRPO trainer updates (PR #217, #218).
- Reporting & endpoints: template/report tweaks and endpoint path loading improvements (PR #206, PR #203, plus follow-ups).
- CLI/UX: make `rich` a default dependency for the eval script (PR #200); eval output refinements.
- Fixes: hotfix for sampling args for `gpt-5`.

## Changes by Area
### CLI and Scripts
- vf-eval
  - Add `rich` as a default dependency to improve output readability (PR #200).
  - Refine eval outputs and result handling (PR #223 and related commits).
- Hotfixes
  - Update sampling args for `gpt-5` (hotfix commit).

### Environments and Examples
- Add a stateful tool environment; load tool information via environment responses (PR #224).
- Rename and consolidate environments, introduce tag metadata for discoverability (PR #222; additional env tag updates).
- Math environment updates and prompt tweaks.
- Remove dead processing code in `environment.py`; general cleanup and type hint improvements.

### Parsers, Rubrics, and Utils
- Caching improvements for JudgeRubric to reduce redundant work (PR #216).
- More robust rule-based math verification and heuristics (PR #213).
- General type-hint and internal cleanup passes.

### Training
- Document Prime-RL training (PR #204).
- Minor updates to GRPO trainer (PR #217, #218).
- Add support for base-model RL flows via `message_type="completions"` (PR #201).

### Reporting and Tooling
- Report generation and template tweaks (PR #206, PR #203).
- Improve endpoint path loading and related tooling.

### Documentation
- README and docs updates (minor) across environments and training utilities; additional guidance for reporting.

## Upgrade Notes
- Environment renames/tags: if you reference environment names or use tags in tooling or scripts, review the updated names and tag metadata (PR #222).


## Reference Commits (since v0.1.2.post0)
- adding stateful toolenv, moving tool json loading to env_response (PR #224)
- Will/eval outputs (PR #223)
- Update grpo_trainer.py (PR #217, PR #218)
- hotfix for gpt-5 sampling args
- Will/rename envs (PR #222)
- Will/judgerubric caching (PR #216)
- More robust rule-based math verification (PR #213)
- Report tweaks and endpoints path loading (PR #206 and follow-ups)
- Integrate and document prime-rl training (PR #204)
- Update report generation and vf-init template (PR #203)
- Add support for base model RL / `message_type="completions"` (PR #201)
- Add `rich` as default dependency for eval script (PR #200)
- Math env updates, prompt tweaks, type hints, and cleanup in `environment.py`

## Full Changelog
- `v0.1.2.post0...HEAD`: https://github.com/willccbb/verifiers/compare/v0.1.2.post0...HEAD
