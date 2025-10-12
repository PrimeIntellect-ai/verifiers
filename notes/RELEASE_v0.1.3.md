# Verifiers v0.1.3 Release Notes

*Date:* 8/26/25

Verifiers v0.1.3 adds a number of features for expanded functionality and ease of use, along with additional library integrations and bug fixes.

## Highlights

- **We now have a TUI!** 🎉 Run `vf-tui` to interactively browse all locally-saved evaluation results in your terminal. 
- Overhauled logging for `vf-eval` evaluation results with tagged JSON artifact folders.
    - Defaults to saving in your environment's project directory under `outputs/` if developing locally; `./outputs` if using an environment installed from elsewhere.
    - The short-lived Markdown report outputs are now deprecated.
- Multimodal-input tasks are supported for evaluation (see `environments/mmmu` for an example)! Official trainer support in verifiers is pending, and can be accessed via HUD's [hud-vf-gym](https://github.com/hud-evals/hud-python/tree/main/rl) project.
- Optional `async` for reward functions, tools, and Environment class methods
    - `maybe_await` pattern for safe accommodation of both sync and async functions
    - Sync extensions of `env_response` and `is_completed` in MultiTurnEnv will work, but with a type warning; users are encouraged to migrate these functions to async for ongoing usage.
- Full JSON sampling args in `vf-eval` via `-S` (#240). 
- Official community examples library under very active development: [prime-environments](https://github.com/PrimeIntellect-ai/prime-environments)
- Native `init`/`push`/`pull`/`install` support in [prime-cli](https://github.com/PrimeIntellect-ai/prime-cli) (and more...)
    - Run `uv tool install prime` for a preview 🙂
- Feature-complete support for training and online evaluations in [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl).
- Improved caching and parallelization for JudgeRubric.
- Bug fixes for tool call sanitization and saving datasets to Huggingface
- Improvements to documentation.
- From the recent `0.1.2.post1` pre-release version: 
    - `StatefulToolEnv` for intercepting function calls for routing and state management (#224)
    - Improved lazy imports for efficient evaluation.
    - Overhauled `MathRubric` for `math-verify` as default reward.
    - Full support restored for `completions` generation (#201, #196).
- New required dependencies since `0.1.2`: `rich`, `textual`, `jinja`.

Thanks to everyone who contributed to this release!
- @laksyaag (#240, #241)
- @cat-state (#238)
- @qgallouedec (#218, #217)
- @vgel (#201, #196)
- @nathom (#200)
- @snellingio (#195, #194)
- @MarwanMashra (#184)
- @alanxmay 
And a special thanks to the entire Prime Intellect team, with PRs this cycle from:
- @JannikSt
- @mikasenghaas
- @samsja

Stay tuned for some big announcements in the coming days 😊

**Full Changelog**: https://github.com/willccbb/verifiers/compare/v0.1.2...v0.1.3