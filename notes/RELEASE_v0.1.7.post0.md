# Verifiers v0.1.7 Release Notes

*Date:* 11/7/25

**Update 0.1.7.post0**: Quick bug fix for `tmux` session setup with `prime-rl`.

Verifiers v0.1.7 features a major overhaul of the included trainer (now called `vf.RLTrainer`), as well as a number of utilities for streamlining the usage of Verifiers with Prime-RL, along with a number of bug fixes, documentation updates, and QoL improvements. 

## Highlights
- Auto-setup of `prime-rl` and config file initialization via the `vf-setup` command (`uv run vf-setup`).
- New `vf.RLTrainer` trainer for single-node training with dense models, overhauled to mirror the algorithmic features of `prime-rl` while keeping the core logic minimal and hackable.
- Updates to the `wiki-search` environment, which is now our primary example for multi-turn tool use.
- Support via `prime-rl` for:
    - Multi-node training
    - LoRA + FFT training (also in `vf-rl`)
    - Training with multiple environments
    - Online evals in with multiple environments
    - Single-file TOML configuration (also in `vf-rl`)
    - Auto-installation of environments from training configs (also in `vf-rl`)
    - Online difficulty buffer filtering
    - PipelineRL/AReaL-style dynamic weight updates
- Refactors of some internal methods to avoid redundancy and simplify downstream integrations.
- Miscellaneous bug fixes, particularly related to `EnvGroup` and Pydantic serialization.

Full changelog: https://github.com/primeintellect-ai/verifiers/compare/v0.1.6...v0.1.7