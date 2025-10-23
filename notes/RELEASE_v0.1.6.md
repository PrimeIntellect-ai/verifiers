<<<<<<< HEAD
# Release v0.1.6

This release is largely focused on improving the training experience with Verifiers, including the built-in trainer as well as features to enable additional training frameworks. The included trainer has been renamed to `vf.RLTrainer` (with an alias to `vf.GRPOTrainer` for backward-compatibility) and supports largely the same features as `GRPOTrainer`, though with a few notable improvements, deprecations, and renamings, and with an emphasis on LoRA-first training, streamlining the codebase, and improving hackability.

## Installing with train support

To install train dependencies, you can now simply run `uv add 'verifiers[train]'` to install the trainer and all dependencies (yes, including `flash-attn`).

## Configs and setup/training scripts (`vf-setup`, `vf-rl`, `vf-train`)

We've added a new CLI command, `vf-setup`, which can be used to quickly setup a Verifiers project with default configs:

```bash
uv run vf-setup
```

After running, you should see the following files in your project:

```bash
configs/
├── endpoints.py # your model endpoints for vf-eval
├── zero3.yaml # your accelerate config
└── rl/
    ├── gsm8k.toml # RL config for GSM8K
    └── wordle.toml # RL config for Wordle
```

We've added support for `toml` configs, and you can now run RL training with a single command:

```bash
uv run vf-rl @ configs/rl/wordle.toml # -s session-name
```

This will create a tmux session with the name `session-name`, and run the RL training in it. It will also install environments from the Environments Hub if your `env.id` is given as `user/env-id` in your config, or from your local project if given simply as `env-id`.

If you wish to bypass the tmux setup and run your inference and training scripts separately, you can still do:

```bash
uv run vf-vllm --model willcb/Qwen3-1.7B-Wordle --data-parallel-size 6 --enforce-eager
```
and then either run:
```bash
uv run vf-train @ configs/rl/wordle.toml
```
or custom Python scripts as before:
```bash
uv run accelerate launch --num-processes 2 --config-file configs/zero3.yaml train_wordle.py
```

## `vf.RLTrainer` ("GRPO-style" RL trainer)

`GRPOTrainer` has been renamed to `RLTrainer` (with an alias to `vf.GRPOTrainer` for backward-compatibility) and supports largely the same features as `GRPOTrainer`, though with a few improvements, deprecations, and :

Based on experimental findings from the broader research community, as well as other integrations, `RLTrainer` is now **LoRA-first**, though full-parameter finetuning is still supported. The default config (`vf.RLConfig()`)

We are deprecating the `vf.lora_defaults()` and `vf.grpo_defaults()` patterns, and are now encouraging users to simply pass `vf.RLConfig()` to the trainer directly, which includes all defaults for LoRA training (enabled by default).


The key batch size parameters are now:
- `rollouts_per_example` (alias for `num_generations`)
- `micro_batch_size` (alias for `per_device_train_batch_size`)
- `batch_size` (total rollouts per global batch, across GPUs and micro-batches)

Users should no longer set `gradient_accumulation_steps` directly, as it will be automatically computed based on `batch_size` and the number of available GPUs. 

We are now primarily exposing run length configuration options in terms of `steps` instead of `epochs`, and are not guaranteeing any behavior around epoch-based training.


Key deprecations:
- We no longer support reference models (or `beta > 0`).
- We no longer support `num_iterations > 1`.
- We have removed `epsilon_high` and `delta` in favor of a single `epsilon (=0.2)` for two-sided clipping.

Other notable changes:
- `vllm_importance_sampling` is now enabled by default (and required).
- LoRA is now enabled by default (`rank=8`), and the default learning rate is now 1e-5.
- Training always uses async level 1 (`max)

Going forward, the philosophy of `vf.RLTrainer` is to be minimal, opinionated, less configurable, more self-contained, and more hackable. We have removed `trl` as a trainer dependency, and import needed functionality directly from `transformers`, `peft`, `vllm`, `deepspeed`, and `accelerate`.

Our intention is *not* to support all algorithmic tweaks that appear in the literature. We are preferring to keep the codebase simple, and encourage users to experiment with their own modifications as necessary. 
Roughly, our criterion for adding new features is whether they should be defaults, i.e. if they unlock major new use cases, or are Pareto improvements over the existing recipe which ought to be broadly adopted.


=======
# Verifiers v0.1.6 Release Notes

*Date:* 10/20/25

Verifiers v0.1.6 primarly focuses on a refactor of the evaluation logic (`vf-eval`) and generation methods (in `vf.Environment`) to track more metadata, streamline duplicated logic, enable intermediate saving of generations, and allow for more flexible evaluation workflows (i.e. importable utilities in `verifiers.utils.eval_utils`). 

The main **breaking change** is that `vf.Environment.generate` and `vf.Environment.evaluate` are now async methods, with `generate_sync` and `evaluate_sync` included as synchronous wrappers. 

We are also migrating towards using the `state` object more explicitly to track information throughout rollouts; existing workflows should be unaffected, but we encourage users to migrate to the new `state` object for better tracking of information throughout rollouts, as eventually other arguments will be deprecated.


**Full Changelog**: https://github.com/willccbb/verifiers/compare/v0.1.5...v0.1.6
>>>>>>> main
