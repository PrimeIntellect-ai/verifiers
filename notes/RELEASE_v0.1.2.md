# What's changed

With the `v0.1.2` release, `verifiers` is significantly more production-ready, and stable to build and train with. We appreciate everyone's patience with the changes and bug fixes thus far as we've addressed a number of long-time requests, and are excited to see what you all build with it! 

Highlights:
- Proper encapsulation of Environments as standalone modules (see `environments/`), which can contain their own dependencies in a `pyproject.toml`, and need only to expose a `load_environment(...) -> vf.Environment` function in order to be trainable. 
- Script flows for initializing (`vf-init`), installing (`vf-install`), and evaluating (`vf-eval`) Environments before training.
- Reorganization of examples and training scripts, removing lots of duplicated logic and creating a cleaner separation between library code and example code.
- Deprecation of the manual dynamically-batched `LLM` inference worker in favor of proper `AsyncLLM` support, allowing full control of native vLLM sampling parameters. 
- Support for native tool call parsing + parallel tool calls in `ToolEnv` (replacing the manual `XMLParser` approach).
- Another trainer! Environments built with `verifiers` are now trainable with `prime-rl` (as of [58ac91f](https://github.com/PrimeIntellect-ai/prime-rl/commit/58ac91fd3e19968e33c12f255de446d959982062) for `v0.1.2`), which supports multi-node FSDP async training, is the primary RL framework used by the Prime Intellect research team, and is under ongoing development and stress-testing in advance of large-scale multi-environment training runs. 
- Pydantic types for core data classes used by Environments.
- Improvements to `GRPOTrainer`, including supporting a single `max_seq_len` option (instead of separate prompt + completion lengths), and configurable turn length limits via `max_tokens`.
- Many more Environment examples.
- Improved logging and evaluation options.
- Overhauled README.md and [docs](https://verifiers.readthedocs.io/en/latest/).