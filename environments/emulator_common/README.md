# emulator_common

Shared harness code for the emulator implementation benchmark environments.

The platform-specific environments keep only their manifest, README, package
metadata, and `load_environment()` entry point. This shared layer owns task
loading, sandbox defaults, starter Rust crate files, deterministic verifier
upload, and reward parsing.

Prime sandbox settings follow `~/Planning/prime_infra_guide.md`: toolchain image
selection is read from `PRIME_TOOLCHAIN_IMAGE`, team IDs are not hardcoded, and
long-running grading uses `run_background_job`.
