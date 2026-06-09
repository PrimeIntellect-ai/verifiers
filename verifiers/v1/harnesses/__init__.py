"""Harnesses: the harness abstraction (`Harness` + `HarnessConfig`).

A harness is the program that runs in a runtime and drives the conversation. Core defines
only the contract here; concrete harnesses are ordinary packages resolved by id via a
`load_harness(config) -> Harness` hook (the shipped `default`/`rlm` live under `packages/`,
custom ones like `compact` under `examples/`). A harness config subclasses `HarnessConfig`
(mirroring how a taskset config subclasses `TasksetConfig`); the concrete type is resolved
by id — narrowed for the CLI, built by `loaders.load_harness` (the mirror of `load_taskset`).
"""

from verifiers.v1.harnesses.base import Harness, HarnessConfig

__all__ = ["Harness", "HarnessConfig"]
