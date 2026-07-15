"""GEPA (Genetic-Pareto) system-prompt optimization for native v1 environments.

`run_gepa(env, config)` drives the third-party `gepa` optimizer against a v1 `Environment`
via `GEPAAdapter`, seeding from and improving the taskset config-layer system prompt
(`--taskset.system-prompt` / `--taskset.system-prompt-file`). The `gepa` console script
(`verifiers.v1.cli.gepa`) is a thin entrypoint over this package.
"""

from verifiers.v1.gepa.adapter import GEPAAdapter
from verifiers.v1.gepa.config import GEPAConfig
from verifiers.v1.gepa.runner import run_gepa

__all__ = ["GEPAAdapter", "GEPAConfig", "run_gepa"]
