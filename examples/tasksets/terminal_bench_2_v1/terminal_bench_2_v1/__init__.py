"""terminal-bench-2-v1 — the harbor taskset pinned to Terminal-Bench 2 (example env).

A thin wrapper over `harbor`: pins `dataset` to "terminal-bench/terminal-bench-2". Needs the
`harbor` CLI (`uv tool install harbor`) and a container runtime (docker/prime), like harbor.
"""

from typing import Literal

from harbor import HarborConfig, HarborTaskset


class TerminalBench2Config(HarborConfig):
    dataset: Literal["terminal-bench/terminal-bench-2"] = (
        "terminal-bench/terminal-bench-2"
    )


def load_taskset(config: TerminalBench2Config) -> HarborTaskset:
    return HarborTaskset(config)
