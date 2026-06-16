"""terminal-bench-2-v1 — the harbor-v1 taskset pinned to Terminal-Bench 2 (example env).

A thin wrapper over `harbor-v1`: pins `dataset` to "terminal-bench/terminal-bench-2". Needs
the `harbor` CLI (`uv tool install harbor`) and a container runtime (docker/prime).
"""

from typing import Literal

import verifiers.v1 as vf
from tasksets.harbor_v1 import HarborConfig, HarborTask, HarborTaskset


class TerminalBench2Config(HarborConfig):
    dataset: Literal["terminal-bench/terminal-bench-2"] = (
        "terminal-bench/terminal-bench-2"
    )


class TerminalBench2Taskset(
    HarborTaskset, vf.Taskset[HarborTask, TerminalBench2Config]
):
    pass
