"""A SHARED, writable tool server that exercises per-rollout state isolation.

It is built once for the whole eval (taskset-scoped, `Taskset.tools`) but written to by every rollout: `roundtrip`
stores a word and reads it back. Each rollout's write lands in its own `self.state` — the per-rollout
shared-state channel the framework tags onto a shared server's URL — so concurrent rollouts never see
each other's word even though they share one process. `roundtrip` sleeps between the write and the
read to force an interleave window, so the reward (read back your OWN word) fails unless `self.state`
truly isolates each rollout.
"""

import asyncio

import verifiers.v1 as vf


class ScratchpadState(vf.State):
    slot: str = ""


class ScratchpadToolsetConfig(vf.SharedToolsetConfig):
    setup_seconds: float = 0.0
    """Simulated expensive one-time `setup` cost (a shared server pays it once for the whole eval)."""


class ScratchpadToolset(vf.Toolset[ScratchpadToolsetConfig, ScratchpadState]):
    TOOL_PREFIX = "scratchpad"

    async def setup(self) -> None:
        # Stand in for an expensive one-time build (corpus/index): paid once for the shared server.
        await asyncio.sleep(self.config.setup_seconds)

    @vf.tool
    async def roundtrip(self, word: str) -> str:
        """Store `word` in the scratchpad, then read it back and return it."""
        self.state.slot = word
        await asyncio.sleep(0.5)  # interleave window for concurrent rollouts
        return self.state.slot


if __name__ == "__main__":
    ScratchpadToolset.run()
