"""A SHARED, writable tool server that exercises per-rollout state isolation.

It is built once for the whole eval (`shared=True`) but written to by every rollout: `roundtrip`
stores a word and reads it back. With `isolate=True` (the default) each rollout's write lands in its
own `self.state` — the per-rollout shared-state channel the framework tags onto a shared server's URL
— so concurrent rollouts never see each other's word. Set `isolate=False` to bypass `self.state` and
write to a process-global slot instead: with one shared process, concurrent rollouts then clobber that
single slot and read back the wrong word — unless `fork=True` gives each rollout its own process. So
`isolate=False, fork=True` is the test that fork (not `self.state`) provides the isolation.
"""

import asyncio

import verifiers.v1 as vf


class ScratchpadState(vf.State):
    slot: str = ""


class ScratchpadToolsetConfig(vf.ToolsetConfig):
    isolate: bool = True
    """Write per-rollout state to `self.state` (isolated by the framework). Set False to write a
    process-global slot instead — corrupts across concurrent rollouts on a shared server unless
    `fork=True`. A config field (not an env var) so it reaches the server in any runtime (it crosses
    in `VF_CONFIG`), including a docker/prime sandbox."""
    setup_seconds: float = 0.0
    """Simulated expensive one-time `setup` cost (a shared server pays it once for the whole eval)."""


# A process-global slot — shared across every rollout on the one shared process. Used only by the
# `isolate=False` corruption demo; the isolated default writes to `self.state` instead.
_GLOBAL_SLOT = {"value": ""}


class ScratchpadToolset(vf.Toolset[ScratchpadToolsetConfig, ScratchpadState]):
    TOOL_PREFIX = "scratchpad"

    async def setup(self) -> None:
        # Stand in for an expensive one-time build (corpus/index): paid once for the shared server.
        await asyncio.sleep(self.config.setup_seconds)

    @vf.tool
    async def roundtrip(self, word: str) -> str:
        """Store `word` in the scratchpad, then read it back and return it."""
        if self.config.isolate:
            self.state.slot = word
            await asyncio.sleep(0.5)  # interleave window for concurrent rollouts
            return self.state.slot
        _GLOBAL_SLOT["value"] = word
        await asyncio.sleep(0.5)
        return _GLOBAL_SLOT["value"]


if __name__ == "__main__":
    ScratchpadToolset.run()
