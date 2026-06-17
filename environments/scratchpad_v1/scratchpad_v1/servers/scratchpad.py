"""A SHARED, writable tool server that exercises per-rollout state isolation.

It is built once for the whole eval (`ToolsetConfig(shared=True)`) but written to by every rollout:
`roundtrip` stores a word and reads it back. With isolation (the default) each rollout's write lands
in its own `self.state` — the per-rollout shared-state channel the framework tags onto a shared
server's URL — so concurrent rollouts never see each other's word. Set `SCRATCHPAD_ISOLATE=0` to
bypass `self.state` and write to a process-global slot instead: with one shared process, concurrent
rollouts then clobber that single slot and read back the wrong word (the corruption isolation
prevents).
"""

import asyncio
import os

import verifiers.v1 as vf


class ScratchpadState(vf.State):
    slot: str = ""


# A process-global slot — shared across every rollout on the one shared process. Used only by the
# `SCRATCHPAD_ISOLATE=0` corruption demo; the isolated default writes to `self.state` instead.
_GLOBAL_SLOT = {"value": ""}


class ScratchpadToolset(vf.Toolset[vf.ToolsetConfig, ScratchpadState]):
    TOOL_PREFIX = "scratchpad"

    async def setup(self) -> None:
        # Stand in for an expensive one-time build (corpus/index): paid once for the shared server.
        await asyncio.sleep(float(os.environ.get("SCRATCHPAD_SETUP_SECONDS", "0")))

    @vf.tool
    async def roundtrip(self, word: str) -> str:
        """Store `word` in the scratchpad, then read it back and return it."""
        if os.environ.get("SCRATCHPAD_ISOLATE", "1") == "1":
            self.state.slot = word
            await asyncio.sleep(0.5)  # interleave window for concurrent rollouts
            return self.state.slot
        _GLOBAL_SLOT["value"] = word
        await asyncio.sleep(0.5)
        return _GLOBAL_SLOT["value"]


if __name__ == "__main__":
    ScratchpadToolset.run()
