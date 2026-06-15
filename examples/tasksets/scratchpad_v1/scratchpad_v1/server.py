"""scratchpad tool server — DELIBERATELY stateful, to exercise per-rollout isolation.

Shared across the whole eval (built once via `ToolsConfig(shared=True)`), it holds mixed
state — a module-level dict AND a file on disk — under a single FIXED slot, so concurrent
rollouts clobber each other unless each is scoped to its own rollout. The server multiplexes
itself by `vf.current_rollout_id()` (the framework injects the rollout id into each request's
URL). Set SCRATCHPAD_MULTIPLEX=0 to disable namespacing and watch rollouts corrupt each other.

With SCRATCHPAD_AUTOFORK=1 the server stays "dumb" (no namespacing) and isolation is handled
*automatically* by `vf.run_mcp_server(multiplex=True)` — a fork-per-rollout multiplexer gives
each rollout its own forked child (copy-on-write memory + private CWD), so ordinary stateful
code is isolated with no rollout-aware logic at all.
"""

import asyncio
import os
import time
from pathlib import Path

import verifiers.v1 as vf
from mcp.server.fastmcp import FastMCP

# Simulated expensive setup — paid ONCE for the shared server (proves "built once": this runs
# a single time for the whole eval, not per rollout).
print("scratchpad: starting expensive setup...", flush=True)
time.sleep(float(os.environ.get("SCRATCHPAD_SETUP_SECONDS", "2")))
print("scratchpad: setup complete", flush=True)

_MULTIPLEX = os.environ.get("SCRATCHPAD_MULTIPLEX", "1") == "1"
# When set, serve as a fork-per-rollout multiplexer (vf.run_mcp_server(multiplex=True)): the
# server stays "dumb" (no namespacing) and the framework isolates each rollout by forking.
_AUTOFORK = os.environ.get("SCRATCHPAD_AUTOFORK", "0") == "1"
# Optional barrier: make every concurrent rollout finish writing before any reads, so a
# non-namespaced shared slot is *deterministically* clobbered (independent of model timing).
# Set SCRATCHPAD_BARRIER=<num concurrent rollouts> for the corruption demo; 0 = just sleep.
# Skipped under autofork: each rollout is its own process, so there's no shared slot to force.
_BARRIER_N = int(os.environ.get("SCRATCHPAD_BARRIER", "0"))
_MEM: dict[str, str] = {}
# Under autofork each child has a private CWD, so a RELATIVE path is isolated for free;
# otherwise an absolute shared dir (collides across rollouts unless namespaced by rollout id).
_BASE = Path() if _AUTOFORK else Path(os.environ.get("SCRATCHPAD_DIR", "/tmp/vf_scratchpad"))
if not _AUTOFORK:
    _BASE.mkdir(parents=True, exist_ok=True)
_FIXED = "slot"  # one fixed slot — collides across rollouts unless isolated
_written = 0
_all_written = asyncio.Event()

mcp = FastMCP("scratchpad")


def _ns() -> str:
    """Per-rollout namespace from the framework-injected id (or a single shared bucket when
    multiplexing is disabled, which is the corruption-demo case)."""
    rid = vf.current_rollout_id() if _MULTIPLEX else None
    return rid or "_shared"


@mcp.tool()
async def roundtrip(word: str) -> str:
    """Store `word` in the scratchpad, then read it back and return `<mem>|<disk>`."""
    global _written
    ns = _ns()
    mem_key = f"{ns}/{_FIXED}"
    disk_path = _BASE / f"{ns}.{_FIXED}"
    _MEM[mem_key] = word
    disk_path.write_text(word)
    if _BARRIER_N and not _AUTOFORK:
        _written += 1
        if _written >= _BARRIER_N:
            _all_written.set()
        try:
            await asyncio.wait_for(_all_written.wait(), timeout=15)
        except asyncio.TimeoutError:
            pass
    else:
        await asyncio.sleep(0.5)  # interleave window for concurrent writers
    return f"{_MEM[mem_key]}|{disk_path.read_text()}"


vf.run_mcp_server(mcp, multiplex=_AUTOFORK)
