"""Benchmark the persistent runtime + warm scoring against the per-rollout default.

Isolates the runtime/scoring overhead (no model generation, which would mask it): runs gsm8k's
`verify.py` N times at a fixed concurrency, both ways, through the real framework API.

  ephemeral:  make_runtime + start() + run_uv_script(cold) + stop() per call
              — exactly today's per-rollout scoring path.
  persistent: RuntimePool.acquire() + run_uv_script(warm=True) + release() per call
              — runtimes (and their warm workers) reused; the math_verify import is paid once
              per pooled runtime, not per call.

Usage: uv run python bench/persistent_runtime.py [N=1000] [CONCURRENCY=128]
"""

import asyncio
import sys
import time
from pathlib import Path

from verifiers.v1.runtimes import RuntimePool, SubprocessConfig, make_runtime

VERIFY = (
    Path(__file__).resolve().parents[1] / "environments/gsm8k_v1/gsm8k_v1/verify.py"
).read_bytes()
GOLD, PRED = "42", "#### 42"


async def ephemeral(n: int, concurrency: int) -> float:
    sem = asyncio.Semaphore(concurrency)

    async def one() -> None:
        async with sem:
            rt = make_runtime(SubprocessConfig())
            await rt.start()
            try:
                r = await rt.run_uv_script(VERIFY, args=[GOLD, PRED])
                assert r.stdout.strip().endswith("1.0"), r
            finally:
                await rt.stop()

    t0 = time.time()
    await asyncio.gather(*(one() for _ in range(n)))
    return time.time() - t0


async def persistent(n: int, concurrency: int) -> float:
    sem = asyncio.Semaphore(concurrency)
    cfg = SubprocessConfig(persistent=True)

    async def one(pool: RuntimePool) -> None:
        async with sem:
            rt = await pool.acquire(cfg)
            try:
                r = await rt.run_uv_script(VERIFY, args=[GOLD, PRED], warm=True)
                assert r.stdout.strip() == "1.0", r
            finally:
                await pool.release(rt)

    async with RuntimePool() as pool:
        t0 = time.time()
        await asyncio.gather(*(one(pool) for _ in range(n)))
        return time.time() - t0


async def main(n: int, concurrency: int) -> None:
    # warm the class-level interpreter cache once so neither mode pays uv-resolve
    rt = make_runtime(SubprocessConfig())
    await rt.start()
    await rt.run_uv_script(VERIFY, args=[GOLD, PRED])
    await rt.stop()

    ephemeral_dt = await ephemeral(n, concurrency)
    persistent_dt = await persistent(n, concurrency)

    print(f"gsm8k verify scoring — n={n} concurrency={concurrency}")
    print(
        f"  ephemeral (runtime + import per call): {ephemeral_dt:7.2f}s  ({1000 * ephemeral_dt / n:6.2f} ms/call)"
    )
    print(
        f"  persistent (pool + warm worker):       {persistent_dt:7.2f}s  ({1000 * persistent_dt / n:6.2f} ms/call)"
    )
    print(
        f"  speedup:                               {ephemeral_dt / persistent_dt:6.1f}x"
    )


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    concurrency = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    asyncio.run(main(n, concurrency))
