"""Load test of the host-wide request gate: many processes, each with many concurrent
requests, through one gate.

    uv run python scripts/request_gate_load.py --processes 100 --waiters 200 \
        --cap 2000 --hold 0.5 2.0 --seconds 60

Each process runs `--waiters` coroutines that loop: acquire a permit, hold it for a time
drawn from `--hold` (a stand-in for the upstream request), release. The parent samples the
daemon's state while they run and reports the admission rate, the most in flight against
the cap, the queue's length, and how long admissions waited (p50/p99/max, and per process,
so a starved process shows).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

WORKER = """
import asyncio, json, random, sys, time
from pathlib import Path
from verifiers.v1.clients.gate import ModelGate
from verifiers.v1.configs.client import RequestGateConfig

directory, waiters, cap, low, high, start, stop = sys.argv[1:]
waiters, cap, low, high, start, stop = int(waiters), int(cap), float(low), float(high), float(start), float(stop)
config = RequestGateConfig(max_concurrent=cap, adaptive=False)

async def main():
    gate = ModelGate(Path(directory), "load", "load-model")
    print("ready", flush=True)
    await asyncio.sleep(max(0.0, start - time.time()))
    waits, inside, held, peak = [], [], 0, 0

    async def loop():
        nonlocal held, peak
        while time.time() < stop:
            asked = time.monotonic()
            slot = await gate.acquire(config)
            waits.append(time.monotonic() - asked)
            if time.time() <= stop:
                inside.append(1)
            held += 1
            peak = max(peak, held)
            try:
                await asyncio.sleep(random.uniform(low, high))
            finally:
                held -= 1
                gate.release(slot, config, "ok")

    await asyncio.gather(*(loop() for _ in range(waiters)))
    print(json.dumps({"admitted": len(inside), "waits": waits, "peak": peak}), flush=True)

asyncio.run(main())
"""


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


async def run(args: argparse.Namespace) -> dict:
    from verifiers.v1.clients import gate as gate_module

    directory = Path(tempfile.mkdtemp(prefix="gate-load-")) / "gates"
    directory.mkdir()
    # start time is set once every worker has imported and registered
    ready_by = time.time() + args.startup
    start = ready_by
    stop = start + args.seconds
    workers = [
        await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            WORKER,
            str(directory),
            str(args.waiters),
            str(args.cap),
            str(args.hold[0]),
            str(args.hold[1]),
            str(start),
            str(stop),
            stdout=subprocess.PIPE,
        )
        for _ in range(args.processes)
    ]
    for worker in workers:
        assert worker.stdout is not None
        line = await worker.stdout.readline()
        assert line == b"ready\n", line
    if time.time() > start:
        raise SystemExit(f"workers took longer than --startup {args.startup}s to start")
    samples = []

    async def sample() -> None:
        while time.time() < stop + args.hold[1] + 1:
            states = await asyncio.to_thread(gate_module.gate_states, directory)
            if states:
                samples.append((time.time(), states[0]))
            await asyncio.sleep(0.25)

    sampler = asyncio.create_task(sample())

    async def read(worker: asyncio.subprocess.Process) -> bytes:
        assert worker.stdout is not None
        return await worker.stdout.read()

    outputs = await asyncio.wait_for(
        asyncio.gather(*(read(worker) for worker in workers)),
        args.seconds + args.hold[1] + 600,
    )
    codes = [await worker.wait() for worker in workers]
    sampler.cancel()
    results = [json.loads(out) for out in outputs]
    waits = [w for r in results for w in r["waits"]]
    admitted = sum(r["admitted"] for r in results)
    during = [s for t, s in samples if start + 5 <= t <= stop]
    in_flight = [s["in_flight"] for _, s in samples]
    queued = [s["waiting"] for s in during]
    per_process_p99 = [quantile(r["waits"], 0.99) for r in results]
    return {
        "processes": args.processes,
        "waiters_per_process": args.waiters,
        "cap": args.cap,
        "hold_seconds": args.hold,
        "seconds": args.seconds,
        "exit_codes_ok": all(code == 0 for code in codes),
        "admitted": admitted,
        "admissions_per_second": round(admitted / args.seconds, 1),
        "ideal_per_second": round(args.cap / statistics.mean(args.hold), 1),
        "max_in_flight": max(in_flight, default=0),
        "mean_in_flight": round(statistics.mean(s["in_flight"] for s in during), 1)
        if during
        else None,
        "mean_queued": round(statistics.mean(queued), 1) if queued else None,
        "max_queued": max(queued, default=0),
        "wait_p50": round(quantile(waits, 0.5), 3),
        "wait_p99": round(quantile(waits, 0.99), 3),
        "wait_max": round(max(waits, default=0.0), 3),
        "per_process_wait_p99_min_max": [
            round(min(per_process_p99), 3),
            round(max(per_process_p99), 3),
        ],
        "per_process_admitted_min_max": [
            min(r["admitted"] for r in results),
            max(r["admitted"] for r in results),
        ],
        "daemon_pid": int((directory / "daemon.pid").read_text()),
        "directory": str(directory),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--processes", type=int, default=100)
    parser.add_argument("--waiters", type=int, default=200)
    parser.add_argument("--cap", type=int, default=2000)
    parser.add_argument("--hold", type=float, nargs=2, default=(0.5, 2.0))
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--startup", type=float, default=60.0)
    args = parser.parse_args()
    random.seed(os.getpid())
    print(json.dumps(asyncio.run(run(args)), indent=2))


if __name__ == "__main__":
    main()
