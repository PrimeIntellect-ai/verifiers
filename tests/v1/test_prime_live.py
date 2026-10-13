"""Live checks against the Prime API: a digest-cached image build in the configured team's
registry, a label sweep, and a chunked transfer through a real sandbox. Each creates and
removes its own resources under a unique label or tag.

    PRIME_API_KEY=... uv run pytest tests/v1/test_prime_live.py -m prime -s
"""

import hashlib
import os
import time
import uuid

import pytest

from verifiers.v1.runtimes import sweep
from verifiers.v1.runtimes.prime import PrimeConfig, PrimeRuntime

pytestmark = [pytest.mark.e2e, pytest.mark.prime]

TRANSFER_BYTES = 300 * 1024 * 1024


def _label() -> str:
    return f"vf-live-{uuid.uuid4().hex[:10]}"


async def test_sweep_deletes_a_labelled_sandbox():
    from prime_sandboxes import AsyncSandboxClient

    label = _label()
    runtime = PrimeRuntime(PrimeConfig(labels=[label]))
    started = time.monotonic()
    try:
        await runtime.start()
        print(f"\nsandbox {runtime.info.id} up in {time.monotonic() - started:.0f}s")
        swept = time.monotonic()
        report = await sweep.sweep([label])
        print(f"sweep {report} in {time.monotonic() - swept:.1f}s")
        assert report.sandboxes == {label: 1}
        client = AsyncSandboxClient()
        try:
            listing = await client.list(labels=[label], exclude_terminated=True)
        finally:
            await client.aclose()
        assert [s.id for s in listing.sandboxes] == []
    finally:
        await runtime.stop()


async def test_chunked_transfer_round_trips_300_mib():
    label = _label()
    runtime = PrimeRuntime(PrimeConfig(labels=[label], disk=5))
    data = os.urandom(TRANSFER_BYTES)
    digest = hashlib.sha256(data).hexdigest()
    try:
        started = time.monotonic()
        await runtime.start()
        print(f"\nsandbox {runtime.info.id} up in {time.monotonic() - started:.0f}s")
        wrote = time.monotonic()
        await runtime.write("blob.bin", data)
        print(f"wrote {TRANSFER_BYTES >> 20} MiB in {time.monotonic() - wrote:.0f}s")
        in_box = await runtime.run(["sha256sum", "/app/blob.bin"], {})
        assert in_box.stdout.split()[0] == digest
        read = time.monotonic()
        back = await runtime.read("blob.bin")
        print(f"read {len(back) >> 20} MiB in {time.monotonic() - read:.0f}s")
        assert hashlib.sha256(back).hexdigest() == digest
        leftovers = await runtime.run(["sh", "-c", "ls /app"], {})
        assert leftovers.stdout.split() == ["blob.bin"]
    finally:
        await runtime.stop()
        await sweep.sweep([label])
