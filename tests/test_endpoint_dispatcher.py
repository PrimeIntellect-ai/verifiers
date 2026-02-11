import asyncio
import sys

import pytest

from verifiers.types import ClientConfig
from verifiers.utils.async_utils import (
    EndpointDispatcher,
    EndpointVariantSlot,
    NullEndpointDispatcher,
)


def _make_config(url: str = "https://a.example/v1") -> ClientConfig:
    return ClientConfig(api_base_url=url)


class TestEndpointVariantSlot:
    def test_available_reflects_capacity(self):
        slot = EndpointVariantSlot(config=_make_config(), max_concurrent=10)
        assert slot.available == 10
        slot.active = 3
        assert slot.available == 7


class TestEndpointDispatcher:
    @pytest.mark.asyncio
    async def test_least_loaded_picks_emptier_variant(self):
        slot_a = EndpointVariantSlot(
            config=_make_config("https://a.example/v1"), max_concurrent=4
        )
        slot_b = EndpointVariantSlot(
            config=_make_config("https://b.example/v1"), max_concurrent=4
        )
        dispatcher = EndpointDispatcher([slot_a, slot_b])

        # Acquire one on each — first should go to whichever has more available
        async with dispatcher.acquire() as got1:
            assert got1 in (slot_a, slot_b)
            # got1 now has 1 active, the other has 0
            other = slot_b if got1 is slot_a else slot_a
            async with dispatcher.acquire() as got2:
                # Should pick the one with more available (the other)
                assert got2 is other

    @pytest.mark.asyncio
    async def test_blocks_when_all_full_then_unblocks(self):
        slot = EndpointVariantSlot(config=_make_config(), max_concurrent=1)
        dispatcher = EndpointDispatcher([slot])

        acquired = asyncio.Event()
        released = asyncio.Event()

        async def holder():
            async with dispatcher.acquire():
                acquired.set()
                await released.wait()

        async def waiter():
            await acquired.wait()
            # This should block until holder releases
            async with dispatcher.acquire() as got:
                assert got is slot

        holder_task = asyncio.create_task(holder())
        waiter_task = asyncio.create_task(waiter())

        # Give tasks time to start
        await asyncio.sleep(0.05)
        assert acquired.is_set()
        assert not waiter_task.done()

        # Release the holder
        released.set()
        await asyncio.wait_for(waiter_task, timeout=2.0)
        await holder_task

    @pytest.mark.asyncio
    async def test_count_parameter_consumes_correct_slots(self):
        slot = EndpointVariantSlot(config=_make_config(), max_concurrent=4)
        dispatcher = EndpointDispatcher([slot])

        async with dispatcher.acquire(count=3) as got:
            assert got is slot
            assert slot.active == 3
            assert slot.available == 1

        assert slot.active == 0
        assert slot.available == 4

    @pytest.mark.asyncio
    async def test_releases_on_exception(self):
        slot = EndpointVariantSlot(config=_make_config(), max_concurrent=2)
        dispatcher = EndpointDispatcher([slot])

        with pytest.raises(RuntimeError, match="boom"):
            async with dispatcher.acquire():
                raise RuntimeError("boom")

        assert slot.active == 0

    @pytest.mark.asyncio
    async def test_oversize_count_waits_for_full_idle(self):
        """When count exceeds every variant's max_concurrent, wait for idle."""
        slot = EndpointVariantSlot(config=_make_config(), max_concurrent=2)
        dispatcher = EndpointDispatcher([slot])

        released = asyncio.Event()

        async def holder():
            async with dispatcher.acquire(count=1):
                await released.wait()

        holder_task = asyncio.create_task(holder())
        await asyncio.sleep(0.05)

        # count=5 exceeds max_concurrent=2 — must wait for full idle
        async def oversize():
            async with dispatcher.acquire(count=5) as got:
                assert got is slot

        oversize_task = asyncio.create_task(oversize())
        await asyncio.sleep(0.05)
        assert not oversize_task.done()

        released.set()
        await asyncio.wait_for(oversize_task, timeout=2.0)
        await holder_task

    def test_empty_variants_raises(self):
        with pytest.raises(ValueError):
            EndpointDispatcher([])


class TestNullEndpointDispatcher:
    @pytest.mark.asyncio
    async def test_round_robins_without_blocking(self):
        configs = [
            _make_config("https://a.example/v1"),
            _make_config("https://b.example/v1"),
            _make_config("https://c.example/v1"),
        ]
        dispatcher = NullEndpointDispatcher(configs)

        urls = []
        for _ in range(6):
            async with dispatcher.acquire() as slot:
                urls.append(slot.config.api_base_url)

        assert urls == [
            "https://a.example/v1",
            "https://b.example/v1",
            "https://c.example/v1",
            "https://a.example/v1",
            "https://b.example/v1",
            "https://c.example/v1",
        ]

    @pytest.mark.asyncio
    async def test_slot_has_maxsize_capacity(self):
        dispatcher = NullEndpointDispatcher([_make_config()])
        async with dispatcher.acquire() as slot:
            assert slot.max_concurrent == sys.maxsize

    def test_empty_configs_raises(self):
        with pytest.raises(ValueError):
            NullEndpointDispatcher([])
