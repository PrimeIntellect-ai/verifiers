"""Tests for update_variants() and ElasticEndpointPool."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from verifiers.types import ClientConfig, EndpointClientConfig
from verifiers.utils.async_utils import EndpointSlot, LeastLoadedDispatcher
from verifiers.utils.elastic import ElasticEndpointPool


def _make_config(url: str = "https://a.example/v1") -> ClientConfig:
    return ClientConfig(api_base_url=url)


def _make_slot(url: str, max_concurrent: int = 4) -> EndpointSlot:
    return EndpointSlot(config=_make_config(url), max_concurrent=max_concurrent)


# ---------------------------------------------------------------------------
# update_variants tests
# ---------------------------------------------------------------------------


class TestUpdateVariants:
    @pytest.mark.asyncio
    async def test_adds_new(self):
        slot_a = _make_slot("https://a.example/v1")
        dispatcher = LeastLoadedDispatcher([slot_a])

        slot_b = _make_slot("https://b.example/v1")
        added, removed = await dispatcher.update_variants(
            [_make_slot("https://a.example/v1"), slot_b]
        )

        assert added == 1
        assert removed == 0

    @pytest.mark.asyncio
    async def test_removes(self):
        slot_a = _make_slot("https://a.example/v1")
        slot_b = _make_slot("https://b.example/v1")
        dispatcher = LeastLoadedDispatcher([slot_a, slot_b])

        added, removed = await dispatcher.update_variants(
            [_make_slot("https://a.example/v1")]
        )

        assert added == 0
        assert removed == 1

    @pytest.mark.asyncio
    async def test_preserves_active(self):
        slot_a = _make_slot("https://a.example/v1", max_concurrent=4)
        dispatcher = LeastLoadedDispatcher([slot_a])

        # Simulate in-flight request
        async with dispatcher.acquire() as got:
            assert got.active == 1

            # Update with same URL — should preserve the slot object and active count
            added, removed = await dispatcher.update_variants(
                [_make_slot("https://a.example/v1", max_concurrent=8)]
            )
            assert added == 0
            assert removed == 0
            # Active count preserved on the original object
            assert got.active == 1
            # max_concurrent updated
            assert got.max_concurrent == 8

    @pytest.mark.asyncio
    async def test_wakes_waiters(self):
        slot_a = _make_slot("https://a.example/v1", max_concurrent=1)
        dispatcher = LeastLoadedDispatcher([slot_a])

        acquired = asyncio.Event()
        unblocked = asyncio.Event()

        async def holder():
            async with dispatcher.acquire():
                acquired.set()
                # Hold slot while waiter tries to acquire
                await unblocked.wait()

        async def waiter():
            await acquired.wait()
            # This blocks because slot_a is full, but adding slot_b should unblock it
            async with dispatcher.acquire() as got:
                assert got.config.api_base_url == "https://b.example/v1"
                unblocked.set()

        holder_task = asyncio.create_task(holder())
        waiter_task = asyncio.create_task(waiter())

        await acquired.wait()
        await asyncio.sleep(0.05)  # let waiter block

        # Add a new endpoint — should wake the waiter
        await dispatcher.update_variants(
            [
                _make_slot("https://a.example/v1", max_concurrent=1),
                _make_slot("https://b.example/v1", max_concurrent=1),
            ]
        )

        await asyncio.wait_for(waiter_task, timeout=2.0)
        await holder_task

    @pytest.mark.asyncio
    async def test_rejects_empty(self):
        dispatcher = LeastLoadedDispatcher([_make_slot("https://a.example/v1")])
        with pytest.raises(ValueError, match="at least one variant"):
            await dispatcher.update_variants([])


# ---------------------------------------------------------------------------
# ElasticEndpointPool tests
# ---------------------------------------------------------------------------


def _write_endpoints_toml(path: Path, entries: list[dict]) -> None:
    """Write a minimal endpoints.toml file."""
    lines: list[str] = []
    for entry in entries:
        lines.append("[[endpoint]]")
        for k, v in entry.items():
            if isinstance(v, int):
                lines.append(f'{k} = {v}')
            else:
                lines.append(f'{k} = "{v}"')
        lines.append("")
    path.write_text("\n".join(lines))


class TestElasticEndpointPool:
    @pytest.mark.asyncio
    async def test_reload_updates_dispatcher(self, tmp_path: Path):
        toml_file = tmp_path / "endpoints.toml"
        _write_endpoints_toml(
            toml_file,
            [
                {
                    "endpoint_id": "my-ep",
                    "url": "https://a.example/v1",
                    "key": "KEY",
                    "model": "m1",
                    "max_concurrent": 4,
                },
            ],
        )

        # Build initial dispatcher
        slot = _make_slot("https://a.example/v1", max_concurrent=4)
        dispatcher = LeastLoadedDispatcher([slot])

        base_config = ClientConfig(
            api_key_var="KEY",
            api_base_url="https://a.example/v1",
            endpoint_configs=[
                EndpointClientConfig(
                    api_key_var="KEY",
                    api_base_url="https://a.example/v1",
                    max_concurrent=4,
                )
            ],
        )

        pool = ElasticEndpointPool(
            dispatcher=dispatcher,
            endpoints_path=str(toml_file),
            endpoint_id="my-ep",
            poll_interval=1.0,
            base_client_config=base_config,
        )

        # Now update the file with a second endpoint
        _write_endpoints_toml(
            toml_file,
            [
                {
                    "endpoint_id": "my-ep",
                    "url": "https://a.example/v1",
                    "key": "KEY",
                    "model": "m1",
                    "max_concurrent": 4,
                },
                {
                    "endpoint_id": "my-ep",
                    "url": "https://b.example/v1",
                    "key": "KEY",
                    "model": "m1",
                    "max_concurrent": 8,
                },
            ],
        )

        await pool._reload()

        # Dispatcher should now have 2 variants
        assert len(dispatcher._variants) == 2

    @pytest.mark.asyncio
    async def test_reload_failure_keeps_previous(self, tmp_path: Path):
        toml_file = tmp_path / "endpoints.toml"
        _write_endpoints_toml(
            toml_file,
            [
                {
                    "endpoint_id": "my-ep",
                    "url": "https://a.example/v1",
                    "key": "KEY",
                    "model": "m1",
                    "max_concurrent": 4,
                },
            ],
        )

        slot = _make_slot("https://a.example/v1", max_concurrent=4)
        dispatcher = LeastLoadedDispatcher([slot])

        base_config = ClientConfig(
            api_key_var="KEY",
            api_base_url="https://a.example/v1",
            endpoint_configs=[
                EndpointClientConfig(
                    api_key_var="KEY",
                    api_base_url="https://a.example/v1",
                    max_concurrent=4,
                )
            ],
        )

        pool = ElasticEndpointPool(
            dispatcher=dispatcher,
            endpoints_path=str(tmp_path / "nonexistent.toml"),
            endpoint_id="my-ep",
            poll_interval=1.0,
            base_client_config=base_config,
        )

        # _reload should not raise — it logs a warning and keeps previous
        await pool._reload()
        assert len(dispatcher._variants) == 1

    @pytest.mark.asyncio
    async def test_start_stop(self, tmp_path: Path):
        toml_file = tmp_path / "endpoints.toml"
        _write_endpoints_toml(
            toml_file,
            [
                {
                    "endpoint_id": "my-ep",
                    "url": "https://a.example/v1",
                    "key": "KEY",
                    "model": "m1",
                    "max_concurrent": 4,
                },
            ],
        )

        slot = _make_slot("https://a.example/v1", max_concurrent=4)
        dispatcher = LeastLoadedDispatcher([slot])

        base_config = ClientConfig(
            api_key_var="KEY",
            api_base_url="https://a.example/v1",
        )

        pool = ElasticEndpointPool(
            dispatcher=dispatcher,
            endpoints_path=str(toml_file),
            endpoint_id="my-ep",
            poll_interval=0.05,
            base_client_config=base_config,
        )

        assert pool._task is None
        pool.start()
        assert pool._task is not None
        assert not pool._task.done()

        await pool.stop()
        assert pool._task is None
