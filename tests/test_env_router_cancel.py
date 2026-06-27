"""Regression: EnvRouter.forward_cancel must clear request bookkeeping (no leak).

A cancel that yields no terminal response would otherwise leave
``request_to_worker`` + the worker's ``active_requests`` populated forever — the
``complete_request`` cleaner runs only on the response path, which never fires
for a request cancelled before it responds. (Upstream #1055; surfaced in the
prime-rl #76 PR1 review as F3.)

Builds the router via ``__new__`` to skip the ZMQ-IPC ``__init__`` —
``forward_cancel`` only touches ``self.workers`` + ``self.request_to_worker``.
"""

import asyncio

from verifiers.serve.server.env_router import ActiveRequestInfo, EnvRouter, WorkerHandle


class FakeSock:
    def __init__(self) -> None:
        self.sent: list = []

    async def send_multipart(self, frames) -> None:
        self.sent.append(frames)


def _router_with_inflight_request():
    router = EnvRouter.__new__(EnvRouter)
    sock = FakeSock()
    router.workers = {
        0: WorkerHandle(worker_id=0, process=None, address="ipc://x", socket=sock)
    }
    router.request_to_worker = {}
    rid, cid = b"req-1", b"cli-1"
    router.workers[0].active_requests[rid] = ActiveRequestInfo(
        client_id=cid, request_id=rid, worker_id=0, payload=b""
    )
    router.request_to_worker[rid] = 0
    return router, sock, rid, cid


def test_forward_cancel_clears_bookkeeping():
    router, sock, rid, cid = _router_with_inflight_request()
    asyncio.run(router.forward_cancel(rid, cid))
    # cancel forwarded to the worker...
    assert sock.sent == [[cid, rid, b""]]
    # ...and both bookkeeping dicts cleared (the leak fix)
    assert rid not in router.request_to_worker
    assert rid not in router.workers[0].active_requests


def test_forward_cancel_unknown_request_is_noop():
    router, sock, rid, _cid = _router_with_inflight_request()
    asyncio.run(router.forward_cancel(b"unknown", b"cli-x"))
    assert sock.sent == []  # nothing forwarded
    assert rid in router.request_to_worker  # the known request is untouched
