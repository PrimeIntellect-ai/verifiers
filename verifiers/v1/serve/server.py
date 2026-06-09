"""The env server: load a taskset once, run rollouts on request by task index.

A ZMQ ROUTER front end (msgpack frames) over a v1 `Environment`. The server
owns the environment — taskset, harness, runtime — and is the only process that ever
loads it. A caller (e.g. the orchestrator) stays env-agnostic: it asks `info` for
the task count + whether group scoring is needed, then `run_rollout` / `run_group`
by task index. Per request the server resolves a `Client` from the request's
`client` config (cached, so a renderer's tokenizer is built once), wraps it in a
`RolloutContext`, and runs `env.episode(task, ctx, n).run()`, returning each
`Trace` as a JSON dict (with its computed `branches`).

Minimal port of `verifiers.serve` (ROUTER + msgpack), single async process: each
request is its own `asyncio.Task`, so many rollouts run concurrently. No worker
pool / heartbeats yet — the rollout's own runtime already isolates execution, and
the structure leaves room to add a pool later. Health is just another request type
(no separate socket), which is enough at this scale.
"""

import asyncio
import contextlib
import logging

import msgpack
import zmq
import zmq.asyncio

from verifiers.v1.clients import RolloutContext, resolve_client
from verifiers.v1.clients.client import Client
from verifiers.v1.clients.config import ClientConfig
from verifiers.v1.decorators import discover_decorated
from verifiers.v1.env import EnvConfig, Environment
from verifiers.v1.serve.types import (
    BaseResponse,
    HealthResponse,
    InfoResponse,
    RunGroupRequest,
    RunGroupResponse,
    RunRolloutRequest,
    RunRolloutResponse,
)
from verifiers.v1.types import SamplingConfig

logger = logging.getLogger(__name__)


class EnvServer:
    """Serve one v1 environment over ZMQ. The only process that loads the env."""

    def __init__(
        self, config: EnvConfig, address: str = "tcp://127.0.0.1:5000"
    ) -> None:
        self.address = address
        self.taskset_id = config.taskset.id
        self.env = Environment(config)
        # Load tasks once; the index range is fixed for the server's lifetime.
        self.tasks = self.env.taskset.load_tasks()
        self.requires_group_scoring = bool(
            discover_decorated(self.env.taskset, "group_reward")
        )
        self._clients: dict[
            tuple[str, str], Client
        ] = {}  # (client_config, model) -> Client

        self.ctx = zmq.asyncio.Context()
        self.frontend = self.ctx.socket(zmq.ROUTER)
        self.frontend.setsockopt(zmq.ROUTER_MANDATORY, 1)
        self.frontend.setsockopt(zmq.SNDHWM, 0)
        self.frontend.setsockopt(zmq.RCVHWM, 0)
        self.frontend.setsockopt(zmq.LINGER, 0)
        self.frontend.bind(self.address)
        # Resolve the concrete endpoint — when bound to an OS-assigned port
        # (address ending in `:0`), this is how we learn the actual port.
        self.address = self.frontend.getsockopt_string(zmq.LAST_ENDPOINT)

    @classmethod
    def run_server(cls, address_queue=None, **kwargs) -> None:
        """Build and run a server (entry point for a spawned process). If
        `address_queue` is given, report the concrete bound address on it (so a
        spawner that passed a `:0` address learns the OS-assigned port) before
        serving."""
        server = cls(**kwargs)
        if address_queue is not None:
            address_queue.put(server.address)
        asyncio.run(server.run())

    def _client(self, client_config: ClientConfig, model: str) -> Client:
        """Resolve (and cache) a `Client` for this config+model. Cached because a
        renderer client builds the model's tokenizer pool on first use — doing that
        per request would be ruinous."""
        key = (client_config.model_dump_json(), model)
        if key not in self._clients:
            self._clients[key] = resolve_client(client_config)
        return self._clients[key]

    def _context(
        self, client_config: ClientConfig, model: str, sampling: SamplingConfig
    ) -> RolloutContext:
        return RolloutContext(
            client=self._client(client_config, model), model=model, sampling=sampling
        )

    async def _run_rollout(self, req: RunRolloutRequest) -> RunRolloutResponse:
        ctx = self._context(req.client, req.model, req.sampling)
        traces = await self.env.episode(self.tasks[req.task_idx], ctx, n=1).run()
        return RunRolloutResponse(trace=traces[0].to_wire())

    async def _run_group(self, req: RunGroupRequest) -> RunGroupResponse:
        ctx = self._context(req.client, req.model, req.sampling)
        traces = await self.env.episode(self.tasks[req.task_idx], ctx, n=req.n).run()
        return RunGroupResponse(traces=[t.to_wire() for t in traces])

    async def _handle(
        self, client_id: bytes, request_id: bytes, method: bytes, payload: bytes
    ) -> None:
        try:
            route = method.decode()
            raw = msgpack.unpackb(payload, raw=False)
            if route == "health":
                response: BaseResponse = HealthResponse()
            elif route == "info":
                response = InfoResponse(
                    num_tasks=len(self.tasks),
                    requires_group_scoring=self.requires_group_scoring,
                )
            elif route == "run_rollout":
                response = await self._run_rollout(
                    RunRolloutRequest.model_validate(raw)
                )
            elif route == "run_group":
                response = await self._run_group(RunGroupRequest.model_validate(raw))
            else:
                response = BaseResponse(
                    success=False, error=f"unknown method {route!r}"
                )
        except (
            Exception
        ) as e:  # a failed request is data, not a crash — report and keep serving
            logger.warning("request failed: %s", e, exc_info=True)
            response = BaseResponse(success=False, error=f"{type(e).__name__}: {e}")
        data = msgpack.packb(response.model_dump(mode="json"), use_bin_type=True)
        try:
            await self.frontend.send_multipart([client_id, request_id, data])
        except zmq.ZMQError as e:
            logger.warning("failed to send response: %s", e)

    async def run(self) -> None:
        logger.info(
            "EnvServer up: taskset=%s address=%s tasks=%d group_scoring=%s",
            self.taskset_id,
            self.address,
            len(self.tasks),
            self.requires_group_scoring,
        )
        poller = zmq.asyncio.Poller()
        poller.register(self.frontend, zmq.POLLIN)
        tasks: set[asyncio.Task] = set()
        try:
            while True:
                events = dict(await poller.poll(timeout=100))
                if self.frontend not in events:
                    continue
                frames = await self.frontend.recv_multipart()
                if len(frames) != 4:
                    logger.warning(
                        "invalid message: expected 4 frames, got %d", len(frames)
                    )
                    continue
                client_id, request_id, method, payload = frames
                task = asyncio.create_task(
                    self._handle(client_id, request_id, method, payload)
                )
                tasks.add(task)
                task.add_done_callback(tasks.discard)
        except (asyncio.CancelledError, KeyboardInterrupt):
            pass
        finally:
            for task in tasks:
                task.cancel()
            for client in self._clients.values():
                with contextlib.suppress(Exception):
                    await client.close()
            self.frontend.close()
            self.ctx.term()
            logger.info("EnvServer down: taskset=%s", self.taskset_id)
