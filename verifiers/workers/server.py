"""
Environment server that runs in a subprocess.

Loads and owns an Environment instance, processing rollout requests
from the main process via IPC.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import queue
from itertools import cycle
from multiprocessing import Process
from multiprocessing.queues import Queue
from typing import Any, AsyncContextManager, Iterator, cast

from openai import AsyncOpenAI

import verifiers as vf
from verifiers.types import RolloutInput
from verifiers.utils.async_utils import maybe_semaphore
from verifiers.utils.type_utils import state_to_result
from verifiers.workers.types import (
    MetadataRequest,
    MetadataResponse,
    RolloutRequest,
    RolloutResponse,
    Shutdown,
)

WorkerRequest = RolloutRequest | MetadataRequest | Shutdown
WorkerResponse = RolloutResponse | MetadataResponse


async def process_request(
    request: RolloutRequest,
    env: vf.Environment,
    client_cycle: Iterator[AsyncOpenAI],
    semaphore: AsyncContextManager,
    logger: logging.Logger,
) -> RolloutResponse:
    """Process a single rollout request.

    The orchestrator is responsible for constructing group_inputs with
    the desired duplication. This worker just runs the group.
    """
    client = next(client_cycle)
    group_inputs = [cast(RolloutInput, inp) for inp in request.group_inputs]

    if request.independent_scoring:
        # Independent scoring: run each input separately with scoring
        states = []
        for rollout_input in group_inputs:
            state = await env.run_rollout(
                input=rollout_input,
                client=client,
                model=request.model_name,
                gen_sampling_args=request.sampling_args,
                gen_sem=semaphore,
                score_sem=semaphore,
                score=True,
            )
            states.append(state)
    else:
        # Group scoring: run all inputs together
        states = await env.run_group(
            group_inputs=group_inputs,
            client=client,
            model=request.model_name,
            gen_sampling_args=request.sampling_args,
            gen_sem=semaphore,
            score_sem=semaphore,
        )

    results = [
        state_to_result(state, state_columns=request.state_columns) for state in states
    ]
    return RolloutResponse(example_id=request.example_id, results=results)


def handle_metadata_request(
    request: MetadataRequest,
    env: vf.Environment,
) -> MetadataResponse:
    """Handle a metadata request - returns dataset and env config."""
    dataset = env.get_eval_dataset(n=request.num_examples)
    return MetadataResponse(
        dataset=dataset.to_list(),
        sampling_args=env.sampling_args,
        max_seq_len=env.max_seq_len,
    )


async def worker_loop(
    request_queue: Queue[WorkerRequest],
    response_queue: Queue[WorkerResponse],
    env: vf.Environment,
    clients: list[AsyncOpenAI],
    max_concurrent: int,
    logger: logging.Logger,
) -> None:
    """Main async loop for processing requests."""
    client_cycle = cycle(clients)
    semaphore = await maybe_semaphore(max_concurrent)

    pending_tasks: dict[asyncio.Task, int] = {}

    def check_for_requests() -> bool:
        """Non-blocking check for new requests. Returns False on shutdown signal."""
        while True:
            try:
                request = request_queue.get_nowait()
            except queue.Empty:
                break

            if isinstance(request, Shutdown):
                return False

            if isinstance(request, MetadataRequest):
                response = handle_metadata_request(request, env)
                response_queue.put(response)
                continue

            task = asyncio.create_task(
                process_request(request, env, client_cycle, semaphore, logger)
            )
            pending_tasks[task] = request.example_id
        return True

    try:
        while True:
            if not check_for_requests():
                break

            if not pending_tasks:
                await asyncio.sleep(0.01)
                continue

            done, _ = await asyncio.wait(
                pending_tasks.keys(), timeout=0.1, return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                pending_tasks.pop(task)
                try:
                    response = task.result()
                    response_queue.put(response)
                except Exception as e:
                    logger.error(f"Task failed: {e}")
    finally:
        for task in pending_tasks:
            task.cancel()


def worker_main(
    request_queue: Queue[WorkerRequest],
    response_queue: Queue[WorkerResponse],
    env_id: str,
    env_args: dict[str, Any],
    client_base_url: str,
    client_api_key: str,
    max_concurrent: int,
    seq_len: int | None,
    interleaved_rollouts: bool,
    worker_name: str,
) -> None:
    """Main entry point for worker process."""
    logger = logging.getLogger(f"verifiers.workers.{worker_name}")
    logger.info(f"Worker {worker_name} starting, loading environment {env_id}")

    env = vf.load_environment(env_id, **env_args)
    if seq_len is not None:
        env.set_max_seq_len(seq_len)
    if interleaved_rollouts:
        env.set_interleaved_rollouts(True)

    client = AsyncOpenAI(base_url=client_base_url, api_key=client_api_key)
    clients = [client]

    logger.info(f"Worker {worker_name} ready")

    asyncio.run(
        worker_loop(
            request_queue,
            response_queue,
            env,
            clients,
            max_concurrent,
            logger,
        )
    )


class EnvServer:
    """Manages a worker subprocess for an environment."""

    def __init__(
        self,
        env_id: str,
        env_args: dict[str, Any],
        client_base_url: str,
        client_api_key: str,
        max_concurrent: int,
        seq_len: int | None = None,
        interleaved_rollouts: bool = False,
        worker_name: str | None = None,
    ):
        self.env_id = env_id
        self.env_args = env_args
        self.client_base_url = client_base_url
        self.client_api_key = client_api_key
        self.max_concurrent = max_concurrent
        self.seq_len = seq_len
        self.interleaved_rollouts = interleaved_rollouts
        self.worker_name = worker_name or f"{env_id}_0"

        self.request_queue: Queue[WorkerRequest] = multiprocessing.Queue()
        self.response_queue: Queue[WorkerResponse] = multiprocessing.Queue()
        self.process: Process | None = None

        self.logger = logging.getLogger(f"verifiers.workers.{self.worker_name}")

    def start(self):
        """Start the worker process."""
        self.logger.info(f"Starting worker process {self.worker_name}")
        self.process = Process(
            target=worker_main,
            args=(
                self.request_queue,
                self.response_queue,
                self.env_id,
                self.env_args,
                self.client_base_url,
                self.client_api_key,
                self.max_concurrent,
                self.seq_len,
                self.interleaved_rollouts,
                self.worker_name,
            ),
            daemon=True,
        )
        self.process.start()

    def stop(self):
        """Stop the worker process."""
        if self.process and self.process.is_alive():
            self.request_queue.put(Shutdown())
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.logger.warning(
                    f"Worker {self.worker_name} did not stop gracefully, terminating"
                )
                self.process.terminate()

    def send_request(self, request: WorkerRequest):
        """Send a request to the worker."""
        self.request_queue.put(request)

    def recv_response(self, timeout: float | None = None) -> WorkerResponse | None:
        """Receive a response from the worker."""
        try:
            return self.response_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def recv_response_nowait(self) -> WorkerResponse | None:
        """Non-blocking receive of response."""
        try:
            return self.response_queue.get_nowait()
        except queue.Empty:
            return None

    @property
    def is_alive(self) -> bool:
        """Check if the worker process is alive."""
        return self.process is not None and self.process.is_alive()
