"""
Environment client that proxies requests to worker subprocesses.

Manages multiple EnvServer instances and provides an async interface
for rollout generation. One worker materializes the dataset, main process
dispatches group inputs to workers.
"""

import asyncio
import logging
from typing import Any

from datasets import Dataset

from verifiers.types import RolloutInput, RolloutResult, SamplingArgs
from verifiers.workers.server import EnvServer
from verifiers.workers.types import (
    MetadataRequest,
    MetadataResponse,
    RolloutRequest,
    RolloutResponse,
)


class EnvClient:
    """Client that manages worker subprocesses for environment execution.

    One worker loads the environment and returns the dataset. The client
    owns the dataset and the caller dispatches group inputs to workers.
    """

    def __init__(
        self,
        env_id: str,
        env_args: dict[str, Any],
        client_base_url: str,
        client_api_key: str,
        num_workers: int = 1,
        max_concurrent: int = -1,
        sampling_args: SamplingArgs | None = None,
        seq_len: int | None = None,
        interleaved_rollouts: bool = False,
        independent_scoring: bool = False,
        state_columns: list[str] | None = None,
    ):
        self.env_id = env_id
        self.env_args = env_args
        self.client_base_url = client_base_url
        self.client_api_key = client_api_key
        self.num_workers = num_workers
        self.max_concurrent = max_concurrent
        self.sampling_args = sampling_args or {}
        self.seq_len = seq_len
        self.interleaved_rollouts = interleaved_rollouts
        self.independent_scoring = independent_scoring
        self.state_columns = state_columns or []

        self.logger = logging.getLogger("verifiers.workers.client")

        self._workers: list[EnvServer] = []
        self._pending_futures: dict[int, tuple[asyncio.Future, int]] = {}
        self._pending_counts: list[int] = []
        self._response_collector_task: asyncio.Task | None = None

        # Populated after start() by querying first worker
        self._dataset: Dataset | None = None
        self._env_sampling_args: SamplingArgs = {}
        self._max_seq_len: int | None = None

    def start(self, num_examples: int = -1):
        """Start all worker processes and fetch dataset from first worker.

        Args:
            num_examples: Number of examples to use. -1 for all.
        """
        self.logger.info(f"Starting {self.num_workers} worker(s) for {self.env_id}")

        # Start first worker to get dataset
        first_worker = self._create_worker(0)
        first_worker.start()
        self._workers.append(first_worker)
        self._pending_counts = [0]

        # Request dataset and metadata from first worker
        first_worker.send_request(MetadataRequest(num_examples=num_examples))
        response = first_worker.recv_response(timeout=120)
        if response is None or not isinstance(response, MetadataResponse):
            raise RuntimeError("Failed to get metadata from worker")

        # Store dataset and metadata
        self._dataset = Dataset.from_list(response.dataset)
        self._env_sampling_args = response.sampling_args
        self._max_seq_len = response.max_seq_len

        self.logger.info(
            f"Got dataset with {len(self._dataset)} examples, "
            f"max_seq_len={self._max_seq_len}"
        )

        # Update sampling args from worker if we didn't specify them
        if not self.sampling_args and self._env_sampling_args:
            self.sampling_args = self._env_sampling_args

        # Start remaining workers
        for i in range(1, self.num_workers):
            worker = self._create_worker(i)
            worker.start()
            self._workers.append(worker)
            self._pending_counts.append(0)

    def _create_worker(self, index: int) -> EnvServer:
        """Create a worker server instance."""
        per_worker = (
            self.max_concurrent // max(1, self.num_workers)
            if self.max_concurrent > 0
            else -1
        )
        return EnvServer(
            env_id=self.env_id,
            env_args=self.env_args,
            client_base_url=self.client_base_url,
            client_api_key=self.client_api_key,
            max_concurrent=per_worker,
            seq_len=self.seq_len,
            interleaved_rollouts=self.interleaved_rollouts,
            worker_name=f"{self.env_id}_{index}",
        )

    def stop(self):
        """Stop all worker processes."""
        self.logger.info(f"Stopping {len(self._workers)} worker(s)")
        if self._response_collector_task:
            self._response_collector_task.cancel()
        for worker in self._workers:
            worker.stop()
        self._workers.clear()

    @property
    def dataset(self) -> Dataset:
        """Get the evaluation dataset."""
        if self._dataset is None:
            raise RuntimeError("Client not started - call start() first")
        return self._dataset

    @property
    def num_examples(self) -> int:
        """Get number of examples in dataset."""
        return len(self.dataset)

    @property
    def max_seq_len(self) -> int | None:
        """Get max sequence length from environment."""
        return self._max_seq_len

    @property
    def env_sampling_args(self) -> SamplingArgs:
        """Get environment's default sampling args."""
        return self._env_sampling_args

    async def submit_group(
        self,
        group_inputs: list[RolloutInput],
        example_id: int,
        model_name: str,
    ) -> asyncio.Future:
        """Submit a group of inputs for rollout and return a future for the response.

        The caller is responsible for constructing `group_inputs` (including any
        duplication needed to get multiple samples per example).
        """
        request = RolloutRequest(
            group_inputs=group_inputs,
            example_id=example_id,
            model_name=model_name,
            sampling_args=self.sampling_args,
            independent_scoring=self.independent_scoring,
            state_columns=self.state_columns,
        )

        worker = self._select_worker()
        worker_idx = self._workers.index(worker)
        worker.send_request(request)

        loop = asyncio.get_event_loop()
        future = loop.create_future()
        self._pending_futures[example_id] = (future, worker_idx)
        self._pending_counts[worker_idx] += 1

        return future

    def _select_worker(self) -> EnvServer:
        """Select the least-loaded worker (by pending request count)."""
        if not self._workers:
            raise RuntimeError("No workers available")
        min_idx = 0
        min_pending = self._pending_counts[0]
        for i, pending in enumerate(self._pending_counts[1:], start=1):
            if pending < min_pending:
                min_idx = i
                min_pending = pending
        return self._workers[min_idx]

    async def collect_responses(self):
        """Background task to collect responses and resolve futures."""
        while True:
            for worker in self._workers:
                while True:
                    response = worker.recv_response_nowait()
                    if response is None:
                        break
                    if isinstance(response, RolloutResponse):
                        self._handle_response(response)
            await asyncio.sleep(0.01)

    def _handle_response(self, response: RolloutResponse):
        """Handle a response from a worker."""
        if response.example_id in self._pending_futures:
            future, worker_idx = self._pending_futures.pop(response.example_id)
            if 0 <= worker_idx < len(self._pending_counts):
                self._pending_counts[worker_idx] = max(
                    0, self._pending_counts[worker_idx] - 1
                )
            if not future.done():
                future.set_result(response.results)

    async def run_groups(
        self,
        groups: list[tuple[list[RolloutInput], int]],
        model_name: str,
    ) -> list[list[RolloutResult]]:
        """Run multiple groups and return all results.

        Args:
            groups: List of (group_inputs, example_id) tuples.
                The caller constructs group_inputs with the desired duplication.
            model_name: Model to use for generation.

        Returns:
            List of result lists, one per group.
        """
        self._response_collector_task = asyncio.create_task(self.collect_responses())

        try:
            futures = []
            for group_inputs, example_id in groups:
                future = await self.submit_group(
                    group_inputs=group_inputs,
                    example_id=example_id,
                    model_name=model_name,
                )
                futures.append(future)

            results = await asyncio.gather(*futures)
            return list(results)
        finally:
            if self._response_collector_task:
                self._response_collector_task.cancel()
                try:
                    await self._response_collector_task
                except asyncio.CancelledError:
                    pass
                self._response_collector_task = None

    @property
    def pending_count(self) -> int:
        """Number of pending requests."""
        return len(self._pending_futures)

    def update_sampling_args(self, sampling_args: SamplingArgs):
        """Update sampling args for future requests."""
        self.sampling_args = sampling_args
