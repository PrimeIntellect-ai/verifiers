import asyncio
import logging
import queue
import threading
import time
from collections import deque
from typing import Any

from pydantic import BaseModel, Field

from verifiers import GenerateOutputs
from verifiers.types import ProcessedOutputs


class BatchRequest(BaseModel):
    """Request for batch generation"""

    batch_id: int
    env_inputs: dict[str, list[Any]]
    processing_class: Any
    mask_env_responses: bool
    max_seq_len: int
    mask_truncated_completions: bool
    zero_truncated_completions: bool
    max_concurrent: int


class BatchResult(BaseModel):
    """Result from batch generation"""

    batch_id: int
    processed_results: ProcessedOutputs
    generation_time: float = 0.0
    all_reward_dict: dict[str, list[float]] = Field(
        default_factory=dict
    )  # All reward scores
    completions: list[Any] = Field(
        default_factory=list
    )  # Store completions for logging
    prompts: list[Any] = Field(default_factory=list)  # Store prompts for logging


class AsyncBatchGenerator:
    """
    Manages asynchronous batch generation for GRPO training.

    This class runs generation in a separate thread, allowing training to continue
    while future batches are being generated. It maintains a queue of pending
    generation requests and completed results.
    """

    def __init__(
        self,
        env,
        client_config,
        model_name: str,
        sampling_args: dict[str, Any],
        num_batches_ahead: int = 1,
        max_queue_size: int | None = None,
        generation_timeout: float = 300.0,  # 5 minutes default
    ):
        self.env = env
        self.client_config = client_config
        self.client = None  # Will be created in worker thread
        self.model_name = model_name
        self.sampling_args = sampling_args
        self.num_batches_ahead = num_batches_ahead
        self.max_queue_size = max_queue_size or max(num_batches_ahead * 2, 4)
        self.generation_timeout = generation_timeout

        # Queues for communication
        self.request_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.is_generating = False

        # Tracking
        self.pending_batches = set()  # batch_ids currently being processed
        self.completed_batches = {}  # batch_id -> BatchResult
        self.next_expected_batch = 0
        self.generation_times = deque(maxlen=100)  # Track recent generation times

        # Thread management
        self.worker_thread = None
        self.stop_event = threading.Event()
        self.logger = logging.getLogger(f"AsyncBatchGenerator-{id(self)}")
        self.is_generating = False  # Track if currently generating
        self.worker_loop = None  # Will be set in worker thread
        self.started = False  # Track if generator is started

        # Synchronization
        self._lock = threading.Lock()

    def start(self):
        """Start the async generation worker thread"""
        if self.started:
            return

        self.worker_thread = threading.Thread(
            target=self._generation_worker, daemon=True, name="AsyncBatchGenerator"
        )
        self.worker_thread.start()
        self.started = True

    def stop(self):
        """Stop the async generation worker thread"""
        if not self.started:
            return

        self.stop_event.set()
        # Send poison pill
        self.request_queue.put(None)

        if self.worker_thread:
            self.worker_thread.join(timeout=10.0)

        self.started = False

    def submit_batch(self, request: BatchRequest) -> bool:
        """
        Submit a batch for async generation.

        Returns:
            bool: True if submitted successfully, False if queue is full
        """
        if not self.started:
            raise RuntimeError("AsyncBatchGenerator not started")

        with self._lock:
            if request.batch_id in self.pending_batches:
                return True  # Already submitted

            if len(self.pending_batches) >= self.max_queue_size:
                return False  # Queue full

            self.pending_batches.add(request.batch_id)

        self.request_queue.put(request)
        return True

    def get_batch(self, batch_id: int, timeout: float | None = None) -> BatchResult:
        """
        Get a completed batch result. Blocks until the batch is ready.

        Args:
            batch_id: The batch ID to retrieve
            timeout: Maximum time to wait (uses generation_timeout if None)

        Returns:
            BatchResult: The completed batch result

        Raises:
            TimeoutError: If batch doesn't complete within timeout
            RuntimeError: If generation failed
        """
        timeout = timeout or self.generation_timeout
        start_time = time.time()

        while True:
            # Check if already completed
            with self._lock:
                if batch_id in self.completed_batches:
                    return self.completed_batches.pop(batch_id)

            # Check for new results
            try:
                result = self.result_queue.get(timeout=0.1)
                with self._lock:
                    self.completed_batches[result.batch_id] = result
                    self.pending_batches.discard(result.batch_id)

                # If this is our batch, return it
                if result.batch_id == batch_id:
                    with self._lock:
                        return self.completed_batches.pop(batch_id)

            except queue.Empty:
                pass

            # Check timeout
            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"Batch {batch_id} generation timed out after {timeout}s"
                )

    def get_pending_count(self) -> int:
        """Get number of batches currently being generated"""
        with self._lock:
            return len(self.pending_batches)

    def get_completed_count(self) -> int:
        """Get number of completed batches waiting to be retrieved"""
        with self._lock:
            return len(self.completed_batches)

    def get_average_generation_time(self) -> float:
        """Get average generation time for recent batches"""
        if not self.generation_times:
            return 0.0
        return sum(self.generation_times) / len(self.generation_times)

    def should_submit_more(self) -> bool:
        """Check if we should submit more batches for generation"""
        with self._lock:
            total_pending = len(self.pending_batches) + len(self.completed_batches)
            return total_pending < self.num_batches_ahead

    def _generation_worker(self):
        """Worker thread that processes generation requests"""
        # Create event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.worker_loop = loop  # Store the event loop reference

        # Create the AsyncOpenAI client within this event loop
        import httpx
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(
            base_url=self.client_config["base_url"],
            api_key=self.client_config["api_key"],
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=self.client_config["http_client_args"]["limits"][
                        "max_connections"
                    ]
                ),
                timeout=self.client_config["http_client_args"]["timeout"],
            ),
        )

        try:
            while not self.stop_event.is_set():
                try:
                    # Get next request
                    request = self.request_queue.get(timeout=0.1)
                    if request is None:  # Poison pill
                        break

                    # Generate batch using the async method
                    start_time = time.time()
                    result = loop.run_until_complete(
                        self._generate_batch_async(request)
                    )
                    generation_time = time.time() - start_time
                    result.generation_time = generation_time
                    self.generation_times.append(generation_time)
                    self.result_queue.put(result)
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"Error in generation worker: {e}")
                    raise e
        finally:
            # Clean up the client
            if self.client:
                loop.run_until_complete(self.client.close())
            # Clean up the event loop
            loop.close()
            asyncio.set_event_loop(None)

    async def _generate_batch_async(self, request: BatchRequest) -> BatchResult:
        """
        Generate a single batch asynchronously.
        """
        # Call environment generation
        self.is_generating = True
        env_results = await self.env.a_generate(
            request.env_inputs,
            client=self.client,
            model=self.model_name,
            sampling_args=self.sampling_args,
            score_rollouts=True,
            max_concurrent=request.max_concurrent,
        )
        self.is_generating = False

        # Extract all reward-related keys
        all_reward_dict = {
            "reward": env_results.reward,
        }
        for k in env_results.metrics:
            all_reward_dict[k] = env_results.metrics[k]

        # Process results
        processed_results = self.env.process_env_results_vllm(
            prompts=env_results.prompt,
            completions=env_results.completion,
            states=env_results.state,
            rewards=env_results.reward,
            processing_class=request.processing_class,
            max_seq_len=request.max_seq_len,
            mask_env_responses=request.mask_env_responses,
            mask_truncated_completions=request.mask_truncated_completions,
            zero_truncated_completions=request.zero_truncated_completions,
        )

        return BatchResult(
            batch_id=request.batch_id,
            processed_results=processed_results,
            all_reward_dict=all_reward_dict,
            completions=env_results.completion,
            prompts=env_results.prompt,
        )

    async def _evaluate_async(self, num_samples: int = -1) -> GenerateOutputs:
        """
        Run evaluation in the worker thread's event loop.
        """
        # Get evaluation dataset
        if self.env.eval_dataset is None:
            self.env.logger.info(
                "eval_dataset is not set, falling back to train dataset"
            )
            assert self.env.dataset is not None
            inputs = self.env.get_dataset(n=num_samples)
        else:
            inputs = self.env.get_eval_dataset(n=num_samples)
        assert inputs is not None, "No dataset found"

        # Run generation on eval dataset
        results = await self.env.a_generate(
            inputs,
            client=self.client,
            model=self.model_name,
            sampling_args=self.sampling_args,
        )
        return results

    def evaluate(self, num_samples: int = -1) -> GenerateOutputs:
        """
        Run evaluation synchronously by creating a separate thread with its own event loop.
        """
        if not self.started:
            raise RuntimeError("AsyncBatchGenerator not started")

        # Run evaluation in a separate thread to avoid event loop conflicts
        result_container = []
        exception_container = []

        def run_evaluation():
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Create a new client for this evaluation
            import httpx
            from openai import AsyncOpenAI

            eval_client = AsyncOpenAI(
                base_url=self.client_config["base_url"],
                api_key=self.client_config["api_key"],
                http_client=httpx.AsyncClient(
                    limits=httpx.Limits(
                        max_connections=self.client_config["http_client_args"][
                            "limits"
                        ]["max_connections"]
                    ),
                    timeout=self.client_config["http_client_args"]["timeout"],
                ),
            )

            async def run_eval():
                try:
                    # Get evaluation dataset
                    if self.env.eval_dataset is None:
                        self.env.logger.info(
                            "eval_dataset is not set, falling back to train dataset"
                        )
                        assert self.env.dataset is not None
                        inputs = self.env.get_dataset(n=num_samples)
                    else:
                        inputs = self.env.get_eval_dataset(n=num_samples)
                    assert inputs is not None, "No dataset found"

                    # Run generation on eval dataset
                    results = await self.env.a_generate(
                        inputs,
                        client=eval_client,
                        model=self.model_name,
                        sampling_args=self.sampling_args,
                    )
                    result_container.append(results)
                except Exception as e:
                    exception_container.append(e)
                finally:
                    await eval_client.close()

            try:
                loop.run_until_complete(run_eval())
            finally:
                loop.close()
                asyncio.set_event_loop(None)

        # Run evaluation in a separate thread
        import threading

        eval_thread = threading.Thread(target=run_evaluation)
        eval_thread.start()
        eval_thread.join(timeout=self.generation_timeout)

        if eval_thread.is_alive():
            raise TimeoutError(f"Evaluation timed out after {self.generation_timeout}s")

        if exception_container:
            raise exception_container[0]

        if not result_container:
            raise RuntimeError("Evaluation completed but no results were returned")

        return result_container[0]
