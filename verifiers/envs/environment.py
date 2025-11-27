import asyncio
import atexit
import inspect
import json
import logging
import signal
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    AsyncContextManager,
    Awaitable,
    Callable,
    List,
    Literal,
    TypeVar,
)

from datasets import Dataset
from openai import AsyncOpenAI, BadRequestError, OpenAI

from verifiers.parsers.parser import Parser
from verifiers.rubrics.rubric import Rubric
from verifiers.types import (
    ChatCompletionToolParam,
    ChatMessage,
    GenerateMetadata,
    GenerateOutputs,
    Messages,
    MessageType,
    ModelResponse,
    RolloutInput,
    RolloutTiming,
    SamplingArgs,
    State,
)
from verifiers.utils.async_utils import maybe_semaphore
from verifiers.utils.eval_utils import make_dataset, save_rollout_results
from verifiers.utils.message_utils import (
    concat_messages,
    get_overlong_prompt_dummy_response,
)
from verifiers.utils.path_utils import get_results_path

if TYPE_CHECKING:
    pass


class Environment(ABC):
    """
    Base class for all environments.
    """

    def __init__(
        self,
        dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        system_prompt: str | None = None,
        few_shot: list[ChatMessage] | None = None,
        parser: Parser | None = None,
        rubric: Rubric | None = None,
        sampling_args: SamplingArgs | None = None,
        message_type: MessageType = "chat",
        oai_tools: list[ChatCompletionToolParam] | None = None,
        max_workers: int = 512,
        env_id: str | None = None,
        env_args: dict | None = None,
        map_kwargs: dict = {},
        max_seq_len: int | None = None,
        **kwargs,
    ):
        self.logger = logging.getLogger(f"verifiers.envs.{self.__class__.__name__}")
        self.message_type: Literal["chat", "completion"] = message_type
        self.oai_tools: list[ChatCompletionToolParam] | None = oai_tools
        self.system_prompt = system_prompt
        self.few_shot = few_shot
        self.parser = parser or Parser()
        self.rubric = rubric or Rubric()
        if self.parser.__class__ != self.rubric.parser.__class__:
            self.logger.warning(
                "The parser and rubric parser are different. This may cause unexpected behavior."
            )

        self.env_id = env_id or ""
        self.env_args = env_args or {}
        self.max_seq_len = max_seq_len
        if self.message_type == "chat":
            if dataset is not None:
                self.dataset = self.format_dataset(
                    dataset, self.system_prompt, self.few_shot, map_kwargs=map_kwargs
                )
            else:
                self.dataset = None
            if eval_dataset is not None:
                self.eval_dataset = self.format_dataset(
                    eval_dataset,
                    self.system_prompt,
                    self.few_shot,
                    map_kwargs=map_kwargs,
                )
            else:
                self.eval_dataset = None
        else:
            if self.system_prompt or self.few_shot:
                raise ValueError(
                    'The fields "system_prompt" and "few_shot" are not supported for completion tasks.'
                    'Please use message_type="chat" instead, or pre-format your dataset '
                    'to contain a "prompt" column.'
                )
            self.dataset = dataset
            self.eval_dataset = eval_dataset

        self.sampling_args = {"n": 1, "extra_body": {}}
        if sampling_args is not None:
            # merge extra_body if provided
            self.sampling_args["extra_body"].update(sampling_args.get("extra_body", {}))  # type: ignore[union-attr]
            # copy other keys
            for key, value in sampling_args.items():
                if key != "extra_body":
                    self.sampling_args[key] = value

        self.max_workers = max_workers
        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.dataset is None and self.eval_dataset is None:
            raise ValueError("Either dataset or eval_dataset must be provided")
        self.rollouts_per_example = None
        self._stop_conditions: list[StopCondition] = []
        self._cleanup_handlers: list[RolloutCleanup] = []
        self._teardown_handlers: list[EnvironmentTeardown] = []

        self.__post_init__()

    def __post_init__(self):
        self._stop_conditions = [
            method
            for _, method in inspect.getmembers(self, predicate=inspect.ismethod)
            if hasattr(method, "stop") and callable(method)
        ]
        self._stop_conditions.sort(
            key=lambda m: (-getattr(m, "stop_priority", 0), m.__name__)
        )

        self._cleanup_handlers = [
            method
            for _, method in inspect.getmembers(self, predicate=inspect.ismethod)
            if hasattr(method, "cleanup") and callable(method)
        ]
        self._cleanup_handlers.sort(
            key=lambda m: (-getattr(m, "cleanup_priority", 0), m.__name__)
        )

        self._teardown_handlers = [
            method
            for _, method in inspect.getmembers(self, predicate=inspect.ismethod)
            if hasattr(method, "teardown") and callable(method)
        ]
        self._teardown_handlers.sort(
            key=lambda m: (-getattr(m, "teardown_priority", 0), m.__name__)
        )

        def _sync_teardown():
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self._teardown())
                else:
                    loop.run_until_complete(self._teardown())
            except RuntimeError:
                asyncio.run(self._teardown())

        atexit.register(_sync_teardown)
        signal.signal(
            signal.SIGINT,
            lambda sig, frame: (
                _sync_teardown(),
                signal.default_int_handler(sig, frame),
            ),
        )
        signal.signal(signal.SIGTERM, lambda _, __: (_sync_teardown(), exit(143)))

    def _ensure_example_id(self, dataset: Dataset) -> Dataset:
        """Ensure example_id column exists and is integer type."""
        if "example_id" in dataset.column_names and not isinstance(
            dataset["example_id"][0], int
        ):
            dataset = dataset.rename_column("example_id", "src_id")
        if "example_id" not in dataset.column_names:
            dataset = dataset.add_column("example_id", range(len(dataset)))  # type: ignore (weird datasets thing)
        return dataset

    def _ensure_prompt(
        self,
        dataset: Dataset,
        system_prompt: str | None = None,
        few_shot: list[ChatMessage] | None = None,
        question_key: str = "question",
        answer_key: str = "answer",
        map_kwargs: dict = {},
    ) -> Dataset:
        """Ensure prompt column exists."""
        if "prompt" not in dataset.column_names:

            def format_prompt_fn(prompt_str: str) -> list[ChatMessage]:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                if few_shot:
                    messages.extend(few_shot)
                messages.append({"role": "user", "content": prompt_str})
                return messages

            if answer_key == "answer":
                dataset = dataset.map(
                    lambda x: {
                        "prompt": format_prompt_fn(x[question_key]),
                    },
                    **map_kwargs,
                )
            else:
                dataset = dataset.map(
                    lambda x: {
                        "prompt": format_prompt_fn(x[question_key]),
                        "answer": x[answer_key],
                    },
                    **map_kwargs,
                )
        return dataset

    def _ensure_task(self, dataset: Dataset, map_kwargs: dict = {}) -> Dataset:
        """Ensure task column exists, set to env_id."""
        if "task" not in dataset.column_names:
            task_value = self.env_id or "default"

            def add_task(example):
                example["task"] = task_value
                return example

            dataset = dataset.map(add_task, **map_kwargs)
        return dataset

    def format_dataset(
        self,
        dataset: Dataset,
        system_prompt: str | None = None,
        few_shot: list[ChatMessage] | None = None,
        question_key: str = "question",
        answer_key: str = "answer",
        map_kwargs: dict = {},
    ) -> Dataset:
        """
        Format dataset by creating example_id and prompt columns, and setting task column.
        """
        dataset = self._ensure_example_id(dataset)
        dataset = self._ensure_prompt(
            dataset, system_prompt, few_shot, question_key, answer_key, map_kwargs
        )
        dataset = self._ensure_task(dataset, map_kwargs)
        return dataset

    def get_dataset(self, n: int = -1, seed: int | None = None) -> Dataset:
        if self.dataset is None:
            raise ValueError("dataset is not set")
        if seed is not None:
            self.dataset = self.dataset.shuffle(seed=seed)
        if n > 0:
            # Cap n to the length of the dataset to prevent IndexError
            n = min(n, len(self.dataset))
            return self.dataset.select(range(n))
        return self.dataset

    def get_eval_dataset(self, n: int = -1, seed: int | None = None) -> Dataset:
        if self.eval_dataset is None:
            self.logger.warning(
                "eval_dataset is not set, falling back to train dataset"
            )
            return self.get_dataset(n, seed)
        if seed is not None:
            self.eval_dataset = self.eval_dataset.shuffle(seed=seed)
        if n > 0:
            # Cap n to the length of the dataset to prevent IndexError
            n = min(n, len(self.eval_dataset))
            return self.eval_dataset.select(range(n))
        return self.eval_dataset

    async def get_model_response(
        self,
        client: AsyncOpenAI,
        model: str,
        prompt: Messages,
        oai_tools: list[ChatCompletionToolParam] | None = None,
        sampling_args: SamplingArgs | None = None,
        message_type: MessageType | None = None,
    ) -> ModelResponse:
        """
        Get model response for a given prompt (chat or completion).

        Convenience function for wrapping (chat, completion) API calls.
        Returns special error messages for context length issues.
        """
        sampling_args = sampling_args or {}
        # resolve message type first
        if message_type is None:
            message_type = self.message_type
        # normalize sampling args:
        # - if max_tokens is provided for chat, rename to max_completion_tokens
        # - drop any None-valued entries to avoid sending to the client
        if "max_tokens" in sampling_args:
            if sampling_args["max_tokens"] is None:
                sampling_args.pop("max_tokens")
            elif message_type == "chat":
                sampling_args["max_completion_tokens"] = sampling_args.pop("max_tokens")
        if (
            "max_completion_tokens" in sampling_args
            and sampling_args["max_completion_tokens"] is None
        ):
            sampling_args.pop("max_completion_tokens")
        clean_sampling_args = {k: v for k, v in sampling_args.items() if v is not None}
        try:
            if message_type == "chat":
                assert isinstance(prompt, list)
                # --- detect audio parts and force text-only modality if caller didn't set one ---
                has_audio = False
                try:
                    for m in prompt:
                        c = m.get("content")
                        if isinstance(c, list):
                            for p in c:
                                if isinstance(p, dict) and str(
                                    p.get("type", "")
                                ).startswith("input_audio"):
                                    has_audio = True
                                    break
                        if has_audio:
                            break
                except Exception:
                    has_audio = False
                if has_audio and "modalities" not in clean_sampling_args:
                    clean_sampling_args = {
                        **clean_sampling_args,
                        "modalities": ["text"],
                    }

                if oai_tools:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=prompt,
                        tools=oai_tools,
                        **clean_sampling_args,
                    )
                else:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=prompt,
                        **clean_sampling_args,
                    )
                return response
            elif message_type == "completion":
                if oai_tools:
                    raise ValueError(
                        "oai_tools are not supported for completion tasks."
                    )
                assert isinstance(prompt, str)
                response = await client.completions.create(
                    model=model, prompt=prompt, **clean_sampling_args
                )
                return response
        except Exception as e:
            # in case of making a request with an overlong prompt, e.g from a too-long
            # environment response, we return a dummy response to with finish_reason "length"
            if isinstance(e, BadRequestError):
                error_text = e.response.text.lower()
                context_length_phrases = [
                    "this model's maximum context length is",
                    "is longer than the model's context length",
                    "exceeds the model's context length",
                    "exceed the configured limit",
                    "exceeds the configured limit",
                    "exceeded model",
                ]
                if any(phrase in error_text for phrase in context_length_phrases):
                    self.logger.debug("Caught overlong prompt.")
                    return get_overlong_prompt_dummy_response(
                        message_type or self.message_type
                    )
            self.logger.error(f"Error getting model response: {e} \n\nExiting...")
            raise e

    async def init_state(
        self,
        input: RolloutInput,
        client: AsyncOpenAI,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ) -> State:
        """
        Create initial state from dataset row.
        Environment-agnostic - just stores the data.

        Creates State with input fields in "input" RolloutInput for structured access,
        but State's forwarding behavior allows backward-compatible direct access.
        """
        state_input = deepcopy(input)
        if "info" in state_input and isinstance(state_input["info"], str):
            state_input["info"] = json.loads(state_input["info"])
        if "task" not in state_input:
            state_input["task"] = self.env_id or "default"
        state = State(input=RolloutInput(**state_input))  # type: ignore[missing-typed-dict-key]
        state["client"] = client
        state["model"] = model
        state["sampling_args"] = sampling_args
        state["is_completed"] = False
        state["oai_tools"] = None
        if "info" in state and hasattr(state["info"], "oai_tools"):
            state["oai_tools"] = state["info"]["oai_tools"]
        elif hasattr(self, "oai_tools"):
            state["oai_tools"] = self.oai_tools
        else:
            state["oai_tools"] = []
        state["trajectory"] = []
        state["reward"] = None
        state["metrics"] = None
        state["timing"] = RolloutTiming(
            generation_ms=0.0,
            scoring_ms=0.0,
            total_ms=0.0,
            start_time=time.time(),
        )
        return state

    @abstractmethod
    async def setup_state(self, state: State) -> State:
        """
        Setup the state.
        """
        return state

    @abstractmethod
    async def rollout(
        self,
        input: RolloutInput,
        client: AsyncOpenAI,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ) -> State:
        """
        Run a rollout for a given input.
        """
        pass

    async def _cleanup(self, state: State):
        """
        Clean up rollout resources.
        """
        for handler in self._cleanup_handlers:
            await handler(state)

    async def _teardown(self):
        """
        Tear down environment resources.
        """
        for handler in self._teardown_handlers:
            await handler()

    async def _render_stop(self, state: State, condition) -> bool:
        if await condition(state):
            state["is_completed"] = True
            state["stop_condition"] = condition.__name__
            return True
        return False

    async def _render_timing(self, state: State):
        start_time = state["timing"]["start_time"]
        end_time = time.time()
        state["timing"]["generation_ms"] = (end_time - start_time) * 1000
        state["timing"]["total_ms"] = (end_time - start_time) * 1000

    async def _render_completion(self, state: State):
        last_prompt = state["trajectory"][-1]["prompt"]
        last_completion = state["trajectory"][-1]["completion"]
        full_conversation = concat_messages([last_prompt, last_completion])
        state["completion"] = full_conversation[len(state["prompt"]) :]

    async def is_completed(self, state: State, **kwargs) -> bool:
        """Check all stop conditions. Sets state.is_completed=True if any condition is met."""
        for condition in self._stop_conditions:
            if await self._render_stop(state, condition):
                await self._render_timing(state)
                await self._render_completion(state)
                await self._cleanup(state)
                return True
        return False

    async def run_rollout(
        self,
        sem: AsyncContextManager,
        input: RolloutInput,
        client: AsyncOpenAI,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ) -> State:
        """
        Run a rollout with a semaphore (generation only, no scoring).
        """
        async with sem:
            state = await self.rollout(
                input,
                client,
                model,
                sampling_args,
            )
        return state

    async def run_group(
        self,
        group_inputs: list[RolloutInput],
        client: AsyncOpenAI,
        model: str,
        gen_sampling_args: SamplingArgs,
        gen_sem: AsyncContextManager,
        score_sem: AsyncContextManager,
        **kwargs,
    ) -> list[State]:
        """Generate and score one group."""
        rollout_tasks = [
            self.run_rollout(
                gen_sem,
                input,
                client,
                model,
                gen_sampling_args,
            )
            for input in group_inputs
        ]
        group_states = await asyncio.gather(*rollout_tasks)
        await self.rubric.score_group(group_states, score_sem=score_sem)
        return list(group_states)

    def _prepare_rollout_results(
        self,
        all_states: list[State],
        model: str,
        client: AsyncOpenAI,
        state_columns: list[str] | None,
        results_path: Path | None,
        gen_sampling_args: SamplingArgs,
        start_time: float,
    ) -> GenerateOutputs:
        """Prepare GenerateOutputs from a list of completed states."""
        # Determine path_to_save
        if results_path is None:
            path_to_save = get_results_path(self.env_id, model)
        else:
            path_to_save = results_path
        prompts = [state["prompt"] for state in all_states]
        completions = [state.get("completion") for state in all_states]
        answers = [state.get("answer", "") for state in all_states]
        tasks = [state.get("task", "default") for state in all_states]
        infos = [state.get("info", {}) for state in all_states]
        example_ids = [state.get("example_id", 0) for state in all_states]
        rewards = [state.get("reward", 0.0) for state in all_states]

        metrics: dict[str, list[float]] = {}
        for state in all_states:
            if state.get("metrics"):
                for metric_name, metric_value in state["metrics"].items():
                    if metric_name not in metrics:
                        metrics[metric_name] = []
                    metrics[metric_name].append(metric_value)

        num_unique_examples = len(set(example_ids)) if example_ids else 0
        rollouts_per_example = (
            len(all_states) // num_unique_examples if num_unique_examples > 0 else 1
        )

        metadata = GenerateMetadata(
            env_id=self.env_id,
            env_args=self.env_args,
            model=model,
            base_url=str(client.base_url) if hasattr(client, "base_url") else "",
            num_examples=num_unique_examples,
            rollouts_per_example=rollouts_per_example,
            sampling_args=gen_sampling_args,
            date=datetime.now().isoformat(),
            time_ms=(time.time() - start_time) * 1000.0,
            avg_reward=sum(rewards) / len(rewards) if rewards else 0.0,
            avg_metrics={
                name: sum(values) / len(values) if values else 0.0
                for name, values in metrics.items()
            },
            state_columns=state_columns or [],
            path_to_save=path_to_save,
        )

        return GenerateOutputs(
            prompt=prompts,
            completion=completions,
            answer=answers,
            state=all_states,
            task=tasks,
            info=infos,
            example_id=example_ids,
            reward=rewards,
            metrics=metrics,
            metadata=metadata,
        )

    async def generate(
        self,
        inputs: Dataset | List[RolloutInput],
        client: AsyncOpenAI,
        model: str,
        sampling_args: SamplingArgs | None = None,
        max_concurrent: int = -1,
        max_concurrent_generation: int | None = None,
        max_concurrent_scoring: int | None = None,
        results_path: Path | None = None,
        state_columns: list[str] | None = None,
        save_results: bool = False,
        save_every: int = -1,
        use_tqdm: bool = True,
    ) -> GenerateOutputs:
        """
        Generate rollouts for a set of inputs by group.
        """
        if isinstance(inputs, Dataset):
            inputs_list = inputs.to_list()
        elif isinstance(inputs, list):
            inputs_list = inputs

        # resolve concurrency knobs
        gen_limit = max_concurrent_generation
        score_limit = max_concurrent_scoring
        if gen_limit is None:
            gen_limit = max_concurrent
        if score_limit is None:
            score_limit = max_concurrent

        # group inputs by example_id
        input_groups: dict[int, list[RolloutInput]] = {}
        for input_item in inputs_list:
            example_id = input_item["example_id"]
            if example_id not in input_groups:
                input_groups[example_id] = []
            input_groups[example_id].append(input_item)
        group_list = list(input_groups.values())

        # set up semaphores
        gen_sem = await maybe_semaphore(gen_limit)
        score_sem = await maybe_semaphore(score_limit)

        # set up sampling args
        gen_sampling_args = deepcopy(self.sampling_args)
        if sampling_args is not None:
            gen_sampling_args.update(sampling_args)

        # create tasks for all groups
        start_time = time.time()
        group_tasks = {
            asyncio.create_task(
                self.run_group(
                    group,
                    client,
                    model,
                    gen_sampling_args,
                    gen_sem,
                    score_sem,
                )
            ): i
            for i, group in enumerate(group_list)
        }

        # process groups as they complete
        pbar = None
        if use_tqdm:
            from tqdm import tqdm

            pbar = tqdm(
                total=len(group_list),
                desc=f"Processing {len(group_list)} groups ({len(inputs_list)} total rollouts)",
            )

        groups_completed = 0
        all_states: list[State] = []
        try:
            for coro in asyncio.as_completed(group_tasks.keys()):
                group_states = await coro
                all_states.extend(group_states)
                groups_completed += 1

                if pbar is not None:
                    pbar.update(1)

                # save intermediate results
                if (
                    save_results
                    and save_every > 0
                    and groups_completed % save_every == 0
                ):
                    temp_results = self._prepare_rollout_results(
                        all_states,
                        model,
                        client,
                        state_columns,
                        results_path,
                        gen_sampling_args,
                        start_time,
                    )
                    self.logger.debug(
                        f"Saving intermediate results to {temp_results['metadata']['path_to_save']}"
                    )
                    save_rollout_results(temp_results)
        finally:
            if pbar is not None:
                pbar.close()

        # sort by example_id to ensure deterministic ordering regardless of completion order
        all_states.sort(key=lambda s: s.get("example_id", 0))

        results = self._prepare_rollout_results(
            all_states,
            model,
            client,
            state_columns,
            results_path,
            gen_sampling_args,
            start_time,
        )

        # Save if requested
        if save_results:
            save_rollout_results(results)

        return results

    def generate_sync(
        self,
        inputs: Dataset | List[RolloutInput],
        client: AsyncOpenAI | OpenAI,
        **kwargs,
    ) -> GenerateOutputs:
        if isinstance(client, OpenAI):
            client = AsyncOpenAI(api_key=client.api_key, base_url=client.base_url)
        coro = self.generate(
            inputs,
            client=client,
            **kwargs,
        )
        # check if we're in existing event loop (e.g. Jupyter)
        try:
            loop = asyncio.get_running_loop()
            import nest_asyncio  # type: ignore

            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        except RuntimeError:
            pass

        # script case: create new loop and executor
        executor = ThreadPoolExecutor(max_workers=self.max_workers)
        loop = asyncio.new_event_loop()
        try:
            loop.set_default_executor(executor)
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            loop.close()
            asyncio.set_event_loop(None)
            # shutdown the executor to prevent thread leaks
            executor.shutdown(wait=False)

    # evaluation
    def get_eval_inputs(
        self, num_examples: int = -1, rollouts_per_example: int = 1
    ) -> List[RolloutInput]:
        if self.eval_dataset is None:
            self.logger.info("eval_dataset is not set, falling back to train dataset")
            assert self.dataset is not None
            inputs = self.get_dataset(n=num_examples)
        else:
            inputs = self.get_eval_dataset(n=num_examples)
        assert inputs is not None, "No dataset found"
        if rollouts_per_example > 1:
            inputs = inputs.repeat(rollouts_per_example)
        return inputs.to_list()

    async def evaluate(
        self,
        client: AsyncOpenAI,
        model: str,
        sampling_args: SamplingArgs | None = None,
        num_examples: int = -1,
        rollouts_per_example: int = 1,
        max_concurrent: int = -1,
        max_concurrent_generation: int | None = None,
        max_concurrent_scoring: int | None = None,
        results_path: Path | None = None,
        state_columns: list[str] | None = None,
        save_results: bool = False,
        save_every: int = -1,
        **kwargs,
    ) -> GenerateOutputs:
        """
        Evaluate model on the Environment evaluation dataset.
        """
        inputs = self.get_eval_inputs(num_examples, rollouts_per_example)
        return await self.generate(
            inputs,
            client=client,
            model=model,
            sampling_args=sampling_args,
            max_concurrent=max_concurrent,
            max_concurrent_generation=max_concurrent_generation,
            max_concurrent_scoring=max_concurrent_scoring,
            results_path=results_path,
            state_columns=state_columns,
            save_results=save_results,
            save_every=save_every,
            **kwargs,
        )

    def evaluate_sync(
        self,
        client: OpenAI | AsyncOpenAI,
        model: str,
        sampling_args: SamplingArgs | None = None,
        num_examples: int = -1,
        rollouts_per_example: int = 1,
        max_concurrent: int = -1,
        max_concurrent_generation: int | None = None,
        max_concurrent_scoring: int | None = None,
        results_path: Path | None = None,
        state_columns: list[str] | None = None,
        save_results: bool = False,
        save_every: int = -1,
    ) -> GenerateOutputs:
        """
        Evaluate model on the Environment evaluation dataset synchronously.
        """
        inputs = self.get_eval_inputs(num_examples, rollouts_per_example)
        return self.generate_sync(
            inputs,
            client=client,
            model=model,
            sampling_args=sampling_args,
            max_concurrent=max_concurrent,
            max_concurrent_generation=max_concurrent_generation,
            max_concurrent_scoring=max_concurrent_scoring,
            results_path=results_path,
            state_columns=state_columns,
            save_results=save_results,
            save_every=save_every,
        )

    def set_max_seq_len(self, max_seq_len: int | None) -> None:
        """Set the maximum sequence length for this environment."""
        self.max_seq_len = max_seq_len

    make_dataset = make_dataset


_EnvT = TypeVar("_EnvT", bound=Environment)
StopCondition = Callable[[State], Awaitable[bool]]
RolloutCleanup = Callable[[State], Awaitable[None]]
EnvironmentTeardown = Callable[[], Awaitable[None]]
