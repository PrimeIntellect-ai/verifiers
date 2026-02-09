from __future__ import annotations

import asyncio
import atexit
import functools
import inspect
import json
import logging
import signal
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from multiprocessing import Process
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    List,
    Literal,
    TypeVar,
    cast,
    final,
)

from openai import AsyncOpenAI, AuthenticationError, BadRequestError, OpenAI

from verifiers.utils.client_utils import (
    resolve_client_config,
    resolve_client_configs,
    setup_client,
)
from verifiers.utils.eval_utils import filter_inputs
from verifiers.utils.path_utils import is_valid_eval_results_path
from verifiers.utils.worker_utils import get_free_port, wait_for_env_server
from verifiers.workers.client.zmq_env_client import ZMQEnvClient
from verifiers.workers.server.zmq_env_server import ZMQEnvServer

if TYPE_CHECKING:
    from datasets import Dataset
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.completion_choice import CompletionChoice

import verifiers as vf
from verifiers.parsers.parser import Parser
from verifiers.rubrics.rubric import Rubric
from verifiers.types import (
    ChatCompletionToolParam,
    ChatMessage,
    ClientConfig,
    DatasetBuilder,
    GenerateMetadata,
    GenerateOutputs,
    LogCallback,
    Messages,
    MessageType,
    ModelResponse,
    ProgressCallback,
    RolloutInput,
    RolloutOutput,
    RolloutTiming,
    SamplingArgs,
    StartCallback,
    State,
    TokenUsage,
)
from verifiers.utils.async_utils import (
    maybe_retry,
    maybe_semaphore,
    with_sem,
)
from verifiers.utils.error_utils import ErrorChain
from verifiers.utils.message_utils import (
    strip_nones_from_content,
)
from verifiers.utils.save_utils import (
    GenerateOutputsBuilder,
    load_outputs,
    make_dataset,
    push_results_to_hf_hub,
    save_metadata,
    save_new_outputs,
    save_outputs,
    state_to_output,
    validate_resume_metadata,
)
from verifiers.utils.token_utils import (
    get_prompt_ids,
    prepare_sampling_args_for_token_prompts,
)
from verifiers.utils.usage_utils import StateUsageTracker
from verifiers.workers.client.env_client import EnvClient

if TYPE_CHECKING:
    pass


class Environment(ABC):
    """
    Base class for all environments.
    """

    def __init__(
        self,
        dataset: Dataset | DatasetBuilder | None = None,
        eval_dataset: Dataset | DatasetBuilder | None = None,
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
        interleaved_rollouts: bool = False,
        score_rollouts: bool = True,
        **kwargs,
    ):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
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
        self.map_kwargs = map_kwargs

        self.set_interleaved_rollouts(interleaved_rollouts)
        self.set_score_rollouts(score_rollouts)

        if self.message_type != "chat" and (self.system_prompt or self.few_shot):
            raise ValueError(
                'The fields "system_prompt" and "few_shot" are not supported for completion tasks.'
                'Please use message_type="chat" instead, or pre-format your dataset '
                'to contain a "prompt" column.'
            )

        self.env_client: EnvClient | None = None
        self.env_server_process: Process | None = None

        # Dataset sources (builders) and built datasets
        # Use get_dataset()/get_eval_dataset() for access; build_dataset() to trigger build
        self.dataset: Dataset | None = None
        self.eval_dataset: Dataset | None = None

        if dataset is not None:
            if callable(dataset):
                self.dataset_source: DatasetBuilder | None = dataset
            else:
                self.dataset_source = lambda ds=dataset: ds
                self.build_dataset()  # Eagerly build for raw datasets (backwards compat)
        else:
            self.dataset_source = None

        if eval_dataset is not None:
            if callable(eval_dataset):
                self.eval_dataset_source: DatasetBuilder | None = eval_dataset
            else:
                self.eval_dataset_source = lambda ds=eval_dataset: ds
                self.build_eval_dataset()  # Eagerly build for raw datasets (backwards compat)
        else:
            self.eval_dataset_source = None

        self.sampling_args = {"n": 1, "extra_body": {}}
        if sampling_args is not None:
            # merge extra_body if provided
            cast(dict[str, Any], self.sampling_args["extra_body"]).update(
                cast(dict[str, Any], sampling_args.get("extra_body", {}))
            )
            # copy other keys
            for key, value in sampling_args.items():
                if key != "extra_body":
                    self.sampling_args[key] = value

        self.max_workers = max_workers
        for key, value in kwargs.items():
            setattr(self, key, value)

        if (
            self.dataset_source is None
            and self.eval_dataset_source is None
            and self.dataset is None
            and self.eval_dataset is None
        ):
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
            dataset = dataset.add_column("example_id", range(len(dataset)))
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

        else:
            if system_prompt is not None:

                def prepend_system_prompt(
                    prompt: list[ChatMessage],
                ) -> list[ChatMessage]:
                    assert isinstance(prompt, list), (
                        f"prompt must be a list of ChatMessages when system_prompt is provided, got {type(prompt)}"
                    )
                    if prompt and prompt[0].get("role") == "system":
                        return prompt
                    sys_msg = cast(
                        ChatMessage, {"role": "system", "content": system_prompt}
                    )
                    return [sys_msg, *prompt]

                dataset = dataset.map(
                    lambda x: {"prompt": prepend_system_prompt(x["prompt"])},
                    **map_kwargs,
                )
            if few_shot is not None:
                self.logger.warning(
                    "Dataset already has a 'prompt' column, so the provided few_shot examples will be ignored."
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

    def _format_dataset(
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

    def _format_completion_dataset(
        self, dataset: Dataset, map_kwargs: dict = {}
    ) -> Dataset:
        """
        Format dataset by creating example_id and prompt columns, and setting task column.
        """
        dataset = self._ensure_example_id(dataset)
        dataset = self._ensure_task(dataset, map_kwargs)
        return dataset

    def _format_dataset_source(self, dataset: Dataset) -> Dataset:
        """Format a dataset based on message_type."""
        if self.message_type == "chat":
            return self._format_dataset(
                dataset,
                self.system_prompt,
                self.few_shot,
                map_kwargs=self.map_kwargs,
            )
        else:
            return self._format_completion_dataset(dataset, map_kwargs=self.map_kwargs)

    def build_dataset(self) -> Dataset | None:
        """Build and cache the training dataset from source if needed."""
        if self.dataset is not None:
            return self.dataset
        if self.dataset_source is None:
            return None
        built = self.dataset_source()
        self.dataset = self._format_dataset_source(built)
        return self.dataset

    def build_eval_dataset(self) -> Dataset | None:
        """Build and cache the evaluation dataset from source if needed."""
        if self.eval_dataset is not None:
            return self.eval_dataset
        if self.eval_dataset_source is None:
            return None
        built = self.eval_dataset_source()
        self.eval_dataset = self._format_dataset_source(built)
        return self.eval_dataset

    @final
    def get_dataset(self, n: int = -1, seed: int | None = None) -> Dataset:
        self.build_dataset()
        if self.dataset is None:
            raise ValueError("dataset is not set")
        if seed is not None:
            self.dataset = self.dataset.shuffle(seed=seed)
        if n > 0:
            n = min(n, len(self.dataset))
            return self.dataset.select(range(n))
        return self.dataset

    @final
    def get_eval_dataset(self, n: int = -1, seed: int | None = None) -> Dataset:
        self.build_eval_dataset()
        if self.eval_dataset is None:
            self.logger.warning(
                "eval_dataset is not set, falling back to train dataset"
            )
            return self.get_dataset(n, seed)
        if seed is not None:
            self.eval_dataset = self.eval_dataset.shuffle(seed=seed)
        if n > 0:
            n = min(n, len(self.eval_dataset))
            return self.eval_dataset.select(range(n))
        return self.eval_dataset

    @final
    def _get_usage_tracker(
        self, state: State, create_if_missing: bool = True
    ) -> StateUsageTracker | None:
        tracker = state.get("usage_tracker")
        if isinstance(tracker, StateUsageTracker):
            return tracker
        if not create_if_missing:
            return None
        tracker = StateUsageTracker()
        state["usage_tracker"] = tracker
        # Expose read-only usage in state for live inspection.
        state["usage"] = tracker.usage
        return tracker

    @final
    def increment_state_usage(
        self,
        state: State,
        input_tokens: int | float = 0,
        output_tokens: int | float = 0,
    ) -> None:
        tracker = self._get_usage_tracker(state, create_if_missing=True)
        assert tracker is not None
        tracker.increment(input_tokens, output_tokens)

    @final
    def increment_state_usage_from_response(
        self, state: State, response: object
    ) -> None:
        tracker = self._get_usage_tracker(state, create_if_missing=True)
        assert tracker is not None
        tracker.increment_from_response(response)

    @final
    def get_state_usage(self, state: State) -> TokenUsage | None:
        tracker = self._get_usage_tracker(state, create_if_missing=False)
        if tracker is not None:
            return tracker.snapshot()
        usage = state.get("usage")
        if isinstance(usage, Mapping):
            try:
                input_tokens = float(usage.get("input_tokens", 0.0))
                output_tokens = float(usage.get("output_tokens", 0.0))
            except (TypeError, ValueError):
                return None
            return {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        return None

    async def get_model_response(
        self,
        state: State,
        prompt: Messages,
        client: AsyncOpenAI | None = None,
        model: str | None = None,
        oai_tools: list[ChatCompletionToolParam] | None = None,
        sampling_args: SamplingArgs | None = None,
        message_type: MessageType | None = None,
    ) -> ModelResponse:
        """
        Get model response for a given prompt (chat or completion).

        Convenience function for wrapping (chat, completion) API calls.
        Returns special error messages for context length issues.

        If interleaved_rollouts is set, the model response is obtained by
        calling a custom token-in endpoint. Note, that this only works if the
        inference server implements this endpoint.  Currently, this is a
        hand-crafted feature for PRIME-RL's vLLM server extension, and is not
        recommended for general use.
        """

        def resolve_optional_args(
            client: AsyncOpenAI | None,
            model: str | None,
            oai_tools: list[ChatCompletionToolParam] | None,
            sampling_args: SamplingArgs | None,
            message_type: MessageType | None,
        ) -> tuple[
            AsyncOpenAI,
            str,
            list[ChatCompletionToolParam] | None,
            SamplingArgs,
            MessageType,
        ]:
            """Resolve optional arguments, fallback to state or class defaults."""
            client = client or state["client"]
            model = model or state["model"]
            assert client is not None and model is not None
            oai_tools = oai_tools or state["oai_tools"]
            sampling_args = cast(
                SamplingArgs, sampling_args or state["sampling_args"] or {}
            )
            message_type = message_type or self.message_type
            return client, model, oai_tools, sampling_args, message_type

        def normalize_sampling_args(sampling_args: SamplingArgs) -> SamplingArgs:
            """
            Normalize sampling arguments. Mainly does 2 things:
            - if max_tokens is provided for chat, rename to max_completion_tokens
            - drop any None-valued entries to avoid sending to the client
            """
            if "max_tokens" in sampling_args:
                if sampling_args["max_tokens"] is None:
                    sampling_args.pop("max_tokens")
                elif message_type == "chat":
                    sampling_args["max_completion_tokens"] = sampling_args.pop(
                        "max_tokens"
                    )
            if (
                "max_completion_tokens" in sampling_args
                and sampling_args["max_completion_tokens"] is None
            ):
                sampling_args.pop("max_completion_tokens")
            return {k: v for k, v in sampling_args.items() if v is not None}

        def handle_overlong_prompt(func):
            """Decorator to handle overlong prompt errors from the model API."""

            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except AuthenticationError:
                    # re-raise auth errors to exit immediately
                    raise
                except BadRequestError as e:
                    # in case of making a request with an overlong prompt, e.g
                    # we raise a special overlong prompt error
                    error_text = e.response.text.lower()
                    context_length_phrases = [
                        "maximum context length is",
                        "is longer than the model's context length",
                        "exceeds the model's context length",
                        "exceed the configured limit",
                        "exceeds the configured limit",
                        "exceeded model",
                    ]
                    if any(phrase in error_text for phrase in context_length_phrases):
                        self.logger.debug("Caught overlong prompt.")
                        raise vf.OverlongPromptError from e
                    raise vf.ModelError from e
                except Exception as e:
                    # in all other case we raise a generic model error
                    raise vf.ModelError from e

            return wrapper

        @handle_overlong_prompt
        async def get_model_response_with_messages(
            client: AsyncOpenAI,
            model: str,
            prompt: Messages,
            oai_tools: list[ChatCompletionToolParam] | None,
            sampling_args: SamplingArgs,
            message_type: MessageType,
        ) -> ModelResponse:
            """Convenience function for wrapping (chat, completion) API calls."""
            if message_type == "chat":
                assert isinstance(prompt, list)
                prompt = strip_nones_from_content(prompt)
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
                if has_audio and "modalities" not in sampling_args:
                    sampling_args = {
                        **sampling_args,
                        "modalities": ["text"],
                    }

                if oai_tools:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=prompt,
                        tools=oai_tools,
                        **sampling_args,
                    )
                else:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=prompt,
                        **sampling_args,
                    )
                return response
            elif message_type == "completion":
                if oai_tools:
                    raise ValueError(
                        "oai_tools are not supported for completion tasks."
                    )
                assert isinstance(prompt, str)
                response = await client.completions.create(
                    model=model, prompt=prompt, **sampling_args
                )
                return response

        @handle_overlong_prompt
        async def get_model_response_with_tokens(
            client: AsyncOpenAI,
            model: str,
            prompt: Messages,
            prompt_ids: list[int],
            oai_tools: list[ChatCompletionToolParam] | None,
            sampling_args: SamplingArgs,
            message_type: MessageType,
        ) -> ModelResponse:
            """
            Get a model response with pre-tokenized prompt from custom
            /v1/chat/completions/tokens endpoint (only available in PRIME-RL's
            vLLM server extension)
            """
            assert message_type == "chat", (
                "get_model_response_with_tokens is only supported for chat tasks."
            )

            extra_body = sampling_args.pop("extra_body", {})
            body = dict(
                model=model,
                messages=prompt,
                tools=oai_tools,
                tokens=prompt_ids,
                **sampling_args,
                **extra_body,
            )

            return await client.post(
                "/chat/completions/tokens",
                body=body,
                cast_to=ChatCompletion,
            )

        client, model, oai_tools, sampling_args, message_type = resolve_optional_args(
            client, model, oai_tools, sampling_args, message_type
        )
        self._get_usage_tracker(state, create_if_missing=True)
        sampling_args = normalize_sampling_args(sampling_args)
        if self.interleaved_rollouts:
            sampling_args = prepare_sampling_args_for_token_prompts(sampling_args)

        if self.interleaved_rollouts and len(state["trajectory"]) > 0:
            prompt_ids = await get_prompt_ids(state, prompt, client)
            response = await get_model_response_with_tokens(
                client=client,
                model=model,
                prompt=prompt,
                prompt_ids=prompt_ids,
                oai_tools=oai_tools,
                sampling_args=sampling_args,
                message_type=message_type,
            )
        else:
            response = await get_model_response_with_messages(
                client=client,
                model=model,
                prompt=prompt,
                oai_tools=oai_tools,
                sampling_args=sampling_args,
                message_type=message_type,
            )

        # Some providers (e.g. OpenRouter) may return None for response or response.choices
        if response is None:
            raise vf.EmptyModelResponseError("Model returned no response")
        self.increment_state_usage_from_response(state, response)
        if response.choices is None:
            raise vf.EmptyModelResponseError("Model returned no response choices")
        if not len(response.choices) == 1:
            raise vf.InvalidModelResponseError(
                f"Model returned {len(response.choices)} choices, expected 1"
            )
        if isinstance(response.choices[0], Choice):
            if not (
                response.choices[0].message.content
                or response.choices[0].message.tool_calls
            ):
                raise vf.EmptyModelResponseError(
                    "Model returned no content and did not call any tools"
                )
        elif isinstance(response.choices[0], CompletionChoice):
            if not response.choices[0].text:
                raise vf.EmptyModelResponseError("Model returned no content")

        return response

    @final
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
        state_input = cast(RolloutInput, deepcopy(input))
        if "info" in state_input and isinstance(state_input["info"], str):
            state_input["info"] = json.loads(state_input["info"])
        if "task" not in state_input:
            state_input["task"] = self.env_id or "default"
        state = State(input=state_input)
        state["client"] = client
        state["model"] = model
        state["sampling_args"] = sampling_args
        state["is_completed"] = False
        state["is_truncated"] = False
        state["oai_tools"] = None
        info = state.get("info")
        if isinstance(info, dict) and "oai_tools" in info:
            state["oai_tools"] = info["oai_tools"]
        elif info is not None and hasattr(info, "oai_tools"):
            state["oai_tools"] = getattr(info, "oai_tools")
        elif hasattr(self, "oai_tools"):
            state["oai_tools"] = self.oai_tools
        else:
            state["oai_tools"] = []
        state["trajectory"] = []
        self._get_usage_tracker(state, create_if_missing=True)
        state["trajectory_id"] = uuid.uuid4().hex
        state["reward"] = None
        state["metrics"] = None
        state["error"] = None
        state["final_env_response"] = None
        state["timing"] = RolloutTiming(
            generation_ms=0.0,
            scoring_ms=0.0,
            total_ms=0.0,
            start_time=time.time(),
        )
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
            state["is_truncated"] = state.get("is_truncated", False) or any(
                step.get("is_truncated", False) for step in state.get("trajectory", [])
            )
            state["stop_condition"] = condition.__name__
            if state.get("stop_condition") == "has_error":
                err = state["error"]
                err_chain = ErrorChain(err)
                self.logger.error(f"Aborted rollout due to {repr(err_chain)}")
            return True
        return False

    async def _render_timing(self, state: State):
        start_time = state["timing"]["start_time"]
        end_time = time.time()
        state["timing"]["generation_ms"] = (end_time - start_time) * 1000
        state["timing"]["total_ms"] = (end_time - start_time) * 1000

    @final
    async def is_completed(self, state: State, **kwargs) -> bool:
        """Check all stop conditions. Sets state.is_completed=True if any condition is met."""
        for condition in self._stop_conditions:
            if await self._render_stop(state, condition):
                await self._render_timing(state)
                await self._cleanup(state)
                return True
        return False

    @final
    async def run_rollout(
        self,
        input: RolloutInput,
        client: AsyncOpenAI | ClientConfig,
        model: str,
        sampling_args: SamplingArgs,
        max_retries: int = 0,
        state_columns: list[str] | None = None,
        env_client: EnvClient | None = None,
    ) -> RolloutOutput:
        """Generate and, optionally, score a rollout."""

        resolved_client_config: ClientConfig | None = None
        if isinstance(client, ClientConfig):
            resolved_client_config = resolve_client_config(client)

        env_client = env_client or self.env_client
        if env_client is not None:  # in server mode
            if resolved_client_config is None:
                raise ValueError(
                    f"client must be have type ClientConfig in server mode, got {type(client)}"
                )
            return await env_client.run_rollout(
                input,
                resolved_client_config,
                model,
                sampling_args,
                max_retries,
                state_columns,
            )

        local_client: AsyncOpenAI
        owned_local_client: AsyncOpenAI | None = None
        if resolved_client_config is not None:
            owned_local_client = setup_client(resolved_client_config)
            local_client = owned_local_client
        else:
            local_client = cast(AsyncOpenAI, client)

        async def run_rollout_attempt() -> State:
            state = await self.rollout(input, local_client, model, sampling_args)

            if self.score_rollouts:
                await self.rubric.score_rollout(state)
            else:
                await self.rubric.dummy_score_rollout(state)

            return state

        try:
            state = await maybe_retry(run_rollout_attempt, max_retries=max_retries)()
        finally:
            if owned_local_client is not None:
                await owned_local_client.close()
        output = state_to_output(state, state_columns or [])
        return output

    @final
    async def run_group(
        self,
        group_inputs: list[RolloutInput],
        client: AsyncOpenAI | ClientConfig,
        model: str,
        sampling_args: SamplingArgs,
        max_retries: int = 0,
        state_columns: list[str] | None = None,
        env_client: EnvClient | None = None,
        **kwargs,
    ) -> list[RolloutOutput]:
        """Generate and, optionally, score one group."""

        resolved_client_config: ClientConfig | None = None
        if isinstance(client, ClientConfig):
            resolved_client_config = resolve_client_config(client)

        env_client = env_client or self.env_client
        if env_client is not None:  # in server mode
            if resolved_client_config is None:
                raise ValueError(
                    f"client must be have type ClientConfig in server mode, got {type(client)}"
                )
            return await env_client.run_group(
                group_inputs,
                resolved_client_config,
                model,
                sampling_args,
                max_retries,
                state_columns,
            )

        local_client: AsyncOpenAI
        owned_local_client: AsyncOpenAI | None = None
        if resolved_client_config is not None:
            owned_local_client = setup_client(resolved_client_config)
            local_client = owned_local_client
        else:
            local_client = cast(AsyncOpenAI, client)

        async def run_group_attempt() -> list[State]:
            rollout_tasks = [
                self.rollout(input, local_client, model, sampling_args)
                for input in group_inputs
            ]
            group_states = await asyncio.gather(*rollout_tasks)

            if self.score_rollouts:
                await self.rubric.score_group(group_states)
            else:
                await self.rubric.dummy_score_group(group_states)
            return group_states

        try:
            group_states = await maybe_retry(
                run_group_attempt, max_retries=max_retries
            )()
        finally:
            if owned_local_client is not None:
                await owned_local_client.close()
        outputs = [
            state_to_output(state, state_columns or []) for state in group_states
        ]
        return outputs

    async def generate(
        self,
        inputs: Dataset | List[RolloutInput],
        client: AsyncOpenAI | ClientConfig,
        model: str,
        sampling_args: SamplingArgs | None = None,
        max_concurrent: int = -1,
        results_path: Path | None = None,
        state_columns: list[str] | None = None,
        save_results: bool = False,
        push_to_hf_hub: bool = False,
        hf_hub_dataset_name: str | None = None,
        independent_scoring: bool = False,
        max_retries: int = 0,
        on_start: StartCallback | None = None,
        on_progress: ProgressCallback | None = None,
        on_log: LogCallback | None = None,
    ) -> GenerateOutputs:
        """
        Generate rollouts for a set of inputs.

        Args:
            client: Can be a single AsyncOpenAI client or a ClientConfig.
        """
        from datasets import Dataset
        from tqdm import tqdm

        pbar: tqdm | None = None

        def default_on_start(
            raw_inputs: list[RolloutInput],
            filtered_inputs: list[RolloutInput] | list[list[RolloutInput]],
        ) -> None:
            """Initializes the progress bar from the raw inputs."""
            nonlocal pbar

            total_rollouts = len(raw_inputs)
            total_groups = len(set([i["example_id"] for i in raw_inputs]))
            rollouts_per_example = (
                total_rollouts // total_groups if total_groups > 0 else 0
            )

            if (
                isinstance(filtered_inputs, list)
                and filtered_inputs
                and isinstance(filtered_inputs[0], list)
            ):
                remaining_rollouts = sum(len(g) for g in filtered_inputs)
            else:
                remaining_rollouts = len(filtered_inputs)
            saved_rollouts = total_rollouts - remaining_rollouts

            if filtered_inputs:
                if isinstance(filtered_inputs[0], list):
                    pbar_total = total_groups
                    pbar_initial = saved_rollouts // rollouts_per_example
                    pbar_desc = f"Processing {total_groups} groups ({total_rollouts} total rollouts)"
                else:
                    pbar_total = total_rollouts
                    pbar_initial = saved_rollouts
                    pbar_desc = f"Processing {total_rollouts} rollouts"

                pbar = tqdm(
                    total=pbar_total,
                    initial=pbar_initial,
                    desc=pbar_desc,
                    postfix=dict(reward="?"),
                )

        def default_on_progress(
            all_outputs: list[RolloutOutput],
            new_outputs: list[RolloutOutput],
            new_metadata: GenerateMetadata,
        ) -> None:
            """Updates the progress bar from the new outputs."""
            nonlocal pbar
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(reward=new_metadata.get("avg_reward"))

        def default_on_log(message: str) -> None:
            """Logs using the environment logger."""
            self.logger.info(message)

        on_start = on_start or cast(StartCallback, default_on_start)
        on_progress = on_progress or cast(ProgressCallback, default_on_progress)
        on_log = on_log or cast(LogCallback, default_on_log)

        if isinstance(inputs, Dataset):
            raw_inputs = cast(list[RolloutInput], inputs.to_list())
        elif isinstance(inputs, list):
            raw_inputs = inputs

        # set up semaphores
        sem = await maybe_semaphore(max_concurrent)

        # set up sampling args
        default_sampling_args = deepcopy(self.sampling_args)
        if sampling_args is not None:
            default_sampling_args.update(sampling_args)
        sampling_args = default_sampling_args

        # initialize outputs builder
        total_rollouts = len(raw_inputs)
        num_examples = len(set([i["example_id"] for i in raw_inputs]))
        rollouts_per_example = total_rollouts // num_examples if num_examples > 0 else 0
        builder = GenerateOutputsBuilder(
            env_id=self.env_id,
            env_args=self.env_args,
            model=model,
            client=client,
            num_examples=num_examples,
            rollouts_per_example=rollouts_per_example,
            state_columns=state_columns,
            sampling_args=sampling_args,
            results_path=results_path,
        )

        single_client: AsyncOpenAI | None = None
        endpoint_client_configs: list[ClientConfig] = []
        endpoint_client_idx = 0
        if isinstance(client, ClientConfig):
            endpoint_client_configs = resolve_client_configs(client)
        else:
            # AsyncOpenAI path
            single_client = client

        local_endpoint_clients: list[AsyncOpenAI] = []

        def get_client_for_group() -> AsyncOpenAI | ClientConfig:
            """Get next client in round-robin order or return the single client."""
            nonlocal endpoint_client_idx
            if self.env_client is not None and endpoint_client_configs:
                config = endpoint_client_configs[
                    endpoint_client_idx % len(endpoint_client_configs)
                ]
                endpoint_client_idx += 1
                return config
            if local_endpoint_clients:
                local_client = local_endpoint_clients[
                    endpoint_client_idx % len(local_endpoint_clients)
                ]
                endpoint_client_idx += 1
                return local_client
            assert single_client is not None
            return single_client

        try:
            if self.env_client is None and endpoint_client_configs:
                for endpoint_config in endpoint_client_configs:
                    local_endpoint_clients.append(setup_client(endpoint_config))

            # load existing results if available
            if results_path is not None and is_valid_eval_results_path(results_path):
                validate_resume_metadata(
                    results_path=results_path,
                    env_id=self.env_id,
                    model=model,
                    num_examples=num_examples,
                    rollouts_per_example=rollouts_per_example,
                )
                on_log(f"Resuming evaluation from {results_path}")
                outputs = load_outputs(results_path)
                builder.add_outputs(outputs)
                filtered_inputs = filter_inputs(
                    raw_inputs, outputs, rollouts_per_example
                )
                if not filtered_inputs:
                    on_log(
                        "No remaining rollouts to evaluate, returning completed outputs"
                    )
                    return builder.build(sort_by_example_id=True)
                on_log(
                    f"Found {len(outputs)} completed rollout(s), {len(filtered_inputs)} remaining rollout(s)"
                )
            else:
                filtered_inputs = raw_inputs

            if save_results:
                on_log(f"Saving results to {builder.results_path}")

            tasks: dict[asyncio.Task, int] = {}
            try:
                # create tasks based on mode
                if independent_scoring:
                    on_start(raw_inputs, filtered_inputs)
                    for i, rollout_input in enumerate(filtered_inputs):
                        task = asyncio.create_task(
                            with_sem(
                                sem,
                                self.run_rollout(
                                    rollout_input,
                                    get_client_for_group(),
                                    model,
                                    sampling_args,
                                    max_retries=max_retries,
                                    state_columns=state_columns,
                                ),
                            ),
                        )
                        tasks[task] = i
                else:
                    group_inputs: dict[int, list[RolloutInput]] = defaultdict(list)
                    for rollout_input in filtered_inputs:
                        example_id = rollout_input["example_id"]
                        group_inputs[example_id].append(rollout_input)
                    filtered_group_inputs = list(group_inputs.values())
                    on_start(raw_inputs, filtered_group_inputs)

                    for i, group_input in enumerate(filtered_group_inputs):
                        # For grouped scoring, keep each group on one endpoint so
                        # rollouts in the same group can benefit from shared KV cache.
                        group_client: AsyncOpenAI | ClientConfig = (
                            get_client_for_group()
                        )
                        task = asyncio.create_task(
                            with_sem(
                                sem,
                                self.run_group(
                                    group_input,
                                    group_client,
                                    model,
                                    sampling_args,
                                    max_retries=max_retries,
                                    state_columns=state_columns,
                                ),
                            ),
                        )
                        tasks[task] = i

                for coro in asyncio.as_completed(tasks.keys()):
                    result = await coro

                    # normalize: independent_scoring returns RolloutOutput, group returns list[RolloutOutput]
                    new_outputs = [result] if independent_scoring else result
                    builder.add_outputs(new_outputs)
                    metadata = builder.build_metadata()

                    on_progress(builder.outputs, new_outputs, metadata)

                    # incrementally save outputs
                    if save_results:
                        save_new_outputs(new_outputs, builder.results_path)
                        save_metadata(metadata, builder.results_path)
            finally:
                # cancel all outstanding tasks and await their completion
                pending = [task for task in tasks.keys() if not task.done()]
                if pending:
                    for task in pending:
                        task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)

            # build final results (sorted by example_id for deterministic ordering)
            results = builder.build(sort_by_example_id=True)

            # save if requested
            if save_results:
                save_outputs(results["outputs"], builder.results_path)
                save_metadata(results["metadata"], builder.results_path)
                if push_to_hf_hub:
                    push_results_to_hf_hub(results, hf_hub_dataset_name)
                if on_log is not None:
                    on_log(
                        f"Saved final results to {results['metadata']['path_to_save']}"
                    )

            return results
        finally:
            if pbar is not None:
                pbar.close()
            if local_endpoint_clients:
                await asyncio.gather(
                    *(client.close() for client in local_endpoint_clients),
                    return_exceptions=True,
                )

    def generate_sync(
        self,
        inputs: Dataset | List[RolloutInput],
        client: AsyncOpenAI | OpenAI | ClientConfig,
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
            import nest_asyncio

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
    def _get_eval_inputs(
        self, num_examples: int = -1, rollouts_per_example: int = 1
    ) -> List[RolloutInput]:
        # get_eval_dataset handles fallback to train dataset if no eval source exists
        inputs = self.get_eval_dataset(n=num_examples)
        assert inputs is not None, "No dataset found"
        if rollouts_per_example > 1:
            inputs = inputs.repeat(rollouts_per_example)
        return inputs.to_list()

    async def evaluate(
        self,
        client: AsyncOpenAI | ClientConfig,
        model: str,
        sampling_args: SamplingArgs | None = None,
        num_examples: int = -1,
        rollouts_per_example: int = 1,
        max_concurrent: int = -1,
        results_path: Path | None = None,
        state_columns: list[str] | None = None,
        save_results: bool = False,
        push_to_hf_hub: bool = False,
        hf_hub_dataset_name: str | None = None,
        independent_scoring: bool = False,
        max_retries: int = 0,
        on_start: StartCallback | None = None,
        on_progress: ProgressCallback | None = None,
        on_log: LogCallback | None = None,
        **kwargs,
    ) -> GenerateOutputs:
        """
        Evaluate model on the Environment evaluation dataset.
        """
        inputs = self._get_eval_inputs(num_examples, rollouts_per_example)
        return await self.generate(
            inputs,
            client=client,
            model=model,
            sampling_args=sampling_args,
            max_concurrent=max_concurrent,
            results_path=results_path,
            state_columns=state_columns,
            save_results=save_results,
            push_to_hf_hub=push_to_hf_hub,
            hf_hub_dataset_name=hf_hub_dataset_name,
            independent_scoring=independent_scoring,
            max_retries=max_retries,
            on_start=on_start,
            on_progress=on_progress,
            on_log=on_log,
            **kwargs,
        )

    def evaluate_sync(
        self,
        client: OpenAI | AsyncOpenAI | ClientConfig,
        model: str,
        sampling_args: SamplingArgs | None = None,
        num_examples: int = -1,
        rollouts_per_example: int = 1,
        max_concurrent: int = -1,
        results_path: Path | None = None,
        state_columns: list[str] | None = None,
        save_results: bool = False,
        push_to_hf_hub: bool = False,
        hf_hub_dataset_name: str | None = None,
        independent_scoring: bool = False,
        max_retries: int = 0,
    ) -> GenerateOutputs:
        """
        Evaluate model on the Environment evaluation dataset synchronously.
        """
        inputs = self._get_eval_inputs(num_examples, rollouts_per_example)
        return self.generate_sync(
            inputs,
            client=client,
            model=model,
            sampling_args=sampling_args,
            max_concurrent=max_concurrent,
            results_path=results_path,
            state_columns=state_columns,
            save_results=save_results,
            push_to_hf_hub=push_to_hf_hub,
            hf_hub_dataset_name=hf_hub_dataset_name,
            independent_scoring=independent_scoring,
            max_retries=max_retries,
        )

    # setters for use by trainers
    def set_kwargs(self, **kwargs) -> None:
        """
        Set environment attributes, using setter methods when available.

        For each kwarg, checks if a `set_{key}` method exists and calls it,
        otherwise falls back to setattr. This ensures proper propagation for
        attributes like `interleaved_rollouts` in EnvGroup.
        """
        for key, value in kwargs.items():
            setter_name = f"set_{key}"
            setter = getattr(self, setter_name, None)
            if setter is not None and callable(setter):
                setter(value)
            else:
                setattr(self, key, value)

    def add_rubric(self, rubric: Rubric) -> None:
        if self.rubric is None:
            self.rubric = rubric
        elif isinstance(self.rubric, vf.RubricGroup):
            self.rubric.rubrics.append(rubric)
        else:
            self.rubric = vf.RubricGroup(rubrics=[self.rubric, rubric])

    def set_max_seq_len(self, max_seq_len: int | None) -> None:
        """Set the maximum sequence length for this environment."""
        self.max_seq_len = max_seq_len

    def set_interleaved_rollouts(self, interleaved_rollouts: bool) -> None:
        """Set the interleaved rollouts flag for this environment."""
        self.interleaved_rollouts = interleaved_rollouts
        if self.interleaved_rollouts:
            self.logger.warning(
                f"{self.__class__.__name__} is configured to use interleaved rollouts. All model responses after the first turn will be pre-tokenized before being sent to the model. Currently, this is a hand-crafted feature for PRIME-RL's vLLM server extension."
            )

    async def start_server(
        self,
        address: str | None = None,
        extra_env_kwargs: dict[str, Any] = {},
        log_level: str | None = None,
        log_file: str | None = None,
        log_file_level: str | None = None,
        startup_timeout: float = 3600,  # 1h
    ) -> None:
        """Start a ZMQ server process for this environment.

        .. warning::
            This method is subject to change. External users should avoid
            depending on it directly.
        """
        address = address or f"tcp://127.0.0.1:{get_free_port()}"
        self.env_server_process = Process(
            target=ZMQEnvServer.run_server,
            args=(
                self.env_id,
                self.env_args,
                extra_env_kwargs,
                log_level,
                log_file,
                log_file_level,
            ),
            kwargs=dict(address=address),
            daemon=True,  # ensure server process is terminated when parent exits
        )
        self.env_server_process.start()
        self.env_client = ZMQEnvClient(address=address)
        await wait_for_env_server(self.env_client, timeout=startup_timeout)

    async def stop_server(self) -> None:
        """Stop the ZMQ server process for this environment.

        .. warning::
            This method is subject to change. External users should avoid
            depending on it directly.
        """
        if self.env_client is not None:
            await self.env_client.close()
            self.env_client = None
        if self.env_server_process is not None:
            self.env_server_process.terminate()
            self.env_server_process.join(timeout=5)
            if self.env_server_process.is_alive():
                self.env_server_process.kill()
                self.env_server_process.join(timeout=5)
            self.env_server_process = None

    def set_score_rollouts(self, score_rollouts: bool) -> None:
        """Set the score rollouts flag for this environment."""
        self.score_rollouts = score_rollouts

    make_dataset = staticmethod(make_dataset)


_EnvT = TypeVar("_EnvT", bound=Environment)
StopCondition = Callable[[State], Awaitable[bool]]
RolloutCleanup = Callable[[State], Awaitable[None]]
EnvironmentTeardown = Callable[[], Awaitable[None]]
