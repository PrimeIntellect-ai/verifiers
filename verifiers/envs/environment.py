import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, AsyncContextManager, Literal

from datasets import Dataset
from openai import AsyncOpenAI, BadRequestError, OpenAI

from verifiers.parsers.parser import Parser
from verifiers.rubrics.rubric import Rubric
from verifiers.types import (
    ChatCompletionToolParam,
    ChatMessage,
    GenerateInputs,
    GenerateMetadata,
    GenerateOutputs,
    Info,
    Messages,
    MessageType,
    ModelResponse,
    ProcessedOutputs,
    RewardFunc,
    SamplingArgs,
    State,
)
from verifiers.utils.async_utils import maybe_semaphore
from verifiers.utils.eval_utils import make_dataset, save_results
from verifiers.utils.message_utils import (
    cleanup_messages,
    get_overlong_prompt_dummy_response,
)
from verifiers.utils.path_utils import get_results_path
from verifiers.utils.processing_utils import (
    process_chat_format_vllm,
    process_completion_format_vllm,
)

if TYPE_CHECKING:
    from transformers.tokenization_utils_base import (  # type: ignore
        PreTrainedTokenizerBase,
    )


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
            self.sampling_args["extra_body"].update(sampling_args.get("extra_body", {}))
            # copy other keys
            for key, value in sampling_args.items():
                if key != "extra_body":
                    self.sampling_args[key] = value

        self.max_workers = max_workers
        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.dataset is None and self.eval_dataset is None:
            raise ValueError("Either dataset or eval_dataset must be provided")

        self.env_id = env_id or ""
        self.env_args = env_args or {}
        self.rollouts_per_example = None

    def format_prompt(
        self,
        prompt_str: str,
        system_prompt: str | None = None,
        few_shot: list[ChatMessage] | None = None,
    ) -> list[ChatMessage]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if few_shot:
            messages.extend(few_shot)
        messages.append({"role": "user", "content": prompt_str})
        return messages

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
        Create `example_id` and `prompt` columns if not present.
        """
        # if "id" column is present and not int, rename it to "src_id"
        if "example_id" in dataset.column_names and not isinstance(
            dataset["example_id"][0], int
        ):
            dataset = dataset.rename_column("example_id", "src_id")
        if "example_id" not in dataset.column_names:
            dataset = dataset.add_column("example_id", range(len(dataset)))  # type: ignore

        # extract format_prompt as a standalone function to avoid capturing self
        def format_prompt_fn(prompt_str: str) -> list[ChatMessage]:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if few_shot:
                messages.extend(few_shot)
            messages.append({"role": "user", "content": prompt_str})
            return messages

        if "prompt" not in dataset.column_names:
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
        assert "example_id" in dataset.column_names
        assert "prompt" in dataset.column_names
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

    def get_reward_funcs(self) -> list[RewardFunc]:
        return self.rubric.get_reward_funcs()

    def get_reward_weights(self) -> list[float]:
        return self.rubric.get_reward_weights()

    async def get_model_response(
        self,
        client: AsyncOpenAI,
        model: str,
        prompt: Messages,
        oai_tools: list[ChatCompletionToolParam] | None = None,
        sampling_args: SamplingArgs | None = None,
        message_type: MessageType | None = None,
        **kwargs,
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
                        c = m.get("content")  # type: ignore[assignment]
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
                        messages=prompt,  # type: ignore
                        tools=oai_tools,
                        **clean_sampling_args,
                    )
                else:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=prompt,  # type: ignore
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
                ]
                if any(phrase in error_text for phrase in context_length_phrases):
                    self.logger.debug("Caught overlong prompt.")
                    return get_overlong_prompt_dummy_response(
                        message_type or self.message_type
                    )
            self.logger.error(f"Error getting model response: {e} \n\nExiting...")
            raise e

    @abstractmethod
    async def rollout(
        self,
        client: AsyncOpenAI,
        model: str,
        prompt: Messages,
        completion: Messages | None = None,
        answer: str = "",
        state: State = {},
        task: str = "default",
        info: Info | None = None,
        example_id: int = 0,
        sampling_args: SamplingArgs | None = None,
        **kwargs,
    ) -> tuple[Messages, State]:
        """
        Run a rollout for a given prompt.
        Returns a tuple of (completion, state).
        """
        pass

    async def run_rollout(
        self,
        sem: AsyncContextManager,
        client: AsyncOpenAI,
        model: str,
        prompt: Messages,
        completion: Messages | None = None,
        answer: str = "",
        state: State = {},
        task: str = "default",
        info: Info | None = None,
        example_id: int = 0,
        sampling_args: SamplingArgs | None = None,
        **kwargs,
    ) -> tuple[Messages, State]:
        """
        Run a rollout with a semaphore.
        """
        async with sem:
            return await self.rollout(
                client,
                model,
                prompt,
                completion,
                answer,
                state,
                task,
                info,
                example_id,
                sampling_args,
                **kwargs,
            )

    async def run_rollouts(
        self,
        client: AsyncOpenAI,
        model: str,
        prompts: list[Messages],
        completions: list[Messages] = [],
        answers: list[str] = [],
        states: list[State] = [],
        tasks: list[str] = [],
        infos: list[Info] = [],
        example_ids: list[int] = [],
        sampling_args: SamplingArgs | None = None,
        max_concurrent: int = -1,
        semaphore: asyncio.Semaphore | None = None,
        use_tqdm: bool = True,
        **kwargs,
    ) -> list[tuple[Messages, State]]:
        """
        Run rollouts for a given list of prompts and return the completions.
        """

        maybe_sem = semaphore or (await maybe_semaphore(max_concurrent))
        if not example_ids:
            example_ids = list(range(len(prompts)))
        if len(completions) == 0:
            completions = [await self.init_completion() for _ in range(len(prompts))]
        if len(states) == 0:
            states = [
                await self.init_state(
                    prompt, completion, answer, task, info, example_id
                )
                for prompt, completion, answer, task, info, example_id in zip(
                    prompts, completions, answers, tasks, infos, example_ids
                )
            ]
        rollout_tasks = [
            self.run_rollout(
                maybe_sem,
                client,
                model,
                prompt,
                completion,
                answer,
                state,
                task,
                info,
                example_id,
                sampling_args,
                **kwargs,
            )
            for prompt, completion, answer, state, task, info, example_id in zip(
                prompts, completions, answers, states, tasks, infos, example_ids
            )
        ]
        if use_tqdm:
            from tqdm.asyncio import tqdm_asyncio

            rollout_results = await tqdm_asyncio.gather(
                *rollout_tasks,
                total=len(prompts),
                desc=f"Running {len(prompts)} rollouts",
            )
        else:
            rollout_results = await asyncio.gather(*rollout_tasks)
        return list(rollout_results)

    async def init_completion(self) -> Messages:
        if self.message_type == "chat":
            return []
        else:
            return ""

    async def init_state(
        self,
        prompt: Messages,
        completion: Messages,
        answer: str,
        task: str,
        info: Info,
        example_id: int,
        **kwargs,
    ) -> State:
        state = {
            "prompt": prompt,
            "completion": completion,
            "answer": answer,
            "task": task,
            "info": info,
            "example_id": example_id,
            "responses": [],
            "turn": 0,
            "timing": {
                "generation_ms": 0.0,
                "scoring_ms": 0.0,
                "total_ms": 0.0,
            },
        }
        return state

    async def generate(
        self,
        inputs: GenerateInputs | Dataset | dict,
        client: AsyncOpenAI,
        model: str,
        sampling_args: SamplingArgs | None = None,
        num_examples: int | None = None,
        rollouts_per_example: int | None = None,
        score_rollouts: bool = True,
        max_concurrent: int = -1,
        max_concurrent_generation: int | None = None,
        max_concurrent_scoring: int | None = None,
        semaphore: asyncio.Semaphore | None = None,
        generation_semaphore: asyncio.Semaphore | None = None,
        scoring_semaphore: asyncio.Semaphore | None = None,
        interleave_scoring: bool = True,
        results_path: Path | None = None,
        state_columns: list[str] | None = None,
        save_every: int = -1,
        use_tqdm: bool = True,
        **kwargs,
    ) -> GenerateOutputs:
        """
        Generate completions and rewards for a given set of inputs.
        """
        if isinstance(inputs, GenerateInputs):
            inputs = inputs.model_dump()
        gen_sampling_args = deepcopy(self.sampling_args)
        if sampling_args is not None:
            gen_sampling_args.update(sampling_args)

        # preprocess dataset or GenerateInputs to GenerateOutputs
        results_dict = {}
        if isinstance(inputs, Dataset):
            # get prompt column
            results_dict = {}
            for col in inputs.column_names:
                if col == "info":
                    # handle info column to ensure mutable dicts
                    if isinstance(inputs[col][0], str):
                        results_dict[col] = [json.loads(item) for item in inputs[col]]
                    else:
                        results_dict[col] = [dict(item) for item in inputs[col]]
                else:
                    results_dict[col] = deepcopy(inputs[col])
        else:
            results_dict = {col: deepcopy(inputs[col]) for col in inputs}
        if "prompt" not in results_dict:
            raise ValueError("prompt column not found in inputs")
        if "answer" not in results_dict and "info" not in results_dict:
            self.logger.warning(
                "Neither 'answer' nor 'info' column found in inputs. "
                "Some environments can evaluate using only prompt/completion/state, "
                "but reward functions requiring ground truth data may return 0.0. "
                "Proceeding with empty values."
            )
        if "example_id" not in results_dict and "id" in results_dict:
            results_dict["example_id"] = deepcopy(results_dict["id"])
        results_dict["prompt"] = [cleanup_messages(p) for p in results_dict["prompt"]]
        n = len(results_dict["prompt"])
        results_dict["completion"] = [await self.init_completion() for _ in range(n)]
        if not results_dict.get("answer"):
            results_dict["answer"] = [""] * n
        if not results_dict.get("task"):
            results_dict["task"] = ["default"] * n
        if not results_dict.get("info"):
            results_dict["info"] = [{}] * n
        for info in results_dict["info"]:
            if isinstance(info, str):
                info = json.loads(info)
            if self.oai_tools and "oai_tools" not in info:
                info["oai_tools"] = self.oai_tools
        if not results_dict.get("example_id"):
            results_dict["example_id"] = list(range(n))
        results_dict["state"] = [{} for _ in range(n)]

        # prepare GenerateOutputs and run rollouts
        num_rollouts = len(results_dict)
        ne_metadata = num_examples or len(list(set(results_dict["example_id"])))
        rpe_metadata = rollouts_per_example or num_rollouts // ne_metadata
        if results_path is None:
            path_to_save = get_results_path(self.env_id, model)
        else:
            path_to_save = results_path
        metadata = GenerateMetadata(
            env_id=self.env_id,
            env_args=self.env_args,
            model=model,
            base_url=str(client.base_url),
            num_examples=ne_metadata,
            rollouts_per_example=rpe_metadata,
            sampling_args=gen_sampling_args,
            avg_reward=0.0,
            avg_metrics={},
            state_columns=state_columns or [],
            path_to_save=path_to_save,
            date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            time_ms=0.0,
        )
        results = GenerateOutputs(
            prompt=results_dict["prompt"],
            completion=results_dict["completion"],
            answer=results_dict["answer"],
            state=results_dict["state"],
            task=results_dict["task"],
            info=results_dict["info"],
            example_id=results_dict["example_id"],
            reward=[0.0] * n,
            metrics={name: [0.0] * n for name in self.rubric.get_reward_func_names()},
            metadata=metadata,
        )
        results.state = [
            await self.init_state(
                prompt=results.prompt[i],
                completion=results.completion[i],
                answer=results.answer[i],
                task=results.task[i],
                info=results.info[i],
                example_id=results.example_id[i],
            )
            for i in range(n)
        ]
        # resolve concurrency knobs
        gen_limit = max_concurrent_generation
        score_limit = max_concurrent_scoring
        if gen_limit is None:
            gen_limit = max_concurrent
        if score_limit is None:
            score_limit = max_concurrent

        # track timing for metadata
        start_time = time.time()

        if interleave_scoring and score_rollouts:
            # interleaved pipeline: separate semaphores for generation and scoring
            # pre-allocate metrics using known reward function names
            maybe_gen_sem = generation_semaphore or (
                semaphore or await maybe_semaphore(gen_limit)
            )
            # TODO: If only a semaphore is provided, we do not have the "sharing" semaphores mechanism
            # as with 'max_concurrent' because its not clear how to "duplicate" the semaphore properly across
            # multiple generate calls. Right now, we just don't support this case.
            maybe_score_sem = scoring_semaphore or (await maybe_semaphore(score_limit))
            num_completed = 0

            async def run_one(i: int) -> None:
                prompt_i = results.prompt[i]
                completion_i = results.completion[i]
                answer_i = results.answer[i]
                state_i = results.state[i]
                task_i = results.task[i]
                info_i = results.info[i]
                example_id_i = results.example_id[i]
                nonlocal num_completed

                # generation stage
                async with maybe_gen_sem:
                    comp_i, state_i = await self.rollout(
                        client,
                        model,
                        prompt_i,
                        completion_i,
                        answer_i,
                        state_i,
                        task_i,
                        info_i,
                        example_id_i,
                        gen_sampling_args,
                        **kwargs,
                    )
                    results.completion[i] = comp_i
                    results.state[i] = state_i
                # scoring stage
                async with maybe_score_sem:
                    rs = await self.rubric.score_rollout(
                        prompt=prompt_i,
                        completion=comp_i,
                        answer=answer_i,
                        state=state_i,
                        task=task_i,
                        info=info_i,
                        example_id=example_id_i,
                        **kwargs,
                    )
                results.reward[i] = rs.reward
                for k, v in rs.metrics.items():
                    # ensure key exists in case of EnvGroup/RubricGroup
                    if k not in results.metrics:
                        results.metrics[k] = [0.0] * n
                    results.metrics[k][i] = v
                num_completed += 1
                if save_every > 0 and num_completed % save_every == 0:
                    self.logger.debug(f"Saving results to {results_path}")
                    save_results(results)

            tasks = [run_one(i) for i in range(n)]

            if use_tqdm:
                from tqdm.asyncio import tqdm_asyncio

                await tqdm_asyncio.gather(
                    *tasks, total=n, desc=f"Running {n} rollouts (interleaved)"
                )
            else:
                await asyncio.gather(*tasks)
        else:
            # non-interleaved: generate all then score all
            if save_every > 0:
                self.logger.warning(
                    (
                        "Intermediate saving is not supported for non-interleaved rollouts. "
                        f"save_every={save_every} will be ignored."
                    )
                )
            rollouts = await self.run_rollouts(
                client=client,
                model=model,
                prompts=results.prompt,
                completions=results.completion,
                answers=results.answer,
                states=results.state,
                tasks=results.task,
                infos=results.info,
                example_ids=results.example_id,
                sampling_args=gen_sampling_args,
                max_concurrent=gen_limit if gen_limit is not None else max_concurrent,
                semaphore=semaphore,
                use_tqdm=use_tqdm,
                **kwargs,
            )
            results.completion = [rollout[0] for rollout in rollouts]
            results.state = [rollout[1] for rollout in rollouts]
            if score_rollouts:
                rollout_scores = await self.rubric.score_rollouts(
                    prompts=results.prompt,
                    completions=results.completion,
                    answers=results.answer,
                    states=results.state,
                    tasks=results.task,
                    infos=results.info,
                    example_ids=results.example_id,
                    max_concurrent=score_limit
                    if score_limit is not None
                    else max_concurrent,
                    apply_weights=True,
                    use_tqdm=use_tqdm,
                )
                results.reward = rollout_scores.reward
                results.metrics = rollout_scores.metrics
            else:
                results.reward = []
                results.metrics = {}

        # update metadata with actual results
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000.0

        avg_reward = 0.0
        avg_metrics = {}
        if score_rollouts and results.reward:
            avg_reward = sum(results.reward) / len(results.reward)
            avg_metrics = {
                name: sum(values) / len(values) if values else 0.0
                for name, values in results.metrics.items()
            }

        results.metadata.time_ms = elapsed_ms
        results.metadata.avg_reward = avg_reward
        results.metadata.avg_metrics = avg_metrics

        return results

    # alias for backward compatibility
    a_generate = generate

    def generate_sync(
        self,
        inputs: GenerateInputs | Dataset,
        client: AsyncOpenAI | OpenAI,
        model: str,
        sampling_args: SamplingArgs | None = None,
        num_examples: int | None = None,
        rollouts_per_example: int | None = None,
        score_rollouts: bool = True,
        max_concurrent: int = -1,
        max_concurrent_generation: int | None = None,
        max_concurrent_scoring: int | None = None,
        semaphore: asyncio.Semaphore | None = None,
        generation_semaphore: asyncio.Semaphore | None = None,
        scoring_semaphore: asyncio.Semaphore | None = None,
        interleave_scoring: bool = True,
        results_path: Path | None = None,
        state_columns: list[str] | None = None,
        save_every: int = -1,
        **kwargs,
    ) -> GenerateOutputs:
        if isinstance(client, OpenAI):
            client = AsyncOpenAI(api_key=client.api_key, base_url=client.base_url)
        coro = self.generate(
            inputs,
            client=client,
            model=model,
            sampling_args=sampling_args,
            num_examples=num_examples,
            rollouts_per_example=rollouts_per_example,
            score_rollouts=score_rollouts,
            max_concurrent=max_concurrent,
            max_concurrent_generation=max_concurrent_generation,
            max_concurrent_scoring=max_concurrent_scoring,
            semaphore=semaphore,
            generation_semaphore=generation_semaphore,
            scoring_semaphore=scoring_semaphore,
            interleave_scoring=interleave_scoring,
            results_path=results_path,
            state_columns=state_columns,
            save_every=save_every,
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
    ) -> Dataset:
        if self.eval_dataset is None:
            self.logger.info("eval_dataset is not set, falling back to train dataset")
            assert self.dataset is not None
            inputs = self.get_dataset(n=num_examples)
        else:
            inputs = self.get_eval_dataset(n=num_examples)
        assert inputs is not None, "No dataset found"
        if rollouts_per_example > 1:
            inputs = inputs.repeat(rollouts_per_example)
        return inputs

    async def evaluate(
        self,
        client: AsyncOpenAI,
        model: str,
        sampling_args: SamplingArgs | None = None,
        num_examples: int = -1,
        rollouts_per_example: int = 1,
        score_rollouts: bool = True,
        max_concurrent: int = -1,
        max_concurrent_generation: int | None = None,
        max_concurrent_scoring: int | None = None,
        interleave_scoring: bool = True,
        results_path: Path | None = None,
        state_columns: list[str] | None = None,
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
            rollouts_per_example=rollouts_per_example,
            score_rollouts=score_rollouts,
            max_concurrent=max_concurrent,
            max_concurrent_generation=max_concurrent_generation,
            max_concurrent_scoring=max_concurrent_scoring,
            interleave_scoring=interleave_scoring,
            results_path=results_path,
            state_columns=state_columns,
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
        score_rollouts: bool = True,
        max_concurrent: int = -1,
        max_concurrent_generation: int | None = None,
        max_concurrent_scoring: int | None = None,
        interleave_scoring: bool = True,
        results_path: Path | None = None,
        state_columns: list[str] | None = None,
        save_every: int = -1,
        **kwargs,
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
            score_rollouts=score_rollouts,
            max_concurrent=max_concurrent,
            max_concurrent_generation=max_concurrent_generation,
            max_concurrent_scoring=max_concurrent_scoring,
            interleave_scoring=interleave_scoring,
            results_path=results_path,
            state_columns=state_columns,
            save_every=save_every,
            **kwargs,
        )

    make_dataset = make_dataset

    # processing results
    def process_env_results_vllm(
        self,
        prompts: list[Messages],
        completions: list[Messages],
        states: list[State],
        rewards: list[float],
        processing_class: "PreTrainedTokenizerBase",
        max_seq_len: int = -1,
        mask_env_responses: bool = False,
        mask_truncated_completions: bool = False,
        zero_truncated_completions: bool = False,
        message_type: MessageType | None = "chat",
    ) -> ProcessedOutputs:
        """
        Process results with vLLM tokens/logprobs.
        """
        is_chat_format = self.message_type == "chat"

        all_prompt_ids = []
        all_prompt_masks = []
        all_completion_ids = []
        all_completion_masks = []
        all_completion_logprobs = []
        all_rewards = []
        all_is_truncated = []
        for i, (prompt, completion, state, reward) in enumerate(
            zip(prompts, completions, states, rewards)
        ):
            # format-specific processing
            if is_chat_format:
                assert isinstance(prompt, list) and isinstance(completion, list)
                (
                    prompt_ids,
                    prompt_mask,
                    completion_ids,
                    completion_mask,
                    completion_logprobs,
                ) = process_chat_format_vllm(
                    prompt, completion, state, processing_class, mask_env_responses
                )
            else:
                assert isinstance(prompt, str) and isinstance(completion, str)
                (
                    prompt_ids,
                    prompt_mask,
                    completion_ids,
                    completion_mask,
                    completion_logprobs,
                ) = process_completion_format_vllm(
                    prompt, completion, state, processing_class, mask_env_responses
                )
            is_truncated = False
            if max_seq_len > 0 and len(prompt_ids) + len(completion_ids) > max_seq_len:
                if len(prompt_ids) > max_seq_len:
                    prompt_ids = prompt_ids[:max_seq_len]
                    prompt_mask = prompt_mask[:max_seq_len]
                completion_ids = completion_ids[: max_seq_len - len(prompt_ids)]
                completion_mask = completion_mask[: max_seq_len - len(prompt_ids)]
                completion_logprobs = completion_logprobs[
                    : max_seq_len - len(prompt_ids)
                ]
                is_truncated = True
            if is_truncated and mask_truncated_completions:
                completion_mask = [0] * len(completion_ids)
            assert len(prompt_ids) == len(prompt_mask), (
                f"Prompt ids: {len(prompt_ids)}, prompt mask: {len(prompt_mask)}"
            )
            assert len(completion_ids) == len(completion_mask), (
                f"Completion ids: {len(completion_ids)}, completion mask: {len(completion_mask)}"
            )
            assert (
                len(completion_mask) == len(completion_ids) == len(completion_logprobs)
            ), (
                f"completion mask: {len(completion_mask)}, completion ids: {len(completion_ids)}, completion logprobs: {len(completion_logprobs)}"
            )
            all_prompt_ids.append(prompt_ids)
            all_prompt_masks.append(prompt_mask)
            all_completion_ids.append(completion_ids)
            all_completion_masks.append(completion_mask)
            all_completion_logprobs.append(completion_logprobs)
            if zero_truncated_completions and is_truncated:
                all_rewards.append(0)
                all_is_truncated.append(True)
            else:
                all_rewards.append(reward)
                all_is_truncated.append(False)
        return ProcessedOutputs(
            prompt_ids=all_prompt_ids,
            prompt_mask=all_prompt_masks,
            completion_ids=all_completion_ids,
            completion_mask=all_completion_masks,
            completion_logprobs=all_completion_logprobs,
            rewards=all_rewards,
            is_truncated=all_is_truncated,
        )

    # alias for process_env_results_vllm
    process_env_results = process_env_results_vllm
