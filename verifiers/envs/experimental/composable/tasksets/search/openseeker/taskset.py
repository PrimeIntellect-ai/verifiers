"""OpenSeeker composable search taskset.

This ports PolarSeeker/OpenSeeker-v1-Data into the composable search taskset
family. OpenSeeker's public evaluator scores final answer semantics against a
gold answer with a binary LLM judge, so this backend preserves that contract
rather than introducing QUEST-style URL-backed verification.
"""

import asyncio
import logging
import math
import os
import re
from typing import Any, NoReturn

import verifiers as vf
from verifiers.utils.sandbox_delete import delete_sandbox_for_rollout
from datasets import Dataset, load_dataset
from openai import (
    APIConnectionError,
    APIResponseValidationError,
    APITimeoutError,
    APIStatusError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    ContentFilterFinishReasonError,
    InternalServerError,
    LengthFinishReasonError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)
from verifiers.envs.experimental.composable import SandboxSpec, SandboxTaskSet
from verifiers.types import ClientConfig
from verifiers.utils.client_utils import setup_openai_client

logger = logging.getLogger(__name__)

DEFAULT_DATASET_NAME = "PolarSeeker/OpenSeeker-v1-Data"
DEFAULT_SPLIT = "train"
DEFAULT_ANSWER_FILE = "/task/answer.txt"
DEFAULT_WORKDIR = "/workspace"
DEFAULT_JUDGE_BASE_URL = "https://api.pinference.ai/api/v1"
DEFAULT_JUDGE_API_KEY_VAR = "PRIME_API_KEY"
DEFAULT_JUDGE_MODEL = "openai/gpt-5.4-mini"
DEFAULT_SANDBOX_IMAGE = "python:3.11-slim"

JUDGE_PROMPT_BC_EN = """
Based on the given question, standard answer, and model-predicted answer, evaluate whether the model's response is correct. Your task is to classify the result as: [CORRECT] or [INCORRECT].

First, we'll list examples for each category, then you'll evaluate a new question's predicted answer.
Here are examples of [CORRECT] responses:
```
Question: What are the names of Barack Obama's children?
Standard Answer: Malia Obama and Sasha Obama
Model Prediction 1: Malia Obama and Sasha Obama
Model Prediction 2: Malia and Sasha
Model Prediction 3: Most would say Malia and Sasha, but I'm not sure, I should verify
Model Prediction 4: Barack Obama has two daughters, Malia Ann and Natasha Marian, commonly known as Malia Obama and Sasha Obama.
```
These responses are all [CORRECT] because they:
    - Fully include the important information from the standard answer.
    - Don't contain any information that contradicts the standard answer.
    - Focus only on semantic content; language, capitalization, punctuation, grammar, and order aren't important.
    - Vague statements or guesses are acceptable as long as they include the standard answer and don't contain incorrect information or contradictions.

Here are examples of [INCORRECT] responses:
```
Question: What are the names of Barack Obama's children?
Standard Answer: Malia Obama and Sasha Obama
Model Prediction 1: Malia
Model Prediction 2: Malia, Sasha and Susan or Sasha Obama or Malia Obama, or Natasha Marian, or Einstein
Model Prediction 3: While I don't know their exact names, I can tell you Barack Obama has two children.
Model Prediction 4: You might be thinking of Betsy and Olivia. But you should verify the details with the latest references. Is that the correct answer?
Model Prediction 5: Barack Obama's children
```
These responses are all [INCORRECT] because they:
    - Contain factual statements that contradict the standard answer.
    - Are empty or merely repeat the question.
    - Enumerate multiple answers or repeat the answer.

Pay special attention to the following:
- The standard answer may contain responses to multiple aspects of the question, and within the same aspect, there might be different descriptions, all of which are correct and are given in the same bracket, connected by commas. For example, for the question "What is the name of ByteDance's AI model?", the standard answer is "[[Doubao, Skylark]]":
    - Predicted answers "Doubao", "Doubao, Skylark", "Skylark", etc. are all [CORRECT].
- For standard answers containing responses to different aspects, the model needs to provide answers to all aspects to be considered correct; otherwise, it's directly judged as [INCORRECT]. There is no [PARTIALLY CORRECT] output option. These answers will be given in different brackets. For example, for the question "Who are the members of TFBOYS?", the standard answer is "[[Wang Junkai][Wang Yuan][Yi Yangqianxi]]":
    - Predicted answers like "Wang Junkai, Wang Yuan, Yi Yangqianxi" that include all answers are [CORRECT].
    - Predicted answers like "Wang Junkai, Yi Yangqianxi" that don't include all answers are [INCORRECT].

Also note the following points:
- For questions with numerical standard answers, the predicted answer should match the standard answer. For example, for the question "What is the total length in meters of the Huangpu River Bridge on the Jinshan Railway?", the standard answer is "3518.17":
    - Predicted answers "3518", "3518.1", "3518.17" are all [CORRECT].
    - Predicted answers "3520" and "3600" are [INCORRECT].
- If the model prediction doesn't directly answer the question, attempts to circumvent or fails to directly provide the standard answer, it's considered an [INCORRECT] answer.
    - For example, for the question "Who is JJ Lin's wife?", with the standard answer "Ding Wenqi", model predictions like "JJ Lin's wife", "JJ Lin's wife should be excellent", "JJ Lin's wife might be a public figure" are all [INCORRECT].
- If the standard answer contains more information than the question asks for, the predicted answer only needs to include the information mentioned in the question.
    - For example, for the question "What is the main chemical component of magnesite?", with the standard answer "Magnesium carbonate (MgCO3)", "Magnesium carbonate" or "MgCO3" are both considered [CORRECT].
- If information omitted in the predicted answer can be clearly inferred from the question, it's considered correct.
    - For example, for the question "The Nuragic ruins of Barumini were listed as a World Cultural Heritage by UNESCO in 1997, so where is this site located?", with the standard answer "Sardinia, Italy", the predicted answer "Sardinia" is considered [CORRECT].
- If it's clear that different translations of a name refer to the same person, it's considered correct.
    - For example, if the standard answer is "Robinson", answers like "Lubinson" or "Lubinsun" are both correct.
- You should focus more on the match between the standard answer and the model prediction, rather than whether the standard answer itself is correct.

Below is a new question example. Please reply with only [CORRECT] or [INCORRECT], without apologies or corrections to your own errors, just evaluate the answer.
```
Question: {question}
Standard Answer: {correct_answer}
Predicted Answer: {response}
```

Evaluate this new question's predicted answer as one of the following:
A. [CORRECT]
B. [INCORRECT]

Return only the option representing [CORRECT] or [INCORRECT], i.e., just return A or B, without adding any other text.
""".strip()

_LABEL_RE = re.compile(r"^\s*([AB])\b")
_CONTEXT_LENGTH_ERROR_PHRASES = (
    "this model's maximum context length is",
    "is longer than the model's context length",
    "is longer than the maximum model length",
    "exceeds the model's context length",
    "exceed the configured limit",
    "exceeds the configured limit",
    "exceeded model",
    "prompt_too_long",
    "context length",
    "maximum model length",
)
_OPENSEEKER_JUDGE_ERROR_TYPES = (
    APIConnectionError,
    APIResponseValidationError,
    APITimeoutError,
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    ContentFilterFinishReasonError,
    InternalServerError,
    LengthFinishReasonError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)


def _is_context_length_error(exc: BadRequestError) -> bool:
    response = getattr(exc, "response", None)
    response_text = getattr(response, "text", "") or ""
    error_text = f"{response_text}\n{exc}".lower()
    return any(phrase in error_text for phrase in _CONTEXT_LENGTH_ERROR_PHRASES)


def _raise_openseeker_judge_error(exc: Exception, *, model: str) -> NoReturn:
    if isinstance(exc, BadRequestError) and _is_context_length_error(exc):
        raise vf.OverlongPromptError(
            f"OpenSeeker judge prompt exceeded model context for {model}: {exc}"
        ) from exc
    if isinstance(
        exc,
        (
            APIConnectionError,
            APITimeoutError,
            RateLimitError,
            InternalServerError,
            ConflictError,
        ),
    ):
        raise vf.InfraError(
            f"OpenSeeker judge transient request failed for {model}: {exc}"
        ) from exc
    if isinstance(exc, APIResponseValidationError):
        raise vf.InvalidModelResponseError(
            f"OpenSeeker judge SDK response validation failed for {model}: {exc}"
        ) from exc
    if isinstance(exc, LengthFinishReasonError):
        raise vf.InvalidModelResponseError(
            f"OpenSeeker judge stopped due to length for {model}: {exc}"
        ) from exc
    if isinstance(
        exc,
        (
            AuthenticationError,
            PermissionDeniedError,
            NotFoundError,
            UnprocessableEntityError,
            ContentFilterFinishReasonError,
            BadRequestError,
            APIStatusError,
        ),
    ):
        raise vf.ModelError(
            f"OpenSeeker judge request failed for {model}: {exc}"
        ) from exc
    raise AssertionError(
        f"Unhandled OpenSeeker judge exception type: {type(exc).__name__}"
    ) from exc


def _single_choice(response: Any) -> Any:
    if response is None:
        raise vf.EmptyModelResponseError("OpenSeeker judge returned no response")
    choices = getattr(response, "choices", None)
    if choices is None:
        raise vf.EmptyModelResponseError(
            "OpenSeeker judge returned no response choices"
        )
    if len(choices) != 1:
        raise vf.InvalidModelResponseError(
            f"OpenSeeker judge returned {len(choices)} choices, expected 1"
        )
    return choices[0]


def _merge_sampling_args(sampling_args: dict[str, Any]) -> dict[str, Any]:
    request_kwargs: dict[str, Any] = {
        "temperature": 0.0,
        "extra_body": {"skip_special_tokens": False},
    }
    for key, value in sampling_args.items():
        if (
            key == "extra_body"
            and isinstance(value, dict)
            and isinstance(request_kwargs.get("extra_body"), dict)
        ):
            request_kwargs["extra_body"] = {
                **request_kwargs["extra_body"],
                **value,
            }
            continue
        request_kwargs[key] = value
    return request_kwargs


def _parse_judge_label(raw: str | None) -> int | None:
    if raw is None:
        return None
    text = str(raw).strip()
    match = _LABEL_RE.match(text)
    if match:
        return 1 if match.group(1) == "A" else 0
    if "</think>" in text:
        after_tag = text.split("</think>", 1)[-1].strip()
        match = _LABEL_RE.match(after_tag)
        if match:
            return 1 if match.group(1) == "A" else 0
    return None


def _completion_text(state: vf.State) -> str:
    completion = state.get("completion")
    if isinstance(completion, str):
        return completion.strip()
    if not isinstance(completion, list):
        return ""
    parts: list[str] = []
    for message in completion:
        role = None
        content = None
        if isinstance(message, dict):
            role = message.get("role")
            content = message.get("content")
        else:
            role = getattr(message, "role", None)
            content = getattr(message, "content", None)
        if role == "assistant" and isinstance(content, str) and content.strip():
            parts.append(content.strip())
    return "\n\n".join(parts).strip()


class OpenSeekerTaskSet(SandboxTaskSet):
    """OpenSeeker v1 search/research taskset."""

    default_workdir = DEFAULT_WORKDIR

    def __init__(
        self,
        dataset_name: str = DEFAULT_DATASET_NAME,
        split: str = DEFAULT_SPLIT,
        include_trajectory: bool = False,
        filter_fn: str | None = None,
        ds_keep_in_memory: bool | None = False,
        ds_num_proc: int | None = None,
        sandbox_image: str = DEFAULT_SANDBOX_IMAGE,
        sandbox_cpu_cores: int = 2,
        sandbox_memory_gb: int = 2,
        sandbox_disk_size_gb: int = 5,
        sandbox_timeout_minutes: int | None = None,
        answer_file: str = DEFAULT_ANSWER_FILE,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        judge_base_url: str | None = DEFAULT_JUDGE_BASE_URL,
        judge_api_key_var: str = DEFAULT_JUDGE_API_KEY_VAR,
        judge_sampling_args: dict[str, Any] | None = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.include_trajectory = include_trajectory
        self.ds_keep_in_memory = ds_keep_in_memory
        self.ds_num_proc = ds_num_proc
        self.answer_file = answer_file
        self._sandbox_spec = SandboxSpec(
            image=sandbox_image,
            cpu_cores=sandbox_cpu_cores,
            memory_gb=sandbox_memory_gb,
            disk_size_gb=sandbox_disk_size_gb,
            timeout_minutes=sandbox_timeout_minutes,
        )
        self._judge_model = judge_model
        self._judge_base_url = judge_base_url
        self._judge_api_key_var = judge_api_key_var
        self._judge_sampling_args = dict(judge_sampling_args or {})
        super().__init__(
            dataset=self._build_dataset,
            name="search/openseeker",
            filter_fn=filter_fn,
        )

    def _build_dataset(self) -> Dataset:
        raw = load_dataset(
            self.dataset_name,
            split=self.split,
            keep_in_memory=self.ds_keep_in_memory,
            num_proc=self.ds_num_proc,
        )
        columns = [
            "question",
            "answer",
            "number of tool calls",
            "trajectory correctness",
        ]
        if self.include_trajectory:
            columns.append("trajectory")
        raw = raw.select_columns([col for col in columns if col in raw.column_names])

        rows: list[dict[str, Any]] = []
        for row_index, row in enumerate(raw):
            correctness = row.get("trajectory correctness")
            tool_calls = row.get("number of tool calls")
            if not isinstance(tool_calls, int):
                tool_calls = None
            question = str(row.get("question") or "").strip()
            answer = str(row.get("answer") or "").strip()
            if not question or not answer:
                continue
            info: dict[str, Any] = {
                "question": question,
                "answer": answer,
                "source_dataset": self.dataset_name,
                "split": self.split,
                "row_index": row_index,
                "number_of_tool_calls": tool_calls,
                "trajectory_correctness": correctness,
                "answer_file": self.answer_file,
            }
            if self.include_trajectory:
                info["trajectory"] = row.get("trajectory")
            rows.append({"question": question, "answer": answer, "info": info})
        return Dataset.from_list(rows)

    def get_instruction(self, info: dict) -> str:
        question = str(info.get("question") or "")
        return (
            f"{question}\n\n"
            f"When you have the final response, write it to {self.answer_file} using a tool call, "
            "then stop. The task is incomplete unless that file exists. Provide the requested answer "
            "directly, include supporting URLs/citations when useful, and do not include scratch "
            "reasoning or tool traces."
        )

    def get_sandbox_spec(self, info: dict) -> SandboxSpec:
        return self._sandbox_spec

    def get_workdir(self, info: dict) -> str:
        return self.default_workdir

    def get_env_vars(self) -> dict[str, str]:
        env_vars: dict[str, str] = {}
        value = os.environ.get("SERPER_API_KEY")
        if value:
            env_vars["SERPER_API_KEY"] = value
        return env_vars

    async def setup(self, state: vf.State) -> None:
        sandbox_client = state["sandbox_client"]
        sandbox_id = state["sandbox_id"]
        await sandbox_client.execute_command(
            sandbox_id, f"mkdir -p {self.default_workdir} /task", timeout=10
        )

    def get_rubric(self) -> vf.Rubric:
        return OpenSeekerRubric(
            answer_file=self.answer_file,
            judge_model=self._judge_model,
            judge_base_url=self._judge_base_url,
            judge_api_key_var=self._judge_api_key_var,
            judge_sampling_args=self._judge_sampling_args,
        )


class OpenSeekerRubric(vf.Rubric):
    """Scores OpenSeeker answers with the upstream binary semantic judge prompt."""

    def __init__(
        self,
        *,
        answer_file: str = DEFAULT_ANSWER_FILE,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        judge_base_url: str | None = DEFAULT_JUDGE_BASE_URL,
        judge_api_key_var: str = DEFAULT_JUDGE_API_KEY_VAR,
        judge_sampling_args: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.answer_file = answer_file
        self.judge_model = judge_model
        self.judge_base_url = judge_base_url
        self.judge_api_key_var = judge_api_key_var
        self.judge_sampling_args = dict(judge_sampling_args or {})
        self._client: AsyncOpenAI | None = None
        self.add_reward_func(self.answer_reward, weight=1.0)

    async def _answer_score_for_state(self, state: vf.State) -> float:
        if state.get("error") is not None:
            return 0.0
        try:
            return await self.answer_reward(state)
        except vf.Error as exc:
            state["error"] = exc
            return 0.0

    async def score_rollout(self, state: vf.State) -> None:
        score = await self._answer_score_for_state(state)
        state["reward"] = score
        state["metrics"] = {"openseeker_reward": score}

    async def score_group(self, states: list[vf.State]) -> None:
        if not states:
            logger.warning("No states to score")
            return
        scores = await asyncio.gather(
            *(self._answer_score_for_state(state) for state in states)
        )
        avg_score = sum(scores) / len(scores)
        for state, score in zip(states, scores):
            state["reward"] = score
            state["advantage"] = score - avg_score
            for turn in state.get("trajectory", []):
                if isinstance(turn, dict):
                    if turn.get("advantage") is None:
                        turn["advantage"] = state["advantage"]
                    if turn.get("reward") is None:
                        turn["reward"] = state["reward"]
            state["metrics"] = {"openseeker_reward": score}

    async def answer_reward(self, state: vf.State, **_: Any) -> float:
        sandbox_client = state.get("sandbox_client")
        sandbox_id = state.get("sandbox_id")
        if not sandbox_client or not sandbox_id:
            raise vf.SandboxError("OpenSeeker scoring requires a live sandbox")
        try:
            result = await sandbox_client.execute_command(
                sandbox_id,
                f"cat {self.answer_file} 2>/dev/null || true",
                working_dir=None,
            )
        except Exception as exc:
            raise vf.SandboxError(
                f"Failed to read OpenSeeker answer file {self.answer_file}"
            ) from exc
        answer = (result.stdout or "").strip()
        answer_source = "answer_file"
        if not answer:
            answer = _completion_text(state)
            answer_source = "completion_fallback" if answer else "missing"
        state["openseeker_answer"] = answer
        state["openseeker_answer_source"] = answer_source
        if not answer:
            state["openseeker_eval_error"] = "empty_answer"
            return 0.0

        info = state.get("info") or {}
        question = str(info.get("question") or "").strip()
        gold_answer = str(info.get("answer") or "").strip()
        if not question or not gold_answer:
            raise vf.InfraError(
                "OpenSeeker task is missing question or answer metadata"
            )
        state["openseeker_gold_answer"] = gold_answer
        state["openseeker_question"] = question

        raw = await self._judge(
            question=question, correct_answer=gold_answer, response=answer
        )
        label = _parse_judge_label(raw)
        state["openseeker_judge_raw"] = raw
        state["openseeker_judge_label"] = label
        if label is None:
            raise vf.InvalidModelResponseError(
                f"OpenSeeker judge returned an invalid label: {raw!r}"
            )
        score = float(label)
        if not math.isfinite(score):
            score = 0.0
        score = max(0.0, min(1.0, score))
        state["openseeker_final_score"] = score
        return score

    async def _judge(self, *, question: str, correct_answer: str, response: str) -> str:
        client = self._get_client()
        prompt = JUDGE_PROMPT_BC_EN.format(
            question=question,
            correct_answer=correct_answer,
            response=response,
        )
        request_kwargs = _merge_sampling_args(self.judge_sampling_args)
        try:
            completion = await client.chat.completions.create(
                model=self.judge_model,
                messages=[
                    {"role": "system", "content": "Judge the response objectively."},
                    {"role": "user", "content": prompt},
                ],
                **request_kwargs,
            )
        except _OPENSEEKER_JUDGE_ERROR_TYPES as exc:
            _raise_openseeker_judge_error(exc, model=self.judge_model)
        choice = _single_choice(completion)
        content = choice.message.content
        if content is None:
            raise vf.EmptyModelResponseError(
                f"OpenSeeker judge returned no text content for {self.judge_model}"
            )
        return content.strip()

    def _get_client(self) -> AsyncOpenAI:
        if self._client is not None:
            return self._client
        api_base_url = self.judge_base_url or "https://api.openai.com/v1"
        self._client = setup_openai_client(
            ClientConfig(
                api_key_var=self.judge_api_key_var,
                api_base_url=api_base_url,
                timeout=1200.0,
            )
        )
        return self._client

    @vf.cleanup
    async def cleanup(self, state: vf.State) -> None:
        sandbox_client = state.get("sandbox_client")
        sandbox_id = state.get("sandbox_id")
        if sandbox_client and sandbox_id:
            await delete_sandbox_for_rollout(sandbox_client, sandbox_id)

    @vf.teardown
    async def teardown(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
