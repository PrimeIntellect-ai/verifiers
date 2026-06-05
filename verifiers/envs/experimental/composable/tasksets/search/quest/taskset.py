"""QUEST composable search taskset.

This ports QUEST objective tasks into the verifiers composable taskset
structure while preserving the generated QUEST eval scripts and objective
verification tree runtime.
"""

import asyncio
import ast
import hashlib
import importlib.util
import logging
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import httpx
import verifiers as vf
from datasets import Dataset, load_dataset
from openai import AsyncOpenAI, OpenAIError
from pydantic import BaseModel
from verifiers.envs.experimental.composable import SandboxSpec, SandboxTaskSet

from .obj_task_eval.utils.cache_filesys import CacheFileSys

logger = logging.getLogger(__name__)

DEFAULT_DATASET_NAME = "osunlp/QUEST-RL-Data"
DEFAULT_SPLIT = "train"
DEFAULT_CATEGORY = "objective"
DEFAULT_ANSWER_FILE = "/task/answer.txt"
DEFAULT_WORKDIR = "/workspace"
DEFAULT_JUDGE_BASE_URL = "https://api.pinference.ai/api/v1"
DEFAULT_JUDGE_API_KEY_VAR = "PRIME_API_KEY"
DEFAULT_JUDGE_MODEL = "openai/gpt-4.1-mini"
DEFAULT_SANDBOX_IMAGE = "python:3.11-slim"


class QuestOpenAIClient:
    """OpenAI-compatible client adapter for QUEST's ``async_response`` API."""

    provider = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None,
        model: str,
        sampling_args: dict[str, Any] | None = None,
        default_headers: dict[str, str] | None = None,
        timeout: float = 1200.0,
    ) -> None:
        self.model = model
        self.sampling_args = dict(sampling_args or {})
        self._httpx_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=1024, max_keepalive_connections=512),
            timeout=httpx.Timeout(timeout),
        )
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers,
            http_client=self._httpx_client,
        )

    async def async_response(self, *, count_token: bool = False, **kwargs: Any) -> Any:
        response_format = kwargs.pop("response_format", None)
        messages = kwargs.pop("messages")
        model = kwargs.pop("model", self.model) or self.model
        request_kwargs = dict(self.sampling_args)
        request_kwargs.update(kwargs)
        try:
            if isinstance(response_format, type) and issubclass(
                response_format, BaseModel
            ):
                response = await self._client.beta.chat.completions.parse(
                    model=model,
                    messages=messages,
                    response_format=response_format,
                    **request_kwargs,
                )
                parsed = response.choices[0].message.parsed
                usage = _usage_dict(response)
                return (parsed, usage) if count_token else parsed
            if response_format is not None:
                request_kwargs["response_format"] = response_format
            response = await self._client.chat.completions.create(
                model=model,
                messages=messages,
                **request_kwargs,
            )
        except OpenAIError as exc:
            raise vf.ModelError(f"QUEST judge model request failed: {exc}") from exc
        content = response.choices[0].message.content or ""
        usage = _usage_dict(response)
        return (content, usage) if count_token else content

    async def close(self) -> None:
        await self._client.close()
        await self._httpx_client.aclose()


def _usage_dict(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    return {
        "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
    }


def _parse_literal(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def _extract_question(prompt: Any, extra_info: Any) -> str:
    if isinstance(prompt, list) and prompt:
        first = prompt[0]
        if isinstance(first, dict) and isinstance(first.get("content"), str):
            return first["content"]
    if isinstance(extra_info, dict) and isinstance(extra_info.get("question"), str):
        return extra_info["question"]
    return ""


def _extract_task_id(reward_model: Any, extra_info: Any) -> str | None:
    if isinstance(reward_model, dict):
        task_id = reward_model.get("task_id")
        if isinstance(task_id, str) and task_id:
            return task_id
        ground_truth = reward_model.get("ground_truth")
        if isinstance(ground_truth, dict):
            task_id = ground_truth.get("task_id")
            if isinstance(task_id, str) and task_id:
                return task_id
    if isinstance(extra_info, dict):
        task_id = extra_info.get("original_task_id")
        if isinstance(task_id, str) and task_id:
            return task_id
    return None


def _safe_module_component(value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    stem = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")[:80]
    return f"{stem}_{digest}" if stem else digest


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


def _load_eval_script(script_path: Path) -> Any:
    if not script_path.is_file():
        raise FileNotFoundError(script_path)
    module_name = f"obj_task_eval_dynamic_{_safe_module_component(script_path.stem)}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load eval script {script_path}")

    # Generated QUEST scripts import the evaluator as top-level
    # ``obj_task_eval``. Expose this vendored package at that import path
    # while executing the script.
    quest_package_parent = Path(__file__).resolve().parent
    added_path = False
    if str(quest_package_parent) not in sys.path:
        sys.path.insert(0, str(quest_package_parent))
        added_path = True
    try:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        if added_path:
            try:
                sys.path.remove(str(quest_package_parent))
            except ValueError:
                pass
    evaluate_answer = getattr(module, "evaluate_answer", None)
    if not asyncio.iscoroutinefunction(evaluate_answer):
        raise TypeError(f"{script_path} does not define async evaluate_answer")
    return evaluate_answer


class QuestTaskSet(SandboxTaskSet):
    """QUEST objective search/research taskset."""

    default_workdir = DEFAULT_WORKDIR

    def __init__(
        self,
        dataset_name: str = DEFAULT_DATASET_NAME,
        split: str = DEFAULT_SPLIT,
        category: str = DEFAULT_CATEGORY,
        filter_fn: str | None = None,
        ds_keep_in_memory: bool | None = True,
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
        quest_cache_dir: str | None = None,
        quest_eval_scripts_dir: str | None = None,
        quest_eval_model: str | None = None,
        quest_eval_concurrency: int = 8,
    ) -> None:
        if category not in {"objective", "open-ended", "all"}:
            raise ValueError(
                "category must be one of 'objective', 'open-ended', or 'all'"
            )
        if category != "objective":
            raise NotImplementedError(
                "Initial QUEST taskset implementation supports category='objective' only"
            )
        self.dataset_name = dataset_name
        self.split = split
        self.category = category
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
        self._quest_cache_dir = quest_cache_dir
        self._quest_eval_scripts_dir = quest_eval_scripts_dir
        self._quest_eval_model = quest_eval_model or judge_model
        self._quest_eval_concurrency = quest_eval_concurrency
        super().__init__(
            dataset=self._build_dataset,
            name=f"search/quest/{category}",
            filter_fn=filter_fn,
        )

    def _build_dataset(self) -> Dataset:
        raw = load_dataset(
            self.dataset_name,
            split=self.split,
            keep_in_memory=self.ds_keep_in_memory,
            num_proc=self.ds_num_proc,
        )
        rows: list[dict[str, Any]] = []
        for row in raw:
            if self.category != "all" and row.get("rl_task_category") != self.category:
                continue
            extra_info = _parse_literal(row.get("extra_info"))
            reward_model = _parse_literal(row.get("reward_model"))
            question = _extract_question(row.get("prompt"), extra_info)
            task_id = _extract_task_id(reward_model, extra_info)
            if self.category == "objective" and not task_id:
                continue
            rows.append(
                {
                    "question": question,
                    "answer": "",
                    "info": {
                        "question": question,
                        "raw_prompt": row.get("prompt"),
                        "data_source": row.get("data_source"),
                        "rl_task_category": row.get("rl_task_category"),
                        "task_id": task_id,
                        "reward_model": reward_model,
                        "reward_model_raw": row.get("reward_model"),
                        "extra_info": extra_info,
                        "extra_info_raw": row.get("extra_info"),
                        "answer_file": self.answer_file,
                    },
                }
            )
        return Dataset.from_list(rows)

    def get_instruction(self, info: dict) -> str:
        question = str(info.get("question") or "")
        return (
            f"{question}\n\n"
            f"When you have the final response, write it to {self.answer_file} using a tool call, "
            "then stop. The task is incomplete unless that file exists. Include supporting URLs/citations "
            "in the file and do not include scratch reasoning or tool traces."
        )

    def get_sandbox_spec(self, info: dict) -> SandboxSpec:
        return self._sandbox_spec

    def get_workdir(self, info: dict) -> str:
        return self.default_workdir

    def get_env_vars(self) -> dict[str, str]:
        env_vars: dict[str, str] = {}
        for key in ("SERPER_API_KEY",):
            value = os.environ.get(key)
            if value:
                env_vars[key] = value
        return env_vars

    async def setup(self, state: vf.State) -> None:
        sandbox_client = state["sandbox_client"]
        sandbox_id = state["sandbox_id"]
        await sandbox_client.execute_command(
            sandbox_id, f"mkdir -p {self.default_workdir} /task", timeout=10
        )

    def get_rubric(self) -> vf.Rubric:
        return QuestRubric(
            answer_file=self.answer_file,
            dataset_name=self.dataset_name,
            eval_scripts_dir=self._quest_eval_scripts_dir,
            cache_dir=self._quest_cache_dir,
            judge_model=self._judge_model,
            judge_base_url=self._judge_base_url,
            judge_api_key_var=self._judge_api_key_var,
            judge_sampling_args=self._judge_sampling_args,
            quest_eval_model=self._quest_eval_model,
            eval_concurrency=self._quest_eval_concurrency,
        )


class QuestRubric(vf.Rubric):
    """Scores QUEST objective tasks using their generated eval scripts."""

    def __init__(
        self,
        *,
        answer_file: str = DEFAULT_ANSWER_FILE,
        dataset_name: str = DEFAULT_DATASET_NAME,
        eval_scripts_dir: str | None = None,
        cache_dir: str | None = None,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        judge_base_url: str | None = DEFAULT_JUDGE_BASE_URL,
        judge_api_key_var: str = DEFAULT_JUDGE_API_KEY_VAR,
        judge_sampling_args: dict[str, Any] | None = None,
        quest_eval_model: str | None = None,
        eval_concurrency: int = 8,
    ) -> None:
        super().__init__()
        self.answer_file = answer_file
        self.dataset_name = dataset_name
        self.eval_scripts_dir = (
            Path(eval_scripts_dir).expanduser() if eval_scripts_dir else None
        )
        self.cache_dir = (
            Path(cache_dir).expanduser()
            if cache_dir
            else Path.home() / ".cache" / "verifiers" / "quest"
        )
        self.judge_model = judge_model
        self.judge_base_url = judge_base_url
        self.judge_api_key_var = judge_api_key_var
        self.judge_sampling_args = dict(judge_sampling_args or {})
        self.quest_eval_model = quest_eval_model or judge_model
        self.eval_concurrency = eval_concurrency
        self._client: QuestOpenAIClient | None = None
        self._semaphore = asyncio.Semaphore(eval_concurrency)
        self._scripts_root: Path | None = None
        self.add_reward_func(self.objective_reward, weight=1.0)

    async def score_rollout(self, state: vf.State) -> None:
        """Score one rollout and preserve QUEST infrastructure failures as ``vf.Error`` values."""
        if state.get("error") is not None:
            state["reward"] = 0.0
            state["metrics"] = {"objective_reward": 0.0}
            return
        try:
            score = await self.objective_reward(state)
        except vf.Error as exc:
            state["error"] = exc
            score = 0.0
        except Exception as exc:
            error = vf.InfraError("QUEST objective scoring failed")
            error.__cause__ = exc
            state["error"] = error
            score = 0.0
        state["reward"] = score
        state["metrics"] = {"objective_reward": score}

    async def objective_reward(self, state: vf.State, **_: Any) -> float:
        sandbox_client = state.get("sandbox_client")
        sandbox_id = state.get("sandbox_id")
        if not sandbox_client or not sandbox_id:
            raise vf.SandboxError("QUEST scoring requires a live sandbox")
        try:
            result = await sandbox_client.execute_command(
                sandbox_id,
                f"cat {self.answer_file} 2>/dev/null || true",
                working_dir=None,
            )
        except Exception as exc:
            raise vf.SandboxError(
                f"Failed to read QUEST answer file {self.answer_file}"
            ) from exc
        answer = (result.stdout or "").strip()
        answer_source = "answer_file"
        if not answer:
            answer = _completion_text(state)
            answer_source = "completion_fallback" if answer else "missing"
        state["quest_answer"] = answer
        state["quest_answer_source"] = answer_source
        if not answer:
            state["quest_eval_error"] = "empty_answer"
            return 0.0
        info = state.get("info") or {}
        task_id = info.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise vf.InfraError("QUEST objective task is missing task_id metadata")
        state["quest_task_id"] = task_id
        script_path = self._script_path(task_id)
        cache = CacheFileSys(str(self.cache_dir / _safe_module_component(task_id)))
        try:
            evaluate_answer = _load_eval_script(script_path)
        except Exception as exc:
            raise vf.InfraError(
                f"Failed to load QUEST eval script for task_id={task_id!r}"
            ) from exc
        client = self._get_client()
        try:
            summary = await evaluate_answer(
                client=client,
                answer=answer,
                agent_name="rlm",
                answer_name=str(state.get("trajectory_id") or task_id),
                cache=cache,
                semaphore=self._semaphore,
                logger=logger,
                model=self.quest_eval_model,
            )
        except vf.Error:
            raise
        except Exception as exc:
            raise vf.InfraError(f"QUEST eval failed for task_id={task_id!r}") from exc
        state["quest_eval_summary"] = summary
        final_score = float(summary.get("final_score", 0.0) or 0.0)
        if not math.isfinite(final_score):
            final_score = 0.0
        final_score = max(0.0, min(1.0, final_score))
        state["quest_final_score"] = final_score
        return final_score

    def _get_client(self) -> QuestOpenAIClient:
        if self._client is not None:
            return self._client
        api_key = os.environ.get(self.judge_api_key_var)
        if not api_key:
            raise vf.ModelError(
                f"{self.judge_api_key_var} environment variable is required for QUEST evaluation"
            )
        headers: dict[str, str] = {}
        if self.judge_api_key_var == "PRIME_API_KEY" and os.environ.get(
            "PRIME_TEAM_ID"
        ):
            headers["X-Prime-Team-ID"] = os.environ["PRIME_TEAM_ID"]
        self._client = QuestOpenAIClient(
            api_key=api_key,
            base_url=self.judge_base_url,
            model=self.judge_model,
            sampling_args=self.judge_sampling_args,
            default_headers=headers or None,
        )
        return self._client

    def _script_path(self, task_id: str) -> Path:
        scripts_root = self._ensure_eval_scripts_dir()
        script_path = scripts_root / "eval_scripts" / f"{task_id}.py"
        if not script_path.is_file():
            raise vf.InfraError(
                f"QUEST eval script not found for task_id={task_id!r}: {script_path}"
            )
        return script_path

    def _ensure_eval_scripts_dir(self) -> Path:
        if self._scripts_root is not None:
            return self._scripts_root
        if self.eval_scripts_dir is not None:
            self._scripts_root = self.eval_scripts_dir
            return self._scripts_root
        try:
            from huggingface_hub import snapshot_download

            root = snapshot_download(
                repo_id=self.dataset_name,
                repo_type="dataset",
                allow_patterns="eval_scripts/*.py",
            )
        except Exception as exc:
            raise vf.InfraError(
                f"Failed to download QUEST eval scripts from {self.dataset_name}"
            ) from exc
        self._scripts_root = Path(root)
        return self._scripts_root

    @vf.cleanup
    async def cleanup(self, state: vf.State) -> None:
        sandbox_client = state.get("sandbox_client")
        sandbox_id = state.get("sandbox_id")
        if sandbox_client and sandbox_id:
            try:
                await sandbox_client.delete(sandbox_id)
            except Exception:
                pass

    @vf.teardown
    async def teardown(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
