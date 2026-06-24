"""QUEST backend — osunlp/QUEST-RL-Data.

QUEST ships two task families:

- **objective**: graded by the dataset's *generated* eval scripts
  (``eval_scripts/{task_id}.py``, fetched from the HF dataset repo) which drive the
  vendored ``obj_task_eval`` verification-tree evaluator;
- **open-ended**: graded by a pairwise rubric (``open_ended.score_open_ended_answer``)
  against the row's ``reward_model`` ground truth.

Both run through the same OpenAI-compatible judge (``QuestOpenAIClient``). The eval
scripts, evaluator tree, and open-ended scorer are vendored verbatim from the
composable taskset; this module is the v1 ``vf.Taskset`` shell around them.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import logging
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import verifiers.v1 as vf
from openai import AsyncOpenAI
from pydantic import BaseModel
from verifiers.types import ClientConfig
from verifiers.utils.client_utils import setup_openai_client

from search_v1._base import SearchTask
from search_v1.quest.obj_task_eval.utils.cache_filesys import CacheFileSys
from search_v1.quest.obj_task_eval.utils.load_eval_script import load_eval_script
from search_v1.quest.open_ended import score_open_ended_answer

if TYPE_CHECKING:
    from search_v1._base import SearchTaskset

logger = logging.getLogger(__name__)

DEFAULT_DATASET_NAME = "osunlp/QUEST-RL-Data"
DEFAULT_SPLIT = "train"
DEFAULT_CATEGORY = "objective"

_EVAL_SCRIPTS_ROOT_CACHE: dict[str, Path] = {}


class QuestJudgeError(RuntimeError):
    """Any QUEST judge / evaluator failure. Caught by the base reward → 0.0,
    matching the composable rubric's resilience."""


# --------------------------------------------------------------------------- #
# OpenAI-compatible judge client (verbatim from the composable taskset, with   #
# the v0 vf.* error types collapsed to QuestJudgeError).                        #
# --------------------------------------------------------------------------- #


def _usage_dict(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    return {
        "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
    }


def _single_choice(response: Any, *, context: str) -> Any:
    if response is None:
        raise QuestJudgeError(f"QUEST judge returned no {context} response")
    choices = getattr(response, "choices", None)
    if choices is None:
        raise QuestJudgeError(f"QUEST judge returned no {context} response choices")
    if len(choices) != 1:
        raise QuestJudgeError(
            f"QUEST judge returned {len(choices)} {context} choices, expected 1"
        )
    return choices[0]


class QuestOpenAIClient:
    """OpenAI-compatible client adapter for QUEST's ``async_response`` API."""

    provider = "openai"

    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        model: str,
        sampling_args: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.sampling_args = dict(sampling_args or {})
        self._client = client

    async def async_response(self, *, count_token: bool = False, **kwargs: Any) -> Any:
        response_format = kwargs.pop("response_format", None)
        messages = kwargs.pop("messages")
        model = kwargs.pop("model", self.model) or self.model
        request_kwargs = dict(self.sampling_args)
        request_kwargs.update(kwargs)
        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            try:
                response = await self._client.beta.chat.completions.parse(
                    model=model,
                    messages=messages,
                    response_format=response_format,
                    **request_kwargs,
                )
            except Exception as exc:
                raise QuestJudgeError(
                    f"QUEST structured judge request failed for {model}: {exc}"
                ) from exc
            choice = _single_choice(response, context="structured")
            parsed = choice.message.parsed
            if parsed is None:
                raise QuestJudgeError(
                    f"QUEST judge returned no parsed structured response for {model}"
                )
            usage = _usage_dict(response)
            return (parsed, usage) if count_token else parsed
        if response_format is not None:
            request_kwargs["response_format"] = response_format
        try:
            response = await self._client.chat.completions.create(
                model=model,
                messages=messages,
                **request_kwargs,
            )
        except Exception as exc:
            raise QuestJudgeError(
                f"QUEST judge request failed for {model}: {exc}"
            ) from exc
        choice = _single_choice(response, context="text")
        content = choice.message.content
        if content is None:
            raise QuestJudgeError(f"QUEST judge returned no text content for {model}")
        usage = _usage_dict(response)
        return (content, usage) if count_token else content

    async def close(self) -> None:
        await self._client.close()


# --------------------------------------------------------------------------- #
# Dataset row parsing (verbatim from the composable taskset).                   #
# --------------------------------------------------------------------------- #


def _parse_ast_literal(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _parse_ast_literal(node.body)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_parse_ast_literal(item) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_parse_ast_literal(item) for item in node.elts)
    if isinstance(node, ast.Dict):
        return {
            _parse_ast_literal(key): _parse_ast_literal(value)
            for key, value in zip(node.keys, node.values)
        }
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand = _parse_ast_literal(node.operand)
        if isinstance(operand, int | float):
            return -operand
    if isinstance(node, ast.Name) and node.id == "object":
        return object
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "array" and len(node.args) == 1:
            return _parse_ast_literal(node.args[0])
    raise ValueError(f"Unsupported QUEST literal syntax: {ast.dump(node)}")


def _parse_literal(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except Exception:
        try:
            return _parse_ast_literal(ast.parse(value, mode="eval"))
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


def _normalize_eval_scripts_root(path: Path) -> Path:
    root = path.expanduser()
    if root.name == "eval_scripts" and root.is_dir():
        return root.parent
    if (root / "eval_scripts").is_dir():
        return root
    raise ValueError(
        f"QUEST eval scripts directory must contain eval_scripts/*.py: {root}"
    )


def _resolve_eval_scripts_root(
    *, dataset_name: str, eval_scripts_dir: str | None
) -> Path:
    if eval_scripts_dir is not None:
        return _normalize_eval_scripts_root(Path(eval_scripts_dir))
    cached = _EVAL_SCRIPTS_ROOT_CACHE.get(dataset_name)
    if cached is not None:
        return cached
    try:
        from huggingface_hub import snapshot_download

        try:
            root = snapshot_download(
                repo_id=dataset_name,
                repo_type="dataset",
                allow_patterns=["eval_scripts/*.py"],
                local_files_only=True,
            )
        except Exception:
            root = snapshot_download(
                repo_id=dataset_name,
                repo_type="dataset",
                allow_patterns=["eval_scripts/*.py"],
            )
    except Exception as exc:
        raise QuestJudgeError(
            f"Failed to resolve QUEST eval scripts from {dataset_name}"
        ) from exc
    scripts_root = _normalize_eval_scripts_root(Path(root))
    _EVAL_SCRIPTS_ROOT_CACHE[dataset_name] = scripts_root
    return scripts_root


# --------------------------------------------------------------------------- #
# Taskset                                                                       #
# --------------------------------------------------------------------------- #


def instruction(ts: "SearchTaskset", question: str) -> str:
    return (
        f"{question}\n\n"
        f"When you have the final response, write it to {ts.config.answer_file} "
        "using a tool call, then stop. The task is incomplete unless that file "
        "exists. Include supporting URLs/citations in the file and do not include "
        "scratch reasoning or tool traces."
    )


def load_tasks(ts: "SearchTaskset") -> list[SearchTask]:
    from datasets import load_dataset

    category = ts.config.category or DEFAULT_CATEGORY
    if category not in {"objective", "open-ended", "all"}:
        raise ValueError("category must be one of 'objective', 'open-ended', or 'all'")
    dataset_name = ts.config.dataset_name or DEFAULT_DATASET_NAME
    split = ts.config.split or DEFAULT_SPLIT
    raw = load_dataset(
        dataset_name,
        split=split,
        keep_in_memory=ts.config.ds_keep_in_memory,
        num_proc=ts.config.ds_num_proc,
    )
    defaults = ts._task_defaults()
    tasks: list[SearchTask] = []
    idx = 0
    for row in raw:
        row_category = row.get("rl_task_category")
        if category != "all" and row_category != category:
            continue
        if row_category not in {"objective", "open-ended"}:
            raise ValueError(f"Unsupported QUEST row category: {row_category!r}")
        extra_info = _parse_literal(row.get("extra_info"))
        reward_model = _parse_literal(row.get("reward_model"))
        question = _extract_question(row.get("prompt"), extra_info)
        task_id = _extract_task_id(reward_model, extra_info)
        if not task_id:
            raise ValueError(f"QUEST {row_category} row missing task_id metadata")
        if row_category == "open-ended" and not isinstance(reward_model, dict):
            raise ValueError("QUEST open-ended row has invalid reward_model")
        tasks.append(
            SearchTask(
                idx=idx,
                name=task_id,
                prompt=instruction(ts, question),
                question=question,
                rl_task_category=str(row_category),
                task_id=task_id,
                reward_model=reward_model if isinstance(reward_model, dict) else None,
                **defaults,
            )
        )
        idx += 1
    return tasks


async def score(
    ts: "SearchTaskset", task: SearchTask, answer: str, trace: vf.Trace
) -> float:
    if task.rl_task_category == "open-ended":
        return await _open_ended(ts, task, answer, trace)
    return await _objective(ts, task, answer, trace)


async def _objective(
    ts: "SearchTaskset", task: SearchTask, answer: str, trace: vf.Trace
) -> float:
    if not task.task_id:
        raise QuestJudgeError("QUEST objective task missing task_id")
    script_path = _script_path(ts, task.task_id)
    cache = CacheFileSys(str(_cache_dir(ts) / _safe_module_component(task.task_id)))
    evaluate_answer = load_eval_script(str(script_path))
    summary = await evaluate_answer(
        client=_client(ts),
        answer=answer,
        agent_name="rlm",
        answer_name=str(task.name or task.task_id),
        cache=cache,
        semaphore=_sem(ts),
        logger=logger,
        model=ts.config.judge_model,
    )
    trace.info["quest_eval_summary"] = summary
    return _clamp_score(summary)


async def _open_ended(
    ts: "SearchTaskset", task: SearchTask, answer: str, trace: vf.Trace
) -> float:
    if not isinstance(task.reward_model, dict):
        raise QuestJudgeError("QUEST open-ended task missing reward_model")
    summary = await score_open_ended_answer(
        client=_client(ts),
        model=ts.config.judge_model,
        semaphore=_sem(ts),
        answer=answer,
        question=task.question,
        reward_model=task.reward_model,
    )
    trace.info["quest_eval_summary"] = summary
    trace.info["quest_upstream_pairwise_score"] = summary.get("upstream_pairwise_score")
    return _clamp_score(summary)


# --- lazily-initialised helpers, cached on the taskset instance ---


def _client(ts: "SearchTaskset") -> QuestOpenAIClient:
    client = getattr(ts, "_quest_client", None)
    if client is None:
        openai_client = setup_openai_client(
            ClientConfig(
                api_key_var=ts.config.judge_api_key_var,
                api_base_url=ts.config.judge_base_url or "https://api.openai.com/v1",
                timeout=1200.0,
            )
        )
        client = QuestOpenAIClient(
            client=openai_client,
            model=ts.config.judge_model,
            sampling_args=ts.config.judge_sampling_args,
        )
        ts._quest_client = client
    return client


def _sem(ts: "SearchTaskset") -> asyncio.Semaphore:
    sem = getattr(ts, "_quest_sem", None)
    if sem is None:
        sem = asyncio.Semaphore(ts.config.quest_eval_concurrency)
        ts._quest_sem = sem
    return sem


def _cache_dir(ts: "SearchTaskset") -> Path:
    if ts.config.quest_cache_dir:
        return Path(ts.config.quest_cache_dir).expanduser()
    return Path.home() / ".cache" / "verifiers" / "quest"


def _script_path(ts: "SearchTaskset", task_id: str) -> Path:
    scripts_root = getattr(ts, "_quest_scripts_root", None)
    if scripts_root is None:
        scripts_root = _resolve_eval_scripts_root(
            dataset_name=ts.config.dataset_name or DEFAULT_DATASET_NAME,
            eval_scripts_dir=ts.config.quest_eval_scripts_dir,
        )
        ts._quest_scripts_root = scripts_root
    script_path = scripts_root / "eval_scripts" / f"{task_id}.py"
    if not script_path.is_file():
        raise FileNotFoundError(
            f"QUEST eval script not found for task_id={task_id!r}: {script_path}"
        )
    return script_path


def _clamp_score(summary: Any) -> float:
    final_score = float((summary or {}).get("final_score", 0.0) or 0.0)
    if not math.isfinite(final_score):
        return 0.0
    return max(0.0, min(1.0, final_score))
