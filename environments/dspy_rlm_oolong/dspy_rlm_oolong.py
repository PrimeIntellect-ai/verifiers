"""
ApiEnv example using DSPy's RLM on Oolong long-context benchmark tasks.

Combines DSPy's RLM (Recursive Language Model) code-execution agent with the
Oolong benchmark datasets and rubrics. The model uses DSPy's sandboxed Python
REPL to programmatically explore large contexts and find information, while
ApiEnv intercepts all LLM calls and routes them to the model under evaluation.

Note: RLM requires Deno for its WASM sandbox. Install with:
    curl -fsSL https://deno.land/install.sh | sh
"""

import ast
import os
import random
from datetime import datetime
from typing import Literal, get_args

import dateutil.parser
import dspy
import httpx

import verifiers as vf
from datasets import load_dataset
from openai import AsyncOpenAI
from verifiers.rubrics.judge_rubric import JudgeRubric
from verifiers.utils.data_utils import extract_boxed_answer

# =============================================================================
# OOLONG-SYNTH dataset names (from oolongbench/oolong-synth on Hugging Face)
# =============================================================================

OolongSynthDatasetName = Literal[
    "agnews",
    "app_reviews",
    "formality",
    "imdb",
    "metaphors",
    "multinli",
    "negation",
    "spam",
    "trec_coarse",
    "yahoo",
]
OOLONG_SYNTH_DATASET_NAMES: frozenset[str] = frozenset(get_args(OolongSynthDatasetName))
OOLONG_SYNTH_DATASET_NAMES_VALIDATION_ONLY: frozenset[str] = frozenset(
    ("spam", "trec_coarse")
)

OOLONG_SYNTH_CONTEXT_LENGTHS: frozenset[int] = frozenset(
    (
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
        131072,
        262144,
        524288,
        1048576,
        2097152,
        4194304,
    )
)

OolongRealConfigName = Literal["dnd", "toy_dnd"]
OOLONG_REAL_CONFIG_NAMES: frozenset[str] = frozenset(get_args(OolongRealConfigName))


def _as_list(x):
    if isinstance(x, (str, int)):
        return [x]
    return list(x)


# =============================================================================
# OOLONG Scoring Helpers
# Ported from https://github.com/abertsch72/oolong/blob/main/src/eval/eval_helpers.py
# =============================================================================


def _synth_attempt_answer_parse(answer: str) -> tuple[str, str]:
    parse_confidence = "low"
    if ":" not in answer:
        if len(answer) < 20:
            return answer, parse_confidence
        else:
            return answer.split()[-1], parse_confidence
    candidate_answer = answer.split(":")[-1].strip()
    candidate_answer = candidate_answer.replace("*", "")
    candidate_answer = candidate_answer.replace("[", "")
    candidate_answer = candidate_answer.replace("]", "")
    parse_confidence = "med"
    if (
        "User:" in answer
        or "Answer:" in answer
        or "Date:" in answer
        or "Label" in answer
    ):
        parse_confidence = "high"
    if len(candidate_answer) < 20:
        parse_confidence = "vhigh"
    elif "more common" in candidate_answer:
        candidate_answer = "more common"
    elif "less common" in candidate_answer:
        candidate_answer = "less common"
    elif "same frequency" in candidate_answer:
        candidate_answer = "same frequency"
    return candidate_answer, parse_confidence


def _synth_score(answer_raw: str, answer_type: str, output: str) -> float:
    gold = (
        ast.literal_eval(answer_raw)[0]
        if "datetime" not in answer_raw
        else datetime.strptime(answer_raw, "[datetime.date(%Y, %m, %d)]")
    )
    trimmed_output, _ = _synth_attempt_answer_parse(output)

    if str(trimmed_output) == str(gold):
        return 1.0
    elif str(trimmed_output) in ["more common", "less common", "same frequency"]:
        if str(trimmed_output) in str(gold):
            return 1.0
    elif answer_type == "ANSWER_TYPE.NUMERIC":
        try:
            return float(0.75 ** abs(int(gold) - int(trimmed_output)))
        except Exception:
            pass
    elif answer_type == "ANSWER_TYPE.DATE":
        try:
            parsed = dateutil.parser.parse(str(trimmed_output))
            return 1.0 if parsed == gold else 0.0
        except Exception:
            pass
    return 0.0


def _dnd_parse_answer(answer: str) -> int | str | list[str]:
    try:
        return int(answer)
    except ValueError:
        pass
    if "," in answer:
        return [item.strip() for item in answer.split(",") if item.strip()]
    return answer


def _dnd_score(answer_raw: str, output: str) -> float:
    gold = _dnd_parse_answer(answer_raw)
    raw = extract_boxed_answer(output) or output or ""
    trimmed_output = _dnd_parse_answer(raw.strip())

    if isinstance(gold, int) and isinstance(trimmed_output, int):
        return float(0.75 ** abs(gold - trimmed_output))
    elif isinstance(gold, str) and isinstance(trimmed_output, str):
        return 1.0 if gold.strip().lower() == trimmed_output.strip().lower() else 0.0
    elif isinstance(gold, list) and isinstance(trimmed_output, list):
        overlap = set(gold) & set(trimmed_output)
        return len(overlap) / len(gold) if gold else 0.0
    return 0.0


# =============================================================================
# Rubrics
# =============================================================================


def oolong_reward(
    completion: vf.Messages, answer: str, info: dict | None = None, **kwargs
) -> float:
    """Deterministic Oolong scoring with partial credit."""
    if not completion:
        return 0.0
    last = str(completion[-1].content or "")
    info = info or {}
    subset = info.get("subset", "synth")

    if subset == "real":
        return _dnd_score(answer, last)
    answer_type = info.get("answer_type", "")
    return _synth_score(answer, answer_type, last)


class OolongJudgeRubric(JudgeRubric):
    """LLM judge rubric for binary correctness scoring."""

    def __init__(
        self,
        judge_model: str = "gpt-4.1-nano",
        judge_api_key_var: str = "OPENAI_API_KEY",
        judge_base_url: str | None = None,
    ):
        httpx_timeout = httpx.Timeout(1200)
        httpx_limits = httpx.Limits(
            max_connections=8192, max_keepalive_connections=8192
        )
        httpx_client = httpx.AsyncClient(limits=httpx_limits, timeout=httpx_timeout)
        judge_client = AsyncOpenAI(
            base_url=judge_base_url,
            api_key=os.getenv(judge_api_key_var) if judge_api_key_var else "EMPTY",
            http_client=httpx_client,
        )
        super().__init__(judge_client=judge_client, judge_model=judge_model)
        self.add_reward_func(self.judge_reward, weight=1.0)

    async def judge_reward(self, state: vf.State, **_kwargs) -> float:
        question = state["info"]["raw_question"]
        response = state.get("final_answer", "")
        ground_truth = state.get("answer", "")
        judge_prompt = self.judge_prompt.format(
            question=question,
            answer=ground_truth,
            response=response,
        )
        judge_result = await self.judge_client.chat.completions.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": judge_prompt}],
        )
        judge_answer = judge_result.choices[0].message.content or ""
        return 1.0 if "yes" in judge_answer.lower() else 0.0


# =============================================================================
# DSPy RLM Agent
# =============================================================================

_ENV_TIPS = """
Strategy for long-context information retrieval:
1. Split the context into chunks (e.g., by paragraphs or fixed character windows with some overlap)
2. Write a prompt describing what to look for, then append it to each chunk to create a list of prompts
3. Call llm_batch() once with all prompts to scan chunks in parallel
4. Aggregate the relevant findings from the responses"""


def make_agent_fn(max_rlm_iterations: int = 10):
    """Create a DSPy RLM agent function with the given iteration limit."""

    async def run_agent(base_url: str, state: vf.State):
        lm = dspy.LM(
            f"openai/{state['model']}",
            api_base=base_url,
            api_key="intercepted",
            cache=False,
        )

        with dspy.context(lm=lm):
            rlm = dspy.RLM(
                "query -> answer",
                max_iterations=max_rlm_iterations,
            )

            query = state["prompt"][-1]["content"]
            result = await rlm.aforward(query=query)
            return result.answer

    return run_agent


# =============================================================================
# Environment Loading
# =============================================================================


def load_environment(
    # Dataset options
    subset: Literal["synth", "synth_with_labels", "real"] = "synth",
    split: Literal["validation", "test"] = "validation",
    dataset_name: str | list[str] | None = None,
    context_len: int | list[int] | None = None,
    shuffle: bool = False,
    seed: int | None = None,
    include_env_tips: bool = False,
    # Reward options
    reward_mode: Literal["oolong", "judge"] = "oolong",
    judge_model: str = "gpt-4.1-nano",
    judge_api_key_var: str = "OPENAI_API_KEY",
    judge_base_url: str | None = None,
    # DSPy RLM options
    max_rlm_iterations: int = 10,
    timeout_seconds: float = 300.0,
) -> vf.ApiEnv:
    """
    Load the DSPy RLM + Oolong long-context evaluation environment.

    Args:
        subset: Which subset to use:
            - "synth": Synthetic dataset with context_window_text
            - "synth_with_labels": Synthetic with context_window_text_with_labels
            - "real": Real-world dataset (dnd/toy_dnd)
        split: Dataset split ("validation" or "test").
        dataset_name: For "real": single config ("dnd"/"toy_dnd"). For "synth":
            one or more dataset names (str or list[str]).
        context_len: Synth only. int or list[int] to filter by context length.
        shuffle: Whether to shuffle the dataset.
        seed: Random seed for shuffling.
        include_env_tips: Include strategy tips in the prompt.
        reward_mode: "oolong" for deterministic scoring, "judge" for LLM judge.
        judge_model: Model for judging (only when reward_mode="judge").
        judge_api_key_var: Env var for judge API key.
        judge_base_url: Base URL for judge API.
        max_rlm_iterations: Max DSPy RLM iterations (REPL turns).
        timeout_seconds: Per-rollout timeout.

    Returns:
        Configured ApiEnv instance.
    """
    # --- Validate dataset args (same logic as oolong-rlm) ---
    names_list: list[str] = []
    context_lens_list: list[int] = []
    if subset == "real":
        if context_len is not None:
            raise ValueError(
                "context_len is only valid for subset 'synth' or 'synth_with_labels'. "
                f"subset 'real' does not support context_len; got context_len={context_len!r}."
            )
        names_list = _as_list(dataset_name) if dataset_name is not None else []
        if names_list:
            if len(names_list) > 1:
                raise ValueError(
                    "For subset 'real', dataset_name must be a single config ('dnd' or 'toy_dnd')."
                )
            if names_list[0] not in OOLONG_REAL_CONFIG_NAMES:
                raise ValueError(
                    f"dataset_name={names_list[0]!r} is not a valid oolong-real config. "
                    f"Must be one of: {sorted(OOLONG_REAL_CONFIG_NAMES)}."
                )
        hf_dataset_name = "oolongbench/oolong-real"
        hf_config_name = names_list[0] if names_list else "dnd"
        context_column = "context_window_text"
    else:
        names_list = _as_list(dataset_name) if dataset_name is not None else []
        context_lens_list = _as_list(context_len) if context_len is not None else []
        test_only_names = (
            OOLONG_SYNTH_DATASET_NAMES - OOLONG_SYNTH_DATASET_NAMES_VALIDATION_ONLY
        )
        for n in names_list:
            if n not in OOLONG_SYNTH_DATASET_NAMES:
                raise ValueError(
                    f"dataset_name={n!r} is not a valid oolong-synth name. "
                    f"Must be one of: {sorted(OOLONG_SYNTH_DATASET_NAMES)}."
                )
            if (
                n in OOLONG_SYNTH_DATASET_NAMES_VALIDATION_ONLY
                and split != "validation"
            ):
                raise ValueError(f"dataset_name={n!r} is only in the validation split.")
            if n in test_only_names and split != "test":
                raise ValueError(f"dataset_name={n!r} is only in the test split.")
        for cl in context_lens_list:
            if cl not in OOLONG_SYNTH_CONTEXT_LENGTHS:
                raise ValueError(
                    f"context_len={cl!r} is not valid. Must be one of: {sorted(OOLONG_SYNTH_CONTEXT_LENGTHS)}."
                )
        hf_dataset_name = "oolongbench/oolong-synth"
        hf_config_name = None
        context_column = (
            "context_window_text"
            if subset == "synth"
            else "context_window_text_with_labels"
        )

    # --- Build dataset ---
    def transform_example(example, idx):
        question = example["question"]
        context = example[context_column]
        answer = example["answer"]

        prompt_content = f"Context:\n{context}\n\nQuestion: {question}"
        if include_env_tips:
            prompt_content += f"\n\n{_ENV_TIPS}"

        info: dict = {
            "subset": subset,
            "raw_question": question,
            "answer_type": example.get("answer_type", ""),
        }
        if subset in ("synth", "synth_with_labels"):
            if "context_len" in example:
                info["context_len"] = example["context_len"]
            if "dataset" in example:
                info["dataset"] = example["dataset"]

        return {
            "example_id": idx,
            "prompt": [{"role": "user", "content": prompt_content}],
            "task": "oolong",
            "answer": answer,
            "info": info,
        }

    def build_dataset():
        raw_dataset = load_dataset(hf_dataset_name, hf_config_name, split=split)

        if subset in ("synth", "synth_with_labels") and (
            names_list or context_lens_list
        ):

            def _filter_synth(example):
                if names_list and example.get("dataset") not in names_list:
                    return False
                if (
                    context_lens_list
                    and example.get("context_len") not in context_lens_list
                ):
                    return False
                return True

            raw_dataset = raw_dataset.filter(
                _filter_synth, desc="filter by dataset_name/context_len"
            )

        dataset = raw_dataset.map(
            transform_example,
            with_indices=True,
            remove_columns=raw_dataset.column_names,
            writer_batch_size=100,
        )

        if shuffle:
            _seed = seed if seed is not None else random.randint(1000, 100_000_000)
            dataset = dataset.shuffle(seed=_seed)

        return dataset

    # --- Rubric ---
    if reward_mode == "judge":
        rubric = OolongJudgeRubric(
            judge_model=judge_model,
            judge_api_key_var=judge_api_key_var,
            judge_base_url=judge_base_url,
        )
    else:
        rubric = vf.Rubric(funcs=[oolong_reward])

    return vf.ApiEnv(
        agent_fn=make_agent_fn(max_rlm_iterations),
        dataset=build_dataset,
        rubric=rubric,
        timeout_seconds=timeout_seconds,
    )
