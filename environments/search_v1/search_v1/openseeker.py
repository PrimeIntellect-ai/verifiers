"""OpenSeeker backend — PolarSeeker/OpenSeeker-v1-Data.

OpenSeeker's public evaluator scores final-answer semantics against a gold answer
with a binary LLM judge ([CORRECT] / [INCORRECT]); this port preserves that exact
prompt and parse so reward stays at parity with the composable taskset.
"""

from __future__ import annotations

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

import verifiers.v1 as vf

from search_v1._base import SearchTask

if TYPE_CHECKING:
    from search_v1._base import SearchTaskset

logger = logging.getLogger(__name__)

DEFAULT_DATASET_NAME = "PolarSeeker/OpenSeeker-v1-Data"
DEFAULT_SPLIT = "train"

# Upstream OpenSeeker binary semantic judge (verbatim from the composable taskset).
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


def parse_judge_label(raw: str | None) -> int | None:
    """Parse the judge's ``A``/``B`` choice into 1 (CORRECT) / 0 (INCORRECT)."""
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


def _merge_sampling_args(sampling_args: dict[str, Any] | None) -> dict[str, Any]:
    request_kwargs: dict[str, Any] = {
        "temperature": 0.0,
        "extra_body": {"skip_special_tokens": False},
    }
    for key, value in (sampling_args or {}).items():
        if (
            key == "extra_body"
            and isinstance(value, dict)
            and isinstance(request_kwargs.get("extra_body"), dict)
        ):
            request_kwargs["extra_body"] = {**request_kwargs["extra_body"], **value}
            continue
        request_kwargs[key] = value
    return request_kwargs


def instruction(ts: "SearchTaskset", question: str) -> str:
    return ts.get_instruction(question)


def load_tasks(ts: "SearchTaskset") -> list[SearchTask]:
    from datasets import load_dataset

    dataset_name = ts.config.dataset_name or DEFAULT_DATASET_NAME
    split = ts.config.split or DEFAULT_SPLIT
    raw = load_dataset(
        dataset_name,
        split=split,
        keep_in_memory=ts.config.ds_keep_in_memory,
        num_proc=ts.config.ds_num_proc,
    )
    columns = ["question", "answer", "number of tool calls", "trajectory correctness"]
    raw = raw.select_columns([c for c in columns if c in raw.column_names])

    defaults = ts._task_defaults()
    tasks: list[SearchTask] = []
    idx = 0
    for row in raw:
        question = str(row.get("question") or "").strip()
        answer = str(row.get("answer") or "").strip()
        if not question or not answer:
            continue
        tool_calls = row.get("number of tool calls")
        tasks.append(
            SearchTask(
                idx=idx,
                name=f"openseeker-{idx}",
                prompt=ts.get_instruction(question),
                question=question,
                answer=answer,
                number_of_tool_calls=tool_calls if isinstance(tool_calls, int) else None,
                trajectory_correctness=row.get("trajectory correctness"),
                **defaults,
            )
        )
        idx += 1
    return tasks


async def score(
    ts: "SearchTaskset", task: SearchTask, answer: str, trace: vf.Trace
) -> float:
    if not task.question or not task.answer:
        raise vf.TasksetError("OpenSeeker task missing question or gold answer")
    raw = await _judge(ts, question=task.question, correct_answer=task.answer, response=answer)
    label = parse_judge_label(raw)
    trace.info["judge_raw"] = raw
    trace.info["judge_label"] = label
    if label is None:
        # Unparseable judge output scored 0.0 in the composable taskset.
        trace.info["eval_error"] = f"unparseable_judge_label: {raw!r}"
        return 0.0
    return float(label)


async def _judge(
    ts: "SearchTaskset", *, question: str, correct_answer: str, response: str
) -> str | None:
    prompt = JUDGE_PROMPT_BC_EN.format(
        question=question, correct_answer=correct_answer, response=response
    )
    request_kwargs = _merge_sampling_args(ts.config.judge_sampling_args)
    try:
        completion = await ts.judge_client().chat.completions.create(
            model=ts.config.judge_model,
            messages=[
                {"role": "system", "content": "Judge the response objectively."},
                {"role": "user", "content": prompt},
            ],
            **request_kwargs,
        )
    except Exception as exc:  # transient/model errors scored 0.0 upstream
        logger.warning("OpenSeeker judge request failed: %s", exc)
        return None
    choices = getattr(completion, "choices", None)
    if not choices or len(choices) != 1:
        return None
    content = choices[0].message.content
    return content.strip() if content else None
