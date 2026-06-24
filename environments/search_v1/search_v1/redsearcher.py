"""REDSearcher backend — Zchu/REDSearcher_RL_1K.

The public artifact is a QA dataset with no verifier scripts, so scoring uses the
paper/repo's answer-matching convention: a normalized exact-match shortcut, then a
DeepTraceHub BROWSECOMP-style LLM judge (with retries) as fallback. Prompt, parse,
normalization, and retry behavior are ported verbatim from the composable taskset.
"""

from __future__ import annotations

from __future__ import annotations

import logging
import re
import unicodedata
from typing import TYPE_CHECKING

import verifiers.v1 as vf

from search_v1._base import SearchTask

if TYPE_CHECKING:
    from search_v1._base import SearchTaskset

logger = logging.getLogger(__name__)

DEFAULT_DATASET_NAME = "Zchu/REDSearcher_RL_1K"
DEFAULT_SPLIT = "train"

# DeepTraceHub's released BROWSECOMP judge prompt (verbatim from the composable taskset);
# the closest public reference for REDSearcher's RL reward.
_JUDGE_PROMPT = """\
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
    - For example, for the question "What is the main chemical component of magnesite?", with the standard answer "Magnesium carbonate (MgCO3)", "Magnesium carbonate" or "MgCO3" are both considered [CORRECT] answers.
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

Return only the option representing [CORRECT] or [INCORRECT], i.e. just return A or B, without adding any other text.
"""


def _normalize_for_match(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def exact_answer_match(*, response: str, answer: str) -> bool:
    normalized_answer = _normalize_for_match(answer)
    normalized_response = _normalize_for_match(response)
    if not normalized_answer or not normalized_response:
        return False
    return normalized_answer == normalized_response


def parse_judge_choice(content: str) -> float | None:
    text = content.strip()
    if not text:
        return None
    first_line = text.splitlines()[0].strip("`*_ \t")
    upper = first_line.upper()
    if re.match(r"^\[?INCORRECT\]?(?:[\s.):\]-]|$)", upper) or re.match(
        r"^B(?:[\s.):\]-]|$)", upper
    ):
        return 0.0
    if re.match(r"^\[?CORRECT\]?(?:[\s.):\]-]|$)", upper) or re.match(
        r"^A(?:[\s.):\]-]|$)", upper
    ):
        return 1.0
    return None


def instruction(ts: "SearchTaskset", question: str) -> str:
    return (
        f"{question}\n\n"
        "This is a REDSearcher long-horizon search task. Break the problem into "
        "search subgoals, cross-check the answer across sources, and synthesize a "
        "concise final response.\n\n"
        f"When you have the final response, write it to {ts.config.answer_file} "
        "using a tool call, then stop. The task is incomplete unless that file "
        "exists. Include the requested answer and supporting URLs/citations in the "
        "file, but do not include scratch reasoning or tool traces."
    )


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
    # `difficulty` filters the row set (the composable env accepted it as a kwarg;
    # here it selects only matching-difficulty rows).
    want_difficulty = (ts.config.difficulty or "").strip().lower()

    defaults = ts._task_defaults()
    tasks: list[SearchTask] = []
    idx = 0
    for row in raw:
        difficulty = str(row.get("difficulty") or "")
        question = str(row.get("problem") or "").strip()
        answer = str(row.get("answer") or "").strip()
        if not question or not answer:
            continue
        if want_difficulty and difficulty.strip().lower() != want_difficulty:
            continue
        tasks.append(
            SearchTask(
                idx=idx,
                name=f"redsearcher-{idx}",
                prompt=instruction(ts, question),
                question=question,
                answer=answer,
                difficulty=difficulty,
                **defaults,
            )
        )
        idx += 1
    return tasks


async def score(
    ts: "SearchTaskset", task: SearchTask, answer: str, trace: vf.Trace
) -> float:
    if not task.answer:
        raise vf.TasksetError("REDSearcher task missing ground-truth answer")
    if ts.config.redsearcher_exact_match_shortcut and exact_answer_match(
        response=answer, answer=task.answer
    ):
        trace.info["match_method"] = "exact_match"
        return 1.0
    result = await _judge_answer(
        ts, question=task.question, response=answer, answer=task.answer, trace=trace
    )
    trace.info["match_method"] = "llm_judge"
    return result


async def _judge_answer(
    ts: "SearchTaskset", *, question: str, response: str, answer: str, trace: vf.Trace
) -> float:
    prompt = _JUDGE_PROMPT.format(
        question=question, response=response, correct_answer=answer
    )
    client = ts.judge_client()
    request_kwargs = dict(ts.config.judge_sampling_args or {})
    last_content = ""
    max_attempts = max(1, ts.config.redsearcher_judge_max_retries)
    for attempt in range(max_attempts):
        try:
            judge_response = await client.chat.completions.create(
                model=ts.config.judge_model,
                messages=[{"role": "user", "content": prompt}],
                **request_kwargs,
            )
        except Exception as exc:  # retry transient, else give up
            if attempt + 1 < max_attempts:
                logger.warning(
                    "REDSearcher judge request failed on attempt %s/%s: %r",
                    attempt + 1, max_attempts, exc,
                )
                continue
            logger.warning("REDSearcher judge request failed terminally: %r", exc)
            trace.info["eval_error"] = f"judge_request_failed: {exc!r}"
            return 0.0
        choices = getattr(judge_response, "choices", None)
        if choices is None or len(choices) != 1:
            last_content = (
                f"invalid choice count: {0 if choices is None else len(choices)}"
            )
        else:
            content = choices[0].message.content
            last_content = content or ""
            parsed = parse_judge_choice(last_content)
            if parsed is not None:
                trace.info["judge_response"] = last_content
                trace.info["judge_result"] = {
                    "correct": "yes" if parsed == 1.0 else "no",
                    "accuracy": parsed,
                }
                return parsed
        logger.warning(
            "Failed to parse REDSearcher judge response on attempt %s/%s: %r",
            attempt + 1, max_attempts, last_content[:200],
        )
    trace.info["eval_error"] = f"unparseable_judge_response: {last_content!r}"
    return 0.0
