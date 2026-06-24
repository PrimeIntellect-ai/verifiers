"""s1-deepresearch: answer deep-research questions by searching the web (web tools + judge).

A v1 port of the ScienceOne-AI ``S1-DeepResearch-15k`` deep-research dataset. The taskset
loads the verifiable (closed-ended) subset of the dataset, exposes ``web_search`` /
``web_visit`` tools over Serper + a page fetcher (a shared ``vf.Toolset``), and scores the
model's final answer against the released ground truth with a normalized exact-match
shortcut plus an LLM-as-judge (the BROWSECOMP answer-match prompt).

The upstream repo declares its ``meta`` column with the ``Json`` feature type, which only
exists in ``datasets>=4.7``; verifiers pins ``datasets<4.7``, so ``load_dataset`` raises
``Feature type 'Json' not found``. We download the raw ``data.jsonl`` (cached via
``huggingface_hub``) and parse it line by line instead.
"""

import json
import re
import unicodedata
from typing import Any

import verifiers.v1 as vf
from verifiers.v1.dialects import ChatDialect

from s1_deepresearch_v1.servers.search import WebSearchConfig, WebSearchToolset

DEFAULT_DATASET = "ScienceOne-AI/S1-DeepResearch-15k"
DEFAULT_DATA_FILE = "data.jsonl"

# The dataset mixes verifiable closed-ended tasks (which carry a ground-truth answer) with
# open-ended exploration tasks (which do not). Only the former can be answer-matched.
VERIFIABLE_TASK_TYPE = "Closed-ended Multi-hop Resolution"

INSTRUCTION = (
    "You are a deep research assistant. Research the question below using the `web_search` "
    "and `web_visit` tools: decompose it into sub-queries, gather and cross-check evidence "
    "across multiple sources, then give a single concise final answer. End your final "
    "message with the answer and the supporting source URLs; do not include scratch "
    "reasoning or tool traces in it."
)

# DeepTraceHub's released BROWSECOMP answer-match judge prompt (the same convention the
# REDSearcher/OpenSeeker backends use). S1-DeepResearch does not publish its own RL reward.
JUDGE_PROMPT = """\
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
- If the standard answer contains more information than the question asks for, the predicted answer only needs to include the information mentioned in the question.
    - For example, for the question "What is the main chemical component of magnesite?", with the standard answer "Magnesium carbonate (MgCO3)", "Magnesium carbonate" or "MgCO3" are both considered [CORRECT].
- If information omitted in the predicted answer can be clearly inferred from the question, it's considered correct.
- If it's clear that different translations of a name refer to the same person, it's considered correct.
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


def _coerce_answer(value: Any) -> str:
    """Normalize a ``meta.answer`` value to a string (a few report tasks carry a dict/list)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False).strip()
    return str(value).strip()


def _normalize_for_match(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _exact_answer_match(*, response: str, answer: str) -> bool:
    normalized_answer = _normalize_for_match(answer)
    normalized_response = _normalize_for_match(response)
    if not normalized_answer or not normalized_response:
        return False
    return normalized_answer == normalized_response


def _parse_judge_choice(content: str) -> float | None:
    text = (content or "").strip()
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


class S1DeepResearchTask(vf.Task):
    question: str
    answer: str
    language: str = ""
    source_id: str | None = None


class JudgeConfig(vf.BaseClientConfig):
    # base_url / api_key_var / Prime team-billing are inherited from BaseClientConfig
    # (default: Prime inference + PRIME_API_KEY), matching the composable backends.
    model: str = "openai/gpt-5.4-mini"


class S1DeepResearchConfig(vf.TasksetConfig):
    dataset_name: str = DEFAULT_DATASET
    data_file: str = DEFAULT_DATA_FILE
    verifiable_only: bool = True
    """Keep only `Closed-ended Multi-hop Resolution` rows (those with a ground-truth answer)."""
    language: str | None = None
    """Optional language filter (`en` or `zh`)."""
    max_examples: int | None = None
    """Optional cap on the number of tasks loaded."""
    use_exact_match_shortcut: bool = True
    judge: JudgeConfig = JudgeConfig()
    # SHARED: the web-search tools are stateless/read-only, so one server serves the whole
    # eval (CLI-tunable, e.g. `--taskset.tools.shared false`).
    tools: WebSearchConfig = WebSearchConfig(shared=True)


class S1DeepResearchTaskset(
    vf.Taskset[S1DeepResearchTask, S1DeepResearchConfig]
):
    def load_tasks(self) -> list[S1DeepResearchTask]:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=self.config.dataset_name,
            filename=self.config.data_file,
            repo_type="dataset",
        )
        tasks: list[S1DeepResearchTask] = []
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    meta = json.loads(line).get("meta")
                except json.JSONDecodeError:
                    continue
                if not isinstance(meta, dict):
                    continue
                task_type = str(meta.get("type") or "").strip()
                if self.config.verifiable_only and task_type != VERIFIABLE_TASK_TYPE:
                    continue
                language = str(meta.get("language") or "").strip()
                if self.config.language is not None and language != self.config.language:
                    continue
                question = str(meta.get("question") or "").strip()
                answer = _coerce_answer(meta.get("answer"))
                if not question or not answer:
                    continue
                tasks.append(
                    S1DeepResearchTask(
                        idx=len(tasks),
                        prompt=f"{INSTRUCTION}\n\nQuestion: {question}",
                        question=question,
                        answer=answer,
                        language=language,
                        source_id=meta.get("id"),
                    )
                )
                if (
                    self.config.max_examples is not None
                    and len(tasks) >= self.config.max_examples
                ):
                    break
        if not tasks:
            raise ValueError(
                f"S1-DeepResearch produced no tasks from {self.config.dataset_name!r} "
                f"(verifiable_only={self.config.verifiable_only}, "
                f"language={self.config.language!r})"
            )
        return tasks

    def tools(self, task: S1DeepResearchTask) -> list[vf.Toolset]:
        return [WebSearchToolset(self.config.tools)]

    @vf.metric()
    async def answered(self, trace: vf.Trace) -> float:
        return float(bool(trace.assistant_messages and trace.assistant_messages[-1].content))

    @vf.reward(weight=1.0)
    async def answer_match(self, task: S1DeepResearchTask, trace: vf.Trace) -> float:
        response = (
            trace.assistant_messages[-1].content if trace.assistant_messages else ""
        ) or ""
        response = response.strip()
        if not response:
            return 0.0
        if self.config.use_exact_match_shortcut and _exact_answer_match(
            response=response, answer=task.answer
        ):
            return 1.0
        prompt = JUDGE_PROMPT.format(
            question=task.question, correct_answer=task.answer, response=response
        )
        client = vf.resolve_client(self.config.judge)
        try:
            verdict = await client.get_response(
                ChatDialect(),
                {"messages": [{"role": "user", "content": prompt}]},
                self.config.judge.model,
                vf.SamplingConfig(),
            )
        finally:
            await client.close()
        parsed = _parse_judge_choice(verdict.message.content or "")
        return parsed if parsed is not None else 0.0
