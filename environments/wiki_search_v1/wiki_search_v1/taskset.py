"""wiki-search: answer trivia by searching a wiki corpus (read-only tools + judge).

The opposite shape from wikispeedia: read-only tools and an LLM-judge reward. The taskset
loads questions from a HuggingFace dataset; the task exposes semantic `search_pages` +
`view_sections`/`read_section` over the full corpus (chroma index) via a `vf.Toolset`. The
expensive corpus + index are built in the toolset's `setup` (runs in the server process), and
the toolset is SHARED — one instance for the whole eval, not rebuilt per rollout. Grading is
a `vf.Judge` called from the task's `correct` reward, with a prompt that also requires
coherence.
"""

import functools

import verifiers.v1 as vf

from wiki_search_v1.servers.wiki import WikiSearchToolset

SYSTEM = (
    "Use the Wikipedia search tools — `wiki_search_pages` to find relevant pages, "
    "`wiki_view_sections` to list a page's sections, and `wiki_read_section` to read "
    "one — to answer the question. When confident, reply with a concise final answer."
)

JUDGE_PROMPT = """Given a ground truth answer and a response, determine if the response \
is both correct and coherent.

Question:
```
{question}
```

Ground truth answer:
```
{answer}
```

Response:
```
{response}
```

Respond either "yes" or "no" only. If a response contains incoherent text, respond \
with "no" even if the correct answer is also present."""

# The question bank and count are fixed properties of this env, not eval-time
# knobs (the searchable corpus is built in `WikiSearchToolset.setup`).
QUESTIONS_DATASET = "willcb/wiki-trivia-questions-v4"
NUM_QUESTIONS = 20


class CorrectnessJudge(vf.Judge[float]):
    """Scores the reply 1/0 against the reference answer (this env's prompt also
    requires coherence); an unparseable verdict raises — a judge failure errors the
    rollout instead of scoring the model 0."""

    prompt = JUDGE_PROMPT

    def parse(self, response: vf.JudgeResponse[float]) -> float:
        return float(vf.judge_verdict(response.text, ("yes", "no")) == "yes")


@functools.cache
def correctness_judge() -> CorrectnessJudge:
    """The eval's one judge instance — it holds an HTTP client, so it is cached here
    rather than constructed per reward call."""
    return CorrectnessJudge(vf.JudgeConfig())


class TriviaTask(vf.Task):
    question: str
    answer: str
    tools: vf.ToolsetConfig = vf.ToolsetConfig(shared=True)
    """Placement for the wiki tool server (from the taskset's `tools` knob)."""

    def load_tools(self) -> list[vf.Toolset]:
        return [WikiSearchToolset(self.tools)]

    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace) -> float:
        response = trace.last_reply
        if not (response or "").strip():
            return 0.0  # nothing to grade — skip the (foregone) judge call
        result = await correctness_judge().evaluate(
            trace=trace, question=self.question, answer=self.answer, response=response
        )
        return float(result.parsed or 0.0)


class WikiSearchConfig(vf.TasksetConfig):
    # SHARED: the chroma corpus is expensive, so one instance serves the whole eval (its own
    # runtime), reused across rollouts rather than rebuilt per rollout. CLI-tunable, e.g.
    # `--taskset.tools.shared false` or `--taskset.tools.runtime.type docker`.
    tools: vf.ToolsetConfig = vf.ToolsetConfig(shared=True)


class WikiSearchTaskset(vf.Taskset[TriviaTask, WikiSearchConfig]):
    def load_tasks(self) -> list[TriviaTask]:
        from datasets import load_dataset

        rows = load_dataset(QUESTIONS_DATASET, split="train")
        return [
            TriviaTask(
                idx=i,
                question=row["question"],
                answer=str(row["answer"]),
                prompt=f"{SYSTEM}\n\nQuestion: {row['question']}",
                tools=self.config.tools,
            )
            for i, row in enumerate(rows.select(range(min(NUM_QUESTIONS, len(rows)))))
        ]
