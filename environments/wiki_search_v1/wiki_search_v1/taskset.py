"""wiki-search: answer trivia by searching a wiki corpus (read-only tools + judge).

The opposite shape from wikispeedia: read-only tools and an LLM-judge reward. The taskset
loads questions from a HuggingFace dataset and exposes semantic `search_pages` +
`view_sections`/`read_section` over the full corpus (chroma index) via a `vf.Toolset`. The
expensive corpus + index are built in the toolset's `setup` (runs in the server process), and
the toolset is SHARED — one instance for the whole eval, not rebuilt per rollout. The reward
comes from a single-call judge agent (`vf.JudgeSpec`, `null` harness): its evidence rides in
its prompt and its reply is the typed verdict. The judge's endpoint binds through the env
model table — run with e.g. `--models.judge.model deepseek/deepseek-v4-flash`.
"""

from pydantic import BaseModel

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

If a response contains incoherent text, it is not correct even if the correct answer is \
also present."""

# The question bank and count are fixed properties of this env, not eval-time
# knobs (the searchable corpus is built in `WikiSearchToolset.setup`).
QUESTIONS_DATASET = "willcb/wiki-trivia-questions-v4"
NUM_QUESTIONS = 20


class TriviaTask(vf.Task):
    question: str
    answer: str


class MatchVerdict(BaseModel):
    correct: bool
    """Whether the response matches the ground truth and is coherent."""


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
            )
            for i, row in enumerate(rows.select(range(min(NUM_QUESTIONS, len(rows)))))
        ]

    def tools(self, task: TriviaTask) -> list[vf.Toolset]:
        return [WikiSearchToolset(self.config.tools)]

    async def judges(self, task: TriviaTask, trace: vf.Trace) -> list[vf.JudgeSpec]:
        return [
            vf.JudgeSpec(
                name="match",
                prompt=JUDGE_PROMPT.format(
                    question=task.question,
                    answer=task.answer,
                    response=trace.last_reply or "",
                ),
                verdict=MatchVerdict,
                harness={"id": "null"},
                model="judge",
            )
        ]

    @vf.reward(weight=1.0)
    async def judged(self, verdicts) -> float:
        return float(verdicts["match"].correct)
