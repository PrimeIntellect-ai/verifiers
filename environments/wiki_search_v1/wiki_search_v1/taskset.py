"""Answer trivia with a worker-shared wiki corpus and a reference judge.

The tool server builds an expensive corpus/index in its process-level `setup`, so
`Taskset.tools` launches one instance per environment worker instead of rebuilding it
per rollout. The tools are read-only; grading comes from the plugged `reference` judge,
whose prompt also rejects incoherent answers.
"""

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


class WikiSearchTaskConfig(vf.TaskConfig):
    # Users can replace or reconfigure this judge through --taskset.task.judges.
    judges: vf.Judges = [
        vf.ReferenceJudgeConfig(prompt=JUDGE_PROMPT, question_field="question")
    ]


class TriviaTaskData(vf.TaskData):
    # These fields feed the reference judge's question and answer template values.
    question: str
    answer: str


class TriviaTask(vf.Task[TriviaTaskData, vf.State, WikiSearchTaskConfig]):
    pass


class WikiSearchConfig(vf.TasksetConfig):
    tools: vf.SharedToolsetConfig = vf.SharedToolsetConfig()
    task: WikiSearchTaskConfig = WikiSearchTaskConfig()


class WikiSearchTaskset(vf.Taskset[TriviaTask, WikiSearchConfig]):
    tools = (WikiSearchToolset,)

    def load(self) -> list[TriviaTask]:
        from datasets import load_dataset

        rows = load_dataset(QUESTIONS_DATASET, split="train")
        return [
            TriviaTask(
                TriviaTaskData(
                    idx=i,
                    question=row["question"],
                    answer=str(row["answer"]),
                    prompt=f"{SYSTEM}\n\nQuestion: {row['question']}",
                ),
                self.config.task,
            )
            for i, row in enumerate(rows.select(range(min(NUM_QUESTIONS, len(rows)))))
        ]
