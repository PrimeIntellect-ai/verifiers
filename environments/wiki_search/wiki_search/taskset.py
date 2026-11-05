"""Answer trivia with a shared wiki corpus and a reference judge.

The tool server builds an expensive corpus/index in its process-level `setup`, so
`shared=True` launches one instance for the serving owner instead of rebuilding it
per rollout. The tools are read-only; grading comes from the plugged wiki-search judge
(class-level prompt; reject incoherent answers).
"""

from pydantic import Field

import verifiers.v1 as vf
from wiki_search.judge import WikiSearchJudgeConfig
from wiki_search.servers.wiki import WikiSearchToolset

SYSTEM = (
    "Use the Wikipedia search tools — `wiki_search_pages` to find relevant pages, "
    "`wiki_view_sections` to list a page's sections, and `wiki_read_section` to read "
    "one — to answer the question. When confident, reply with a concise final answer."
)

# The question bank and count are fixed properties of this env, not eval-time
# knobs (the searchable corpus is built in `WikiSearchToolset.setup`).
QUESTIONS_DATASET = "willcb/wiki-trivia-questions-v4"
NUM_QUESTIONS = 20


class WikiSearchTaskConfig(vf.TaskConfig):
    tools: vf.ToolsetConfig = vf.ToolsetConfig(shared=True)
    # Users can replace or reconfigure this judge through --taskset.task.judges.
    judges: vf.Judges = Field(default_factory=lambda: [WikiSearchJudgeConfig()])


class TriviaTaskData(vf.TaskData):
    # These fields feed the reference judge's question and answer template values.
    question: str
    answer: str


class TriviaTask(vf.Task[TriviaTaskData, vf.State, WikiSearchTaskConfig]):
    @classmethod
    def toolsets(cls, config: WikiSearchTaskConfig) -> list[vf.Toolset]:
        return [WikiSearchToolset(config.tools)]


class WikiSearchConfig(vf.TasksetConfig):
    task: WikiSearchTaskConfig = WikiSearchTaskConfig()


class WikiSearchTaskset(vf.Taskset[TriviaTask, WikiSearchConfig]):
    def load(self) -> list[TriviaTask]:
        from datasets import load_dataset

        rows = load_dataset(QUESTIONS_DATASET, split="train")
        return [
            TriviaTask(
                TriviaTaskData(
                    question=row["question"],
                    answer=str(row["answer"]),
                    prompt=f"{SYSTEM}\n\nQuestion: {row['question']}",
                ),
                self.config.task,
            )
            for row in rows.select(range(min(NUM_QUESTIONS, len(rows))))
        ]
