"""wiki-search: answer trivia by searching a wiki corpus (read-only tools + judge).

The opposite shape from wikispeedia: read-only tools and an LLM-judge reward. The taskset
loads questions from a HuggingFace dataset and exposes semantic `search_pages` +
`view_sections`/`read_section` over the full corpus (chroma index) via a `vf.Toolset`. The
expensive corpus + index are built in the toolset's `setup` (runs in the server process), and
the toolset is SHARED — one instance for the whole eval, not rebuilt per rollout. Grading is
the plugged built-in `reference` judge (a default `judges` entry, overridable per eval) with a
prompt that also requires coherence.
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
    # The built-in reference judge, plugged by default with this env's prompt (which also
    # requires coherence). Fully eval-tunable: `--taskset.task.judges.0.model ...`, or
    # replaced wholesale from the TOML's `[[taskset.task.judges]]`.
    judges: vf.Judges = [
        vf.ReferenceJudgeConfig(prompt=JUDGE_PROMPT, question_field="question")
    ]
    # SHARED: the chroma corpus is expensive, so one instance serves the whole eval (its own
    # runtime), reused across rollouts rather than rebuilt per rollout. CLI-tunable, e.g.
    # `--taskset.task.tools.shared false` or `--taskset.task.tools.runtime.type docker`.
    tools: vf.ToolsetConfig = vf.ToolsetConfig(shared=True)


class TriviaTaskData(vf.TaskData):
    question: str
    answer: str
    # `question`/`answer` feed the plugged reference judge (it reads them off the row by
    # field name: `question_field`/`answer_field`).


class TriviaTask(vf.Task[TriviaTaskData, vf.State, WikiSearchTaskConfig]):
    tools = (WikiSearchToolset,)
    # Built with the task config's `tools` field (SHARED by default; placement stays
    # CLI-tunable), resolved by `Task.server_config`.


class WikiSearchConfig(vf.TasksetConfig):
    task: WikiSearchTaskConfig = WikiSearchTaskConfig()


class WikiSearchTaskset(vf.Taskset[TriviaTask, WikiSearchConfig]):
    def load(self) -> list[TriviaTaskData]:
        from datasets import load_dataset

        rows = load_dataset(QUESTIONS_DATASET, split="train")
        return [
            TriviaTaskData(
                idx=i,
                question=row["question"],
                answer=str(row["answer"]),
                prompt=f"{SYSTEM}\n\nQuestion: {row['question']}",
            )
            for i, row in enumerate(rows.select(range(min(NUM_QUESTIONS, len(rows)))))
        ]
