"""wiki-search: answer trivia by searching a wiki corpus (read-only tools + judge).

The opposite shape from wikispeedia: read-only tools and an LLM-judge reward. The taskset
loads questions from a HuggingFace dataset and exposes semantic `search_pages` +
`view_sections`/`read_section` over the full corpus (chroma index) via a `vf.Toolset`. The
expensive corpus + index are built in the toolset's `setup` (runs in the server process), and
the toolset is SHARED — one instance for the whole eval, not rebuilt per rollout. The reward
asks a judge model whether the harness's answer matches the ground truth.
"""

import verifiers.v1 as vf
from verifiers.v1.dialects import ChatDialect

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
# knobs (the corpus dataset lives in corpus.py).
QUESTIONS_DATASET = "willcb/wiki-trivia-questions-v4"
NUM_QUESTIONS = 20


class TriviaTask(vf.Task):
    question: str
    answer: str


class JudgeConfig(vf.BaseClientConfig):
    # base_url / api_key_var / Prime team-billing are inherited from BaseClientConfig.
    model: str = "deepseek/deepseek-v4-flash"


class WikiSearchConfig(vf.TasksetConfig):
    judge: JudgeConfig = JudgeConfig()
    # SHARED: the chroma corpus is expensive, so one instance serves the whole eval (its own
    # runtime), reused across rollouts rather than rebuilt per rollout. CLI-tunable, e.g.
    # `--taskset.tools.shared false` or `--taskset.tools.runtime.type docker`.
    tools: vf.ToolsetConfig = vf.ToolsetConfig(shared=True)


class WikiSearchToolset(vf.Toolset[vf.ToolsetConfig]):
    """Read-only search/view/read over the wiki corpus. The corpus + chroma index (expensive)
    are built once in `setup`, in the server process; every tool call is a read. No per-server
    knobs, so it uses the base `vf.ToolsetConfig` (placement only)."""

    name = "wiki"  # the model sees `wiki_search_pages` / `wiki_view_sections` / `wiki_read_section`
    deps = ["chromadb", "datasets"]

    async def setup(self, task) -> None:
        from wiki_search_v1.corpus import collection, corpus

        self.pages = corpus()  # global state; the shared server ignores the per-task `task`
        self.collection = collection()

    @vf.tool
    def search_pages(self, query: str) -> list[dict]:
        """Search the wiki for the 10 most relevant pages (by title). Returns
        `[{page_id, title}, ...]` — pass a page_id to view_sections."""
        result = self.collection.query(query_texts=[query], n_results=10)
        return [
            {"page_id": pid, "title": self.pages[pid]["title"]}
            for pid in result["ids"][0]
        ]

    @vf.tool
    def view_sections(self, page_id: str) -> list[dict]:
        """List a page's sections. Returns `[{section_id, section_name}, ...]` — pass a
        section_id to read_section."""
        from wiki_search_v1.corpus import normalize_id

        content = self.pages[page_id]["content"]
        sections = [
            {
                "section_id": f"{page_id}:{normalize_id(line.lstrip('#').strip())}",
                "section_name": line.lstrip("#").strip(),
            }
            for line in content.split("\n")
            if line.startswith("#")
        ]
        return sections or [
            {"section_id": f"{page_id}:full", "section_name": "Full Page"}
        ]

    @vf.tool
    def read_section(self, section_id: str) -> str:
        """Read the text of a section (`page_id:section`; use `page_id:full` for all)."""
        from wiki_search_v1.corpus import normalize_id

        page_id, section = section_id.split(":", 1)
        content = self.pages[page_id]["content"]
        if section == "full":
            return content
        lines = content.split("\n")
        start = end = None
        for i, line in enumerate(lines):
            if line.startswith("#"):
                if normalize_id(line.lstrip("#").strip()) == section and start is None:
                    start = i
                elif start is not None:
                    end = i
                    break
        if start is None:
            return f"Section not found: {section_id}"
        return "\n".join(lines[start : end or len(lines)])


class WikiSearchTaskset(vf.Taskset[TriviaTask, WikiSearchConfig]):
    def load_tasks(self) -> list[TriviaTask]:
        from datasets import load_dataset

        rows = load_dataset(QUESTIONS_DATASET, split="train")
        return [
            TriviaTask(
                idx=i,
                question=row["question"],
                answer=str(row["answer"]),
                instruction=f"{SYSTEM}\n\nQuestion: {row['question']}",
            )
            for i, row in enumerate(rows.select(range(min(NUM_QUESTIONS, len(rows)))))
        ]

    def tools(self, task: TriviaTask) -> list[vf.Toolset]:
        return [WikiSearchToolset(self.config.tools)]

    @vf.reward(weight=1.0)
    async def judged(
        self, task: TriviaTask, trace: vf.Trace, runtime: vf.Runtime
    ) -> float:
        response = (
            trace.assistant_messages[-1].content if trace.assistant_messages else ""
        )
        prompt = JUDGE_PROMPT.format(
            question=task.question, answer=task.answer, response=response or ""
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
        return float("yes" in (verdict.message.content or "").lower())


def load_taskset(config: WikiSearchConfig) -> WikiSearchTaskset:
    return WikiSearchTaskset(config)
