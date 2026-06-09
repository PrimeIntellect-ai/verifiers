"""wiki-search: answer trivia by searching a wiki corpus (read-only tools + judge).

The opposite shape from wikispeedia: read-only tools and an LLM-judge reward. The
taskset loads questions from a HuggingFace dataset and, per rollout, launches a
host-side server exposing semantic `search_pages` + `view_sections`/`read_section`
over the full corpus (chroma index, host-side). The reward asks a judge model
whether the harness's answer matches the ground truth.
"""

import sys

import verifiers.v1 as vf

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
    # SHARED: the chroma corpus is expensive to build, so the server runs as ONE
    # instance for the whole eval (in its own `tools.runtime`), reused across rollouts
    # rather than rebuilt per rollout. (colocated and shared are mutually exclusive.)
    tools: vf.ToolsConfig = vf.ToolsConfig(colocated=False, shared=True)
    judge: JudgeConfig = JudgeConfig()


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

    def tool_servers(self, task: TriviaTask) -> list[vf.ToolServer]:
        return [
            vf.ToolServer(
                name="wiki", command=[sys.executable, "-m", "wiki_search_v1.server"]
            )
        ]

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
                [vf.UserMessage(content=prompt)],
                self.config.judge.model,
                vf.SamplingConfig(),
            )
        finally:
            await client.close()
        return float("yes" in (verdict.message.content or "").lower())


def load_taskset(config: WikiSearchConfig) -> WikiSearchTaskset:
    return WikiSearchTaskset(config)
