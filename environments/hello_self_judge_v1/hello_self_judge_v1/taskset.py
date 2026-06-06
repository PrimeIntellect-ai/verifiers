import verifiers.v1 as vf

SYSTEM_PROMPT = """Answer the question concisely and include a Sources: line."""

TASKS: list[vf.JsonData] = [
    {
        "task_id": "example-domains",
        "question": "Explain what the example domains are reserved for.",
        "seed_urls": ["https://www.iana.org/domains/reserved"],
        "answer_hint": "reserved for use in documentation and examples",
    },
    {
        "task_id": "rfc-9110-404",
        "question": "Explain what HTTP status code 404 means.",
        "seed_urls": ["https://www.rfc-editor.org/rfc/rfc9110.txt"],
        "answer_hint": "target resource was not found",
    },
    {
        "task_id": "python-json",
        "question": "Summarize what json.dumps does.",
        "seed_urls": ["https://docs.python.org/3/library/json.html"],
        "answer_hint": "serializes an object to a JSON formatted string",
    },
]


class SelfJudgeTasksetConfig(vf.TasksetConfig):
    system_prompt: str = SYSTEM_PROMPT
    num_examples: int = -1


class SelfJudgeHarnessConfig(vf.HarnessConfig):
    max_turns: int = 1


class SelfJudgeTask(vf.Task):
    question: str
    seed_urls: list[str]
    answer_hint: str


class SelfJudgeTaskset(vf.Taskset[SelfJudgeTasksetConfig]):
    task_type = SelfJudgeTask

    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        if split == "eval":
            return []
        rows = (
            TASKS if self.config.num_examples < 0 else TASKS[: self.config.num_examples]
        )
        return [
            {
                **row,
                "row_id": index,
                "prompt": [{"role": "user", "content": str(row["question"])}],
                "max_turns": 1,
            }
            for index, row in enumerate(rows)
        ]

    @vf.update(priority=10)
    async def evidence_review(self, task: SelfJudgeTask, state: vf.State) -> None:
        response = assistant_text(state)
        urls = [str(url) for url in task.seed_urls]
        findings = {
            "has_answer": bool(response.strip()),
            "mentions_source": any(url in response for url in urls),
            "mentions_sources_line": "sources:" in response.lower(),
            "mentions_hint": task.answer_hint.lower() in response.lower(),
        }
        state.scratch["judge"] = findings
        state.artifacts["judge_findings"] = findings

    @vf.metric
    async def source_mentions(self, state: vf.State) -> float:
        judge = state.scratch.get("judge")
        if not isinstance(judge, dict):
            return 0.0
        return float(bool(judge.get("mentions_source")))

    @vf.reward(weight=1.0)
    async def self_consistency_score(self, state: vf.State) -> float:
        judge = state.scratch.get("judge")
        if not isinstance(judge, dict):
            return 0.0
        checks = [
            bool(judge.get("has_answer")),
            bool(judge.get("mentions_sources_line")),
            bool(judge.get("mentions_hint")),
        ]
        return sum(float(check) for check in checks) / len(checks)


def assistant_text(state: vf.State) -> str:
    messages = vf.get_messages(state.completion or [], role="assistant")
    return str(messages[-1].content or "") if messages else ""


def load_taskset(config: SelfJudgeTasksetConfig) -> SelfJudgeTaskset:
    return SelfJudgeTaskset(config=config)


def load_harness(config: SelfJudgeHarnessConfig) -> vf.Harness:
    return vf.Harness(config=config)
