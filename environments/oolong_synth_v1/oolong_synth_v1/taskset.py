"""oolong-synth-v1 — Oolong synthetic long-context questions solved by an agent in a sandbox.

Each task's long context window is uploaded to `/workspace/context.txt` so the agent can scan it
from a REPL instead of spending tokens on the whole document. The agent writes its final answer
(a single token / word / date / label) to `/workspace/answer.txt`; the reward reads it back
(falling back to the agent's last message) and scores with the official Oolong synth rules
(deterministic; partial credit for numeric/date answers) or, when a `judge` is configured, a
host-side binary LLM judge.

Dataset: `oolongbench/oolong-synth`.
"""

import ast
from datetime import datetime, timezone
from typing import Literal

import verifiers.v1 as vf
from oolong_synth_v1.judge import OolongJudge, OolongJudgeConfig

WORKDIR = "/workspace"
CONTEXT_PATH = f"{WORKDIR}/context.txt"
ANSWER_PATH = f"{WORKDIR}/answer.txt"

# Valid oolong-synth context lengths (tokens). The two smallest are the most tractable; the real
# long-context regime is the larger buckets.
ContextLen = Literal[1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304]

# Answer types the scorer gives partial credit for; every other type is exact-match (`""`).
AnswerType = Literal["", "ANSWER_TYPE.NUMERIC", "ANSWER_TYPE.DATE"]

INSTRUCTIONS = (
    f"The context window is in `{CONTEXT_PATH}`. Scan it from a REPL, then write your final "
    f"answer — and ONLY your final answer (a single token / word / date / label) — to "
    f"`{ANSWER_PATH}` (also output it as your last message)."
)


def attempt_answer_parse(answer: str) -> str:
    """Extract the candidate answer from the agent's output."""
    if ":" not in answer:
        return answer
    candidate = answer.split(":")[-1].strip().replace("*", "").replace("[", "").replace("]", "")
    for phrase in ("more common", "less common", "same frequency"):
        if phrase in candidate:
            return phrase
    return candidate


def parse_gold(answer_raw: str):
    """Parse the dataset's serialized gold answer (a list literal, or a wrapped date)."""
    if "datetime" not in answer_raw:
        return ast.literal_eval(answer_raw)[0]
    return datetime.strptime(answer_raw, "[datetime.date(%Y, %m, %d)]").replace(tzinfo=timezone.utc)


def score(answer_raw: str, answer_type: str, output: str) -> float:
    gold = parse_gold(answer_raw)
    trimmed_output = attempt_answer_parse(output)
    if str(trimmed_output) == str(gold):
        return 1.0
    elif str(trimmed_output) in ["more common", "less common", "same frequency"]:
        if str(trimmed_output) in str(gold):
            return 1.0
    elif answer_type == "ANSWER_TYPE.NUMERIC":
        try:
            return float(0.75 ** abs(int(gold) - int(trimmed_output)))
        except (TypeError, ValueError):
            return 0.0
    elif answer_type == "ANSWER_TYPE.DATE":
        try:
            import dateutil.parser

            return 1.0 if dateutil.parser.parse(str(trimmed_output)) == gold else 0.0
        except (TypeError, ValueError):
            return 0.0
    return 0.0


class OolongSynthData(vf.TaskData):
    question: str
    """Raw dataset question (passed to the judge; the prompt wraps it with task instructions)."""
    answer: str
    """Raw gold answer from the dataset (parsed by the Oolong scorer)."""
    context: str
    """The long context window, uploaded to `CONTEXT_PATH` before the agent runs."""
    answer_type: AnswerType = ""
    """Answer type that drives partial credit; `""` for every other type (exact match)."""


class OolongSynthTaskConfig(vf.TaskConfig):
    judge: OolongJudgeConfig | None = None
    """When set, score with this LLM judge instead of the deterministic Oolong rules
    (None = deterministic)."""


class OolongSynthTask(vf.Task[OolongSynthData, vf.State, OolongSynthTaskConfig]):
    NEEDS_CONTAINER = True

    async def setup(self, runtime: vf.Runtime) -> None:
        await runtime.run(["mkdir", "-p", WORKDIR], {})
        await runtime.write(CONTEXT_PATH, self.data.context.encode())

    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        response = await vf.read_answer_file_or_last_reply(runtime, ANSWER_PATH, trace)
        if self.config.judge is not None:
            judge = OolongJudge(self.config.judge)
            result = await judge.evaluate(
                trace=trace,
                question=self.data.question,
                answer=str(parse_gold(self.data.answer)),
                response=response,
            )
            return 1.0 if result.parsed else 0.0
        return score(self.data.answer, self.data.answer_type, response)


class OolongSynthConfig(vf.TasksetConfig):
    split: Literal["validation", "test"] = "validation"
    with_labels: bool = False
    """Use the label-augmented context window (`context_window_text_with_labels`)."""
    context_len: ContextLen = 262144
    """Which `context_len` (token) bucket to evaluate (default: 256K)."""
    task: OolongSynthTaskConfig = OolongSynthTaskConfig()


class OolongSynthTaskset(vf.Taskset[OolongSynthTask, OolongSynthConfig]):
    def load(self) -> list[OolongSynthTask]:
        from datasets import load_dataset

        cfg = self.config
        context_column = "context_window_text_with_labels" if cfg.with_labels else "context_window_text"
        rows = load_dataset("oolongbench/oolong-synth", split=cfg.split, streaming=True)
        tasks: list[OolongSynthTask] = []
        for i, row in enumerate(rows):
            if row.get("context_len") != cfg.context_len:
                continue
            answer_type = row.get("answer_type", "")
            if answer_type not in ("ANSWER_TYPE.NUMERIC", "ANSWER_TYPE.DATE"):
                answer_type = ""
            tasks.append(
                OolongSynthTask(
                    OolongSynthData(
                        idx=i,
                        prompt=f"{row['question']}\n\n{INSTRUCTIONS}",
                        question=row["question"],
                        answer=row["answer"],
                        context=row[context_column],
                        answer_type=answer_type,
                        workdir=WORKDIR,
                    ),
                    self.config.task,
                )
            )
        return tasks
