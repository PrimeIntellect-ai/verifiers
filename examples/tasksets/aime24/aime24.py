"""aime24: AIME 2024 competition math (single-turn, in-runtime scoring).

The eval analog of `math-env`: load the 30 AIME 2024 problems, prompt the model to reason
and box its integer answer, and score by math equivalence (`math-verify`) of the boxed
answer against the gold — the same in-runtime uv-script verifier as `math-env`.
"""

from pathlib import Path

import verifiers.v1 as vf

INSTRUCTION = (
    "Solve the following math problem. Explain your reasoning and put the final answer "
    "in \\boxed{}.\n\n"
)
VERIFY = (Path(__file__).parent / "verify.py").read_bytes()


class AIME24Task(vf.Task):
    answer: str
    """The ground-truth integer answer (0–999), as a string."""


class AIME24Config(vf.TasksetConfig):
    dataset_name: str = "HuggingFaceH4/aime_2024"
    dataset_split: str = "train"
    dataset_revision: str = "2fe88a2f1091d5048c0f36abc874fb997b3dd99a"
    math_verify_timeout: int = 5
    """Per-answer wall-clock budget for the math-verify comparison, in seconds."""


class AIME24Taskset(vf.Taskset[AIME24Task, AIME24Config]):
    def load_tasks(self) -> list[AIME24Task]:
        from datasets import load_dataset

        rows = load_dataset(
            self.config.dataset_name,
            split=self.config.dataset_split,
            revision=self.config.dataset_revision,
        )
        return [
            AIME24Task(
                idx=i,
                instruction=INSTRUCTION + row["problem"],
                answer=str(int(row["answer"])),
            )
            for i, row in enumerate(rows)
        ]

    @vf.stop
    async def single_turn(self, trace: vf.Trace) -> bool:
        # AIME is single-turn: refuse a second turn so the model answers once.
        return trace.num_turns >= 1

    @vf.reward(weight=1.0)
    async def correct(
        self, task: AIME24Task, trace: vf.Trace, runtime: vf.Runtime
    ) -> float:
        prediction = (
            trace.assistant_messages[-1].content if trace.assistant_messages else ""
        )
        result = await runtime.run_uv_script(
            VERIFY,
            args=[task.answer, prediction or "", str(self.config.math_verify_timeout)],
        )
        if result.exit_code != 0:
            raise vf.ProgramError(f"verify.py failed: {result.stderr.strip()[-500:]}")
        lines = result.stdout.strip().splitlines()
        return float(lines[-1]) if lines else 0.0


def load_taskset(config: AIME24Config) -> AIME24Taskset:
    return AIME24Taskset(config)
