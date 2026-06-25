"""math500: MATH-500 competition math (single-turn, in-runtime scoring).

The eval analog of `math-env`: load the 500 MATH-500 problems, prompt the model to reason
and box its final answer, and score by math equivalence (`math-verify`) of the boxed answer
against the gold — the same in-runtime uv-script verifier as `aime24` / `math-env`.
"""

from pathlib import Path

import verifiers.v1 as vf

# Appended after the problem to match the v0 `math500` env's prompt.
INSTRUCTION = (
    "\nPlease reason step by step, and put your final answer within \\boxed{}."
)
VERIFY = (Path(__file__).parent / "verify.py").read_bytes()


class Math500Task(vf.Task):
    answer: str
    """The ground-truth answer — math-verify compares the model's boxed answer to this."""


class Math500Config(vf.TasksetConfig):
    dataset_name: str = "HuggingFaceH4/MATH-500"
    dataset_split: str = "test"
    dataset_revision: str = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
    math_verify_timeout: int = 5
    """Per-answer wall-clock budget for the math-verify comparison, in seconds."""


class Math500Taskset(vf.Taskset[Math500Task, Math500Config]):
    def load_tasks(self) -> list[Math500Task]:
        from datasets import load_dataset

        rows = load_dataset(
            self.config.dataset_name,
            split=self.config.dataset_split,
            revision=self.config.dataset_revision,
        )
        return [
            Math500Task(
                idx=i,
                prompt=row["problem"] + INSTRUCTION,
                answer=str(row["answer"]),
            )
            for i, row in enumerate(rows)
        ]

    @vf.stop
    async def single_turn(self, trace: vf.Trace) -> bool:
        # MATH-500 is single-turn: refuse a second turn so the model answers once.
        return trace.num_turns >= 1

    @vf.reward(weight=1.0)
    async def correct(
        self, task: Math500Task, trace: vf.Trace, runtime: vf.Runtime
    ) -> float:
        prediction = (
            trace.assistant_messages[-1].content if trace.assistant_messages else ""
        )
        result = await runtime.run_uv_script(
            VERIFY,
            args=[task.answer, prediction or "", str(self.config.math_verify_timeout)],
        )
        if result.exit_code != 0:
            raise RuntimeError(f"verify.py failed: {result.stderr.strip()[-500:]}")
        lines = result.stdout.strip().splitlines()
        return float(lines[-1]) if lines else 0.0
