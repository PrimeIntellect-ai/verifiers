"""math-env: math problems with a boxed final answer (single-turn, in-runtime scoring).

Resembles prime-rl's verifiers `math-env`: load a HF dataset of `{question, answer}`,
prompt the model to reason and box its final answer, and score by math equivalence
(`math-verify`) of the boxed answer against the gold. The verifier runs as a uv script
(`verify.py`, with `math-verify` as an isolated dependency) IN the rollout's runtime, so
its deps never touch the eval process and it works on the subprocess/docker/prime runtimes.
"""

from pathlib import Path

import verifiers.v1 as vf

INSTRUCTION = (
    "Solve the following math problem. Explain your reasoning and put the final answer "
    "in \\boxed{}.\n\n"
)
VERIFY = (Path(__file__).parent / "verify.py").read_bytes()


class MathTask(vf.Task):
    answer: str
    """The ground-truth answer — math-verify compares the model's boxed answer to this."""


class MathConfig(vf.TasksetConfig):
    dataset_name: str = "PrimeIntellect/Hendrycks-Math"
    dataset_subset: str = "default"
    dataset_split: str = "train"
    question_key: str = "question"
    answer_key: str = "answer"
    math_verify_timeout: int = 5
    """Per-answer wall-clock budget for the math-verify comparison, in seconds."""


class MathTaskset(vf.Taskset[MathTask, MathConfig]):
    def load_tasks(self) -> list[MathTask]:
        from datasets import load_dataset

        rows = load_dataset(
            self.config.dataset_name,
            self.config.dataset_subset,
            split=self.config.dataset_split,
        )
        return [
            MathTask(
                idx=i,
                prompt=INSTRUCTION + row[self.config.question_key],
                answer=str(row[self.config.answer_key]),
            )
            for i, row in enumerate(rows)
        ]

    @vf.stop
    async def single_turn(self, trace: vf.Trace) -> bool:
        # Math is single-turn: refuse a second turn so the model answers once.
        return trace.num_turns >= 1

    @vf.reward(weight=1.0)
    async def correct(
        self, task: MathTask, trace: vf.Trace, runtime: vf.Runtime
    ) -> float:
        prediction = (
            trace.assistant_messages[-1].content if trace.assistant_messages else ""
        )
        result = await runtime.run_uv_script(
            VERIFY,
            args=[task.answer, prediction or "", str(self.config.math_verify_timeout)],
        )
        if result.exit_code != 0:
            raise vf.SandboxError(f"verify.py failed: {result.stderr.strip()[-500:]}")
        lines = result.stdout.strip().splitlines()
        return float(lines[-1]) if lines else 0.0
