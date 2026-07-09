"""gsm8k: grade-school math word problems (single-turn, in-runtime verification).

Loads questions from GSM8K; the model reasons and gives a final number. The reward
runs a uv script (`verify.py`, with `math-verify` as an isolated dependency) IN the
rollout's runtime via `runtime.write` + `runtime.run` — so the verifier's deps never
touch the eval process, and it works identically on the subprocess, docker, and
prime runtimes. (A reward is thus either a pure function of the trace or, like this
one, runtime read/write/exec.)
"""

from pathlib import Path
from typing import Literal

import verifiers.v1 as vf

SYSTEM = (
    "Solve the grade-school math problem. Reason step by step, then give the final "
    "answer as a single number on the last line, prefixed with '#### ' (e.g. '#### 42')."
)
VERIFY = (Path(__file__).parent / "verify.py").read_bytes()


class GSM8KData(vf.TaskData):
    answer: str
    """The ground-truth final answer (the value after GSM8K's `####`)."""


class GSM8KTask(vf.Task[GSM8KData]):
    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        prediction = trace.last_reply
        result = await runtime.run_uv_script(
            VERIFY, args=[self.data.answer, prediction or ""]
        )
        if result.exit_code != 0:
            raise RuntimeError(f"verify.py failed: {result.stderr.strip()[-500:]}")
        lines = result.stdout.strip().splitlines()
        return float(lines[-1]) if lines else 0.0

    async def validate(self, runtime: vf.Runtime) -> bool:
        """Valid iff the verifier accepts the ground-truth answer: run `verify.py` on the gold
        answer as a well-formed `#### N` prediction and require a 1.0 score — catching rows the
        verifier can't parse or grade (the model-free counterpart of the `correct` reward)."""
        result = await runtime.run_uv_script(
            VERIFY, args=[self.data.answer, f"#### {self.data.answer}"]
        )
        if result.exit_code != 0:
            raise RuntimeError(f"verify.py failed: {result.stderr.strip()[-500:]}")
        lines = result.stdout.strip().splitlines()
        return bool(lines) and float(lines[-1]) == 1.0


class GSM8KConfig(vf.TasksetConfig):
    split: Literal["train", "test"] = "test"


class GSM8KTaskset(vf.Taskset[GSM8KTask, GSM8KConfig]):
    def load(self) -> list[GSM8KData]:
        from datasets import load_dataset

        rows = load_dataset("openai/gsm8k", "main", split=self.config.split)
        return [
            GSM8KData(
                idx=i,
                prompt=f"{SYSTEM}\n\n{row['question']}",
                answer=row["answer"].split("####")[-1].strip(),
            )
            for i, row in enumerate(rows)
        ]
