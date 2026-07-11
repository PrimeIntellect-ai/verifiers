"""code_golf: write a Python program and evaluate it in the rollout runtime.

Each task asks for a tiny program with a known output. Runtime-dependent measurements
are recorded per trace:

  - `evaluate`      per-rollout `@metric`: runs the program once in that rollout's
                    runtime and records `passed` + `latency`. (task, trace, runtime)
  - `correct`       per-rollout `@reward`: reads `passed` off the trace.      (trace)
Length and latency remain metrics for analysis. Relative comparison across independent
programs is a training-algorithm concern.
"""

import re
import time

import verifiers.v1 as vf

SYSTEM = (
    "Write a single self-contained Python program that prints EXACTLY the requested "
    "output and nothing else. Put the program in one ```python code block."
)


def extract_program(trace: vf.Trace) -> str:
    """The python from the last assistant message's code block (or its raw text)."""
    text = trace.last_reply
    match = re.search(r"```(?:python)?\n(.*?)```", text or "", re.DOTALL)
    return (match.group(1) if match else text or "").strip()


class CodeGolfData(vf.TaskData):
    expected: str
    """The exact stdout the program must produce."""


class CodeGolfTask(vf.Task[CodeGolfData]):
    @vf.metric
    async def evaluate(self, trace: vf.Trace, runtime: vf.Runtime) -> dict[str, float]:
        """Run the program once in the rollout's runtime; record correctness + latency."""
        program = extract_program(trace)
        if not program:
            return {"passed": 0.0, "latency": 1e6, "length": 0.0}
        await runtime.write("solution.py", program.encode())
        start = time.perf_counter()
        result = await runtime.run(["python3", "solution.py"], {})
        latency = time.perf_counter() - start
        passed = float(
            result.exit_code == 0 and result.stdout.strip() == self.data.expected
        )
        return {"passed": passed, "latency": latency, "length": float(len(program))}

    @vf.reward
    async def correct(self, trace: vf.Trace) -> float:
        return trace.metrics.get("passed", 0.0)


class CodeGolfTaskset(vf.Taskset[CodeGolfTask, vf.TasksetConfig]):
    # (name, description, expected stdout)
    SPECS = [
        ("simple-sum", "the sum of the integers 1 to 100", "5050"),
        (
            "fibonacci",
            "the first 10 Fibonacci numbers space-separated on one line",
            "0 1 1 2 3 5 8 13 21 34",
        ),
        ("reverse-str", "the string HELLO reversed", "OLLEH"),
    ]

    def load(self) -> list[CodeGolfTask]:
        return [
            CodeGolfTask(
                CodeGolfData(
                    idx=i,
                    name=name,
                    prompt=f"{SYSTEM}\n\nPrint {description}.",
                    expected=expected,
                ),
                self.config.task,
            )
            for i, (name, description, expected) in enumerate(self.SPECS)
        ]
