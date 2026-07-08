"""code_golf: write a short, fast Python program — showcases GROUP rewards.

Each task asks for a tiny program with a known output. We sample a *group* of rollouts
per task (run with `-r 2` for the intended pairwise effect) and score them. Anything
that needs the runtime is measured per rollout, as a `@metric`, into the trace; the
group rewards then just compare that trace metadata across the task's rollouts:

  - `evaluate`      per-rollout `@metric`: runs the program once in that rollout's
                    runtime and records `passed` + `latency`. (task, trace, runtime)
  - `correct`       per-rollout `@reward`: reads `passed` off the trace.      (trace)
  - `most_concise`  `@group_reward`: of the group, the shortest source wins.  (traces)
  - `fastest`       `@group_reward`: of the group, the lowest `latency`
                    wins — a comparison of recorded trace metadata.           (traces)

So a group of 2 produces, per rollout: did it work, was it the shorter one, was it the
quicker one — the relative signals you can only get by comparing siblings.
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


class CodeGolfTask(vf.Task):
    expected: str
    """The exact stdout the program must produce."""

    @vf.metric
    async def evaluate(self, trace: vf.Trace, runtime: vf.Runtime) -> dict[str, float]:
        """Run the program once in the rollout's runtime; record correctness + latency."""
        program = extract_program(trace)
        if not program:
            return {"passed": 0.0, "latency": 1e6}
        await runtime.write("solution.py", program.encode())
        start = time.perf_counter()
        result = await runtime.run(["python3", "solution.py"], {})
        latency = time.perf_counter() - start
        passed = float(result.exit_code == 0 and result.stdout.strip() == self.expected)
        return {"passed": passed, "latency": latency}

    @vf.reward
    async def correct(self, trace: vf.Trace) -> float:
        return trace.metrics.get("passed", 0.0)

    @vf.group_reward(weight=0.5)
    async def most_concise(self, traces: list[vf.Trace]) -> list[float]:
        """The shortest program in the group wins; ties share."""
        lengths = [len(extract_program(t)) or 10**9 for t in traces]
        best = min(lengths)
        return [1.0 if length == best else 0.0 for length in lengths]

    @vf.group_reward(weight=0.5)
    async def fastest(self, traces: list[vf.Trace]) -> list[float]:
        """The lowest recorded `latency` in the group wins; ties share."""
        times = [t.metrics.get("latency", 1e6) for t in traces]
        best = min(times)
        return [1.0 if t == best else 0.0 for t in times]


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
                idx=i,
                name=name,
                prompt=f"{SYSTEM}\n\nPrint {description}.",
                expected=expected,
            )
            for i, (name, description, expected) in enumerate(self.SPECS)
        ]
