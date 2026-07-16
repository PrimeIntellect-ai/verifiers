"""code_golf: write a short, fast Python program — the sibling-comparison recipe env.

Each task asks for a tiny program with a known output. The env fans one env-rollout
into `--env.attempts` independent attempts by the same "golfer" role and scores them
against each other. Anything that needs the runtime is measured per attempt, box-live,
into that attempt's trace; the env's `score()` then just compares the recorded
metadata across the finished siblings:

  - `evaluate`      per-attempt `@metric`: runs the program once in that attempt's
                    runtime and records `passed` + `latency`. (task, trace, runtime)
  - `correct`       per-attempt `@reward`: reads `passed` off the trace.      (trace)
  - `most_concise`  env `score()`: of the attempts, the shortest source wins.
  - `fastest`       env `score()`: of the attempts, the lowest recorded
                    `latency` wins — a comparison of trace metadata.

So one env-rollout produces, per attempt: did it work, was it the shorter one, was it
the quicker one — the relative signals you can only get by comparing siblings.
"""

import asyncio
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
        """Run the program once in the attempt's runtime; record correctness + latency."""
        program = extract_program(trace)
        if not program:
            return {"passed": 0.0, "latency": 1e6}
        await runtime.write("solution.py", program.encode())
        start = time.perf_counter()
        result = await runtime.run(["python3", "solution.py"], {})
        latency = time.perf_counter() - start
        passed = float(
            result.exit_code == 0 and result.stdout.strip() == self.data.expected
        )
        return {"passed": passed, "latency": latency}

    @vf.reward
    async def correct(self, trace: vf.Trace) -> float:
        return trace.metrics.get("passed", 0.0)


class CodeGolfParams(vf.EnvParams):
    golfer: vf.AgentConfig = vf.AgentConfig()
    attempts: int = 2
    """Independent attempts per env-rollout, scored against each other."""


class CodeGolfEnv(vf.Environment[CodeGolfParams]):
    def roles(self):
        return {"golfer": self.params.golfer}

    async def rollout(self, task, agents):
        return list(
            await asyncio.gather(
                *(agents["golfer"].run(task) for _ in range(self.params.attempts))
            )
        )

    async def score(self, task, traces):
        """The sibling comparison: shortest source and lowest latency win (ties share)."""
        lengths = [len(extract_program(t)) or 10**9 for t in traces]
        times = [t.metrics.get("latency", 1e6) for t in traces]
        for trace, length, latency in zip(traces, lengths, times):
            trace.record_reward("most_concise", float(length == min(lengths)), 0.5)
            trace.record_reward("fastest", float(latency == min(times)), 0.5)


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
