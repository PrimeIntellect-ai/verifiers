"""echo: ask the model to repeat a short phrase back verbatim (single-turn).

The smallest possible reward-1 taskset — no dataset, no tools, no reasoning required — so an
end-to-end eval run is fast and deterministic. The reward is 1.0 when the phrase appears in
the model's reply. It's a fixture taskset for the v1 e2e suite (in tests/v1/fixtures, resolved
by id `echo-v1` via pytest's `pythonpath`).
"""

import verifiers.v1 as vf

SYSTEM = "Repeat the user's message back to them exactly, with no extra words."


def _key(text: str) -> str:
    """Lenient comparison key: lowercase, alphanumerics only."""
    return "".join(c for c in text.casefold() if c.isalnum())


def lenient_match(answer: str, text: str) -> bool:
    """True if `answer` appears in `text` ignoring case, spacing, and punctuation — so a
    reformatted echo ("Hello, World!") still counts."""
    return _key(answer) in _key(text)


class EchoTask(vf.Task):
    answer: str
    """The phrase the model should echo back."""


class EchoConfig(vf.TasksetConfig):
    phrases: list[str] = ["hello world", "ping", "verifiers"]


class EchoTaskset(vf.Taskset[EchoTask, EchoConfig]):
    def load_tasks(self) -> list[EchoTask]:
        return [
            # Keep coding-agent harnesses on the direct-response path instead of
            # spending this single-turn smoke task on a tool call.
            EchoTask(
                idx=i,
                prompt=(
                    "Do not call tools or execute code. Reply immediately and include "
                    f"this exact phrase in your final response: {phrase}"
                ),
                system_prompt=SYSTEM,
                answer=phrase,
            )
            for i, phrase in enumerate(self.config.phrases)
        ]

    @vf.stop
    async def single_turn(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= 1

    @vf.reward(weight=1.0)
    async def echoed(self, task: EchoTask, trace: vf.Trace) -> float:
        reply = trace.assistant_messages[-1].content if trace.assistant_messages else ""
        return float(lenient_match(task.answer, reply or ""))


__all__ = ["EchoTaskset"]
