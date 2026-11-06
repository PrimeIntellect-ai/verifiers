"""echo: ask the model to repeat a short phrase back verbatim (single-turn).

The smallest possible reward-1 taskset — no dataset, no tools, no reasoning required — so an
end-to-end eval run is fast and deterministic. The reward is 1.0 when the phrase appears in
the model's reply. It's a fixture taskset for the integration suite (in tests/integration/fixtures,
resolved by id `echo` via pytest's `pythonpath`).
"""

from pydantic import Field

import verifiers.v1 as vf


def _key(text: str) -> str:
    """Lenient comparison key: lowercase, alphanumerics only."""
    return "".join(c for c in text.casefold() if c.isalnum())


def lenient_match(answer: str, text: str) -> bool:
    """True if `answer` appears in `text` ignoring case, spacing, and punctuation — so a
    reformatted echo ("Hello, World!") still counts."""
    return _key(answer) in _key(text)


class EchoData(vf.TaskData):
    answer: str
    """The phrase the model should echo back."""
    finish_tool: bool = False


class EchoTask(vf.Task[EchoData]):
    @property
    def key(self) -> str:
        return f"echo:{self.data.answer}"

    @vf.stop
    async def single_turn(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= 1

    @vf.reward(weight=1.0)
    async def echoed(self, trace: vf.Trace) -> float:
        reply = trace.assistant_messages[-1].content if trace.assistant_messages else ""
        if self.data.finish_tool:
            reply = trace.last_reply
        return float(lenient_match(self.data.answer, reply or ""))


class EchoConfig(vf.TasksetConfig):
    finish_tool: bool = False
    """Request the answer through finish for harnesses that require it."""
    phrases: list[str] = Field(
        default_factory=lambda: ["hello world", "ping", "verifiers"]
    )


class EchoTaskset(vf.Taskset[EchoTask, EchoConfig]):
    def load(self) -> list[EchoTask]:
        return [
            # Default to a direct response; finish_tool requests native completion
            # from harnesses that require a finish call. Phrased as an
            # instruction rather than a prohibition: OpenAI's content filter refuses
            # "Do not call tools ... include this exact phrase" as an injection.
            EchoTask(
                EchoData(
                    idx=i,
                    prompt=(
                        f"Call your finish tool with reason exactly {phrase!r} and paths=[]."
                        if self.config.finish_tool
                        else "Answer directly without using tools or running code. Your reply "
                        f"must contain this exact phrase: {phrase}"
                    ),
                    answer=phrase,
                    finish_tool=self.config.finish_tool,
                ),
                self.config.task,
            )
            for i, phrase in enumerate(self.config.phrases)
        ]


__all__ = ["EchoTaskset"]
