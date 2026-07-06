"""The compacting harness: the default harness (chat loop + bash/edit/search/MCP tools) plus
rlm-style automatic context compaction.

`DefaultHarness` owns the whole chat-loop program; this subclass only turns on the program's
compaction path by appending `--compact-at-tokens` (and, when configured, the two prompt
overrides) to the program argv. Compaction itself lives in the program: when a turn's reported
`usage.prompt_tokens` reaches the threshold, it asks the model for a handoff summary with the
full conversation still in context — the summary turn is the last turn of the old context —
then rebuilds its messages as `[system, user(framing + summary)]`. The rewrite forks the trace
into a new branch (see `verifiers.v1.graph`), which is the boundary context-training (TTT)
techniques key on.
"""

from pydantic import model_validator

from verifiers.v1.harness import Harness
from verifiers.v1.harnesses.default.harness import DefaultHarness, DefaultHarnessConfig


class CompactingHarnessConfig(DefaultHarnessConfig):
    """The default harness's knobs plus the compaction threshold and prompt overrides."""

    compact_at_tokens: int
    """Compact once a turn's prompt token count reaches this many tokens. The check runs after
    each turn's tool results are appended (so they're summarized too) and fires at most once
    per loop turn; the turn that crosses the threshold completes first, so the cap is soft by
    one turn."""

    checkpoint_prompt: str | None = None
    """Override the user message that requests the handoff summary. None uses the program's
    built-in prompt (mirroring rlm's CHECKPOINT_COMPACTION_PROMPT)."""

    compaction_framing: str | None = None
    """Override the framing text that wraps the summary in the post-compaction user message.
    None uses the program's built-in framing (mirroring rlm's POST_COMPACTION_FRAMING)."""

    @model_validator(mode="after")
    def validate_threshold(self) -> "CompactingHarnessConfig":
        if self.compact_at_tokens <= 0:
            raise ValueError("`compact_at_tokens` must be positive.")
        return self


class CompactingHarness(DefaultHarness, Harness[CompactingHarnessConfig]):
    config: CompactingHarnessConfig

    def extra_program_args(self) -> list[str]:
        args = [f"--compact-at-tokens={self.config.compact_at_tokens}"]
        if self.config.checkpoint_prompt is not None:
            args.append(f"--checkpoint-prompt={self.config.checkpoint_prompt}")
        if self.config.compaction_framing is not None:
            args.append(f"--compaction-framing={self.config.compaction_framing}")
        return args
