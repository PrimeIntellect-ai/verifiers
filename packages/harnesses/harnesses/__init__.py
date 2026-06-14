"""Built-in harnesses, resolved by id (`--harness.id <id>`) as `harnesses.<id>`.

Re-exports each harness's class + config off the package."""

from harnesses.claude_code import ClaudeCodeHarness, ClaudeCodeHarnessConfig
from harnesses.codex import CodexHarness, CodexHarnessConfig
from harnesses.default import DefaultHarness, DefaultHarnessConfig
from harnesses.rlm import RLMHarness, RLMHarnessConfig

__all__ = [
    "ClaudeCodeHarness",
    "ClaudeCodeHarnessConfig",
    "CodexHarness",
    "CodexHarnessConfig",
    "DefaultHarness",
    "DefaultHarnessConfig",
    "RLMHarness",
    "RLMHarnessConfig",
]
