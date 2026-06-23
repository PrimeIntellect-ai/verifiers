"""Regression: the in-sandbox runner program (``runner_source()``) must apply
verifier-facing ``state_appends`` BEFORE raising on a failed (``ok: False``)
tool result inside ``call_sandbox_local_tool``.

``call_sandbox_local_tool`` lives inside the ``runner_source()`` program string
(executed inside the per-rollout sandbox, not importable here), so this guards
the ordering at the source level. A malformed/invalid tool call returns
``ok: False`` plus a tagged ``state_appends.browser_tool_results`` event; if the
program raised before applying the appends, the cross-turn ``@vf.stop`` caps
(malformed-computer / invalid-tool consecutive caps) would never see the event
and never fire — the exact bug this guards against.
"""

from verifiers.v1.utils import sandbox_program_utils as spu


def _call_sandbox_local_tool_body(source: str) -> str:
    marker = "async def call_sandbox_local_tool("
    start = source.index(marker)
    # Slice to the next top-level ``async def``/``def`` after the function start.
    rest = source[start + len(marker) :]
    next_def = rest.find("\nasync def ")
    alt_def = rest.find("\ndef ")
    candidates = [i for i in (next_def, alt_def) if i != -1]
    end = min(candidates) if candidates else len(rest)
    return rest[:end]


def test_state_appends_applied_before_failure_raise_in_runner_source():
    source = spu.runner_source()
    assert "async def call_sandbox_local_tool(" in source

    body = _call_sandbox_local_tool_body(source)
    appends_idx = body.index('appends = payload.get("state_appends")')
    raise_idx = body.index('payload.get("ok") is False')

    assert appends_idx < raise_idx, (
        "call_sandbox_local_tool must apply state_appends BEFORE raising on "
        "ok is False, so failed/invalid tool events still reach rollout state "
        "for the @vf.stop consecutive caps."
    )
