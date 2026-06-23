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

import ast

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


def _run_base_body(source: str) -> str:
    marker = "async def run_base("
    start = source.index(marker)
    rest = source[start + len(marker) :]
    next_def = rest.find("\nasync def ")
    alt_def = rest.find("\ndef ")
    candidates = [i for i in (next_def, alt_def) if i != -1]
    end = min(candidates) if candidates else len(rest)
    return rest[:end]


def test_runner_source_is_valid_python():
    # The runner program is shipped as a source string executed in-sandbox; a
    # syntax error would only surface at rollout time, so compile it here.
    ast.parse(spu.runner_source())


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


def test_run_base_enforces_invalid_tool_caps_in_sandbox():
    # The host @vf.stop reads host state, but in-sandbox tool events only reach the
    # host at end-of-rollout, so the cap MUST be enforced inside run_base off the
    # sandbox-local browser_tool_results. Assert the loop sets the stop condition
    # and breaks on the cap, both as a top-of-loop guard and right after the
    # per-tool dispatch (before the existing check_stop backstop).
    source = spu.runner_source()
    body = _run_base_body(source)

    # The cap is read from the forwarded runner config.
    assert 'config.get("invalid_tool_caps")' in body

    # The cap check sets the stop condition and breaks.
    assert 'set_stop_condition(state, "invalid_tool_call_cap")' in body
    cap_call = "invalid_tool_caps_triggered(state, invalid_tool_caps)"
    assert body.count(cap_call) >= 2, (
        "expected an in-sandbox cap check both at the top of the loop and after "
        "the per-tool dispatch"
    )

    # The post-dispatch cap check must precede the existing check_stop backstop so
    # the loop is cut at the cap, not one turn later.
    dispatch_marker = '"tool_call_id": tool_call["id"],'
    dispatch_idx = body.index(dispatch_marker)
    tail = body[dispatch_idx:]
    cap_idx = tail.index(cap_call)
    check_stop_idx = tail.index("if await check_stop(state):")
    assert cap_idx < check_stop_idx, (
        "the in-sandbox cap check must run before the check_stop backstop after "
        "appending the tool message"
    )


def test_invalid_tool_caps_logic_resets_on_valid_call():
    # Execute the runner's cap helpers in isolation to prove the consecutive-run
    # logic: it triggers on the cap, resets on a valid in-scope call, and ignores
    # out-of-scope events. The helpers are pure, so exec the source and call them.
    source = spu.runner_source()
    # The runner program ends with ``asyncio.run(main())``; exec only the pure cap
    # helpers (they depend on nothing but builtins).
    helpers_start = source.index("def consecutive_tag_cap_triggered(")
    helpers_end = source.index("async def run_base(")
    namespace: dict[str, object] = {}
    exec(  # noqa: S102
        compile(source[helpers_start:helpers_end], "<runner_helpers>", "exec"),
        namespace,
    )
    triggered = namespace["invalid_tool_caps_triggered"]
    caps = {"malformed_computer": 1, "invalid_tool": 2}

    # Rule 1: a single malformed computer call trips the K=1 cap.
    assert triggered(
        {"browser_tool_results": [{"action": "computer", "malformed_computer": True}]},
        caps,
    )
    # A valid computer call resets the malformed-computer run (shown with K=2 so the
    # reset is observable; with K=1 the first malformed call already trips it).
    caps_k2 = {"malformed_computer": 2, "invalid_tool": 2}
    assert not triggered(
        {
            "browser_tool_results": [
                {"action": "computer", "malformed_computer": True},
                {"action": "computer"},
                {"action": "computer", "malformed_computer": True},
            ]
        },
        caps_k2,
    )
    # Two consecutive malformed computer calls trip the K=2 cap.
    assert triggered(
        {
            "browser_tool_results": [
                {"action": "computer", "malformed_computer": True},
                {"action": "computer", "malformed_computer": True},
            ]
        },
        caps_k2,
    )
    # A non-computer event between malformed computer calls is ignored (does not
    # reset the computer-scoped run) -> still trips K=2.
    assert triggered(
        {
            "browser_tool_results": [
                {"action": "computer", "malformed_computer": True},
                {"action": "click"},
                {"action": "computer", "malformed_computer": True},
            ]
        },
        caps_k2,
    )
    # Rule 2: two consecutive any-tool invalids trip the K=2 cap (incl form_input).
    assert triggered(
        {
            "browser_tool_results": [
                {"action": "form_input", "invalid_tool_call": True},
                {"action": "click", "invalid_tool_call": True},
            ]
        },
        caps,
    )
    # One invalid then a valid call resets -> no stop.
    assert not triggered(
        {
            "browser_tool_results": [
                {"action": "form_input", "invalid_tool_call": True},
                {"action": "click"},
            ]
        },
        caps,
    )
    # A runtime/execution error (no invalid tag) does NOT count toward Rule 2.
    assert not triggered(
        {
            "browser_tool_results": [
                {"action": "click", "ok": False},
                {"action": "click", "ok": False},
            ]
        },
        caps,
    )
    # Caps of 0 disable enforcement.
    assert not triggered(
        {"browser_tool_results": [{"action": "computer", "malformed_computer": True}]},
        {"malformed_computer": 0, "invalid_tool": 0},
    )
