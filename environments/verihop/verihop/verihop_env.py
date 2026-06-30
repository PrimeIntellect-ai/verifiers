from __future__ import annotations

from typing import Any, Callable, cast

import verifiers as vf
from verifiers.types import AssistantMessage, Messages, State, UserMessage
from verifiers.utils.message_utils import maybe_normalize_messages

from .rubrics import VeriHopRubric
from .synthesizer import parse_hop_answer, synthesize
from .tools.visual_tools import make_visual_tools


def _text_from_assistant(msg: Any) -> str:
    if isinstance(msg, dict):
        if msg.get("role") != "assistant":
            return ""
        c = msg.get("content")
    else:
        if getattr(msg, "role", None) != "assistant":
            return ""
        c = getattr(msg, "content", None)
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        chunks: list[str] = []
        for block in c:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    chunks.append(str(block.get("text", "")))
            elif getattr(block, "type", None) == "text":
                chunks.append(str(getattr(block, "text", "")))
        return "\n".join(chunks)
    return ""


def _last_assistant_text(messages: Messages) -> str:
    for msg in reversed(messages):
        role = getattr(msg, "role", None) or (
            msg.get("role") if isinstance(msg, dict) else None
        )
        if role == "assistant":
            return _text_from_assistant(msg)
    return ""


def _advance_hop(messages: Messages, state: State) -> Messages:
    text = _last_assistant_text(messages)
    hop_val = parse_hop_answer(text) or ""
    cast(list[str], state["verihop_collected"]).append(hop_val)
    vh = cast(dict[str, Any], state["info"]["verihop"])
    hops = cast(list[dict[str, Any]], vh["hops"])
    idx = int(state["verihop_hop_idx"])
    if idx >= len(hops) - 1:
        state["final_env_response"] = []
        if "verihop_all_hops_done" in state:
            state["verihop_all_hops_done"] = True
        return []
    state["verihop_hop_idx"] = idx + 1
    nxt = hops[idx + 1]["question"]
    return maybe_normalize_messages(
        [UserMessage(role="user", content=nxt)], field_name="verihop_followup"
    )


class VeriHopEnv(vf.MultiTurnEnv):
    """
    Multi-hop visual QA: the first user turn includes the image; later hops arrive
    as plain user text. The model must use ``<hop_answer>...</hop_answer>`` each hop
    and ``\\boxed{}`` on the final hop.
    """

    def __init__(self, max_turns: int = 16, **kwargs: Any):
        super().__init__(max_turns=max_turns, **kwargs)

    async def setup_state(self, state: State) -> State:
        state = await super().setup_state(state)
        info = state.get("info")
        if not isinstance(info, dict) or "verihop" not in info:
            raise ValueError("Dataset row must include info['verihop'] from verihop.synthesize()")
        state["verihop_collected"] = []
        state["verihop_hop_idx"] = 0
        return state

    async def env_response(self, messages: Messages, state: State, **kwargs) -> Messages:
        return _advance_hop(messages, state)


class VeriHopToolEnv(vf.StatefulToolEnv):
    """
    Like ``VeriHopEnv`` but allows sandboxed PIL tools between hops. A hop advances
    when the assistant sends a message **without** tool calls.
    """

    def __init__(
        self,
        tools: list[Callable[..., Any]] | None = None,
        max_turns: int = 48,
        **kwargs: Any,
    ):
        super().__init__(tools=[], max_turns=max_turns, **kwargs)
        for t in tools or make_visual_tools():
            self.add_tool(t, args_to_skip=["_pil_image"])

    async def setup_state(self, state: State) -> State:
        state = await super().setup_state(state)
        info = state.get("info")
        if not isinstance(info, dict) or "verihop" not in info:
            raise ValueError("Dataset row must include info['verihop']")
        state["verihop_collected"] = []
        state["verihop_hop_idx"] = 0
        state["verihop_all_hops_done"] = False
        import base64
        from io import BytesIO

        from PIL import Image

        b64 = cast(dict[str, Any], state["info"]["verihop"]).get("image_b64")
        if isinstance(b64, str):
            state["verihop_pil_image"] = Image.open(BytesIO(base64.b64decode(b64)))
        else:
            state["verihop_pil_image"] = None
        return state

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict,
        messages: vf.Messages,
        state: vf.State,
        **kwargs,
    ) -> dict:
        out = dict(tool_args)
        img = state.get("verihop_pil_image")
        if img is not None:
            out["_pil_image"] = img
        return out

    async def env_response(self, messages: Messages, state: State, **kwargs) -> Messages:
        last = messages[-1]
        if not isinstance(last, AssistantMessage):
            return []
        if last.tool_calls:
            return await super().env_response(messages, state, **kwargs)
        return _advance_hop(messages, state)

    @vf.stop
    async def no_tools_called(self, state: State) -> bool:
        if len(state["trajectory"]) == 0:
            return False
        last = state["trajectory"][-1]["completion"][-1]
        if not isinstance(last, AssistantMessage):
            return False
        if last.tool_calls:
            return False
        return bool(state.get("verihop_all_hops_done", False))


def load_environment(
    num_samples: int = 5000,
    max_hops: int = 3,
    seed: int = 42,
    use_tools: bool = False,
    process_weight: float = 0.4,
    outcome_weight: float = 0.6,
    **kwargs: Any,
) -> vf.Environment:
    dataset = synthesize(
        num_samples=num_samples,
        max_hops=max_hops,
        min_hops=max_hops,
        seed=seed,
    )
    rubric = VeriHopRubric(
        process_weight=process_weight,
        outcome_weight=outcome_weight,
    )
    cls = VeriHopToolEnv if use_tools else VeriHopEnv
    return cls(
        dataset=dataset,
        rubric=rubric,
        parser=rubric.parser,
        max_turns=48 if use_tools else 16,
        env_id="verihop",
        **kwargs,
    )
