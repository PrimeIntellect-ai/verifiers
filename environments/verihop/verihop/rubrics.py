from __future__ import annotations

import re
from typing import Any

import verifiers as vf
from verifiers.utils.data_utils import extract_boxed_answer

from .synthesizer import _iou, parse_grounding_boxes


def _assistant_segments(completion: vf.Messages) -> list[str]:
    parts: list[str] = []
    for msg in completion:
        role = getattr(msg, "role", None)
        if role is None and isinstance(msg, dict):
            role = msg.get("role")
        if role != "assistant":
            continue
        c = getattr(msg, "content", None) if not isinstance(msg, dict) else msg.get("content")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            for block in c:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(str(block.get("text", "")))
                elif getattr(block, "type", None) == "text":
                    parts.append(str(getattr(block, "text", "")))
    return parts


def _norm_num(s: str) -> str:
    m = re.search(r"-?\d+", s)
    return m.group(0) if m else s.strip().lower()


class VeriHopRubric(vf.Rubric):
    """
    Combines a strict final outcome (boxed integer) with optional dense per-hop
    scoring using ``state['verihop_collected']`` and grounding IoU when metadata
    provides normalized boxes.
    """

    def __init__(
        self,
        process_weight: float = 0.4,
        outcome_weight: float = 0.6,
        parser: vf.Parser | None = None,
        min_grounding_iou: float = 0.15,
    ):
        self.process_weight = process_weight
        self.outcome_weight = outcome_weight
        self.min_grounding_iou = min_grounding_iou
        super().__init__(
            funcs=[self.outcome_reward, self.process_reward],
            weights=[outcome_weight, process_weight],
            parser=parser or vf.Parser(),
        )

    async def outcome_reward(self, completion: vf.Messages, answer: str, **kwargs) -> float:
        segs = _assistant_segments(completion)
        if not segs:
            return 0.0
        last = segs[-1]
        got = extract_boxed_answer(last) or ""
        return 1.0 if _norm_num(got) == _norm_num(str(answer)) else 0.0

    async def process_reward(
        self,
        completion: vf.Messages,
        info: dict[str, Any],
        state: vf.State,
        **kwargs,
    ) -> float:
        vh = info.get("verihop") or {}
        hops = vh.get("hops") or []
        collected = state.get("verihop_collected")
        if not hops or not collected:
            return 0.0
        n = min(len(hops), len(collected))
        if n == 0:
            return 0.0
        segs = _assistant_segments(completion)
        full_text = "\n".join(segs)
        scores: list[float] = []
        align_grounding = len(segs) == len(collected)
        for i in range(n):
            gt = _norm_num(hops[i]["ground_truth"])
            pred_raw = collected[i]
            pred = _norm_num(pred_raw or "")
            hop_score = 1.0 if pred == gt else 0.0
            gmeta = hops[i].get("grounding_norm")
            if gmeta is not None:
                text_scope = segs[i] if align_grounding else full_text
                boxes = parse_grounding_boxes(text_scope)
                geo = 0.0
                if boxes:
                    best = max(_iou(b, gmeta) for b in boxes)
                    geo = 1.0 if best >= self.min_grounding_iou else best / max(
                        self.min_grounding_iou, 1e-6
                    )
                hop_score = 0.5 * hop_score + 0.5 * geo
            scores.append(hop_score)
        return sum(scores) / len(scores)
