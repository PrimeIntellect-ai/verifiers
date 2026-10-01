"""Message-in training using provider-returned exact token arrays."""

import math
from collections.abc import Mapping

from pydantic import ValidationError

from verifiers.v1.clients.eval import EvalClient
from verifiers.v1.dialects import ChatDialect, Dialect
from verifiers.v1.errors import model_error
from verifiers.v1.graph import PendingTurn
from verifiers.v1.types import Response, SamplingConfig, SamplingMask, TurnTokens


def tokens_from_engine_data(raw: dict) -> TurnTokens:
    """Require engine-authoritative arrays; never reconstruct tokens from response text."""
    try:
        nvext = raw["nvext"]
        data = nvext["engine_data"]
        prompt = data["prompt_token_ids"]
        completion = data["completion_token_ids"]
        logprobs = data["completion_logprobs"]
        if not all(isinstance(ids, list) for ids in (prompt, completion, logprobs)):
            raise ValueError("token arrays must be lists")
        if any(type(token) is not int or token < 0 for token in (*prompt, *completion)):
            raise ValueError("token IDs must be nonnegative integers")
        if len(completion) != len(logprobs):
            raise ValueError("completion IDs and logprobs must align")
        if any(
            type(lp) not in (int, float) or not math.isfinite(lp) for lp in logprobs
        ):
            raise ValueError("completion logprobs must be finite numbers")
        mask = data.get("sampling_mask", nvext.get("sampling_mask"))
        if mask is not None and (
            not isinstance(mask, list)
            or len(mask) != len(completion)
            or any(
                not isinstance(row, list)
                or any(type(token) is not int or token < 0 for token in row)
                for row in mask
            )
        ):
            raise ValueError("sampling masks must align with completion IDs")
        routing = data.get("routed_experts", nvext.get("routed_experts"))
        tokens = TurnTokens(
            prompt_ids=prompt,
            completion_ids=completion,
            completion_logprobs=logprobs,
            routed_experts=routing,
            sampling_mask=SamplingMask.from_sampling_mask(mask)
            if mask is not None
            else None,
        )
        # The wire can carry dtype beyond RoutedExperts' historical uint8 fields.
        typed_routing = tokens.routed_experts
        if routing is not None and typed_routing is not None and "dtype" in routing:
            tokens = tokens.model_copy(
                update={
                    "routed_experts": {
                        **typed_routing,
                        "dtype": routing["dtype"],
                    }
                }
            )
        return tokens
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        raise model_error(
            "missing or invalid nvext.engine_data training tokens"
        ) from exc


class MITOTrainClient(EvalClient):
    """Reuse native chat HTTP plumbing, but require exact tokens for every training turn."""

    async def get_response(
        self,
        dialect: Dialect,
        body: dict,
        sampling: SamplingConfig,
        session_id: str | None = None,
        turn: PendingTurn | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Response:
        if not isinstance(dialect, ChatDialect):
            raise model_error(
                "MITO training only supports chat completions", status_code=400
            )
        if body.get("n", 1) != 1:
            raise model_error("MITO training requires n=1", status_code=400)
        nvext = dict(body.get("nvext") or {})
        extra_fields = list(nvext.get("extra_fields") or [])
        if "engine_data" not in extra_fields:
            extra_fields.append("engine_data")
        request = {
            **body,
            "stream": False,
            "logprobs": True,
            "nvext": {**nvext, "extra_fields": extra_fields},
        }
        response = await super().get_response(
            dialect,
            request,
            sampling,
            session_id=session_id,
            turn=turn,
            headers=headers,
        )
        raw = response.raw
        if raw is None or len(raw.get("choices", [])) != 1:
            raise model_error("MITO training requires one response choice")
        return response.model_copy(update={"tokens": tokens_from_engine_data(raw)})
