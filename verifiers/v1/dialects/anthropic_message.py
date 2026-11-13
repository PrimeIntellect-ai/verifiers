"""The Anthropic SDK's `Message`, widened for Anthropic-compatible gateways. Kept apart from
`dialects.anthropic` so the SDK, which is slow to import, loads only when a response is parsed."""

from anthropic.types import Message, Usage


class ModdedUsage(Usage):
    """The SDK closes `service_tier` to a fixed Literal, but Anthropic-compatible gateways
    report their own tiers (e.g. Prime's `provisioned`). Widen to a plain string — we don't
    consume it — so parsing stays lenient about the label instead of dropping it."""

    service_tier: str | None = None  # type: ignore[assignment]


class ModdedAnthropicMessage(Message):
    usage: ModdedUsage  # type: ignore[assignment]
