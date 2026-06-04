"""Internal harness utilities."""


def split_versioned_agent_spec(spec: str) -> tuple[str, str | None]:
    """Split an agent install spec written as name[@version]."""
    spec = spec.strip()
    name, _, version = spec.rpartition("@")
    if not name:
        return spec, None
    return name, version or None
