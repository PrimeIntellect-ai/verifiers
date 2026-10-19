"""Resolve process environment templates without persisting host values."""

import os
import re
from collections.abc import Mapping

_TEMPLATE = re.compile(r"\$\{([^}:]+)(?::-(.*))?\}")


def resolve_env(values: Mapping[str, str]) -> dict[str, str]:
    """Expand whole-value ${VAR} or ${VAR:-default} templates, as Harbor does.

    Defaults apply to absent host variables; an explicitly empty value stays empty.
    Other strings are literals. Call only when binding a process environment.
    """
    resolved = {}
    for key, value in values.items():
        match = _TEMPLATE.fullmatch(value)
        if match is None:
            resolved[key] = value
            continue
        name, default = match.groups()
        if name in os.environ:
            resolved[key] = os.environ[name]
        elif default is not None:
            resolved[key] = default
        else:
            raise ValueError(
                f"Environment variable '{name}' not found in host environment"
            )
    return resolved
