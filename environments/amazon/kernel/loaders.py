"""YAML loaders for dependency-graph contract/spec files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .types import (
    GraphContractSpec,
    SamplingRequestDoc,
    TaskSpecsDoc,
)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    data = yaml.safe_load(p.read_text())
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML object at root for {p}")
    return data


def load_graph_contract(path: str | Path) -> GraphContractSpec:
    """Load and validate a graph contract YAML file."""
    return GraphContractSpec.model_validate(_load_yaml(path))


def load_task_specs(path: str | Path) -> TaskSpecsDoc:
    """Load and validate task specs YAML file."""
    return TaskSpecsDoc.model_validate(_load_yaml(path))


def load_sampling_request(path: str | Path) -> SamplingRequestDoc:
    """Load and validate sampling request YAML file."""
    return SamplingRequestDoc.model_validate(_load_yaml(path))
