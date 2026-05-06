from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def load_bfcl_module() -> ModuleType:
    path = Path(__file__).parents[1] / "environments" / "bfcl_v3" / "bfcl_v3.py"
    spec = importlib.util.spec_from_file_location("bfcl_v3_test_module", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bfcl_prefers_hinted_function_schemas() -> None:
    bfcl = load_bfcl_module()
    task = {
        "function": [{"name": "plain"}],
        "function_with_hints": [{"name": "hinted"}],
    }

    assert bfcl.bfcl_functions(task) == [{"name": "hinted"}]


def test_bfcl_prefers_hinted_holdout_function_schemas() -> None:
    bfcl = load_bfcl_module()
    task = {
        "missed_function": {"1": [{"name": "plain"}]},
        "missed_function_with_hints": {"1": [{"name": "hinted"}]},
    }

    assert bfcl.bfcl_missed_function(task) == {"1": [{"name": "hinted"}]}


def test_bfcl_row_preserves_hinted_holdout_functions() -> None:
    bfcl = load_bfcl_module()
    entry = {
        "id": "case",
        "question": ["call tools"],
        "function": [{"name": "plain"}],
        "missed_function": {"1": [{"name": "plain_holdout"}]},
    }
    hinted_entry = {
        "function": [{"name": "hinted"}],
        "missed_function": {"1": [{"name": "hinted_holdout"}]},
    }

    row = bfcl.bfcl_row("multi_turn", entry, hinted_entry, None)

    assert row["function_with_hints"] == [{"name": "hinted"}]
    assert row["missed_function"] == {"1": [{"name": "plain_holdout"}]}
    assert row["missed_function_with_hints"] == {"1": [{"name": "hinted_holdout"}]}
