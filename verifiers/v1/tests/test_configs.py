"""Tests for the typed v1 callable-config layer.

These exercise:
- ``ImportRef`` validation at config-construction time.
- ``model_validator`` coercion of bare strings into ``{"fn": ...}``.
- TasksetConfig / HarnessConfig list field types (the previously-untyped
  ``list[object]`` for rewards/metrics/etc. is now ``list[RewardConfig]`` /
  ``list[SignalConfig]`` / ``list[CallableConfig]``).
- The metadata-only fields (``priority``, ``stage``, ``weight``) round-trip
  through the config → wrapper boundary.
"""

from __future__ import annotations

import pytest

from verifiers.v1.config import (
    CallableConfig,
    HarnessConfig,
    ImportRef,  # noqa: F401  (imported for visibility check)
    RewardConfig,
    SignalConfig,
    TasksetConfig,
    callable_config_item,
)


# --------------------------------------------------------------------------- #
# ImportRef validation.
# --------------------------------------------------------------------------- #


def test_callable_spec_accepts_module_object_ref() -> None:
    spec = CallableConfig(fn="pkg.mod:obj")
    assert spec.fn == "pkg.mod:obj"


def test_callable_spec_rejects_non_ref_string() -> None:
    with pytest.raises(ValueError, match="must use 'module:object'"):
        CallableConfig(fn="just-a-name")


def test_callable_spec_rejects_missing_module() -> None:
    with pytest.raises(ValueError, match="must use 'module:object'"):
        CallableConfig(fn=":obj")


def test_callable_spec_rejects_missing_attr() -> None:
    with pytest.raises(ValueError, match="must use 'module:object'"):
        CallableConfig(fn="pkg.mod:")


# --------------------------------------------------------------------------- #
# Bare-string coercion via @model_validator(mode="before").
# --------------------------------------------------------------------------- #


def test_callable_spec_coerces_bare_string_to_fn() -> None:
    spec = CallableConfig.model_validate("pkg.mod:obj")
    assert spec.fn == "pkg.mod:obj"
    assert spec.priority == 0


def test_reward_spec_coerces_bare_string_with_defaults() -> None:
    spec = RewardConfig.model_validate("pkg.mod:obj")
    assert spec.fn == "pkg.mod:obj"
    assert spec.weight == 1.0
    assert spec.stage == "rollout"


def test_callable_spec_passes_through_existing_instance() -> None:
    original = CallableConfig(fn="pkg.mod:obj", priority=5)
    coerced = CallableConfig.model_validate(original)
    assert coerced is original


# --------------------------------------------------------------------------- #
# Subclass hierarchy: SignalConfig adds `stage`, RewardConfig adds `weight`.
# --------------------------------------------------------------------------- #


def test_signal_spec_has_stage() -> None:
    spec = SignalConfig(fn="pkg.mod:obj", stage="group")
    assert spec.stage == "group"
    assert spec.priority == 0


def test_signal_spec_rejects_unknown_stage() -> None:
    with pytest.raises(ValueError):
        SignalConfig(fn="pkg.mod:obj", stage="never")  # type: ignore[arg-type]


def test_reward_spec_weight_validates() -> None:
    spec = RewardConfig(fn="pkg.mod:obj", weight=0.5)
    assert spec.weight == 0.5


def test_callable_spec_rejects_unknown_field() -> None:
    # extra="forbid" inherited from Config base.
    with pytest.raises(ValueError):
        CallableConfig(fn="pkg.mod:obj", weight=1.0)  # type: ignore[call-arg]


# --------------------------------------------------------------------------- #
# TasksetConfig / HarnessConfig list field typings.
# --------------------------------------------------------------------------- #


def test_taskset_config_rewards_typed_as_reward_spec_list() -> None:
    cfg = TasksetConfig(rewards=[{"fn": "pkg.mod:obj", "weight": 0.5}])
    assert isinstance(cfg.rewards[0], RewardConfig)
    assert cfg.rewards[0].weight == 0.5


def test_taskset_config_metrics_typed_as_signal_spec_list() -> None:
    cfg = TasksetConfig(metrics=[{"fn": "pkg.mod:obj", "stage": "group"}])
    assert isinstance(cfg.metrics[0], SignalConfig)
    assert cfg.metrics[0].stage == "group"


def test_taskset_config_setups_typed_as_callable_spec_list() -> None:
    cfg = TasksetConfig(setups=[{"fn": "pkg.mod:obj", "priority": 10}])
    assert isinstance(cfg.setups[0], CallableConfig)
    assert cfg.setups[0].priority == 10


def test_taskset_config_accepts_bare_string_in_lists() -> None:
    cfg = TasksetConfig(rewards=["pkg.mod:obj"], metrics=["pkg.mod:obj"])
    assert cfg.rewards[0].fn == "pkg.mod:obj"
    assert cfg.metrics[0].fn == "pkg.mod:obj"


def test_harness_config_rewards_typed_as_reward_spec_list() -> None:
    cfg = HarnessConfig(rewards=[{"fn": "pkg.mod:obj", "weight": 0.5}])
    assert isinstance(cfg.rewards[0], RewardConfig)
    assert cfg.rewards[0].weight == 0.5


def test_harness_config_setups_typed_as_callable_spec_list() -> None:
    cfg = HarnessConfig(setups=[{"fn": "pkg.mod:obj"}])
    assert isinstance(cfg.setups[0], CallableConfig)


# --------------------------------------------------------------------------- #
# Spec -> callable resolution.
# --------------------------------------------------------------------------- #


def _example_reward() -> float:
    return 0.5


def test_callable_config_item_resolves_spec_to_callable() -> None:
    spec = RewardConfig(fn=f"{__name__}:_example_reward", weight=0.7, priority=5)
    fn = callable_config_item(spec, "reward")
    assert getattr(fn, "reward") is True
    assert getattr(fn, "reward_weight") == 0.7
    assert getattr(fn, "reward_priority") == 5


def test_callable_config_item_preserves_identity_for_default_spec() -> None:
    spec = CallableConfig(fn=f"{__name__}:_example_reward")
    fn = callable_config_item(spec, "setup")
    # No non-default metadata → unwrapped callable returned as-is.
    assert fn is _example_reward


def test_callable_config_item_wraps_when_priority_explicit() -> None:
    spec = CallableConfig(fn=f"{__name__}:_example_reward", priority=10)
    fn = callable_config_item(spec, "setup")
    assert fn is not _example_reward
    assert getattr(fn, "setup_priority") == 10


# --------------------------------------------------------------------------- #
# CallableConfig.import_fn().
# --------------------------------------------------------------------------- #


def test_import_fn_resolves_ref_to_callable() -> None:
    spec = CallableConfig(fn=f"{__name__}:_example_reward")
    assert spec.import_fn() is _example_reward


def test_import_fn_raises_when_module_missing() -> None:
    spec = CallableConfig(fn="not_a_real_module_xyz:obj")
    with pytest.raises(ModuleNotFoundError):
        spec.import_fn()


def test_import_fn_raises_when_attr_missing() -> None:
    spec = CallableConfig(fn=f"{__name__}:nonexistent_attr")
    with pytest.raises(AttributeError):
        spec.import_fn()


_not_callable = 42


def test_import_fn_raises_when_target_not_callable() -> None:
    spec = CallableConfig(fn=f"{__name__}:_not_callable")
    with pytest.raises(TypeError, match="expected a callable"):
        spec.import_fn()
