from copy import deepcopy

from .artifact import ArtifactsConfig
from .model import model_config_data
from .sandbox import SandboxConfig
from .toolset import VisibilityConfig
from .utils.task_freeze_utils import assert_serializable, freeze_value
from .utils.prompt_utils import normalize_prompt, normalize_system_prompt
from .types import JsonData, JsonValue


class Task(dict):
    _vf_state_contract = "v1"

    def __init__(self, task: JsonData | None = None):
        super().__init__(deepcopy(dict(task or {})))
        self._frozen = False

    def freeze(self) -> "Task":
        if "runtime" in self:
            raise TypeError(
                "task.runtime is not supported; use top-level task fields or state.runtime."
            )
        if "prompt" in self:
            super().__setitem__(
                "prompt", normalize_prompt(self["prompt"], field_name="task.prompt")
            )
        if "system_prompt" in self:
            super().__setitem__(
                "system_prompt",
                normalize_system_prompt(
                    self["system_prompt"], field_name="task.system_prompt"
                ),
            )
        if "tools" in self:
            super().__setitem__(
                "tools",
                {
                    name: config.model_dump(mode="json", exclude_none=True)
                    for name, config in self.tools_config().items()
                },
            )
        if "toolsets" in self:
            super().__setitem__(
                "toolsets",
                self.toolsets_config().model_dump(mode="json", exclude_none=True),
            )
        sandbox_config = self.sandbox_config()
        if sandbox_config is not None:
            super().__setitem__(
                "sandbox",
                sandbox_config.data(fill_defaults=False),
            )
        if "program" in self and not isinstance(self["program"], dict):
            raise TypeError("task.program must be a mapping.")
        if "artifacts" in self:
            super().__setitem__(
                "artifacts",
                {
                    name: artifact.data()
                    for name, artifact in self.artifacts_config()
                    .artifacts("task.artifacts")
                    .items()
                },
            )
        if "model" in self:
            super().__setitem__("model", model_config_data(self["model"]))
        if "max_turns" in self and (
            not isinstance(self["max_turns"], int)
            or isinstance(self["max_turns"], bool)
        ):
            raise TypeError("task.max_turns must be an integer.")
        for key, value in list(self.items()):
            super().__setitem__(key, freeze_value(value))
        assert_serializable(self)
        self._frozen = True
        return self

    @property
    def frozen(self) -> bool:
        return self._frozen

    def toolsets_config(self) -> VisibilityConfig:
        raw_toolsets = self.get("toolsets") or {}
        if not isinstance(raw_toolsets, dict):
            raise TypeError("task.toolsets must be a mapping.")
        return VisibilityConfig.model_validate(raw_toolsets)

    def tools_config(self) -> dict[str, VisibilityConfig]:
        raw_tools = self.get("tools") or {}
        if not isinstance(raw_tools, dict):
            raise TypeError("task.tools must be a toolset-keyed mapping.")
        if "show" in raw_tools or "hide" in raw_tools:
            raise ValueError("task.tools must be keyed by toolset name.")
        configs: dict[str, VisibilityConfig] = {}
        for name, raw_filter in raw_tools.items():
            if not isinstance(name, str):
                raise TypeError("task.tools keys must be toolset names.")
            configs[name] = VisibilityConfig.model_validate(raw_filter)
        return configs

    def sandbox_config(self) -> SandboxConfig | None:
        raw_sandbox = self.get("sandbox")
        if raw_sandbox is None:
            return None
        if not isinstance(raw_sandbox, dict):
            raise TypeError("task.sandbox must be a mapping.")
        return SandboxConfig.model_validate(raw_sandbox)

    def artifacts_config(self) -> ArtifactsConfig:
        raw_artifacts = self.get("artifacts") or {}
        if not isinstance(raw_artifacts, dict):
            raise TypeError("task.artifacts must be a mapping.")
        return ArtifactsConfig.model_validate(raw_artifacts)

    def __setitem__(self, key: str, value: object) -> None:
        self._raise_if_frozen()
        super().__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        self._raise_if_frozen()
        super().__delitem__(key)

    def update(self, *args: object, **kwargs: object) -> None:
        self._raise_if_frozen()
        super().update(*args, **kwargs)

    def setdefault(self, key: str, default: object = None) -> object:
        self._raise_if_frozen()
        return super().setdefault(key, default)

    def pop(self, key: str, default: object = None) -> object:
        self._raise_if_frozen()
        return super().pop(key, default)

    def popitem(self) -> tuple[str, JsonValue]:
        raise TypeError("Task.popitem() is not supported.")

    def clear(self) -> None:
        self._raise_if_frozen()
        super().clear()

    def __ior__(self, value: object, /) -> "Task":
        self._raise_if_frozen()
        self.update(value)
        return self

    def _raise_if_frozen(self) -> None:
        if self._frozen:
            raise TypeError("Task is immutable after freeze.")
