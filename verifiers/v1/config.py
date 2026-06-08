from collections.abc import Mapping
from typing import TypeAlias

from pydantic import BaseModel
from pydantic_config import BaseConfig

from .utils.config_utils import (
    import_config_ref as import_config_ref,
    resolve_config_object as resolve_config_object,
)


ConfigSource: TypeAlias = BaseModel | Mapping[str, object] | None


Config = BaseConfig
