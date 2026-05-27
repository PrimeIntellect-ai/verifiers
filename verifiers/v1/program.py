from typing import cast

from .config import ConfigSource, ProgramConfig
from .types import ConfigData
from .utils.config_utils import coerce_config, explicit_config_data

PROGRAM_DEFAULT_DUMP_DATA = ProgramConfig().model_dump(exclude_none=True)
PROGRAM_DEFAULT_DUMP_KEYS = set(PROGRAM_DEFAULT_DUMP_DATA)


def program_config_data(config: ProgramConfig) -> ConfigData:
    data = explicit_config_data(config)
    if PROGRAM_DEFAULT_DUMP_KEYS.issubset(config.model_fields_set):
        data = {
            key: value
            for key, value in data.items()
            if value != PROGRAM_DEFAULT_DUMP_DATA.get(key)
        }
    return data


class Program:
    config: ProgramConfig

    def __init__(self, config: ConfigSource = None):
        self.config = cast(ProgramConfig, coerce_config(ProgramConfig, config))

    def data(self) -> ConfigData:
        data = program_config_data(self.config)
        if data:
            return data
        return {
            "base": True,
        }
