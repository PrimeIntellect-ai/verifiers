from collections.abc import Mapping


def serializable(value: object) -> object:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(exclude_none=True)
    if isinstance(value, list):
        return [serializable(item) for item in value]
    if isinstance(value, tuple):
        return [serializable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): serializable(item) for key, item in value.items()}
    return value
