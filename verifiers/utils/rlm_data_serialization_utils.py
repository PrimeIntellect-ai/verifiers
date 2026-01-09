from __future__ import annotations

import base64
import inspect
import json
import math
import textwrap
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class SerializedData:
    dtype: str
    inline_data: Any | None
    file_bytes: bytes | None
    file_name: str | None
    metadata: dict[str, Any]
    format: str | None = None
    encoding: str | None = None
    deserializer_code: str | None = None
    deserializer_function: str | None = None


@dataclass(frozen=True)
class DataSerializer:
    dtype: str
    serialize: Callable[[Any], SerializedData]
    can_handle: Callable[[Any], bool] | None = None
    deserializer_code: str | None = None
    deserializer_function: str | None = None


@dataclass(frozen=True)
class PreparedContextData:
    context_dict: dict[str, Any]
    payload_bytes: bytes | None
    payload_path: str | None
    payload_name: str | None


class SerializerRegistry:
    def __init__(self, serializers: list[DataSerializer] | None = None) -> None:
        self.serializers_by_dtype: dict[str, DataSerializer] = {}
        self.serializer_order: list[DataSerializer] = []
        if serializers:
            for serializer in serializers:
                self.register(serializer)

    def register(
        self, serializer: DataSerializer, allow_override: bool = False
    ) -> None:
        existing = self.serializers_by_dtype.get(serializer.dtype)
        if existing and not allow_override:
            raise ValueError(
                f"Serializer for dtype '{serializer.dtype}' already registered. "
                "Use allow_override=True to replace it."
            )
        if existing:
            for index, item in enumerate(self.serializer_order):
                if item.dtype == serializer.dtype:
                    self.serializer_order[index] = serializer
                    break
        else:
            self.serializer_order.append(serializer)
        self.serializers_by_dtype[serializer.dtype] = serializer

    def get(self, dtype: str) -> DataSerializer:
        serializer = self.serializers_by_dtype.get(dtype)
        if serializer is None:
            supported = ", ".join(sorted(self.serializers_by_dtype))
            raise ValueError(
                f"Unsupported dtype '{dtype}'. Supported dtypes: {supported}."
            )
        return serializer

    def list_dtypes(self) -> list[str]:
        return list(self.serializers_by_dtype.keys())

    def all(self) -> list[DataSerializer]:
        return list(self.serializer_order)

    def resolve(self, data: Any, dtype: str | None) -> DataSerializer:
        if dtype:
            return self.get(dtype)

        if isinstance(data, str):
            serializer = self.serializers_by_dtype.get("text")
            if serializer is not None:
                return serializer

        matches: list[DataSerializer] = []
        for serializer in self.serializer_order:
            if serializer.can_handle and serializer.can_handle(data):
                matches.append(serializer)

        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            matched = ", ".join(sorted({s.dtype for s in matches}))
            raise ValueError(
                f"Ambiguous data type {type(data)} matched multiple serializers: {matched}. "
                "Specify dtype or provide a custom serializer."
            )

        raise ValueError(
            f"Unsupported data type {type(data)}. Specify dtype or provide a custom serializer."
        )


def build_default_data_serializers() -> list[DataSerializer]:
    return build_default_serializer_registry().all()


def build_default_serializer_registry() -> SerializerRegistry:
    registry = SerializerRegistry()
    registry.register(build_builtin_serializer())
    registry.register(
        DataSerializer(
            dtype="text",
            serialize=serialize_text_data,
            can_handle=None,
        )
    )
    registry.register(
        DataSerializer(
            dtype="json",
            serialize=serialize_json_data,
            can_handle=None,
        )
    )
    return registry


def build_deserializer_spec(
    deserializer: Callable[[Any, dict[str, Any]], Any] | None,
    deserializer_code: str | None,
    deserializer_function: str | None,
) -> tuple[str | None, str | None]:
    if deserializer is not None and (deserializer_code or deserializer_function):
        raise ValueError(
            "Provide either a deserializer callable or "
            "deserializer_code/deserializer_function, not both."
        )
    if deserializer is not None:
        try:
            source = inspect.getsource(deserializer)
        except OSError as exc:
            raise ValueError(
                "Unable to extract deserializer source; pass deserializer_code instead."
            ) from exc
        source = textwrap.dedent(source)
        source = "from __future__ import annotations\n\n" + source
        name = getattr(deserializer, "__name__", None)
        if not name:
            raise ValueError(
                "Deserializer must be a named function; pass deserializer_code instead."
            )
        return source, name
    if (deserializer_code is None) != (deserializer_function is None):
        raise ValueError("Provide both deserializer_code and deserializer_function.")
    return deserializer_code, deserializer_function


def build_custom_serializer(
    dtype: str,
    dump: Callable[[Any], bytes | str],
    *,
    can_handle: Callable[[Any], bool] | None = None,
    file_name: str | None = None,
    format: str | None = None,
    encoding: str | None = None,
    metadata: dict[str, Any] | None = None,
    deserializer: Callable[[Any, dict[str, Any]], Any] | None = None,
    deserializer_code: str | None = None,
    deserializer_function: str | None = None,
) -> DataSerializer:
    deserializer_spec = build_deserializer_spec(
        deserializer, deserializer_code, deserializer_function
    )

    def serialize(data: Any) -> SerializedData:
        payload = dump(data)
        if isinstance(payload, str):
            payload_bytes = payload.encode(encoding or "utf-8")
        elif isinstance(payload, bytes):
            payload_bytes = payload
        else:
            raise ValueError(
                f"Serializer dump must return bytes or str, got {type(payload)}."
            )

        data_metadata = build_base_metadata(data)
        if metadata:
            data_metadata.update(metadata)

        return SerializedData(
            dtype=dtype,
            inline_data=None,
            file_bytes=payload_bytes,
            file_name=file_name,
            metadata=data_metadata,
            format=format,
            encoding=encoding,
            deserializer_code=deserializer_spec[0],
            deserializer_function=deserializer_spec[1],
        )

    return DataSerializer(
        dtype=dtype,
        serialize=serialize,
        can_handle=can_handle,
        deserializer_code=deserializer_spec[0],
        deserializer_function=deserializer_spec[1],
    )


def resolve_data_serializer(
    data: Any,
    dtype: str | None,
    serializers: SerializerRegistry | list[DataSerializer],
) -> DataSerializer:
    if isinstance(serializers, SerializerRegistry):
        return serializers.resolve(data, dtype)

    if dtype:
        for serializer in serializers:
            if serializer.dtype == dtype:
                return serializer
        supported = ", ".join(sorted({s.dtype for s in serializers}))
        raise ValueError(
            f"Unsupported dtype '{dtype}' for data type {type(data)}. "
            f"Supported dtypes: {supported}. Provide a custom serializer or use a supported dtype."
        )

    if isinstance(data, str):
        for serializer in serializers:
            if serializer.dtype == "text":
                return serializer

    matches: list[DataSerializer] = []
    for serializer in serializers:
        if serializer.can_handle and serializer.can_handle(data):
            matches.append(serializer)

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        matched = ", ".join(sorted({s.dtype for s in matches}))
        raise ValueError(
            f"Ambiguous data type {type(data)} matched multiple serializers: {matched}. "
            "Specify dtype or provide a custom serializer."
        )

    raise ValueError(
        f"Unsupported data type {type(data)}. Specify dtype or provide a custom serializer."
    )


def prepare_context_data(
    data: Any,
    dtype: str | None,
    serializers: SerializerRegistry | list[DataSerializer],
    max_payload_bytes: int | None,
) -> PreparedContextData:
    if data is None:
        metadata = build_base_metadata(data)
        context_dict = {
            "input_data_spec": None,
            "input_data_metadata": metadata,
        }
        return PreparedContextData(context_dict, None, None, None)

    serializer = resolve_data_serializer(data, dtype, serializers)
    serialized = serializer.serialize(data)

    if serialized.inline_data is not None:
        raise ValueError(
            "Inline payloads are not supported. Provide file bytes via the serializer."
        )
    if serialized.file_bytes is None:
        raise ValueError("Serialized data must include file bytes.")

    deserializer_code = serialized.deserializer_code or serializer.deserializer_code
    deserializer_function = (
        serialized.deserializer_function or serializer.deserializer_function
    )
    if serialized.dtype not in {"text", "json"} and not (
        deserializer_code and deserializer_function
    ):
        raise ValueError(
            f"Custom dtype '{serialized.dtype}' requires a deserializer. "
            "Provide deserializer_code and deserializer_function."
        )

    payload_bytes = serialized.file_bytes
    payload_name = validate_file_name(
        serialized.file_name or default_payload_file_name(serialized)
    )
    payload_path = f"/tmp/{payload_name}"
    payload_size = len(payload_bytes)
    ensure_payload_size(payload_size, max_payload_bytes)

    metadata = build_metadata(
        data,
        serialized,
        payload_path=payload_path,
        payload_size=payload_size,
    )

    spec = {
        "dtype": serialized.dtype,
        "format": serialized.format,
        "payload_path": payload_path,
        "payload_encoding": serialized.encoding,
        "deserializer_code": deserializer_code,
        "deserializer_function": deserializer_function,
        "metadata": metadata,
    }

    context_dict = {
        "input_data_spec": spec,
        "input_data_metadata": metadata,
    }

    return PreparedContextData(context_dict, payload_bytes, payload_path, payload_name)


def serialize_text_data(data: Any) -> SerializedData:
    if not isinstance(data, str):
        raise ValueError("Text serializer expects a string input.")

    metadata = build_base_metadata(data)
    payload_bytes = data.encode("utf-8")
    return SerializedData(
        dtype="text",
        inline_data=None,
        file_bytes=payload_bytes,
        file_name="rlm_input_data.txt",
        metadata=metadata,
        format="text",
        encoding="utf-8",
    )


def serialize_json_data(data: Any) -> SerializedData:
    if not is_json_compatible(data, allow_str=True):
        raise ValueError(
            "JSON serializer supports nested Python primitives (dict/list/tuple + "
            "str/int/float/bool/None)."
        )
    metadata = build_base_metadata(data)
    payload_text = json.dumps(data, ensure_ascii=True)
    payload_bytes = payload_text.encode("utf-8")
    return SerializedData(
        dtype="json",
        inline_data=None,
        file_bytes=payload_bytes,
        file_name="rlm_input_data.json",
        metadata=metadata,
        format="json",
        encoding="utf-8",
    )


def build_base_metadata(data: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {"type": str(type(data))}
    if data is None:
        metadata["size"] = 0
        return metadata
    size_value = get_length_if_available(data)
    if size_value is not None:
        metadata["size"] = size_value
    return metadata


_BUILTIN_TAG = "__rlm_builtin__"


def build_builtin_serializer() -> DataSerializer:
    def dump_builtin(data: Any) -> bytes:
        if not is_builtin_data(data):
            raise ValueError(
                "Builtin serializer supports Python primitives (None, bool, int, float, str) "
                "and builtin containers (list, tuple, set, frozenset, dict) plus bytes-like "
                "types, range, slice, and complex."
            )
        encoded = encode_builtin_value(data)
        return json.dumps(encoded, ensure_ascii=True, allow_nan=False).encode("utf-8")

    return build_custom_serializer(
        dtype="builtin",
        dump=dump_builtin,
        can_handle=is_builtin_data,
        file_name="rlm_input_data.json",
        format="json",
        encoding="utf-8",
        deserializer=deserialize_builtin,
    )


def is_builtin_data(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (bool, int, float, str, bytes, bytearray, complex, range)):
        return True
    if isinstance(value, slice):
        return True
    if isinstance(value, memoryview):
        return True
    if isinstance(value, (list, tuple, set, frozenset)):
        return all(is_builtin_data(item) for item in value)
    if isinstance(value, dict):
        return all(is_builtin_data(k) and is_builtin_data(v) for k, v in value.items())
    return False


def encode_builtin_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if math.isnan(value):
            return {_BUILTIN_TAG: "float", "value": "nan"}
        if value > 0:
            return {_BUILTIN_TAG: "float", "value": "inf"}
        return {_BUILTIN_TAG: "float", "value": "-inf"}
    if isinstance(value, bytes):
        return {
            _BUILTIN_TAG: "bytes",
            "data": base64.b64encode(value).decode("ascii"),
        }
    if isinstance(value, bytearray):
        return {
            _BUILTIN_TAG: "bytearray",
            "data": base64.b64encode(bytes(value)).decode("ascii"),
        }
    if isinstance(value, memoryview):
        return {
            _BUILTIN_TAG: "memoryview",
            "data": base64.b64encode(value.tobytes()).decode("ascii"),
        }
    if isinstance(value, complex):
        return {_BUILTIN_TAG: "complex", "real": value.real, "imag": value.imag}
    if isinstance(value, range):
        return {
            _BUILTIN_TAG: "range",
            "args": [value.start, value.stop, value.step],
        }
    if isinstance(value, slice):
        return {
            _BUILTIN_TAG: "slice",
            "args": [value.start, value.stop, value.step],
        }
    if isinstance(value, tuple):
        return {
            _BUILTIN_TAG: "tuple",
            "items": [encode_builtin_value(item) for item in value],
        }
    if isinstance(value, set):
        return {
            _BUILTIN_TAG: "set",
            "items": [encode_builtin_value(item) for item in value],
        }
    if isinstance(value, frozenset):
        return {
            _BUILTIN_TAG: "frozenset",
            "items": [encode_builtin_value(item) for item in value],
        }
    if isinstance(value, list):
        return [encode_builtin_value(item) for item in value]
    if isinstance(value, dict):
        if all(isinstance(key, str) for key in value) and _BUILTIN_TAG not in value:
            return {key: encode_builtin_value(item) for key, item in value.items()}
        return {
            _BUILTIN_TAG: "dict",
            "items": [
                [encode_builtin_value(key), encode_builtin_value(item)]
                for key, item in value.items()
            ],
        }
    raise ValueError(f"Unsupported builtin type: {type(value)}")


def deserialize_builtin(payload: Any, spec: dict[str, Any]) -> Any:
    import base64
    import json

    builtin_tag = "__rlm_builtin__"
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    data = json.loads(payload)

    def decode(value: Any) -> Any:
        if isinstance(value, list):
            return [decode(item) for item in value]
        if isinstance(value, dict):
            tag = value.get(builtin_tag)
            if tag == "bytes":
                return base64.b64decode(value["data"].encode("ascii"))
            if tag == "bytearray":
                return bytearray(base64.b64decode(value["data"].encode("ascii")))
            if tag == "memoryview":
                return memoryview(base64.b64decode(value["data"].encode("ascii")))
            if tag == "float":
                float_value = value.get("value")
                if float_value == "nan":
                    return float("nan")
                if float_value == "inf":
                    return float("inf")
                if float_value == "-inf":
                    return float("-inf")
                raise ValueError(f"Unsupported float marker: {float_value}")
            if tag == "complex":
                return complex(value["real"], value["imag"])
            if tag == "range":
                return range(*value["args"])
            if tag == "slice":
                return slice(*value["args"])
            if tag == "tuple":
                return tuple(decode(item) for item in value["items"])
            if tag == "set":
                return set(decode(item) for item in value["items"])
            if tag == "frozenset":
                return frozenset(decode(item) for item in value["items"])
            if tag == "dict":
                return {
                    decode(key): decode(item) for key, item in value.get("items", [])
                }
            return {key: decode(item) for key, item in value.items()}
        return value

    return decode(data)


def is_json_compatible(value: Any, *, allow_str: bool) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return allow_str
    if isinstance(value, (bool, int, float)):
        return True
    if isinstance(value, (list, tuple)):
        return all(is_json_compatible(item, allow_str=True) for item in value)
    if isinstance(value, dict):
        for key, item in value.items():
            if not is_json_key(key):
                return False
            if not is_json_compatible(item, allow_str=True):
                return False
        return True
    return False


def is_json_key(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def build_metadata(
    data: Any,
    serialized: SerializedData,
    payload_path: str | None,
    payload_size: int | None,
) -> dict[str, Any]:
    metadata = dict(serialized.metadata)
    metadata.setdefault("type", str(type(data)))

    size_value = get_length_if_available(data)
    if size_value is not None and "size" not in metadata:
        metadata["size"] = size_value

    metadata["dtype"] = serialized.dtype
    if serialized.format:
        metadata["format"] = serialized.format

    if payload_path:
        metadata["path"] = payload_path
    if payload_size is not None:
        metadata["file_size"] = payload_size
    return normalize_metadata(metadata)


def get_length_if_available(value: Any) -> int | None:
    try:
        return len(value)
    except Exception:
        return None


def normalize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (int, str)):
            normalized[key] = value
        else:
            normalized[key] = str(value)
    return normalized


def default_payload_file_name(serialized: SerializedData) -> str:
    if serialized.format == "json":
        ext = "json"
    elif serialized.format == "text" or serialized.dtype == "text":
        ext = "txt"
    else:
        ext = "bin"
    return f"rlm_input_data.{ext}"


def validate_file_name(file_name: str) -> str:
    if "/" in file_name or "\\" in file_name:
        raise ValueError("File name must not include path separators.")
    return file_name


def ensure_payload_size(payload_size: int, max_payload_bytes: int | None) -> None:
    if max_payload_bytes is None:
        return
    if payload_size > max_payload_bytes:
        raise ValueError(
            "Payload exceeds sandbox storage limit: "
            f"{payload_size} bytes > {max_payload_bytes} bytes."
        )
