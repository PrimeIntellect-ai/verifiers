import io
import random
from typing import Literal

from datasets import Dataset

import verifiers as vf
from verifiers.envs.experimental.rlm_env import RLMEnv
from verifiers.types import State
from verifiers.utils.rlm_data_serialization_utils import (
    DataSerializer,
    build_custom_serializer,
)

ContextDType = Literal[
    "text",
    "list",
    "tuple",
    "nested_list",
    "nested_dict",
    "mixed",
    "large_list",
    "polars",
]


def deserialize_polars(payload, spec):
    import io
    import polars as pl

    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    if isinstance(payload, bytes):
        buffer = io.BytesIO(payload)
        return pl.read_parquet(buffer)
    return payload


def build_polars_serializer() -> DataSerializer:
    import polars as pl

    def dump_polars(data):
        if not isinstance(data, pl.DataFrame):
            raise ValueError("Polars serializer expects a polars.DataFrame input.")
        buffer = io.BytesIO()
        data.write_parquet(buffer)
        return buffer.getvalue()

    return build_custom_serializer(
        dtype="polars",
        dump=dump_polars,
        can_handle=lambda value: isinstance(value, pl.DataFrame),
        file_name="dataframe.parquet",
        format="parquet",
        deserializer=deserialize_polars,
    )


def build_dataframe(rng: random.Random, num_rows: int = 200):
    import polars as pl

    return pl.DataFrame(
        {
            "a": [rng.randint(-100, 100) for _ in range(num_rows)],
            "b": [rng.randint(-100, 100) for _ in range(num_rows)],
        }
    )


class ArbitraryDataRLMEnv(RLMEnv):
    def __init__(self, context_data, **kwargs):
        super().__init__(**kwargs)
        self._context_data = context_data

    async def setup_state(self, state: State, **kwargs) -> State:
        info = state.get("info") or {}
        if not isinstance(info, dict):
            info = {}
        info = dict(info)
        info[self.context_key] = self._context_data
        state["info"] = info
        return await super().setup_state(state, **kwargs)


def _build_dataset(question: str, answer: str) -> Dataset:
    return Dataset.from_list(
        [
            {
                "question": question,
                "answer": answer,
                "info": {},
            }
        ]
    )


def load_environment(context_dtype: ContextDType = "text", **kwargs) -> vf.Environment:
    rng_seed = kwargs.pop("rng_seed", None)
    rng = random.Random(rng_seed)
    serializers: list[DataSerializer] = []
    serializer_dtype = context_dtype

    if context_dtype == "text":
        values = [rng.randint(-50, 50) for _ in range(rng.randint(4, 10))]
        values_text = ", ".join(str(value) for value in values)
        context_data = f"Numbers: {values_text}"
        question = "Sum the numbers in extra_data and reply with the integer."
        answer = str(sum(values))
        pip_install_packages = ""
    elif context_dtype == "list":
        values = [rng.randint(-50, 50) for _ in range(rng.randint(4, 12))]
        context_data = values
        serializer_dtype = "builtin"
        question = "Sum the numbers in extra_data and reply with the integer."
        answer = str(sum(values))
        pip_install_packages = ""
    elif context_dtype == "tuple":
        values = tuple(rng.randint(-50, 50) for _ in range(rng.randint(4, 12)))
        context_data = values
        serializer_dtype = "builtin"
        question = "Sum the numbers in extra_data and reply with the integer."
        answer = str(sum(values))
        pip_install_packages = ""
    elif context_dtype == "nested_list":
        outer_size = rng.randint(2, 5)
        inner_min = 3
        inner_max = 8
        nested_values = []
        total = 0
        for _ in range(outer_size):
            inner = [
                rng.randint(-50, 50) for _ in range(rng.randint(inner_min, inner_max))
            ]
            nested_values.append(inner)
            total += sum(inner)
        context_data = nested_values
        serializer_dtype = "builtin"
        question = (
            "Sum all numbers in the nested list extra_data and reply with the integer."
        )
        answer = str(total)
        pip_install_packages = ""
    elif context_dtype == "nested_dict":
        group_count = rng.randint(2, 4)
        nested_dict: dict[str, object] = {}
        total = 0
        for index in range(group_count):
            key = f"group_{index + 1}"
            values = [rng.randint(-50, 50) for _ in range(rng.randint(3, 7))]
            total += sum(values)
            if rng.random() < 0.5:
                nested_dict[key] = values
            else:
                nested_dict[key] = {"values": values}
        context_data = nested_dict
        serializer_dtype = "builtin"
        question = "Sum all integers contained anywhere in extra_data and reply with the integer."
        answer = str(total)
        pip_install_packages = ""
    elif context_dtype == "mixed":
        list_values = [rng.randint(-50, 50) for _ in range(rng.randint(3, 7))]
        tuple_values = tuple(rng.randint(-50, 50) for _ in range(rng.randint(3, 7)))
        set_values = set(rng.randint(-50, 50) for _ in range(rng.randint(3, 7)))
        inner_values = [rng.randint(-50, 50) for _ in range(rng.randint(3, 7))]
        context_data = {
            "list": list_values,
            "tuple": tuple_values,
            "set": set_values,
            "dict": {"inner": inner_values},
        }
        serializer_dtype = "builtin"
        question = "Sum all integers contained anywhere in extra_data and reply with the integer."
        answer = str(
            sum(list_values) + sum(tuple_values) + sum(set_values) + sum(inner_values)
        )
        pip_install_packages = ""
    elif context_dtype == "large_list":
        values = [rng.randint(-100, 100) for _ in range(rng.randint(200, 500))]
        context_data = values
        serializer_dtype = "builtin"
        question = "Sum the numbers in extra_data and reply with the integer."
        answer = str(sum(values))
        pip_install_packages = ""
    elif context_dtype == "polars":
        num_rows = rng.randint(80, 220)
        dataframe = build_dataframe(rng, num_rows=num_rows)
        context_data = dataframe
        serializers.append(build_polars_serializer())
        total = int((dataframe["a"] + dataframe["b"]).sum())
        question = "Compute the sum of columns a and b in extra_data and reply with the integer."
        answer = str(total)
        pip_install_packages = "polars>=0.20.0"
    else:
        raise ValueError(f"Unsupported context_dtype: {context_dtype}")

    dataset = _build_dataset(question, answer)

    async def exact_answer(state, answer, **_kwargs) -> float:
        final_answer = (state.get("final_answer") or "").strip()
        return 1.0 if final_answer == answer else 0.0

    rubric = vf.Rubric(funcs=[exact_answer])

    env = ArbitraryDataRLMEnv(
        dataset=dataset,
        rubric=rubric,
        context_dtype=serializer_dtype,
        data_serializers=serializers,
        context_data=context_data,
        pip_install_packages=pip_install_packages,
        **kwargs,
    )

    return env
