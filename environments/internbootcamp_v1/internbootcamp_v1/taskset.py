"""InternBootcamp tasks backed by the benchmark's native generators and scorers.

The selected Bootcamp generates seeded single-turn tasks on the environment
host. Scoring happens inside the rollout container because some upstream verifiers
evaluate model-produced expressions; model output must never reach them in the host
process. The scorer dependency is pinned to the same Apache-2.0 snapshot as the task
generator, so generation and verification cannot drift independently.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import math
import random
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field

import verifiers.v1 as vf

DEFAULT_SYSTEM_PROMPT = "Think step by step to solve the puzzle."
DEFAULT_BOOTCAMP = "game24"
MAX_COMPLETION_BYTES = 100_000
VERIFY = (Path(__file__).parent / "verify.py").read_bytes()


def _canonical_key(class_name: str) -> str:
    base = re.sub(r"bootcamp$", "", class_name, flags=re.IGNORECASE)
    return re.sub(r"[^0-9a-z]+", "", base.lower())


@lru_cache(maxsize=1)
def _discover_bootcamps() -> dict[str, type]:
    import internbootcamp

    classes: dict[str, type] = {}
    for name, candidate in vars(internbootcamp).items():
        if (
            inspect.isclass(candidate)
            and name.lower().endswith("bootcamp")
            and callable(getattr(candidate, "case_generator", None))
            and callable(getattr(candidate, "prompt_func", None))
            and callable(getattr(candidate, "verify_score", None))
        ):
            key = getattr(candidate, "canonical_name", None) or _canonical_key(name)
            classes[_canonical_key(str(key))] = candidate
    return classes


def _resolve_bootcamp(name: str) -> tuple[str, type]:
    key = _canonical_key(name)
    classes = _discover_bootcamps()
    if key not in classes:
        sample = ", ".join(sorted(classes)[:25])
        raise ValueError(
            f"unknown InternBootcamp task {name!r}; examples include: {sample}"
        )
    return key, classes[key]


def _new_bootcamp(name: str, seed: int):
    key, cls = _resolve_bootcamp(name)
    signature = inspect.signature(cls)
    required = [
        parameter.name
        for parameter in signature.parameters.values()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ]
    kwargs: dict[str, Any] = {}
    if "seed" in signature.parameters:
        kwargs["seed"] = seed
        required = [name for name in required if name != "seed"]
    if required:
        raise ValueError(
            f"InternBootcamp task {key!r} requires constructor arguments {required}; "
            "select a task with a default constructor"
        )
    return key, cls(**kwargs)


def _json_value(value: Any) -> Any:
    """Convert an upstream identity to stable JSON without lossy string fallbacks."""
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("InternBootcamp identity contains a non-finite number")
        return value
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("InternBootcamp identity contains a non-string object key")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, set):
        return sorted((_json_value(item) for item in value), key=repr)
    if dataclasses.is_dataclass(value):
        return _json_value(dataclasses.asdict(value))

    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return _json_value(value.tolist())
        if isinstance(value, np.generic):
            return _json_value(value.item())
    except ImportError:
        pass
    raise TypeError(
        f"InternBootcamp identity contains unsupported {type(value).__name__}; "
        "choose a JSON-serializable Bootcamp"
    )


class InternBootcampTaskConfig(vf.TaskConfig):
    verifier_timeout: int = Field(default=180, ge=30, le=600)
    """Maximum wall time for one upstream scoring call inside the container."""


class InternBootcampConfig(vf.TasksetConfig):
    bootcamp: str = DEFAULT_BOOTCAMP
    """Bootcamp class name or canonical key (for example ``Game24`` or ``game24``)."""

    num_examples: int = Field(default=50, ge=1, le=10_000)
    seed: int = 0
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    task: InternBootcampTaskConfig = InternBootcampTaskConfig()


class InternBootcampData(vf.TaskData):
    bootcamp: str
    """Canonical upstream Bootcamp key used by the isolated scorer."""

    identity: dict[str, Any]
    """JSON form of the generated case consumed by the upstream scorer."""


class InternBootcampTask(
    vf.Task[InternBootcampData, vf.State, InternBootcampTaskConfig]
):
    NEEDS_CONTAINER = True

    @vf.stop
    async def single_turn(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= 1

    async def _score(self, completion: str, runtime: vf.Runtime) -> float:
        raw = completion.encode("utf-8", "replace")
        if len(raw) > MAX_COMPLETION_BYTES:
            return 0.0

        digest = hashlib.sha256(raw).hexdigest()[:16]
        stem = f"/tmp/internbootcamp/{self.data.idx}-{digest}"
        identity_path = f"{stem}.identity.json"
        completion_path = f"{stem}.completion.txt"
        await runtime.write(
            identity_path,
            json.dumps(
                self.data.identity,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            ).encode(),
        )
        await runtime.write(completion_path, raw)
        result = await runtime.run_uv_script(
            VERIFY,
            args=[self.data.bootcamp, identity_path, completion_path],
            env={"INTERNBOOTCAMP_VERIFY_TIMEOUT": str(self.config.verifier_timeout)},
        )
        if result.exit_code != 0:
            detail = (
                result.stderr or result.stdout or "unknown verifier error"
            ).strip()
            raise RuntimeError(f"InternBootcamp verifier failed: {detail[-2000:]}")
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError("InternBootcamp verifier produced no result")
        payload = json.loads(lines[-1])
        score = float(payload["score"])
        return min(1.0, max(0.0, score)) if math.isfinite(score) else 0.0

    @vf.reward(weight=1.0)
    async def upstream_score(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        return await self._score(trace.last_reply or "", runtime)

    async def validate(self, runtime: vf.Runtime) -> bool:
        """Preflight the pinned scorer on this generated identity without a model call."""
        score = await self._score("", runtime)
        return 0.0 <= score <= 1.0


class InternBootcampTaskset(vf.Taskset[InternBootcampTask, InternBootcampConfig]):
    def load(self) -> list[InternBootcampTask]:
        config = self.config
        random.seed(config.seed)
        try:
            import numpy as np

            np.random.seed(config.seed)
        except ImportError:
            pass
        try:
            from faker import Faker

            Faker.seed(config.seed)
        except ImportError:
            pass

        key, bootcamp = _new_bootcamp(config.bootcamp, config.seed)
        tasks: list[InternBootcampTask] = []
        resources = vf.TaskResources(cpu=2, memory=4, disk=4)
        for idx in range(config.num_examples):
            raw_identity = bootcamp.case_generator()
            prompt = bootcamp.prompt_func(raw_identity)
            identity = _json_value(raw_identity)
            if not isinstance(identity, dict):
                raise TypeError(
                    f"InternBootcamp task {key!r} returned a non-object identity"
                )
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(
                    f"InternBootcamp task {key!r} returned an empty prompt"
                )
            tasks.append(
                InternBootcampTask(
                    InternBootcampData(
                        idx=idx,
                        name=f"{key}_{idx:05d}",
                        prompt=prompt,
                        system_prompt=config.system_prompt,
                        resources=resources,
                        bootcamp=key,
                        identity=identity,
                    ),
                    config.task,
                )
            )
        return tasks


__all__ = [
    "InternBootcampConfig",
    "InternBootcampData",
    "InternBootcampTask",
    "InternBootcampTaskset",
]
