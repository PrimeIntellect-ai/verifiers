"""Run identity shared by the CLIs that write a run directory (`validate`, `gepa`)."""

from uuid import uuid4

from pydantic import PrivateAttr
from pydantic_config import BaseConfig

from verifiers.v1.configs.env import EnvConfig


def default_run_name(env: EnvConfig, model: str) -> str:
    """The auto-generated run name: `<env>--<model>--<harness>--<short-id>`, a
    descriptive leaf for the run directory `output_dir / run.name`. The short-id
    suffix keeps repeated invocations from colliding."""
    taskset = env.taskset
    name = taskset.name if taskset.id else "no-taskset"
    if taskset.id and env.id:
        # Same compounding as `EnvConfig.env_id`: a `best-of-n+gsm8k` run must
        # not share a name with a plain `gsm8k` one.
        name = f"{env.id}+{name}"
    # Every seat's resolved harness, distinct, in role order.
    harness = "+".join(dict.fromkeys(h.name for h in env.agent_harnesses().values()))
    slug = (
        f"{name}--{model.replace('/', '--')}--{harness or 'default'}--{uuid4().hex[:8]}"
    )
    return slug.lower()


class RunConfig(BaseConfig):
    name: str | None = None
    """Run name. Auto-generated as `<env>--<model>--<harness>--<short-id>` when unset."""

    dir: str | None = None
    """Run directory name — the run writes to `output_dir / dir`. Defaults to `run.name`;
    set it only when the directory should differ from the display name."""

    # TODO: fetch the id from the Prime SDK once runs are registered there.
    _id: str = PrivateAttr(default_factory=lambda: str(uuid4()))

    @property
    def id(self) -> str:
        return self._id
