# tasksets

Reusable Verifiers v1 tasksets.

```python
from tasksets import HarborTaskset, HarborTasksetConfig

taskset = HarborTaskset(config=HarborTasksetConfig())
```

Install TextArena-backed tasksets with:

```bash
uv add "tasksets[textarena]"
```

Install OpenEnv-backed tasksets with:

```bash
uv add "tasksets[openenv]"
```

Install OpenReward-backed tasksets with:

```bash
uv add "tasksets[openreward]"
```
