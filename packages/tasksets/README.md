# tasksets

Reusable tasksets for Verifiers.

```python
from tasksets import HarborTaskset, HarborTasksetConfig

taskset = HarborTaskset(config=HarborTasksetConfig(bundle_package=__name__))
```

Install only the backend extras you need:

```bash
uv add "tasksets[openenv,openreward,ta]"
```
