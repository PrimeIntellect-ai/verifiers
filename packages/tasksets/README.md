# tasksets

Reusable tasksets for Verifiers.

```python
from tasksets import HarborTaskset, HarborTasksetConfig

taskset = HarborTaskset(config=HarborTasksetConfig(bundle_package=__name__))
```

Bundled tasksets include their upstream runtime dependencies by default:

```bash
uv add tasksets
```
