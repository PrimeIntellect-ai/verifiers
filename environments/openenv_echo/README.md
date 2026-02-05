# OpenEnv Echo

Minimal example showing how to use `OpenEnvEnv` against an OpenEnv environment.

By default this points at the Hugging Face Space `openenv/echo-env`:

```python
import verifiers as vf

env = vf.load_environment("openenv-echo")
```

You can also point at a local OpenEnv project:

```python
from pathlib import Path
import verifiers as vf

env = vf.OpenEnvEnv(openenv_project=Path("/path/to/openenv_project"))
```

If the project includes a Dockerfile, build/register the image once:

```bash
vf-openenv-build --path /path/to/openenv_project
```
