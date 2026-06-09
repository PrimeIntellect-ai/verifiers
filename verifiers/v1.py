"""`verifiers.v1` is vf-nano.

vf-nano (the `verifiers.nano` package) is vendored as a submodule under ``deps/vf-nano`` and
made importable via this package's extended ``__path__`` (see ``verifiers/__init__.py``). This
module aliases ``verifiers.v1`` to ``verifiers.nano`` 1:1, so ``verifiers.v1.Trace``,
``verifiers.v1.serve.EnvServer``, ``verifiers.v1.EnvConfig``, etc. are exactly the nano objects.
"""

import importlib
import sys

_nano = importlib.import_module("verifiers.nano")

# Alias the package and every already-imported submodule so `verifiers.v1.<sub>` is the same
# module object as `verifiers.nano.<sub>` (not a re-execution).
sys.modules[__name__] = _nano
for _name, _module in list(sys.modules.items()):
    if _name.startswith("verifiers.nano."):
        sys.modules["verifiers.v1." + _name[len("verifiers.nano.") :]] = _module
