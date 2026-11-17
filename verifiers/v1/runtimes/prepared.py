"""Reuse static setup recorded in an image or filesystem checkpoint."""

from __future__ import annotations

import hashlib
import json
import shlex
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from verifiers.v1.runtimes import Runtime

# The interpreter belongs to a prepared environment. Neither loading nor saving
# imports the framework inside the sandbox, or captures an active session.
_LOAD = r"""
import hashlib, json, os, pathlib, sys
records = {}
for path in pathlib.Path(sys.argv[1]).glob('*.json'):
    try:
        if path.stat().st_size > 65536:
            continue
        record = json.loads(path.read_text())
        if record['schema'] != 1 or not os.access(record['python'], os.X_OK):
            continue
        value = record['value']
        if not isinstance(record['key'], str) or not isinstance(value, dict):
            continue
        if not isinstance(record['files'], dict) or not record['files']:
            continue
        if 'script' in value:
            if value['script'] != record['key']:
                continue
        elif not all(isinstance(value[k], str) for k in ('distribution', 'digest', 'marker')):
            continue
        if not all(hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() == digest
                   for p, digest in record['files'].items()):
            continue
        records[record['key']] = record
    except (OSError, ValueError, KeyError, TypeError):
        continue
print(json.dumps(records))
"""

_SAVE = r"""
import hashlib, importlib.metadata, json, os, pathlib, sys, uuid
directory, key, value = pathlib.Path(sys.argv[1]), sys.argv[2], json.loads(sys.argv[3])
directory.mkdir(parents=True, exist_ok=True)
def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_bytes(data)
    os.replace(temporary, path)
files = []
if 'script' in value:
    files.append(pathlib.Path(value['script']))
    config = pathlib.Path(sys.prefix) / 'pyvenv.cfg'
    if config.is_file():
        files.append(config)
    for dist in importlib.metadata.distributions():
        files.extend(pathlib.Path(p.locate()) for p in (dist.files or [])
                     if p.name == 'RECORD' and p.parent.suffix == '.dist-info')
else:
    dist = importlib.metadata.distribution(value['distribution'])
    files.append(next(pathlib.Path(p.locate()) for p in dist.files
                      if p.name == 'RECORD' and p.parent.suffix == '.dist-info'))
    marker = pathlib.Path(value['marker'])
    write(marker, value['digest'].encode())
    files.append(marker)
record = dict(schema=1, key=key, python=sys.executable, value=value,
              files={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
write(directory / (hashlib.sha256(key.encode()).hexdigest() + '.json'),
      json.dumps(record).encode())
write(directory / 'python', sys.executable.encode())
"""


def script_directory(runtime: Runtime) -> str:
    # User/home bindings can resolve uv's environments differently.
    directory = runtime.backend.scripts_dir
    if runtime.user is not None:
        directory += f"-{runtime.user}"
    if runtime.home is not None:
        directory += "-" + hashlib.sha256(runtime.home.encode()).hexdigest()[:16]
    return directory


def directory(runtime: Runtime) -> str:
    return f"{script_directory(runtime)}-prepared"


async def load_prepared(runtime: Runtime) -> None:
    cache = runtime.backend.cache
    root = directory(runtime)
    if root in cache.prepared_loaded:
        return
    async with cache.prepared_lock:
        if root in cache.prepared_loaded:
            return
        pointer = shlex.quote(f"{root}/python")
        command = (
            f"if [ -r {pointer} ]; then "
            f'python=$(cat {pointer}); if [ -x "$python" ]; then '
            f'"$python" -I -c {shlex.quote(_LOAD)} {shlex.quote(root)}; fi; fi'
        )
        result = await runtime.run(["sh", "-c", command], {})
        # Missing or damaged preparation is just a cold setup, including a
        # deleted interpreter that prevented the manifest reader from starting.
        if result.exit_code == 0 and result.stdout.strip():
            try:
                records = json.loads(result.stdout)
                for key, record in records.items():
                    cache.prepared[key] = record
                    if script := record["value"].get("script"):
                        cache.uv_interpreters[script] = record["python"]
            except (ValueError, KeyError, TypeError, AttributeError):
                pass
        cache.prepared_loaded.add(root)


def record_arguments(runtime: Runtime, key: str, value: dict) -> list[str]:
    """Publish readiness after a successful install, within its existing command."""
    return ["-I", "-c", _SAVE, directory(runtime), key, json.dumps(value)]
