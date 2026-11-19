"""Reuse static setup recorded in an image or filesystem checkpoint."""

from __future__ import annotations

import hashlib
import json
import shlex
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from verifiers.v1.runtimes import Runtime
    from verifiers.v1.runtimes.base import RuntimeCache

# The interpreter belongs to a prepared environment. Neither loading nor saving
# imports the framework inside the sandbox, or captures an active session.
_LOAD = r"""
import hashlib, json, os, pathlib, subprocess, sys
records = {}
for path in pathlib.Path(sys.argv[1]).glob('*.json'):
    try:
        if path.stat().st_size > 65536:
            continue
        record = json.loads(path.read_text())
        if record['schema'] != 1:
            continue
        value = record['value']
        if not isinstance(record['key'], str) or not isinstance(value, dict):
            continue
        if 'install' in value:
            if not isinstance(value['install'], str) or not os.path.isdir(value['install']):
                continue
            ready = value.get('ready')
            if ready and subprocess.call(
                ['sh', '-c', ready],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ) != 0:
                continue
        else:
            if not os.access(record['python'], os.X_OK):
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
if 'install' in value:
    record = dict(schema=1, key=key, python='', value=value, files={})
    write(directory / (hashlib.sha256(key.encode()).hexdigest() + '.json'),
          json.dumps(record).encode())
    sys.exit(0)
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


def load_prepared_command(root: str) -> str:
    pointer = shlex.quote(f"{root}/python")
    root_q = shlex.quote(root)
    return (
        f"if [ -d {root_q} ]; then "
        f'python=""; if [ -r {pointer} ]; then python=$(cat {pointer}); fi; '
        'if [ ! -x "$python" ]; then python=$(command -v python3 || command -v python || true); fi; '
        f'if [ -n "$python" ] && [ -x "$python" ]; then "$python" -I -c {shlex.quote(_LOAD)} {root_q}; fi; fi'
    )


def apply_prepared(cache: RuntimeCache, root: str, output: str) -> None:
    if output.strip():
        try:
            records = json.loads(output)
            for key, record in records.items():
                cache.prepared[key] = record
                if script := record["value"].get("script"):
                    cache.uv_interpreters[script] = record["python"]
                elif record["value"].get("install"):
                    cache.installed.add(key)
        except (ValueError, KeyError, TypeError, AttributeError):
            pass
    cache.prepared_loaded.add(root)


async def load_prepared(runtime: Runtime) -> None:
    cache = runtime.backend.cache
    root = directory(runtime)
    if root in cache.prepared_loaded:
        return
    async with cache.prepared_lock:
        if root in cache.prepared_loaded:
            return
        result = await runtime.run(["sh", "-c", load_prepared_command(root)], {})
        # Missing or damaged preparation is just a cold setup, including a
        # deleted interpreter that prevented the manifest reader from starting.
        apply_prepared(cache, root, result.stdout if result.exit_code == 0 else "")


def record_arguments(runtime: Runtime, key: str, value: dict) -> list[str]:
    """Publish readiness after a successful install, within its existing command."""
    return ["-I", "-c", _SAVE, directory(runtime), key, json.dumps(value)]
