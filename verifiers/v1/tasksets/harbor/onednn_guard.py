"""A guard against oneDNN's SIGFPE on guests that report their CPU topology inconsistently.

oneDNN's bundled Xbyak sizes per-core caches as size / coresSharingDataCache. A KVM guest
(Prime's AMD EPYC 9255 boxes) whose CPUID leaf 0x8000001D says L1 is shared by 2 threads
while leaf 0xB says 1 thread and 1 core makes Xbyak compute 0 sharers for L2 and L3, and
torch's first oneDNN convolution dies with SIGFPE (exit 136). No oneDNN or OpenMP setting
avoids it.

Harbor setup runs this file with each of the box's Python interpreters that has torch
(`install`): on a CPU in that state it copies itself into the interpreter's site-packages
with a `.pth` that imports it at startup. Imported there, it only registers a one-shot
finder; when torch is imported it turns torch's oneDNN backend off
(`torch.backends.mkldnn.enabled = False`). Nothing is imported eagerly, and a CPU that
reports its topology consistently keeps oneDNN. `VF_ONEDNN_GUARD=0` disables it, `=1`
forces it.
"""

import os
import sys

_MODE = os.environ.get("VF_ONEDNN_GUARD", "auto").strip().lower()


def _cpuid_fn():
    import ctypes
    import mmap

    code = bytes(
        [
            0x53,
            0x49,
            0x89,
            0xD0,
            0x89,
            0xF8,
            0x89,
            0xF1,
            0x0F,
            0xA2,
            0x41,
            0x89,
            0x00,
            0x41,
            0x89,
            0x58,
            0x04,
            0x41,
            0x89,
            0x48,
            0x08,
            0x41,
            0x89,
            0x50,
            0x0C,
            0x5B,
            0xC3,
        ]
    )
    buf = mmap.mmap(
        -1, mmap.PAGESIZE, prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC
    )
    buf.write(code)
    addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
    fn = ctypes.CFUNCTYPE(
        None, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)
    )(addr)

    def cpuid(leaf, sub=0):
        out = (ctypes.c_uint32 * 4)()
        fn(leaf, sub, out)
        return tuple(out)

    cpuid._keep = buf  # keep the executable page alive
    return cpuid


def _xbyak_divides_by_zero():
    """Replicates Xbyak's (oneDNN v3.x) AMD setNumCores/setCacheHierarchy; True if a cache level gets 0 sharers."""
    import platform

    if platform.machine().lower() not in ("x86_64", "amd64"):
        return False
    try:
        with open("/proc/cpuinfo") as f:
            if "AuthenticAMD" not in f.read(4096):
                return False
    except OSError:
        return False
    cpuid = _cpuid_fn()
    max_leaf = cpuid(0)[0]
    if cpuid(0x80000000)[0] < 0x8000001D or max_leaf < 0xB:
        return False
    a, b, _, _ = cpuid(0xB, 0)
    if a == 0 and b == 0:
        return False  # legacy path; not the failure we know about
    cores = [0, 0]
    for i in range(4):
        _, b, c, _ = cpuid(0xB, i)
        level = (c >> 8) & 0xFF
        if level in (1, 2):
            cores[level - 1] = b & 0xFFFF
    cores[0] = max(1, cores[0])
    cores[1] = max(cores[0], cores[1])
    sharing = []
    sub = 0
    while len(sharing) < 10 and sub < 16:
        eax = cpuid(0x8000001D, sub)[0]
        ctype = eax & 0x1F
        if ctype == 0:
            break
        if ctype != 2:
            n = ((eax >> 14) & 0xFFF) + 1
            if sub > 0:
                n = min(n, cores[1]) // max(1, sharing[0])
            sharing.append(n)
        sub += 1
    return any(n == 0 for n in sharing[1:])


def _apply(torch_mod):
    try:
        if _MODE in ("1", "on", "force", "true") or _xbyak_divides_by_zero():
            torch_mod._C._set_mkldnn_enabled(False)
            if os.environ.get("VF_ONEDNN_GUARD_VERBOSE"):
                sys.stderr.write(
                    "[verifiers] torch.backends.mkldnn disabled (oneDNN cache-topology SIGFPE guard)\n"
                )
    except Exception:  # noqa: BLE001, S110 - a guard in a task's interpreter never raises
        pass


class _Finder:
    def find_spec(self, name, path=None, target=None):
        if name != "torch":
            return None
        try:
            sys.meta_path.remove(self)
        except ValueError:
            return None
        import importlib.util

        try:
            spec = importlib.util.find_spec("torch")
        except Exception:  # noqa: BLE001 - leave torch's import to Python
            return None
        if (
            spec is None
            or spec.loader is None
            or not hasattr(spec.loader, "exec_module")
        ):
            return spec
        loader = spec.loader
        orig = loader.exec_module

        def exec_module(module):
            orig(module)
            _apply(module)

        try:
            loader.exec_module = exec_module
        except Exception:  # noqa: BLE001, S110 - an unwrappable loader keeps oneDNN
            pass
        return spec


NAME = "_vf_onednn_guard"


def guard():
    if _MODE in ("0", "off", "false", "no"):
        return
    if "torch" in sys.modules:
        _apply(sys.modules["torch"])
        return
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


def install() -> str:
    """Copy this guard into this interpreter's site-packages when it has torch and the
    CPU is in the faulting state; returns what it did."""
    import importlib.util
    import shutil
    import sysconfig

    if importlib.util.find_spec("torch") is None:
        return "no torch"
    if not _xbyak_divides_by_zero():
        return "cpu ok"
    target = sysconfig.get_paths()["purelib"]
    shutil.copyfile(__file__, os.path.join(target, NAME + ".py"))
    with open(os.path.join(target, "zz" + NAME + ".pth"), "w") as f:
        f.write("import " + NAME + "\n")
    return "installed in " + target


if __name__ == NAME:
    guard()
elif __name__ == "__main__":
    print(install())
