import subprocess
import sys
import textwrap


def test_v1_imports_without_optional_runtime_dependencies():
    script = textwrap.dedent(
        """
        import asyncio
        import builtins

        real_import = builtins.__import__

        def block_optional(name, *args, **kwargs):
            if name == "modal" or name.startswith("modal."):
                raise ModuleNotFoundError(f"No module named {name!r}", name=name)
            if name == "renderers" or name.startswith("renderers."):
                raise ModuleNotFoundError(f"No module named {name!r}", name=name)
            return real_import(name, *args, **kwargs)

        builtins.__import__ = block_optional

        import verifiers.v1 as vf
        from verifiers.v1.clients import TrainClientConfig
        from verifiers.v1.runtimes.modal import ModalConfig, ModalRuntime
        from verifiers.v1.types import MultiModalData

        assert MultiModalData().is_empty()

        try:
            from verifiers.v1.clients import TrainClient
        except ImportError as exc:
            assert "verifiers[renderers]" in str(exc)
        else:
            raise AssertionError(f"unexpected TrainClient: {TrainClient}")

        try:
            vf.resolve_client(TrainClientConfig())
        except ImportError as exc:
            assert "verifiers[renderers]" in str(exc)
        else:
            raise AssertionError("TrainClientConfig resolved without renderers")

        try:
            asyncio.run(ModalRuntime(ModalConfig()).start())
        except vf.ProgramError as exc:
            assert "verifiers[modal]" in str(exc)
        else:
            raise AssertionError("ModalRuntime started without modal")
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True)
