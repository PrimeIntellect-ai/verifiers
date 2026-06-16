import subprocess
import sys
import textwrap


def test_v1_imports_without_optional_runtime_dependencies():
    script = textwrap.dedent(
        """
        import asyncio
        import builtins
        import sys

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
        assert "modal" not in sys.modules
        assert "renderers" not in sys.modules

        try:
            vf.resolve_client(TrainClientConfig())
        except ModuleNotFoundError as exc:
            assert exc.name == "renderers"
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


def test_train_client_normalizes_renderer_types_at_the_boundary():
    from renderers import AutoRendererConfig
    from renderers.base import MultiModalData as RendererMultiModalData
    from renderers.base import PlaceholderRange as RendererPlaceholderRange

    from verifiers.v1.clients import TrainClientConfig
    from verifiers.v1.clients.train import TrainClient, response_from_generate
    from verifiers.v1.types import MultiModalData

    config = TrainClientConfig(renderer={"name": "auto", "preserve_all_thinking": True})
    client = TrainClient(None, config=config.renderer)
    assert isinstance(client.config, AutoRendererConfig)

    response = response_from_generate(
        {
            "content": "ok",
            "prompt_ids": [1],
            "completion_ids": [2],
            "multi_modal_data": RendererMultiModalData(
                mm_hashes={"image": ["hash"]},
                mm_placeholders={
                    "image": [RendererPlaceholderRange(offset=0, length=1)]
                },
                mm_items={"image": [{"pixel_values": "pixels"}]},
            ),
        },
        "model",
    )
    assert response.tokens is not None
    assert isinstance(response.tokens.multi_modal_data, MultiModalData)
