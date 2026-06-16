import sys

import pytest

from verifiers.v1.runtimes import ModalConfig, ModalRuntime


async def test_modal_runtime_requires_modal_extra(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "modal", None)

    with pytest.raises(ModuleNotFoundError, match=r"verifiers\[modal\]"):
        await ModalRuntime(ModalConfig()).start()
