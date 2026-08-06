import numpy as np
from renderers.base import MultiModalData

import verifiers.v1 as vf
from verifiers.v1.episode import EnvInfo
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import KeptTokens
from verifiers.v1.utils.platform import _build_samples


def test_native_wrapper_excludes_tensor_fields():
    trace = vf.Trace(
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="test")),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        nodes=[
            MessageNode(
                message=vf.UserMessage(content="test"),
                multi_modal_data=MultiModalData(),
                routed_experts=np.array([[[1]]], dtype=np.uint8),
                kept_tokens=KeptTokens(
                    ids=np.array([1], dtype=np.int32),
                    counts=np.array([1], dtype=np.int32),
                ),
            )
        ],
    )
    episode = vf.Episode(env=EnvInfo(id="test"), ok=True, traces=[trace])

    node = _build_samples([episode])[0]["info"]["native_wrapper"]["traces"][0]["nodes"][
        0
    ]

    assert "multi_modal_data" not in node
    assert "routed_experts" not in node
    assert "kept_tokens" not in node
