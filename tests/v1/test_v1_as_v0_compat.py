from __future__ import annotations

import verifiers as vf
import compat_taskset_v1
from verifiers.types import (
    Response,
    ResponseMessage,
    ResponseTokens,
    Usage,
)
from verifiers.v1.compat import V0ClientAsV1Client, V1AsV0Environment, build_env_config
from verifiers.v1.dialects import ChatDialect
from verifiers.v1.types import SamplingConfig
from verifiers.v1.utils.multimodal import ImageOffloadStats


class FakeV0Client:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def get_response(
        self,
        prompt,
        model: str,
        sampling_args: dict,
        tools=None,
        **kwargs,
    ) -> Response:
        del tools
        self.calls.append(
            {
                "prompt": prompt,
                "model": model,
                "sampling_args": sampling_args,
                "state": kwargs.get("state"),
            }
        )
        text = f"echo: {_last_user_text(prompt)}"
        prompt_ids = [1, 2, 3, 4]
        completion_ids = [10 + len(self.calls), 20 + len(self.calls)]
        return Response(
            id=f"fake-{len(self.calls)}",
            created=0,
            model=model,
            usage=Usage(
                prompt_tokens=len(prompt_ids),
                reasoning_tokens=0,
                completion_tokens=len(completion_ids),
                total_tokens=len(prompt_ids) + len(completion_ids),
            ),
            message=ResponseMessage(
                content=text,
                reasoning_content=None,
                finish_reason="stop",
                is_truncated=False,
                tool_calls=None,
                tokens=ResponseTokens(
                    prompt_ids=prompt_ids,
                    prompt_mask=[0] * len(prompt_ids),
                    completion_ids=completion_ids,
                    completion_mask=[1] * len(completion_ids),
                    completion_logprobs=[-0.1, -0.2],
                ),
            ),
        )


async def test_v1_taskset_loads_as_v0_and_runs_rollout() -> None:
    env = vf.load_environment(
        "compat-taskset-v1",
        phrase="alpha",
        harness={"id": "compat-harness-v1"},
    )
    try:
        assert isinstance(env, V1AsV0Environment)
        row = env.get_dataset(n=1).to_list()[0]

        out = await env.run_rollout(
            row,
            client=FakeV0Client(),
            model="fake/model",
            sampling_args={"max_tokens": 8, "temperature": 0},
            state_columns=["trajectory"],
        )

        assert out["example_id"] == 0
        assert out["reward"] == 1.0
        assert out["metrics"]["turns"] == 1.0
        assert out["token_usage"]["final_input_tokens"] == 4.0
        assert out["token_usage"]["final_output_tokens"] == 2.0
        assert out["completion"][0]["content"] == "echo: alpha"
        assert out["trajectory"][0]["tokens"]["prompt_ids"] == [1, 2, 3, 4]
        assert out["trajectory"][0]["tokens"]["completion_ids"] == [11, 21]
        assert out["info"]["v1"]["task_idx"] == 0
    finally:
        await env._teardown()


async def test_v1_group_reward_runs_through_v0_run_group() -> None:
    env = vf.load_environment(
        "compat-group-taskset-v1",
        harness={"id": "compat-harness-v1"},
    )
    try:
        assert isinstance(env, V1AsV0Environment)
        assert env.rubric.has_group_rewards
        assert any(
            env.rubric._is_group_func(func) for func in env.rubric._get_reward_funcs()
        )
        row = env.get_dataset(n=1).to_list()[0]

        outs = await env.run_group(
            [row, row],
            client=FakeV0Client(),
            model="fake/model",
            sampling_args={"max_tokens": 8, "temperature": 0},
            state_columns=["trajectory"],
        )

        assert [out["reward"] for out in outs] == [2.0, 3.0]
        assert all(out["trajectory"] for out in outs)
    finally:
        await env._teardown()


async def test_v1_as_v0_set_kwargs_accepts_prime_rl_limit_sentinels() -> None:
    env = vf.load_environment(
        "compat-taskset-v1",
        phrase="alpha",
        harness={"id": "compat-harness-v1"},
    )
    try:
        assert isinstance(env, V1AsV0Environment)
        original_limits = env.v1_env.limits

        env.set_kwargs(
            max_total_completion_tokens=-1,
            max_seq_len=4096,
            max_input_tokens=2048,
            max_turns=0,
            timeout_seconds=123,
        )

        assert env.v1_env.limits is not original_limits
        assert env.v1_env.config.max_output_tokens is None
        assert env.v1_env.limits.max_output_tokens is None
        assert env.v1_env.config.max_total_tokens == 4096
        assert env.v1_env.limits.max_total_tokens == 4096
        assert env.v1_env.config.max_input_tokens == 2048
        assert env.v1_env.limits.max_input_tokens == 2048
        assert env.v1_env.config.max_turns is None
        assert env.v1_env.limits.max_turns is None
        assert env.v1_env.config.timeout.rollout == 123
        assert env.v1_env.harness_timeout == 123
    finally:
        await env._teardown()


async def test_v0_client_as_v1_offloads_chat_images_before_v0_client(
    monkeypatch,
) -> None:
    def fake_offload_images_inplace(value, *, image_dir=None):
        del image_dir
        stats = ImageOffloadStats()

        def visit(item):
            if isinstance(item, dict):
                if item.get("type") == "image_url":
                    source = item.get("image_url") or {}
                    url = source.get("url")
                    if isinstance(url, str) and url.startswith("data:image"):
                        source["url"] = "file:///tmp/vf-offloaded-image.png"
                        stats.images_rewritten += 1
                        stats.bytes_written += 100
                for child in item.values():
                    visit(child)
                return
            if isinstance(item, list | tuple):
                for child in item:
                    visit(child)
                return
            if getattr(item, "type", None) == "image_url":
                source = getattr(item, "image_url", None)
                url = getattr(source, "url", None)
                if isinstance(url, str) and url.startswith("data:image"):
                    source.url = "file:///tmp/vf-offloaded-image.png"
                    stats.images_rewritten += 1
                    stats.bytes_written += 100
                return
            content = getattr(item, "content", None)
            if isinstance(content, list | tuple):
                visit(content)

        visit(value)
        return stats

    monkeypatch.setattr(
        "verifiers.v1.compat.offload_images_inplace",
        fake_offload_images_inplace,
    )
    v0_client = FakeV0Client()
    client = V0ClientAsV1Client(v0_client)
    dialect = ChatDialect()
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,abc123"},
                    },
                ],
            }
        ]
    }

    prepared = await client.prepare_request_body(dialect, body)
    await client.get_response(
        dialect,
        prepared,
        model="fake/model",
        sampling_args=SamplingConfig(max_tokens=8),
    )

    prompt = v0_client.calls[0]["prompt"]
    image_source = prompt[0].content[1].image_url
    assert image_source.url == "file:///tmp/vf-offloaded-image.png"


def test_module_load_taskset_annotation_does_not_require_taskset_plugin() -> None:
    config = build_env_config(compat_taskset_v1, "standalone-compat-v1", {})

    assert type(config.taskset).__name__ == "CompatTasksetConfig"
    assert config.taskset.id == "standalone-compat-v1"


def _last_user_text(prompt) -> str:
    for message in reversed(prompt):
        role = getattr(message, "role", None)
        if role != "user":
            continue
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                str(part.get("text", "")) for part in content if isinstance(part, dict)
            )
    return ""
