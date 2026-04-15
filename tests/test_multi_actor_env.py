from __future__ import annotations

from typing import Any

import verifiers as vf


def test_multi_actor_types_construct():
    sample = vf.EpisodeSpec(
        base_example_id=1,
        episode_id="sample-0",
        input={"prompt": []},
    )
    member = vf.Member(
        member_id="member-a",
        role_id="debater",
        seat_id="A",
    )
    request = vf.TurnReq(
        episode_id="sample-0",
        member_id="member-a",
        turn_id="req-0",
        prompt=[],
        stop_sequences=["</answer>"],
    )
    response = vf.TurnResp(
        episode_id="sample-0",
        member_id="member-a",
        turn_id="req-0",
        content=[],
        token_count=42,
    )
    start = vf.EpisodeStart(
        episode=sample,
        members=[member],
        ready_turns=[request],
    )
    member_output = vf.MemberResult(
        member_id="member-a",
        role_id="debater",
        seat_id="A",
        trajectory=[],
    )
    output = vf.EpisodeResult(
        base_example_id=1,
        episode_id="sample-0",
        members=[member_output],
    )

    assert sample.episode_id == "sample-0"
    assert member.seat_id == "A"
    assert request.turn_id == "req-0"
    assert response.token_count == 42
    assert start.episode.episode_id == "sample-0"
    assert output.members[0].member_id == "member-a"


def test_multi_actor_env_protocol_is_runtime_checkable():
    class DummyMultiActorEnv:
        def start_episode(
            self,
            example: dict[str, Any],
            sample_index: int,
        ) -> vf.EpisodeStart:
            episode = vf.EpisodeSpec(
                base_example_id=example.get("example_id", 0),
                episode_id=f"sample-{sample_index}",
                input=example,
            )
            return vf.EpisodeStart(episode=episode, members=[], ready_turns=[])

        async def submit_ready_turns(
            self,
            responses: list[vf.TurnResp],
        ) -> list[vf.TurnReq]:
            return []

        async def finalize_episode(
            self,
            episode_id: str,
        ) -> vf.EpisodeResult:
            return vf.EpisodeResult(
                base_example_id=0,
                episode_id=episode_id,
                members=[],
            )

    dummy = DummyMultiActorEnv()
    assert isinstance(dummy, vf.MultiActorEnv)
