"""Tests for the arc-agi-3 environment."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from arc_agi_3_agents.structs import FrameData, GameAction, GameState, Scorecard

from verifiers.types import State


# Mock the entire module to avoid import errors if not installed
@pytest.fixture
def mock_arc_env_module():
    """Mock the arc-agi-3 module."""
    with patch("sys.modules") as mock_modules:
        yield mock_modules


@pytest.fixture
def sample_frame():
    """Create a sample FrameData for testing."""
    return FrameData(
        game_id="test_game",
        guid="test_guid",
        frame=[[[0, 1], [2, 3]]],
        score=100,
        state=GameState.NOT_FINISHED,
        available_actions=[GameAction.ACTION1, GameAction.ACTION2],
    )


@pytest.fixture
def mock_arc_client():
    """Create a mock ARC API client."""
    client = AsyncMock()
    client.scorecard_url = "https://test.arcprize.org/scorecards"
    client.open_scorecard = AsyncMock(return_value="test_card_id")
    client.close_scorecard = AsyncMock(
        return_value={"card_id": "test_card_id", "total_score": 100}
    )
    client.aclose = AsyncMock()
    return client


class TestArcAgi3Environment:
    """Test cases for ARC-AGI-3 environment."""

    @pytest.mark.asyncio
    async def test_environment_loads(self):
        """Test that the arc-agi-3 environment can be loaded."""
        import sys
        from pathlib import Path

        # Add the environment to the path
        env_path = Path(__file__).parent.parent / "environments" / "arc_agi_3"
        sys.path.insert(0, str(env_path))

        try:
            from arc_agi_3 import load_environment

            # Mock the API key
            with patch.dict("os.environ", {"ARC_API_KEY": "test_key"}):
                env = load_environment(max_actions=5)
                assert env is not None
                assert len(env.dataset) > 0
                assert hasattr(env, "max_actions")
                assert env.max_actions == 5
        finally:
            sys.path.pop(0)

    @pytest.mark.asyncio
    async def test_environment_requires_api_key(self):
        """Test that environment raises error without API key."""
        import sys
        from pathlib import Path

        env_path = Path(__file__).parent.parent / "environments" / "arc_agi_3"
        sys.path.insert(0, str(env_path))

        try:
            from arc_agi_3 import load_environment

            with patch.dict("os.environ", {"ARC_API_KEY": ""}, clear=True):
                with pytest.raises(ValueError, match="ARC_API_KEY is required"):
                    load_environment()
        finally:
            sys.path.pop(0)

    @pytest.mark.asyncio
    async def test_game_normalization(self):
        """Test that games are normalized correctly."""
        import sys
        from pathlib import Path

        env_path = Path(__file__).parent.parent / "environments" / "arc_agi_3"
        sys.path.insert(0, str(env_path))

        try:
            from arc_agi_3 import _normalize_games

            # Test with string list
            result = _normalize_games(["game1", "game2"])
            assert len(result) == 2
            assert result[0] == {"game_id": "game1"}
            assert result[1] == {"game_id": "game2"}

            # Test with dict list
            result = _normalize_games([{"game_id": "game1", "tags": ["test"]}])
            assert len(result) == 1
            assert result[0]["game_id"] == "game1"
            assert result[0]["tags"] == ["test"]

            # Test with None/empty
            result = _normalize_games(None)
            assert len(result) == 1  # Should return default game
            assert "game_id" in result[0]
        finally:
            sys.path.pop(0)

    @pytest.mark.asyncio
    async def test_action_command_validation(self):
        """Test ActionCommand validation."""
        import sys
        from pathlib import Path

        env_path = Path(__file__).parent.parent / "environments" / "arc_agi_3"
        sys.path.insert(0, str(env_path))

        try:
            from arc_agi_3 import ActionCommand

            # Valid action
            cmd = ActionCommand(action="reset", reasoning="Starting game")
            assert cmd.action == "RESET"  # Should be uppercased

            # Action with coordinates
            cmd = ActionCommand(action="action6", x=10, y=20, reasoning="Move")
            assert cmd.action == "ACTION6"
            assert cmd.x == 10
            assert cmd.y == 20

            # Invalid coordinates (out of range)
            with pytest.raises(Exception):  # Pydantic validation error
                ActionCommand(action="action6", x=100, y=20)
        finally:
            sys.path.pop(0)

    @pytest.mark.asyncio
    async def test_json_extraction(self):
        """Test JSON extraction from responses."""
        import sys
        from pathlib import Path

        env_path = Path(__file__).parent.parent / "environments" / "arc_agi_3"
        sys.path.insert(0, str(env_path))

        try:
            from arc_agi_3 import _extract_json_object

            # Valid JSON
            text = 'Some text {"action": "RESET", "reasoning": "test"} more text'
            result = _extract_json_object(text)
            assert result["action"] == "RESET"

            # No JSON
            with pytest.raises(ValueError):
                _extract_json_object("No JSON here")

            # Invalid JSON
            with pytest.raises(ValueError):
                _extract_json_object("{ invalid json }")
        finally:
            sys.path.pop(0)

    @pytest.mark.asyncio
    async def test_success_reward_function(self):
        """Test the success reward function."""
        import sys
        from pathlib import Path

        env_path = Path(__file__).parent.parent / "environments" / "arc_agi_3"
        sys.path.insert(0, str(env_path))

        try:
            from arc_agi_3 import success

            # Win state
            state: State = {"arc": {"final_state": "WIN"}}
            reward = await success(state)
            assert reward == 1.0

            # Game over state
            state = {"arc": {"final_state": "GAME_OVER"}}
            reward = await success(state)
            assert reward == 0.0

            # No state
            state = {"arc": {}}
            reward = await success(state)
            assert reward == 0.0
        finally:
            sys.path.pop(0)

    @pytest.mark.asyncio
    async def test_environment_initialization(self, mock_arc_client):
        """Test ArcAgi3Env initialization."""
        import sys
        from pathlib import Path

        env_path = Path(__file__).parent.parent / "environments" / "arc_agi_3"
        sys.path.insert(0, str(env_path))

        try:
            from datasets import Dataset

            from arc_agi_3 import ArcAgi3Env
            from verifiers import Rubric

            dataset = Dataset.from_dict(
                {"question": ["Play game"], "info": [{"game_id": "test_game"}]}
            )
            rubric = Rubric()

            with patch.dict("os.environ", {"ARC_API_KEY": "test_key"}):
                env = ArcAgi3Env(
                    base_url="https://test.arcprize.org",
                    api_key="test_key",
                    dataset=dataset,
                    rubric=rubric,
                    max_actions=10,
                )
                assert env.max_actions == 10
                assert env.base_url == "https://test.arcprize.org"
                assert env.api_key == "test_key"
        finally:
            sys.path.pop(0)

    def test_frame_rendering(self):
        """Test grid rendering."""
        import sys
        from pathlib import Path

        env_path = Path(__file__).parent.parent / "environments" / "arc_agi_3"
        sys.path.insert(0, str(env_path))

        try:
            from arc_agi_3 import FrameData, _render_grid

            # Create a simple frame
            frame = FrameData(
                game_id="test",
                frame=[[[0, 1], [2, 3]]],
                score=0,
                state=GameState.NOT_FINISHED,
            )
            result = _render_grid(frame)
            assert "Layer 0:" in result
            assert "00 01" in result
            assert "02 03" in result

            # Empty frame
            frame = FrameData(game_id="test", frame=[], score=0, state=GameState.NOT_FINISHED)
            result = _render_grid(frame)
            assert result == "(empty frame)"
        finally:
            sys.path.pop(0)

