"""ARC-AGI-3 environment that wraps the official competition API."""

from __future__ import annotations

import asyncio
import json
import os
import textwrap
from typing import Any, Dict, Iterable, List, Optional

import httpx
from arc_agi_3_agents.structs import (  # type: ignore
    FrameData,
    GameAction,
    GameState,
    Scorecard,
)
from datasets import Dataset
from pydantic import BaseModel, Field, ValidationError, field_validator

import verifiers as vf
from verifiers.types import Messages, State

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are controlling an ARC-AGI-3 agent through JSON commands.
    Every assistant message MUST be a single JSON object with double-quoted keys.

    When playing the game:
    - Provide an action using {"action": <NAME>, "reasoning": ...}.
    - ACTION6 additionally requires integer fields "x" and "y" in [0, 63].
    - The environment calls the official ARC-AGI-3 API on your behalf and will
      return the next frame after every valid action.

    Once the environment reports that the game finished, respond with a final
    summary in the format
      {"final": {"state": <WIN|GAME_OVER>, "score": <int>, "summary": "...",
                  "actions": ["RESET", "ACTION1", ...]}}
    Do not emit any extra text before or after the JSON.
    """
)

DEFAULT_BASE_URL = "https://three.arcprize.org"
DEFAULT_GAME_ID = "ls20-fa137e247ce6"


class ArcAgi3APIError(RuntimeError):
    """Raised when the ARC-AGI-3 API returns an error payload."""


class ArcAgi3RateLimitError(ArcAgi3APIError):
    """Raised when the ARC-AGI-3 API rate limit is exceeded."""


def _coerce_game_action(value: Any) -> GameAction:
    """Convert raw API or model values into a ``GameAction`` instance."""

    if isinstance(value, GameAction):
        return value
    if isinstance(value, str):
        try:
            return GameAction[value.upper()]
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Unknown action name: {value}") from exc
    try:
        return GameAction(int(value))
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unknown action id: {value}") from exc


def _requires_coordinates(action: GameAction) -> bool:
    if hasattr(action, "is_complex"):
        return bool(action.is_complex())
    return action.name == "ACTION6"


class ActionCommand(BaseModel):
    action: str
    reasoning: Optional[Any] = None
    x: Optional[int] = Field(default=None, ge=0, le=63)
    y: Optional[int] = Field(default=None, ge=0, le=63)

    @field_validator("action", mode="before")
    @classmethod
    def _upper_action(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("action must be a string")
        return value.upper()


class FinalPayload(BaseModel):
    state: str
    score: Optional[int] = None
    summary: str
    actions: List[str] | None = None


class FinalMessage(BaseModel):
    final: FinalPayload


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Extract the first JSON object from the assistant message."""

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Expected a JSON object with braces in the response.")
    snippet = text[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError as exc:  # pragma: no cover - formatting guard
        raise ValueError(f"Invalid JSON: {exc.msg}") from exc


def _render_grid(frame: FrameData) -> str:
    if not frame.frame:
        return "(empty frame)"
    layers: List[str] = []
    for idx, grid in enumerate(frame.frame):
        rows = [" ".join(f"{cell:02d}" for cell in row) for row in grid]
        header = f"Layer {idx}:"
        layers.append("\n".join([header, *rows]))
    return "\n\n".join(layers)


def _format_available(actions: Iterable[GameAction]) -> str:
    names = [action.name for action in actions]
    return ", ".join(names) if names else "(no available actions)"


def _initial_instruction(game_id: str, max_actions: int) -> str:
    return textwrap.dedent(
        f"""
        Game `{game_id}` is ready. Start by issuing a RESET action. You may take up
        to {max_actions} actions before a summary is required.

        Reply with JSON only. Example: {{"action": "RESET", "reasoning": "start"}}
        After ACTION6 supply integer fields "x" and "y" (0-63).
        Wait for the environment to acknowledge each action before sending the next.
        """
    ).strip()


def _frame_update_message(
    frame: FrameData,
    action_name: str,
    actions_taken: int,
    max_actions: int,
) -> str:
    remaining = max(max_actions - actions_taken, 0)
    grid_text = _render_grid(frame)
    available = _format_available(frame.available_actions)
    return textwrap.dedent(
        f"""
        Game `{frame.game_id}` update after `{action_name}`:
        - Score: {frame.score}
        - State: {frame.state.value}
        - Actions used: {actions_taken} (remaining before summary: {remaining})
        - Next available actions: {available}

        Current frame:
        {grid_text}

        Respond with the next JSON action.
        """
    ).strip()


def _summary_prompt(
    frame: FrameData,
    actions_taken: int,
    scorecard_url: str,
    limit_reached: bool = False,
) -> str:
    status_line = (
        "Maximum action limit reached; finalize the session."
        if limit_reached
        else f"Game finished with state {frame.state.value}."
    )
    return textwrap.dedent(
        f"""
        {status_line}
        Final score: {frame.score}. Total actions sent: {actions_taken}.
        Scorecard URL: {scorecard_url}

        Provide the final JSON summary using the required format.
        """
    ).strip()


class ArcAgi3Client:
    """Thin wrapper around the official ARC-AGI-3 HTTP API."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float,
        max_retries: int = 3,
        initial_backoff: float = 1.0,
    ) -> None:
        base = base_url.rstrip("/")
        if not base:
            raise ValueError("A non-empty base_url is required")
        self._client = httpx.AsyncClient(
            base_url=base,
            headers={
                "X-API-Key": api_key,
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            timeout=timeout,
            cookies=httpx.Cookies(),
            follow_redirects=True,
        )
        self._base_url = base
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff

    @property
    def scorecard_url(self) -> str:
        return f"{self._base_url}/scorecards"

    async def open_scorecard(self, tags: List[str]) -> str:
        response = await self._request_with_retry(
            lambda: self._client.post("/api/scorecard/open", json={"tags": tags})
        )
        data = self._parse_response(response)
        return str(data["card_id"])

    async def close_scorecard(self, card_id: str) -> Dict[str, Any]:
        response = await self._request_with_retry(
            lambda: self._client.post("/api/scorecard/close", json={"card_id": card_id})
        )
        data = self._parse_response(response)
        if Scorecard is not None:
            try:
                return Scorecard.model_validate(data).model_dump()
            except ValidationError as exc:  # pragma: no cover - defensive
                raise ArcAgi3APIError(
                    f"ARC API returned an invalid scorecard payload: {exc}"
                ) from exc
        return data

    async def send_action(
        self,
        card_id: str,
        game_id: str,
        guid: Optional[str],
        action: GameAction,
        payload: ActionCommand,
    ) -> FrameData:
        body: Dict[str, Any] = {"game_id": game_id}
        if action is GameAction.RESET:
            body["card_id"] = card_id
        if guid:
            body["guid"] = guid
        if payload.reasoning is not None:
            body["reasoning"] = payload.reasoning
        if _requires_coordinates(action):
            if payload.x is None or payload.y is None:
                raise ValueError("ACTION6 requires both x and y fields")
            body["x"] = payload.x
            body["y"] = payload.y
        response = await self._request_with_retry(
            lambda: self._client.post(f"/api/cmd/{action.name}", json=body)
        )
        data = self._parse_response(response)
        frame = FrameData.model_validate(data)
        return frame

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _request_with_retry(self, request_func) -> httpx.Response:
        """Execute request with exponential backoff for transient failures."""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = await request_func()
                if response.status_code == 429:
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.initial_backoff * (2**attempt))
                        continue
                    raise ArcAgi3RateLimitError(
                        "Rate limit exceeded. API limit: 600 RPM"
                    )
                return response
            except httpx.RequestError as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.initial_backoff * (2**attempt))
                    continue
                raise ArcAgi3APIError(
                    f"Request failed after {self.max_retries} attempts: {exc}"
                ) from exc

        raise ArcAgi3APIError(f"Request failed: {last_error}") from last_error

    @staticmethod
    def _parse_response(response: httpx.Response) -> Dict[str, Any]:
        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise ArcAgi3APIError(
                f"ARC API returned non-JSON (status {response.status_code})"
            ) from exc
        if response.status_code == 429:
            error_msg = (
                data.get("message", "Rate limit exceeded")
                if isinstance(data, dict)
                else "Rate limit exceeded"
            )
            raise ArcAgi3RateLimitError(error_msg)
        if isinstance(data, dict) and data.get("error"):
            error_code = data.get("error")
            error_msg = data.get("message", str(error_code))
            raise ArcAgi3APIError(f"{error_code}: {error_msg}")
        if response.is_error:
            raise ArcAgi3APIError(
                f"HTTP {response.status_code}: {data if isinstance(data, dict) else 'error'}"
            )
        if not isinstance(data, dict):
            raise ArcAgi3APIError("Unexpected payload from ARC API")
        return data


class ArcAgi3Env(vf.MultiTurnEnv):
    """Multi-turn environment that mirrors the official ARC agent loop."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        dataset: Dataset,
        rubric: vf.Rubric,
        max_actions: int = 80,
        request_timeout: float = 10.0,
        tags: Optional[List[str]] = None,
        system_prompt: str = SYSTEM_PROMPT,
        few_shot: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> None:
        self.base_url = base_url.rstrip("/") or DEFAULT_BASE_URL
        self.api_key = api_key
        self.max_actions = max_actions
        self.request_timeout = request_timeout
        self.tags = sorted(tags or [])
        self._clients: Dict[int, ArcAgi3Client] = {}
        super().__init__(
            dataset=dataset,
            rubric=rubric,
            system_prompt=system_prompt,
            few_shot=few_shot or [],
            max_turns=max_actions + 5,
            **kwargs,
        )

    async def setup_state(self, state: State, **kwargs: Any) -> State:
        game_id = state.get("info", {}).get("game_id")
        if not game_id:
            raise ValueError("Dataset entries must include an info['game_id'] field")
        tag_values = sorted(set(self.tags + state.get("info", {}).get("tags", [])))
        client = ArcAgi3Client(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.request_timeout,
        )
        state_key = id(state)
        self._clients[state_key] = client
        scorecard_task = asyncio.create_task(client.open_scorecard(tag_values))

        arc_state = {
            "state_key": state_key,
            "card_id": None,
            "scorecard_task": scorecard_task,
            "scorecard_ready": False,
            "game_id": game_id,
            "guid": None,
            "frames": [],
            "actions": [],
            "phase": "awaiting_action",
            "allowed_actions": [GameAction.RESET.name],
            "final_state": None,
            "final_score": None,
            "final_frame": None,
            "final_report": None,
            "scorecard": None,
            "tags": tag_values,
            "actions_taken": 0,
            "max_actions": self.max_actions,
            "scorecard_url": None,
            "errors": [],
        }
        state["arc"] = arc_state
        prompt = state.get("prompt", [])
        if isinstance(prompt, list):
            prompt.append(
                {
                    "role": "user",
                    "content": _initial_instruction(game_id, self.max_actions),
                }
            )
        return state

    async def _ensure_scorecard_ready(self, arc_state: Dict[str, Any]) -> None:
        """Wait for scorecard to be ready (only awaits on first call)."""
        if arc_state.get("scorecard_ready"):
            return

        scorecard_task = arc_state.get("scorecard_task")
        if scorecard_task is None:
            raise ValueError("Scorecard task not found in state")

        try:
            card_id = await scorecard_task
            arc_state["card_id"] = card_id
            arc_state["scorecard_ready"] = True
            client = self._clients.get(arc_state["state_key"])
            if client:
                arc_state["scorecard_url"] = f"{client.scorecard_url}/{card_id}"
        except Exception as exc:
            raise ArcAgi3APIError(f"Failed to open scorecard: {exc}") from exc

    async def is_completed(
        self, messages: Messages, state: State, **kwargs: Any
    ) -> bool:
        arc = state.get("arc", {})
        return arc.get("phase") == "done"

    async def env_response(
        self, messages: Messages, state: State, **kwargs: Any
    ) -> tuple[Messages, State]:
        arc = state.get("arc", {})
        phase = arc.get("phase")
        if not phase:
            return [], state
        last_message = messages[-1]
        assert isinstance(last_message, dict)
        content = last_message.get("content")
        assert isinstance(content, str)
        try:
            payload_dict = _extract_json_object(content)
        except ValueError as exc:
            arc.setdefault("errors", []).append(str(exc))
            return (
                [
                    {
                        "role": "user",
                        "content": f"Response error: {exc}. Please reply with JSON only.",
                    }
                ],
                state,
            )

        if phase == "awaiting_action":
            try:
                action_payload = ActionCommand.model_validate(payload_dict)
            except ValidationError as exc:
                arc.setdefault("errors", []).append(exc.errors())
                return (
                    [
                        {
                            "role": "user",
                            "content": (
                                "Invalid action payload. Ensure fields match the schema. "
                                f"Details: {exc}"
                            ),
                        }
                    ],
                    state,
                )
            try:
                action = _coerce_game_action(action_payload.action)
            except ValueError as exc:
                arc.setdefault("errors", []).append(str(exc))
                return (
                    [
                        {
                            "role": "user",
                            "content": (
                                f"Unknown action '{action_payload.action}'. "
                                f"Allowed: {', '.join(arc.get('allowed_actions', []))}."
                            ),
                        }
                    ],
                    state,
                )
            if action.name not in arc.get("allowed_actions", [GameAction.RESET.name]):
                return (
                    [
                        {
                            "role": "user",
                            "content": (
                                f"Action {action.name} not allowed. "
                                f"Valid options: {', '.join(arc.get('allowed_actions', []))}."
                            ),
                        }
                    ],
                    state,
                )
            if arc["actions_taken"] >= arc["max_actions"]:
                arc["phase"] = "awaiting_summary"
                last_frame_data = arc.get("frames", [])[-1] if arc.get("frames") else {}
                frame = (
                    FrameData.model_validate(last_frame_data)
                    if last_frame_data
                    else FrameData(game_id=arc["game_id"])
                )
                if not arc.get("scorecard_ready"):
                    await self._ensure_scorecard_ready(arc)
                summary = _summary_prompt(
                    frame,
                    arc["actions_taken"],
                    arc["scorecard_url"],
                    limit_reached=True,
                )
                return ([{"role": "user", "content": summary}], state)

            if not arc.get("scorecard_ready"):
                await self._ensure_scorecard_ready(arc)

            client = self._clients.get(arc["state_key"])
            if client is None:  # pragma: no cover - defensive guard
                arc.setdefault("errors", []).append("Missing client for state")
                return (
                    [
                        {
                            "role": "user",
                            "content": "Internal error: client not initialised.",
                        }
                    ],
                    state,
                )
            try:
                frame = await client.send_action(
                    card_id=arc["card_id"],
                    game_id=arc["game_id"],
                    guid=arc.get("guid"),
                    action=action,
                    payload=action_payload,
                )
            except (ArcAgi3APIError, ValueError) as exc:
                arc.setdefault("errors", []).append(str(exc))
                return (
                    [
                        {
                            "role": "user",
                            "content": (
                                f"API error while performing {action.name}: {exc}. "
                                "Please choose a different action or RESET."
                            ),
                        }
                    ],
                    state,
                )
            arc["guid"] = frame.guid
            arc["actions_taken"] += 1
            arc["allowed_actions"] = [act.name for act in frame.available_actions]
            arc.setdefault("frames", []).append(frame.model_dump())
            arc.setdefault("actions", []).append(
                {"action": action.name, **action_payload.model_dump(exclude_none=True)}
            )
            arc["last_state"] = frame.state.value
            arc["last_score"] = frame.score
            arc.setdefault("metrics", {})["final_score"] = frame.score
            arc.setdefault("metrics", {})["action_count"] = arc["actions_taken"]

            if frame.state in {GameState.WIN, GameState.GAME_OVER}:
                arc["phase"] = "awaiting_summary"
                arc["final_state"] = frame.state.value
                arc["final_score"] = frame.score
                arc["final_frame"] = frame.model_dump()
                summary_message = _summary_prompt(
                    frame,
                    arc["actions_taken"],
                    arc["scorecard_url"],
                )
                return ([{"role": "user", "content": summary_message}], state)

            update = _frame_update_message(
                frame=frame,
                action_name=action.name,
                actions_taken=arc["actions_taken"],
                max_actions=arc["max_actions"],
            )
            return ([{"role": "user", "content": update}], state)

        if phase == "awaiting_summary":
            try:
                summary_payload = FinalMessage.model_validate(payload_dict)
            except ValidationError as exc:
                arc.setdefault("errors", []).append(exc.errors())
                return (
                    [
                        {
                            "role": "user",
                            "content": (
                                'Invalid final summary. Expected {"final": {...}}. '
                                f"Details: {exc}"
                            ),
                        }
                    ],
                    state,
                )
            arc["final_report"] = summary_payload.model_dump()
            client = self._clients.pop(arc["state_key"], None)
            if client is not None:
                try:
                    scorecard = await client.close_scorecard(arc["card_id"])
                    arc["scorecard"] = scorecard
                except ArcAgi3APIError as exc:
                    arc.setdefault("errors", []).append(str(exc))
                finally:
                    await client.aclose()
            arc["phase"] = "done"
            return ([], state)

        return ([], state)


def _build_dataset(games: List[Dict[str, Any]]) -> Dataset:
    questions: List[str] = []
    infos: List[Dict[str, Any]] = []
    for game in games:
        game_id = game.get("game_id")
        if not game_id:
            raise ValueError("Each game entry must include a 'game_id'")
        prompt = game.get(
            "prompt",
            f"Play ARC-AGI-3 game '{game_id}'. Wait for the board updates before responding.",
        )
        questions.append(prompt)
        info: Dict[str, Any] = {"game_id": game_id}
        if "tags" in game:
            info["tags"] = list(game["tags"])
        infos.append(info)
    return Dataset.from_dict({"question": questions, "info": infos})


def _normalize_games(games: Optional[Iterable[Any]]) -> List[Dict[str, Any]]:
    if not games:
        return [{"game_id": DEFAULT_GAME_ID}]
    normalized: List[Dict[str, Any]] = []
    for entry in games:
        if isinstance(entry, str):
            normalized.append({"game_id": entry})
        elif isinstance(entry, dict):
            normalized.append(entry)
        else:  # pragma: no cover - defensive
            raise TypeError("games must be a list of strings or dictionaries")
    return normalized


async def success(state: State) -> float:
    arc = state.get("arc", {})
    return 1.0 if arc.get("final_state") == GameState.WIN.value else 0.0


DEFAULT_GAME_IDS = ["ls20-fa137e247ce6"]


def load_environment(
    *,
    games: Iterable[Any] = DEFAULT_GAME_IDS,
    max_actions: int = 5,
    request_timeout: float = 3.0,
) -> vf.Environment:
    """Factory for the ARC-AGI-3 environment."""

    game_entries = _normalize_games(games)
    dataset = _build_dataset(game_entries)
    api_key = os.getenv("ARC_API_KEY")
    if not api_key:
        raise ValueError("ARC_API_KEY is required to call the ARC-AGI-3 API")
    rubric = vf.Rubric()
    rubric.add_reward_func(success, weight=1.0)
    env = ArcAgi3Env(
        base_url=DEFAULT_BASE_URL,
        api_key=api_key,
        dataset=dataset,
        rubric=rubric,
        max_actions=max_actions,
        request_timeout=request_timeout,
        system_prompt=SYSTEM_PROMPT,
    )
    return env
