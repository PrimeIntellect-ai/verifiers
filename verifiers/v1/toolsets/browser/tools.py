import asyncio
from typing import Annotated, Literal

from pydantic import Field

from .session import BrowserSession, ScreenshotContent

Coordinate = Annotated[
    list[int], Field(description="[x, y] pixel coordinate in the viewport.")
]

_MAX_WAIT_SECONDS = 10.0


def _xy(coordinate: list[int] | None, name: str = "coordinate") -> tuple[int, int]:
    if not coordinate or len(coordinate) != 2:
        raise ValueError(f"{name} must be an [x, y] pair.")
    return int(coordinate[0]), int(coordinate[1])


ComputerAction = Literal[
    "key",
    "type",
    "mouse_move",
    "left_click",
    "left_click_drag",
    "right_click",
    "middle_click",
    "double_click",
    "triple_click",
    "scroll",
    "cursor_position",
    "screenshot",
    "wait",
]


async def computer(
    action: Annotated[
        ComputerAction,
        Field(description="The computer-use action to perform."),
    ],
    coordinate: Annotated[
        list[int] | None,
        Field(description="[x, y] target for click/move/scroll actions."),
    ] = None,
    text: Annotated[
        str | None,
        Field(description="Text to type, or key chord (e.g. 'ctrl+s') to press."),
    ] = None,
    start_coordinate: Annotated[
        list[int] | None,
        Field(description="[x, y] start point for left_click_drag."),
    ] = None,
    scroll_direction: Annotated[
        Literal["up", "down", "left", "right"] | None,
        Field(description="Scroll direction."),
    ] = None,
    scroll_amount: Annotated[
        int | None,
        Field(description="Number of scroll steps."),
    ] = None,
    duration: Annotated[
        float | None,
        Field(description="Seconds to wait for the 'wait' action."),
    ] = None,
    *,
    session: BrowserSession,
) -> ScreenshotContent | str:
    """Control the browser via mouse, keyboard, scrolling and screenshots (viewport pixels)."""
    if action == "screenshot":
        return await session.screenshot()
    if action == "cursor_position":
        x, y = session.cursor_position
        return f"{x}, {y}"
    if action == "wait":
        await asyncio.sleep(min(float(duration or 1.0), _MAX_WAIT_SECONDS))
        return await session.screenshot()
    if action == "mouse_move":
        await session.move_mouse(*_xy(coordinate))
        return await session.screenshot()
    if action == "key":
        if not text:
            raise ValueError("key action requires 'text' (the key chord).")
        await session.press_key(text)
        return await session.screenshot()
    if action == "type":
        if text is None:
            raise ValueError("type action requires 'text'.")
        await session.type_text(text)
        return await session.screenshot()
    if action == "left_click_drag":
        await session.drag(_xy(start_coordinate, "start_coordinate"), _xy(coordinate))
        return await session.screenshot()
    if action == "scroll":
        x, y = _xy(coordinate) if coordinate else session.cursor_position
        await session.scroll(scroll_direction or "down", scroll_amount or 3, x, y)
        return await session.screenshot()
    # Click family.
    button = {"left_click": "left", "right_click": "right", "middle_click": "middle"}
    counts = {"double_click": 2, "triple_click": 3}
    if action in button:
        cx, cy = _xy(coordinate) if coordinate else session.cursor_position
        await session.click(cx, cy, button=button[action])
        return await session.screenshot()
    if action in counts:
        cx, cy = _xy(coordinate) if coordinate else session.cursor_position
        await session.click(cx, cy, button="left", count=counts[action])
        return await session.screenshot()
    raise ValueError(f"Unknown action: {action!r}")


async def screenshot(*, session: BrowserSession) -> ScreenshotContent:
    """Capture and return a screenshot of the current viewport."""
    return await session.screenshot()


async def navigate(
    url: Annotated[str, Field(description="URL to load.")],
    *,
    session: BrowserSession,
) -> ScreenshotContent:
    """Navigate the browser to a URL and return a screenshot."""
    await session.navigate(url)
    return await session.screenshot()


async def mouse_move(
    coordinate: Coordinate, *, session: BrowserSession
) -> ScreenshotContent:
    """Move the pointer to a coordinate."""
    await session.move_mouse(*_xy(coordinate))
    return await session.screenshot()


async def left_click(
    coordinate: Coordinate, *, session: BrowserSession
) -> ScreenshotContent:
    """Left-click at a coordinate."""
    await session.click(*_xy(coordinate), button="left")
    return await session.screenshot()


async def right_click(
    coordinate: Coordinate, *, session: BrowserSession
) -> ScreenshotContent:
    """Right-click at a coordinate."""
    await session.click(*_xy(coordinate), button="right")
    return await session.screenshot()


async def middle_click(
    coordinate: Coordinate, *, session: BrowserSession
) -> ScreenshotContent:
    """Middle-click at a coordinate."""
    await session.click(*_xy(coordinate), button="middle")
    return await session.screenshot()


async def double_click(
    coordinate: Coordinate, *, session: BrowserSession
) -> ScreenshotContent:
    """Double-click at a coordinate."""
    await session.click(*_xy(coordinate), button="left", count=2)
    return await session.screenshot()


async def triple_click(
    coordinate: Coordinate, *, session: BrowserSession
) -> ScreenshotContent:
    """Triple-click at a coordinate."""
    await session.click(*_xy(coordinate), button="left", count=3)
    return await session.screenshot()


async def left_click_drag(
    start_coordinate: Annotated[list[int], Field(description="[x, y] drag start.")],
    coordinate: Coordinate,
    *,
    session: BrowserSession,
) -> ScreenshotContent:
    """Press at the start coordinate, drag to the end coordinate and release."""
    await session.drag(_xy(start_coordinate, "start_coordinate"), _xy(coordinate))
    return await session.screenshot()


async def type_text(
    text: Annotated[str, Field(description="Text to type at the current focus.")],
    *,
    session: BrowserSession,
) -> ScreenshotContent:
    """Type literal text into the focused element."""
    await session.type_text(text)
    return await session.screenshot()


async def key(
    text: Annotated[
        str, Field(description="Key chord, e.g. 'Return', 'ctrl+s', 'shift+Tab'.")
    ],
    *,
    session: BrowserSession,
) -> ScreenshotContent:
    """Press a key or key chord."""
    await session.press_key(text)
    return await session.screenshot()


async def scroll(
    coordinate: Coordinate,
    scroll_direction: Annotated[
        Literal["up", "down", "left", "right"], Field(description="Scroll direction.")
    ],
    scroll_amount: Annotated[int, Field(description="Number of scroll steps.")] = 3,
    *,
    session: BrowserSession,
) -> ScreenshotContent:
    """Scroll the page at a coordinate in the given direction."""
    x, y = _xy(coordinate)
    await session.scroll(scroll_direction, scroll_amount, x, y)
    return await session.screenshot()


async def wait(
    duration: Annotated[float, Field(description="Seconds to wait.")] = 1.0,
    *,
    session: BrowserSession,
) -> ScreenshotContent:
    """Wait for the page to settle, then return a screenshot."""
    await asyncio.sleep(min(float(duration), _MAX_WAIT_SECONDS))
    return await session.screenshot()


async def cursor_position(*, session: BrowserSession) -> str:
    """Return the current pointer position as 'x, y'."""
    x, y = session.cursor_position
    return f"{x}, {y}"


DECOMPOSED_TOOLS = [
    screenshot,
    navigate,
    mouse_move,
    left_click,
    right_click,
    middle_click,
    double_click,
    triple_click,
    left_click_drag,
    type_text,
    key,
    scroll,
    wait,
    cursor_position,
]
