import asyncio

from verifiers.types import ContentPart
from verifiers.v1.types import ConfigData

from .backends import BrowserBackend
from .cdp import CDPClient, CDPError
from .keymap import parse_chord

# A tool returns message content; a screenshot is a single image content part.
ScreenshotContent = list[ContentPart]


def _screenshot_content(b64_png: str) -> ScreenshotContent:
    return [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64_png}"},
        }
    ]


class BrowserSession:
    """A backend-provisioned CDP page plus the input/scroll/screenshot primitives."""

    def __init__(
        self,
        backend: BrowserBackend,
        *,
        width: int = 1280,
        height: int = 800,
        start_url: str | None = None,
    ):
        self.backend = backend
        self.width = width
        self.height = height
        self.start_url = start_url
        self._client: CDPClient | None = None
        self._provider_session_id: str | None = None
        self._target_id: str | None = None
        self._session_id: str | None = None
        self._cursor = (width // 2, height // 2)

    async def start(self) -> "BrowserSession":
        """Provision, connect, and attach to a fresh page target. Idempotent."""
        if self._client is not None:
            return self
        handle = await self.backend.create()
        self._provider_session_id = handle.session_id
        self._client = CDPClient(handle.cdp_ws_url)
        await self._client.connect()
        await self._attach_page(create=True)
        if self.start_url:
            await self.navigate(self.start_url)
        return self

    async def _attach_page(self, *, create: bool) -> None:
        # create=True opens a fresh page; otherwise re-attach to the newest live
        # page (recovering when the current one was replaced or closed).
        assert self._client is not None
        target_id: str | None = None
        if not create:
            targets = await self._client.send("Target.getTargets")
            infos = targets.get("targetInfos")
            pages = (
                [t for t in infos if isinstance(t, dict) and t.get("type") == "page"]
                if isinstance(infos, list)
                else []
            )
            if pages:
                target_id = str(pages[-1].get("targetId"))
        if target_id is None:
            created = await self._client.send(
                "Target.createTarget", {"url": "about:blank"}
            )
            target_id = str(created["targetId"])
        attached = await self._client.send(
            "Target.attachToTarget", {"targetId": target_id, "flatten": True}
        )
        self._target_id = target_id
        self._session_id = str(attached["sessionId"])
        # Prepare the freshly attached page. These three commands are
        # independent, so pipeline them rather than paying three round-trips.
        await asyncio.gather(
            self._client.send("Page.enable", session_id=self._session_id),
            self._client.send("Runtime.enable", session_id=self._session_id),
            self._client.send(
                "Emulation.setDeviceMetricsOverride",
                {
                    "width": self.width,
                    "height": self.height,
                    "deviceScaleFactor": 1,
                    "mobile": False,
                },
                session_id=self._session_id,
            ),
        )

    async def aclose(self) -> None:
        if self._client is not None:
            if self._target_id is not None:
                try:
                    await self._client.send(
                        "Target.closeTarget", {"targetId": self._target_id}
                    )
                except Exception:  # noqa: BLE001 - best effort
                    pass
            await self._client.close()
            self._client = None
        if self._provider_session_id is not None:
            await self.backend.close(self._provider_session_id)
            self._provider_session_id = None
        self._target_id = None
        self._session_id = None

    async def _ensure(self) -> CDPClient:
        if self._client is None:
            await self.start()
        assert self._client is not None
        return self._client

    @staticmethod
    def _is_detached_error(error: CDPError) -> bool:
        text = str(error).lower()
        return any(
            marker in text
            for marker in (
                "not attached",
                "no target with given id",
                "session with given id",
                "target closed",
                "navigated or closed",
                "cannot find context",
            )
        )

    async def _send(self, method: str, params: ConfigData | None = None) -> ConfigData:
        """Send a page-scoped CDP command, recovering once if the page detached."""
        client = await self._ensure()
        try:
            return await client.send(method, params, session_id=self._session_id)
        except CDPError as error:
            if not self._is_detached_error(error):
                raise
        # The page went away; re-attach to a live one and retry exactly once.
        await self._attach_page(create=False)
        return await client.send(method, params, session_id=self._session_id)

    @property
    def cursor_position(self) -> tuple[int, int]:
        return self._cursor

    async def move_mouse(self, x: int, y: int) -> None:
        self._cursor = (int(x), int(y))
        await self._send(
            "Input.dispatchMouseEvent", {"type": "mouseMoved", "x": int(x), "y": int(y)}
        )

    async def click(
        self,
        x: int | None = None,
        y: int | None = None,
        *,
        button: str = "left",
        count: int = 1,
    ) -> None:
        if x is not None and y is not None:
            self._cursor = (int(x), int(y))
        cx, cy = self._cursor
        for press in range(1, count + 1):
            base = {"x": cx, "y": cy, "button": button, "clickCount": press}
            await self._send(
                "Input.dispatchMouseEvent", {"type": "mousePressed", **base}
            )
            await self._send(
                "Input.dispatchMouseEvent", {"type": "mouseReleased", **base}
            )

    async def drag(
        self, start: tuple[int, int], end: tuple[int, int], *, button: str = "left"
    ) -> None:
        sx, sy = int(start[0]), int(start[1])
        ex, ey = int(end[0]), int(end[1])
        await self._send(
            "Input.dispatchMouseEvent",
            {
                "type": "mousePressed",
                "x": sx,
                "y": sy,
                "button": button,
                "clickCount": 1,
            },
        )
        await self._send(
            "Input.dispatchMouseEvent",
            {"type": "mouseMoved", "x": ex, "y": ey, "button": button},
        )
        await self._send(
            "Input.dispatchMouseEvent",
            {
                "type": "mouseReleased",
                "x": ex,
                "y": ey,
                "button": button,
                "clickCount": 1,
            },
        )
        self._cursor = (ex, ey)

    async def scroll(
        self,
        direction: str,
        amount: int = 3,
        x: int | None = None,
        y: int | None = None,
    ) -> None:
        if x is not None and y is not None:
            self._cursor = (int(x), int(y))
        cx, cy = self._cursor
        step = 100 * int(amount)  # ~one notch per unit, like a wheel tick.
        delta_x = {"left": -step, "right": step}.get(direction, 0)
        delta_y = {"up": -step, "down": step}.get(direction, 0)
        await self._send(
            "Input.dispatchMouseEvent",
            {
                "type": "mouseWheel",
                "x": cx,
                "y": cy,
                "deltaX": delta_x,
                "deltaY": delta_y,
            },
        )

    async def type_text(self, text: str) -> None:
        await self._send("Input.insertText", {"text": text})

    async def press_key(self, chord: str) -> None:
        modifiers, key = parse_chord(chord)
        event: ConfigData = {
            "modifiers": modifiers,
            "key": key.key,
            "code": key.code,
            "windowsVirtualKeyCode": key.key_code,
            "nativeVirtualKeyCode": key.key_code,
        }
        down = {"type": "keyDown", **event}
        # Printable keys with no command modifier should also produce text.
        if len(key.key) == 1 and not (modifiers & 0b0110):  # not ctrl/meta
            down["text"] = key.key
        await self._send("Input.dispatchKeyEvent", down)
        await self._send("Input.dispatchKeyEvent", {"type": "keyUp", **event})

    async def navigate(self, url: str) -> None:
        await self._send("Page.navigate", {"url": url})

    async def screenshot(self) -> ScreenshotContent:
        result = await self._send(
            "Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False}
        )
        return _screenshot_content(str(result["data"]))


async def open_browser_session(
    backend: BrowserBackend,
    width: int = 1280,
    height: int = 800,
    start_url: str | None = None,
) -> BrowserSession:
    """Toolset object factory: build and start a rollout browser session."""
    session = BrowserSession(backend, width=width, height=height, start_url=start_url)
    await session.start()
    return session
