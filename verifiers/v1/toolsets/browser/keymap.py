from dataclasses import dataclass

# CDP modifier bitmask.
ALT = 1
CTRL = 2
META = 4
SHIFT = 8

_MODIFIER_ALIASES: dict[str, int] = {
    "alt": ALT,
    "option": ALT,
    "ctrl": CTRL,
    "control": CTRL,
    "meta": META,
    "super": META,
    "cmd": META,
    "command": META,
    "win": META,
    "shift": SHIFT,
}


@dataclass(frozen=True)
class KeyDef:
    key: str
    code: str
    key_code: int


# Standalone modifier keys, used when a chord is just a modifier name.
_MODIFIER_KEYDEFS: dict[int, KeyDef] = {
    ALT: KeyDef("Alt", "AltLeft", 18),
    CTRL: KeyDef("Control", "ControlLeft", 17),
    META: KeyDef("Meta", "MetaLeft", 91),
    SHIFT: KeyDef("Shift", "ShiftLeft", 16),
}


# Named (non-printable) keys, keyed by lowercased xdotool/X11 name.
_NAMED_KEYS: dict[str, KeyDef] = {
    "return": KeyDef("Enter", "Enter", 13),
    "enter": KeyDef("Enter", "Enter", 13),
    "kp_enter": KeyDef("Enter", "NumpadEnter", 13),
    "tab": KeyDef("Tab", "Tab", 9),
    "space": KeyDef(" ", "Space", 32),
    "backspace": KeyDef("Backspace", "Backspace", 8),
    "delete": KeyDef("Delete", "Delete", 46),
    "escape": KeyDef("Escape", "Escape", 27),
    "esc": KeyDef("Escape", "Escape", 27),
    "home": KeyDef("Home", "Home", 36),
    "end": KeyDef("End", "End", 35),
    "page_up": KeyDef("PageUp", "PageUp", 33),
    "prior": KeyDef("PageUp", "PageUp", 33),
    "page_down": KeyDef("PageDown", "PageDown", 34),
    "next": KeyDef("PageDown", "PageDown", 34),
    "left": KeyDef("ArrowLeft", "ArrowLeft", 37),
    "up": KeyDef("ArrowUp", "ArrowUp", 38),
    "right": KeyDef("ArrowRight", "ArrowRight", 39),
    "down": KeyDef("ArrowDown", "ArrowDown", 40),
    "insert": KeyDef("Insert", "Insert", 45),
    "minus": KeyDef("-", "Minus", 189),
    "plus": KeyDef("+", "Equal", 187),
    "equal": KeyDef("=", "Equal", 187),
    "period": KeyDef(".", "Period", 190),
    "comma": KeyDef(",", "Comma", 188),
    "slash": KeyDef("/", "Slash", 191),
    "backslash": KeyDef("\\", "Backslash", 220),
    "semicolon": KeyDef(";", "Semicolon", 186),
}
for _n in range(1, 13):  # Function keys F1-F12.
    _NAMED_KEYS[f"f{_n}"] = KeyDef(f"F{_n}", f"F{_n}", 111 + _n)


def _printable_keydef(char: str) -> KeyDef | None:
    if len(char) != 1:
        return None
    if char.isalpha():
        return KeyDef(char, f"Key{char.upper()}", ord(char.upper()))
    if char.isdigit():
        return KeyDef(char, f"Digit{char}", ord(char))
    return KeyDef(char, "", ord(char))


def parse_chord(chord: str) -> tuple[int, KeyDef]:
    """Parse an xdotool-style chord into (modifier_mask, KeyDef)."""
    parts = [part for part in chord.split("+") if part]
    if not parts:
        raise ValueError(f"Empty key chord: {chord!r}")
    # Leading tokens are modifiers; the final token is the primary key.
    modifiers = 0
    for part in parts[:-1]:
        alias = _MODIFIER_ALIASES.get(part.lower())
        if alias is None:
            raise ValueError(f"Unrecognized modifier {part!r} in chord {chord!r}")
        modifiers |= alias
    last = parts[-1]
    token = last.lower()
    if token in _NAMED_KEYS:
        primary = _NAMED_KEYS[token]
    elif token in _MODIFIER_ALIASES:
        # The chord is (or ends in) a bare modifier, e.g. ``shift``.
        primary = _MODIFIER_KEYDEFS[_MODIFIER_ALIASES[token]]
    else:
        primary = _printable_keydef(last)
    if primary is None:
        raise ValueError(f"Unrecognized key chord: {chord!r}")
    return modifiers, primary
