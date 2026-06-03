from verifiers.v1.toolsets.browser.keymap import ALT, CTRL, META, SHIFT, parse_chord


def test_named_key():
    mods, key = parse_chord("Return")
    assert mods == 0
    assert key.key == "Enter" and key.code == "Enter"


def test_single_letter():
    mods, key = parse_chord("a")
    assert mods == 0
    assert key.key == "a" and key.code == "KeyA" and key.key_code == ord("A")


def test_modifier_chord():
    mods, key = parse_chord("ctrl+s")
    assert mods == CTRL
    assert key.code == "KeyS"


def test_multiple_modifiers():
    mods, key = parse_chord("ctrl+shift+t")
    assert mods == (CTRL | SHIFT)
    assert key.code == "KeyT"


def test_aliases():
    assert parse_chord("cmd+a")[0] == META
    assert parse_chord("option+Tab")[0] == ALT
    assert parse_chord("control+c")[0] == CTRL


def test_navigation_keys():
    assert parse_chord("Page_Down")[1].key == "PageDown"
    assert parse_chord("shift+Tab") == (SHIFT, parse_chord("Tab")[1])


def test_bare_modifier():
    mods, key = parse_chord("shift")
    assert key.key == "Shift"


def test_unknown_raises():
    import pytest

    with pytest.raises(ValueError):
        parse_chord("")
