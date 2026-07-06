"""The eval dashboard `Pager`: which page of overflowing rollout rows is on screen.

Auto-advance is anchored to the first paged frame, so paging always opens on page 1 and rotates
from there, and the arrows are inert until more than one page exists.
"""

from verifiers.v1.cli.dashboard.eval import _PAGE_SECONDS, Pager


def test_opens_on_first_page_regardless_of_wall_clock() -> None:
    # The first paged frame is page 1 (index 0) whatever the wall clock reads — the timer is
    # anchored to that frame, not to `now % count`, which would open on an arbitrary page.
    for now in (1000.0, 1002.5, 1005.0, 1007.5, 1_751_000_006.0):
        pager = Pager()
        pager.count = 3
        assert pager.index(now) == 0


def test_auto_advances_then_wraps() -> None:
    pager = Pager()
    pager.count = 3
    assert pager.index(100.0) == 0  # first frame anchors the timer
    assert pager.index(100.0 + _PAGE_SECONDS) == 1
    assert pager.index(100.0 + 2 * _PAGE_SECONDS) == 2
    assert pager.index(100.0 + 3 * _PAGE_SECONDS) == 0  # wraps back around


def test_arrows_inert_until_more_than_one_page() -> None:
    pager = Pager()
    pager.count = 1  # everything fits — a stray arrow must not switch off auto-advance
    pager.on_key("right")
    assert pager.manual is False
    assert pager.index(123.0) == 0


def test_manual_takeover_continues_from_current_page_and_clamps() -> None:
    pager = Pager()
    pager.count = 3
    assert pager.index(100.0) == 0
    pager.on_key("right")  # user takes over from what's on screen (page 1 -> page 2)
    assert pager.manual is True
    assert pager.index(100_000.0) == 1  # stays put; auto-advance no longer drives it
    pager.on_key("right")
    pager.on_key("right")  # would be index 3; clamped to the last page
    assert pager.index(100_000.0) == 2
    pager.on_key("left")
    pager.on_key("left")
    pager.on_key("left")  # would be index -1; clamped to the first page
    assert pager.index(100_000.0) == 0
