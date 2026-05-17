def fps(frames: int, elapsed_seconds: float) -> float:
    if elapsed_seconds <= 0:
        return 0.0
    return float(frames) / float(elapsed_seconds)


def stable_within(values: list[float], tolerance: float) -> bool:
    if not values:
        return False
    return max(values) - min(values) <= tolerance + 1e-9
