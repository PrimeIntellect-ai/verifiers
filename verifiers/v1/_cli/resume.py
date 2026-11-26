"""Resume primitives shared by eval-like CLIs."""


def distribute(
    selected_keys: list[str], owed: dict[str, int], num_results: int
) -> list[int]:
    """Spread each key's owed results over its selected instances, in order."""
    remaining = dict(owed)
    counts: list[int] = []
    for key in selected_keys:
        take = min(num_results, remaining.get(key, 0))
        if take:
            remaining[key] -= take
        counts.append(take)
    return counts
