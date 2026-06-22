from verifiers.types import State


def compute_discounted_returns(
    step_rewards: list[float],
    gamma: float = 1.0,
) -> list[float]:
    """Compute γ-discounted future returns for each step in a trajectory."""
    n = len(step_rewards)
    if n == 0:
        return []
    returns = [0.0] * n
    returns[n - 1] = step_rewards[n - 1]
    for t in range(n - 2, -1, -1):
        returns[t] = step_rewards[t] + gamma * returns[t + 1]
    return returns


def apply_step_advantages(
    states: list[State],
    gamma: float = 1.0,
) -> None:
    """Assign per-step discounted advantages normalised across a group."""
    all_returns: list[float] = []
    index_map: list[tuple[int, int]] = []

    for state_idx, state in enumerate(states):
        trajectory = state.get("trajectory", [])
        if not isinstance(trajectory, list):
            continue
        step_rewards = []
        for step in trajectory:
            if not isinstance(step, dict):
                continue
            r = step.get("reward")
            step_rewards.append(float(r) if r is not None else 0.0)

        if not step_rewards:
            continue

        returns = compute_discounted_returns(step_rewards, gamma)
        for step_idx, ret in enumerate(returns):
            all_returns.append(ret)
            index_map.append((state_idx, step_idx))

    if not all_returns:
        return

    n = len(all_returns)
    mean = sum(all_returns) / n
    variance = sum((r - mean) ** 2 for r in all_returns) / n
    std = variance**0.5
    eps = 1e-8

    for pos, (state_idx, step_idx) in enumerate(index_map):
        advantage = (all_returns[pos] - mean) / (std + eps)
        state = states[state_idx]
        trajectory = state["trajectory"]
        trajectory[step_idx]["advantage"] = advantage
        trajectory[step_idx]["reward"] = all_returns[pos]

    for state_idx, state in enumerate(states):
        trajectory = state.get("trajectory", [])
        if not isinstance(trajectory, list) or not trajectory:
            continue
        step_advantages = [
            s["advantage"]
            for s in trajectory
            if isinstance(s, dict) and s.get("advantage") is not None
        ]
        if step_advantages:
            state["advantage"] = sum(step_advantages) / len(step_advantages)
