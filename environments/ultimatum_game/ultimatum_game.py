"""
Ultimatum Game: Two-agent negotiation with asymmetric incentives.

Proposer has $10 and offers a split.
Responder accepts or rejects.
  Accept: both get their share.
  Reject: both get nothing.
"""

import re

from datasets import Dataset

from verifiers.envs.agent import Agent
from verifiers.envs.multiagent_env import MultiAgentEnv
from verifiers.rubrics.multiagent_rubric import MultiAgentRubric
from verifiers.envs.taskset import TaskSet
from verifiers.types import Messages, State
from openai import AsyncOpenAI


# =============================================================================
# Model Configuration
# =============================================================================

TOTAL_AMOUNT = 10


# =============================================================================
# Rubric
# =============================================================================

def create_rubric() -> MultiAgentRubric:
    rubric = MultiAgentRubric()

    def proposer_reward(state, **kwargs) -> float:
        return state.get("extras", {}).get("proposer_payoff", 0.0)

    def responder_reward(state, **kwargs) -> float:
        return state.get("extras", {}).get("responder_payoff", 0.0)

    def deal_rate_metric(state, **kwargs) -> float:
        return 1.0 if state.get("extras", {}).get("accepted", False) else 0.0

    def offer_metric(state, **kwargs) -> float:
        return float(state.get("extras", {}).get("offer", 0))

    rubric.add_actor_reward_func("proposer", proposer_reward, weight=1.0)
    rubric.add_actor_reward_func("responder", responder_reward, weight=1.0)
    rubric.add_reward_func(deal_rate_metric, weight=0.0)
    rubric.add_reward_func(offer_metric, weight=0.0)
    return rubric


# =============================================================================
# TaskSet
# =============================================================================

class UltimatumTask(TaskSet):

    def __init__(self, num_examples: int = -1):
        dataset = self._create_dataset()
        if num_examples > 0:
            dataset = dataset.select(range(min(num_examples, len(dataset))))

        super().__init__(
            name="ultimatum_game",
            dataset=dataset,
            rubric=create_rubric(),
            roles=["proposer", "responder"],
        )

    @staticmethod
    def _create_dataset() -> Dataset:
        items = [
            {
                "prompt": [{"role": "user", "content": "Negotiate a deal."}],
                "answer": "",
                "info": {"total": TOTAL_AMOUNT},
                "example_id": i,
                "task": "ultimatum_game",
            }
            for i in range(20)
        ]
        return Dataset.from_list(items)

    # ---- State ----

    async def setup_state(self, state: State) -> State:
        state["extras"]["offer"] = 0
        state["extras"]["accepted"] = False
        state["extras"]["proposer_payoff"] = 0.0
        state["extras"]["responder_payoff"] = 0.0
        return state

    # ---- Prompts ----

    async def build_prompt(self, role: str, state: State) -> Messages:
        if role == "proposer":
            return [
                {"role": "user", "content": (
                    f"You have ${TOTAL_AMOUNT} to split with another player. "
                    "You must propose how much to give them. "
                    "If they reject your offer, you BOTH get nothing.\n\n"
                    "Make a fair offer so they accept.\n\n"
                    "Output ONLY a single number (0-10) representing how much you offer the other player.\n\n"
                    f"You have ${TOTAL_AMOUNT}. How much do you offer the other player?"
                )},
            ]
        else:  # responder
            offer = state["extras"]["offer"]
            return [
                {"role": "user", "content": (
                    f"Someone is splitting ${TOTAL_AMOUNT} with you. "
                    f"They are offering you ${offer} (they keep ${TOTAL_AMOUNT - offer}).\n\n"
                    "Decide whether this offer is fair enough to accept.\n\n"
                    "Output ONLY 'Accept' or 'Reject'.\n\n"
                    f"/no_think They offer you ${offer} out of ${TOTAL_AMOUNT}. Accept or Reject?"
                )},
            ]

    # ---- Game Logic ----

    async def on_turn_complete(self, state: State) -> None:
        if not state["trajectory"]:
            return

        last_step = state["trajectory"][-1]
        actor_id = last_step["extras"].get("actor_id", "")
        completion = last_step.get("completion", [])
        if not completion:
            return
        content = completion[-1].get("content", "").strip()
        content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL).strip()

        if actor_id == "proposer":
            numbers = re.findall(r'\d+', content)
            if numbers:
                offer = min(int(numbers[0]), TOTAL_AMOUNT)
                offer = max(0, offer)
            else:
                # No number found = invalid offer, proposer gives everything away
                offer = TOTAL_AMOUNT
            state["extras"]["offer"] = offer

        elif actor_id == "responder":
            accepted = "accept" in content.lower()
            state["extras"]["accepted"] = accepted
            offer = state["extras"]["offer"]
            if accepted:
                state["extras"]["proposer_payoff"] = (TOTAL_AMOUNT - offer) / TOTAL_AMOUNT
                state["extras"]["responder_payoff"] = offer / TOTAL_AMOUNT
            else:
                state["extras"]["proposer_payoff"] = 0.0
                state["extras"]["responder_payoff"] = 0.0

    async def should_stop(self, state: State) -> bool:
        actor_history = state.get("extras", {}).get("actor_history", [])
        return len(actor_history) >= 2

    async def on_game_end(self, state: State) -> None:
        pass


# =============================================================================
# Environment Loader
# =============================================================================

def load_environment(num_examples: int = -1, actor_endpoints: dict[str, str] | None = None):
    task = UltimatumTask(num_examples=num_examples)
    actor_endpoints = actor_endpoints or {}

    proposer_url = actor_endpoints.get("proposer")
    responder_url = actor_endpoints.get("responder")

    proposer = Agent(
        id="proposer",
        max_tokens=32,
        is_trainable=True,
        model="Qwen/Qwen3-4B-Instruct-2507",
        client=AsyncOpenAI(base_url=proposer_url, api_key="EMPTY") if proposer_url else None,
    )

    responder = Agent(
        id="responder",
        max_tokens=32,
        is_trainable=True,
        model="Qwen/Qwen3-4B",
        client=AsyncOpenAI(base_url=responder_url, api_key="EMPTY") if responder_url else None,
    )

    return MultiAgentEnv(
        task=task,
        agents={"proposer": proposer, "responder": responder},
        max_turns=4,
    )
